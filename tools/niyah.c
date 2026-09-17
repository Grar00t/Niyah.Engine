#include "niyah/checkpoint.h"
#include "niyah/dataset.h"
#include "niyah/decode.h"
#include "niyah/evidence.h"
#include "niyah/generate.h"
#include "niyah/ir.h"
#include "niyah/receipt.h"
#include "niyah/optimizer.h"
#include "niyah/tokenizer.h"

#ifdef NIYAH_CLI_ENABLE_CUDA
#include "niyah_cuda_matvec.h"
#endif

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NIYAH_PREPARE_RECORD_STREAM 0
#define NIYAH_PREPARE_RECORD_BLANK_LINE 1

typedef struct NiyahPrepareOptions {
    const char *corpus_path;
    const char *tokenizer_out;
    const char *shard_out;
    uint32_t target_vocab_size;
    uint32_t min_pair_frequency;
    size_t sequence_length;
    int record_mode;
    const char **response_delimiters;
    size_t response_delimiter_count;
    int have_target_vocab_size;
    int have_min_pair_frequency;
    int have_sequence_length;
} NiyahPrepareOptions;

typedef struct NiyahRunOptions {
    const char *tokenizer_path;
    const char *checkpoint_path;
    const char *prompt;
    size_t max_new_tokens;
    float temperature;
    uint64_t seed;
    int have_max_new_tokens;
    int use_cuda;
    int execute_ir;
    int emit_receipt;
    const char *evidence_out_path;
} NiyahRunOptions;

typedef struct NiyahVerifyEvidenceOptions {
    const char *file_path;
    const char *checkpoint_path;
    const char *tokenizer_path;
} NiyahVerifyEvidenceOptions;

static void usage(FILE *stream)
{
    fprintf(stream,
        "Usage:\n"
        "  niyah prepare --corpus FILE --tokenizer-out TOK --shard-out SHARD\n"
        "      --target-vocab N --min-pair-frequency N --sequence-length N\n"
        "      [--record-mode stream|blank-line]\n"
        "      [--response-delimiter TEXT ...]\n"
        "\n"
        "  niyah run --tokenizer TOK --checkpoint CKPT --prompt TEXT\n"
        "      --max-new-tokens N [--temperature F] [--seed N]\n"
        "      [--backend cpu|cuda]\n");
}

static const char *status_name(NiyahStatus status)
{
    switch (status) {
        case NIYAH_OK: return "NIYAH_OK";
        case NIYAH_ERR_INVALID_ARGUMENT: return "NIYAH_ERR_INVALID_ARGUMENT";
        case NIYAH_ERR_INVALID_CONFIG: return "NIYAH_ERR_INVALID_CONFIG";
        case NIYAH_ERR_OVERFLOW: return "NIYAH_ERR_OVERFLOW";
        case NIYAH_ERR_OUT_OF_MEMORY: return "NIYAH_ERR_OUT_OF_MEMORY";
        case NIYAH_ERR_BUFFER_TOO_SMALL: return "NIYAH_ERR_BUFFER_TOO_SMALL";
        case NIYAH_ERR_IO: return "NIYAH_ERR_IO";
        case NIYAH_ERR_CORRUPT_DATA: return "NIYAH_ERR_CORRUPT_DATA";
        case NIYAH_ERR_UNSUPPORTED_VERSION: return "NIYAH_ERR_UNSUPPORTED_VERSION";
        default: return "NIYAH_ERR_UNKNOWN";
    }
}

static int fail_status(const char *stage, NiyahStatus status)
{
    fprintf(stderr,
            "error_stage=%s status=%s(%d)\n",
            stage,
            status_name(status),
            (int)status);
    return 1;
}

static int write_hex_field(
    FILE *stream,
    const char *name,
    const uint8_t *bytes,
    size_t size)
{
    size_t i;

    if (stream == NULL ||
        name == NULL ||
        bytes == NULL) {
        return 0;
    }

    if (fprintf(stream, "%s=", name) < 0) {
        return 0;
    }

    for (i = 0U; i < size; ++i) {
        if (fprintf(
                stream,
                "%02x",
                (unsigned)bytes[i]) < 0) {
            return 0;
        }
    }

    return fputc('\n', stream) != EOF;
}

static int write_evidence_document(
    FILE *stream,
    const uint8_t *receipt_hash,
    const uint8_t *checkpoint_hash,
    const uint8_t *tokenizer_hash,
    const uint8_t *evidence_root,
    const char *receipt_text,
    size_t receipt_length)
{
    if (stream == NULL ||
        receipt_hash == NULL ||
        checkpoint_hash == NULL ||
        tokenizer_hash == NULL ||
        evidence_root == NULL ||
        receipt_text == NULL) {
        return 0;
    }

    if (fputs("NIYAH_EVIDENCE_V1\n", stream) == EOF ||
        !write_hex_field(
            stream,
            "receipt_sha256",
            receipt_hash,
            NIYAH_RECEIPT_SHA256_SIZE) ||
        !write_hex_field(
            stream,
            "checkpoint_sha256",
            checkpoint_hash,
            NIYAH_CHECKPOINT_IDENTITY_SHA256_SIZE) ||
        !write_hex_field(
            stream,
            "tokenizer_sha256",
            tokenizer_hash,
            NIYAH_TOKENIZER_IDENTITY_SHA256_SIZE) ||
        !write_hex_field(
            stream,
            "evidence_root_sha256",
            evidence_root,
            NIYAH_EVIDENCE_ROOT_SHA256_SIZE)) {
        return 0;
    }

    return fwrite(
        receipt_text,
        1U,
        receipt_length,
        stream) == receipt_length;
}

static int parse_u64(const char *text, uint64_t *out)
{
    char *end = NULL;
    unsigned long long value;

    if (text == NULL || text[0] == '\0' || text[0] == '-') {
        return 0;
    }

    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return 0;
    }

    *out = (uint64_t)value;
    return 1;
}

static int parse_size(const char *text, size_t *out)
{
    uint64_t value;

    if (!parse_u64(text, &value) ||
        value > (uint64_t)SIZE_MAX) {
        return 0;
    }

    *out = (size_t)value;
    return 1;
}

static int parse_u32(const char *text, uint32_t *out)
{
    uint64_t value;

    if (!parse_u64(text, &value) ||
        value > (uint64_t)UINT32_MAX) {
        return 0;
    }

    *out = (uint32_t)value;
    return 1;
}

static int parse_float_value(const char *text, float *out)
{
    char *end = NULL;
    float value;

    if (text == NULL || text[0] == '\0') {
        return 0;
    }

    errno = 0;
    value = strtof(text, &end);
    if (errno != 0 ||
        end == text ||
        *end != '\0' ||
        !isfinite(value)) {
        return 0;
    }

    *out = value;
    return 1;
}

static const char *next_value(int argc, char **argv, int *index)
{
    if (*index + 1 >= argc) {
        return NULL;
    }

    *index += 1;
    return argv[*index];
}

static int parse_prepare_options(
    int argc,
    char **argv,
    NiyahPrepareOptions *options)
{
    int i;

    memset(options, 0, sizeof(*options));

    options->response_delimiters =
        (const char **)calloc(
            (size_t)argc,
            sizeof(*options->response_delimiters));
    if (options->response_delimiters == NULL) {
        return 0;
    }

    for (i = 2; i < argc; ++i) {
        const char *key = argv[i];
        const char *value;

        if (strcmp(key, "--corpus") == 0) {
            value = next_value(argc, argv, &i);
            if (value == NULL) return 0;
            options->corpus_path = value;
        } else if (strcmp(key, "--tokenizer-out") == 0) {
            value = next_value(argc, argv, &i);
            if (value == NULL) return 0;
            options->tokenizer_out = value;
        } else if (strcmp(key, "--shard-out") == 0) {
            value = next_value(argc, argv, &i);
            if (value == NULL) return 0;
            options->shard_out = value;
        } else if (strcmp(key, "--target-vocab") == 0) {
            value = next_value(argc, argv, &i);
            if (value == NULL ||
                !parse_u32(value, &options->target_vocab_size)) {
                return 0;
            }
            options->have_target_vocab_size = 1;
        } else if (strcmp(key, "--min-pair-frequency") == 0) {
            value = next_value(argc, argv, &i);
            if (value == NULL ||
                !parse_u32(value, &options->min_pair_frequency)) {
                return 0;
            }
            options->have_min_pair_frequency = 1;
        } else if (strcmp(key, "--sequence-length") == 0) {
            value = next_value(argc, argv, &i);
            if (value == NULL ||
                !parse_size(value, &options->sequence_length)) {
                return 0;
            }
            options->have_sequence_length = 1;
        } else if (strcmp(key, "--record-mode") == 0) {
            value = next_value(argc, argv, &i);
            if (value == NULL) return 0;

            if (strcmp(value, "stream") == 0) {
                options->record_mode =
                    NIYAH_PREPARE_RECORD_STREAM;
            } else if (strcmp(value, "blank-line") == 0) {
                options->record_mode =
                    NIYAH_PREPARE_RECORD_BLANK_LINE;
            } else {
                return 0;
            }
        } else if (strcmp(key, "--response-delimiter") == 0) {
            size_t j;

            value = next_value(argc, argv, &i);
            if (value == NULL || value[0] == '\0') {
                return 0;
            }

            for (j = 0U;
                 j < options->response_delimiter_count;
                 ++j) {
                if (strcmp(
                        options->response_delimiters[j],
                        value) == 0) {
                    return 0;
                }
            }

            if (options->response_delimiter_count >=
                (size_t)argc) {
                return 0;
            }

            options->response_delimiters[
                options->response_delimiter_count++] =
                    value;
        } else {
            return 0;
        }
    }

    return options->corpus_path != NULL &&
           options->tokenizer_out != NULL &&
           options->shard_out != NULL &&
           strcmp(options->tokenizer_out, options->shard_out) != 0 &&
           options->have_target_vocab_size &&
           options->target_vocab_size >= NIYAH_TOKENIZER_BASE_VOCAB_SIZE &&
           options->have_min_pair_frequency &&
           options->min_pair_frequency > 0U &&
           options->have_sequence_length &&
           options->sequence_length > 0U &&
           (options->response_delimiter_count == 0U ||
            options->record_mode ==
                NIYAH_PREPARE_RECORD_BLANK_LINE);
}

static FILE *niyah_cli_fopen(const char *path, const char *mode)
{
#if defined(_MSC_VER)
    FILE *file = NULL;
    if (fopen_s(&file, path, mode) != 0) {
        return NULL;
    }
    return file;
#else
    return fopen(path, mode);
#endif
}

static int path_exists(const char *path)
{
    FILE *file;

    if (path == NULL) {
        return 0;
    }

    file = niyah_cli_fopen(path, "rb");
    if (file == NULL) {
        return 0;
    }

    (void)fclose(file);
    return 1;
}

static int read_file_bytes(
    const char *path,
    uint8_t **out_bytes,
    size_t *out_size)
{
    FILE *file = NULL;
    uint8_t *bytes = NULL;
    long end;
    size_t size;

    if (path == NULL || out_bytes == NULL || out_size == NULL) {
        return 0;
    }

    *out_bytes = NULL;
    *out_size = 0U;

    file = niyah_cli_fopen(path, "rb");
    if (file == NULL) {
        return 0;
    }

    if (fseek(file, 0L, SEEK_END) != 0) {
        (void)fclose(file);
        return 0;
    }

    end = ftell(file);
    if (end <= 0L) {
        (void)fclose(file);
        return 0;
    }

    if ((uint64_t)end > (uint64_t)SIZE_MAX) {
        (void)fclose(file);
        return 0;
    }

    size = (size_t)end;

    if (fseek(file, 0L, SEEK_SET) != 0) {
        (void)fclose(file);
        return 0;
    }

    bytes = (uint8_t *)malloc(size);
    if (bytes == NULL) {
        (void)fclose(file);
        return 0;
    }

    if (fread(bytes, 1U, size, file) != size) {
        free(bytes);
        (void)fclose(file);
        return 0;
    }

    if (fclose(file) != 0) {
        free(bytes);
        return 0;
    }

    *out_bytes = bytes;
    *out_size = size;
    return 1;
}

static int blank_line_records(
    const uint8_t *text,
    size_t text_size,
    size_t *offsets,
    size_t *lengths,
    size_t capacity,
    size_t *out_count)
{
    size_t record_start = 0U;
    size_t position = 0U;
    size_t count = 0U;

    if (text == NULL ||
        text_size == 0U ||
        out_count == NULL)
        return 0;

    while (position < text_size) {
        const size_t line_start = position;
        size_t line_end;
        int blank;

        while (position < text_size &&
               text[position] != (uint8_t)'\n')
            position += 1U;

        line_end = position;

        if (line_end > line_start &&
            text[line_end - 1U] == (uint8_t)'\r')
            line_end -= 1U;

        blank = line_end == line_start;

        if (position < text_size)
            position += 1U;

        if (blank) {
            size_t record_end = line_start;

            while (record_end > record_start &&
                   (text[record_end - 1U] == (uint8_t)'\n' ||
                    text[record_end - 1U] == (uint8_t)'\r'))
                record_end -= 1U;

            if (record_end > record_start) {
                if (offsets != NULL) {
                    if (count >= capacity)
                        return 0;
                    offsets[count] = record_start;
                    lengths[count] =
                        record_end - record_start;
                }
                count += 1U;
            }

            record_start = position;
        }
    }

    {
        size_t record_end = text_size;

        while (record_end > record_start &&
               (text[record_end - 1U] == (uint8_t)'\n' ||
                text[record_end - 1U] == (uint8_t)'\r'))
            record_end -= 1U;

        if (record_end > record_start) {
            if (offsets != NULL) {
                if (count >= capacity)
                    return 0;
                offsets[count] = record_start;
                lengths[count] =
                    record_end - record_start;
            }
            count += 1U;
        }
    }

    *out_count = count;
    return count != 0U;
}

static int supervised_records_from_delimiters(
    const uint8_t *text,
    size_t text_size,
    const size_t *record_offsets,
    const size_t *record_lengths,
    size_t record_count,
    const char *const *delimiters,
    size_t delimiter_count,
    NiyahDatasetSupervisedRecord *records)
{
    size_t record_index;

    if (text == NULL ||
        text_size == 0U ||
        record_offsets == NULL ||
        record_lengths == NULL ||
        record_count == 0U ||
        delimiters == NULL ||
        delimiter_count == 0U ||
        records == NULL) {
        return 0;
    }

    for (record_index = 0U;
         record_index < record_count;
         ++record_index) {
        const size_t record_start =
            record_offsets[record_index];
        const size_t record_length =
            record_lengths[record_index];
        size_t record_end;
        size_t matched_end = 0U;
        size_t match_count = 0U;
        size_t delimiter_index;

        if (record_start > text_size ||
            record_length >
                text_size - record_start) {
            return 0;
        }

        record_end =
            record_start + record_length;

        for (delimiter_index = 0U;
             delimiter_index < delimiter_count;
             ++delimiter_index) {
            const char *delimiter =
                delimiters[delimiter_index];
            const size_t delimiter_length =
                strlen(delimiter);
            size_t position;

            if (delimiter_length == 0U ||
                delimiter_length > record_length) {
                continue;
            }

            for (position = record_start;
                 position <=
                     record_end - delimiter_length;
                 ++position) {
                if (memcmp(
                        text + position,
                        delimiter,
                        delimiter_length) == 0) {
                    match_count += 1U;
                    matched_end =
                        position + delimiter_length;

                    if (match_count > 1U) {
                        return 0;
                    }
                }
            }
        }

        if (match_count != 1U ||
            matched_end <= record_start ||
            matched_end >= record_end) {
            return 0;
        }

        records[record_index].prompt_offset =
            record_start;
        records[record_index].prompt_length =
            matched_end - record_start;
        records[record_index].response_offset =
            matched_end;
        records[record_index].response_length =
            record_end - matched_end;
    }

    return 1;
}

static int prepare_command(int argc, char **argv)
{
    NiyahPrepareOptions options;
    NiyahTokenizerTrainConfig tokenizer_config;
    NiyahTokenizer *tokenizer = NULL;
    NiyahDatasetShard shard;
    uint8_t *corpus = NULL;
    size_t corpus_size = 0U;
    size_t *record_offsets = NULL;
    size_t *record_lengths = NULL;
    NiyahDatasetSupervisedRecord *supervised_records = NULL;
    size_t record_count = 0U;
    NiyahStatus status;
    int tokenizer_created = 0;
    int shard_created = 0;
    int exit_code = 1;

    memset(&tokenizer_config, 0, sizeof(tokenizer_config));
    memset(&shard, 0, sizeof(shard));

    if (!parse_prepare_options(argc, argv, &options)) {
        usage(stderr);
        free(options.response_delimiters);
        return 2;
    }

    if (path_exists(options.tokenizer_out) ||
        path_exists(options.shard_out)) {
        fprintf(stderr, "error_stage=output_exists status=NIYAH_ERR_INVALID_ARGUMENT\n");
        return 2;
    }

    if (!read_file_bytes(
            options.corpus_path,
            &corpus,
            &corpus_size)) {
        exit_code = fail_status("corpus_read", NIYAH_ERR_IO);
        goto cleanup;
    }

    tokenizer_config.target_vocab_size =
        options.target_vocab_size;
    tokenizer_config.min_pair_frequency =
        options.min_pair_frequency;

    status = niyah_tokenizer_train(
        corpus,
        corpus_size,
        &tokenizer_config,
        &tokenizer);
    if (status != NIYAH_OK) {
        exit_code = fail_status("tokenizer_train", status);
        goto cleanup;
    }

    if (options.record_mode ==
        NIYAH_PREPARE_RECORD_BLANK_LINE) {
        size_t geometry_bytes;

        if (!blank_line_records(
                corpus,
                corpus_size,
                NULL,
                NULL,
                0U,
                &record_count)) {
            exit_code = fail_status(
                "record_split",
                NIYAH_ERR_INVALID_CONFIG);
            goto cleanup;
        }

        if (record_count > SIZE_MAX / sizeof(size_t)) {
            exit_code = fail_status(
                "record_allocation",
                NIYAH_ERR_OVERFLOW);
            goto cleanup;
        }

        geometry_bytes =
            record_count * sizeof(size_t);

        record_offsets =
            (size_t *)malloc(geometry_bytes);
        record_lengths =
            (size_t *)malloc(geometry_bytes);

        if (record_offsets == NULL ||
            record_lengths == NULL) {
            exit_code = fail_status(
                "record_allocation",
                NIYAH_ERR_OUT_OF_MEMORY);
            goto cleanup;
        }

        if (!blank_line_records(
                corpus,
                corpus_size,
                record_offsets,
                record_lengths,
                record_count,
                &record_count)) {
            exit_code = fail_status(
                "record_split",
                NIYAH_ERR_INVALID_CONFIG);
            goto cleanup;
        }

        if (options.response_delimiter_count != 0U) {
            size_t supervised_bytes;

            if (record_count >
                SIZE_MAX /
                    sizeof(*supervised_records)) {
                exit_code = fail_status(
                    "response_allocation",
                    NIYAH_ERR_OVERFLOW);
                goto cleanup;
            }

            supervised_bytes =
                record_count *
                sizeof(*supervised_records);

            supervised_records =
                (NiyahDatasetSupervisedRecord *)
                    malloc(supervised_bytes);

            if (supervised_records == NULL) {
                exit_code = fail_status(
                    "response_allocation",
                    NIYAH_ERR_OUT_OF_MEMORY);
                goto cleanup;
            }

            if (!supervised_records_from_delimiters(
                    corpus,
                    corpus_size,
                    record_offsets,
                    record_lengths,
                    record_count,
                    options.response_delimiters,
                    options.response_delimiter_count,
                    supervised_records)) {
                exit_code = fail_status(
                    "response_split",
                    NIYAH_ERR_INVALID_CONFIG);
                goto cleanup;
            }

            status =
                niyah_dataset_shard_build_supervised_records(
                    tokenizer,
                    corpus,
                    corpus_size,
                    supervised_records,
                    record_count,
                    options.sequence_length,
                    &shard);
        } else {
            status = niyah_dataset_shard_build_records(
                tokenizer,
                corpus,
                corpus_size,
                record_offsets,
                record_lengths,
                record_count,
                options.sequence_length,
                &shard);
        }
    } else {
        status = niyah_dataset_shard_build_text(
            tokenizer,
            corpus,
            corpus_size,
            options.sequence_length,
            &shard);
    }

    if (status != NIYAH_OK) {
        exit_code = fail_status("shard_build", status);
        goto cleanup;
    }

    status = niyah_tokenizer_save(
        tokenizer,
        options.tokenizer_out);
    if (status != NIYAH_OK) {
        exit_code = fail_status("tokenizer_save", status);
        goto cleanup;
    }
    tokenizer_created = 1;

    status = niyah_dataset_shard_save(
        &shard,
        tokenizer,
        options.shard_out);
    if (status != NIYAH_OK) {
        exit_code = fail_status("shard_save", status);
        goto cleanup;
    }
    shard_created = 1;

    fprintf(
        stdout,
        "P8C_PREPARE=PASS vocab=%zu merges=%zu tokens=%zu samples=%zu\n",
        niyah_tokenizer_vocab_size(tokenizer),
        niyah_tokenizer_merge_count(tokenizer),
        shard.token_count,
        shard.sample_count);

    if (options.response_delimiter_count != 0U) {
        fprintf(
            stdout,
            "P8F_SUPERVISED_PREPARE=PASS delimiters=%zu\n",
            options.response_delimiter_count);
    }

    exit_code = 0;

cleanup:
    if (exit_code != 0) {
        if (shard_created) {
            (void)remove(options.shard_out);
        }
        if (tokenizer_created) {
            (void)remove(options.tokenizer_out);
        }
    }

    niyah_dataset_shard_destroy(&shard);
    niyah_tokenizer_destroy(tokenizer);
    free(supervised_records);
    free(record_lengths);
    free(record_offsets);
    free(options.response_delimiters);
    free(corpus);
    return exit_code;
}

static int parse_run_options(
    int argc,
    char **argv,
    NiyahRunOptions *options)
{
    int i;

    memset(options, 0, sizeof(*options));
    options->temperature = 0.0f;
    options->seed = UINT64_C(1);

    for (i = 2; i < argc; ++i) {
        const char *key = argv[i];
        const char *value;

        if (strcmp(key, "--tokenizer") == 0) {
            value = next_value(argc, argv, &i);
            if (value == NULL) return 0;
            options->tokenizer_path = value;
        } else if (strcmp(key, "--checkpoint") == 0) {
            value = next_value(argc, argv, &i);
            if (value == NULL) return 0;
            options->checkpoint_path = value;
        } else if (strcmp(key, "--prompt") == 0) {
            value = next_value(argc, argv, &i);
            if (value == NULL) return 0;
            options->prompt = value;
        } else if (strcmp(key, "--max-new-tokens") == 0) {
            value = next_value(argc, argv, &i);
            if (value == NULL ||
                !parse_size(value, &options->max_new_tokens)) {
                return 0;
            }
            options->have_max_new_tokens = 1;
        } else if (strcmp(key, "--temperature") == 0) {
            value = next_value(argc, argv, &i);
            if (value == NULL ||
                !parse_float_value(value, &options->temperature)) {
                return 0;
            }
        } else if (strcmp(key, "--seed") == 0) {
            value = next_value(argc, argv, &i);
            if (value == NULL ||
                !parse_u64(value, &options->seed)) {
                return 0;
            }
        } else if (strcmp(key, "--backend") == 0) {
            value = next_value(argc, argv, &i);
            if (value == NULL) {
                return 0;
            }
            if (strcmp(value, "cpu") == 0) {
                options->use_cuda = 0;
            } else if (strcmp(value, "cuda") == 0) {
                options->use_cuda = 1;
            } else {
                return 0;
            }
        } else if (strcmp(key, "--execute-ir") == 0) {
            options->execute_ir = 1;
        } else if (strcmp(key, "--receipt") == 0) {
            options->emit_receipt = 1;
        } else if (strcmp(key, "--evidence-out") == 0) {
            value = next_value(argc, argv, &i);
            if (value == NULL || value[0] == '\0') {
                return 0;
            }
            options->evidence_out_path = value;
        } else {
            return 0;
        }
    }

    return options->tokenizer_path != NULL &&
           options->checkpoint_path != NULL &&
           options->prompt != NULL &&
           options->prompt[0] != '\0' &&
           options->have_max_new_tokens &&
           options->max_new_tokens > 0U &&
           (!options->emit_receipt ||
            options->execute_ir) &&
           (options->evidence_out_path == NULL ||
            options->emit_receipt);
}

static int parse_verify_evidence_options(
    int argc,
    char **argv,
    NiyahVerifyEvidenceOptions *options)
{
    int i;

    memset(options, 0, sizeof(*options));

    for (i = 2; i < argc; ++i) {
        const char *key = argv[i];
        const char *value;

        if (strcmp(key, "--file") == 0) {
            value = next_value(argc, argv, &i);
            if (value == NULL) return 0;
            options->file_path = value;
        } else if (
            strcmp(key, "--checkpoint") == 0) {
            value = next_value(argc, argv, &i);
            if (value == NULL) return 0;
            options->checkpoint_path = value;
        } else if (
            strcmp(key, "--tokenizer") == 0) {
            value = next_value(argc, argv, &i);
            if (value == NULL) return 0;
            options->tokenizer_path = value;
        } else {
            return 0;
        }
    }

    return options->file_path != NULL &&
           options->checkpoint_path != NULL &&
           options->tokenizer_path != NULL;
}

static NiyahStatus write_evidence_file_atomic(
    const char *path,
    const uint8_t *receipt_hash,
    const uint8_t *checkpoint_hash,
    const uint8_t *tokenizer_hash,
    const uint8_t *evidence_root,
    const char *receipt_text,
    size_t receipt_length)
{
    FILE *stream = NULL;
    char *temporary = NULL;
    size_t path_length;
    NiyahStatus status = NIYAH_ERR_IO;

    if (path == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    path_length = strlen(path);

    if (path_length > SIZE_MAX - 5U) {
        return NIYAH_ERR_OVERFLOW;
    }

    temporary = (char *)malloc(path_length + 5U);
    if (temporary == NULL) {
        return NIYAH_ERR_OUT_OF_MEMORY;
    }

    if (snprintf(
            temporary,
            path_length + 5U,
            "%s.tmp",
            path) < 0) {
        free(temporary);
        return NIYAH_ERR_IO;
    }

    (void)remove(temporary);

    stream = fopen(temporary, "wb");
    if (stream == NULL) {
        free(temporary);
        return NIYAH_ERR_IO;
    }

    if (!write_evidence_document(
            stream,
            receipt_hash,
            checkpoint_hash,
            tokenizer_hash,
            evidence_root,
            receipt_text,
            receipt_length)) {
        goto cleanup;
    }

    if (fflush(stream) != 0) {
        goto cleanup;
    }

    if (fclose(stream) != 0) {
        stream = NULL;
        goto cleanup;
    }

    stream = NULL;

    if (rename(temporary, path) != 0) {
        goto cleanup;
    }

    status = NIYAH_OK;

cleanup:
    if (stream != NULL) {
        (void)fclose(stream);
    }

    if (status != NIYAH_OK) {
        (void)remove(temporary);
    }

    free(temporary);
    return status;
}

static int run_command(int argc, char **argv)
{
    NiyahRunOptions options;
    NiyahTokenizer *tokenizer = NULL;
    NiyahModel model;
    NiyahAdamWState optimizer_state;
    NiyahAdamWConfig optimizer_config;
    NiyahKVCache cache;
    NiyahGenerationConfig generation_config;
    NiyahGenerationResult generation_result;
#ifdef NIYAH_CLI_ENABLE_CUDA
    NiyahCudaModelState cuda_model;
    NiyahCudaDecodeState cuda_decode;
#endif
    uint32_t *prompt_tokens = NULL;
    uint32_t *generated_tokens = NULL;
    float *workspace = NULL;
#ifdef NIYAH_CLI_ENABLE_CUDA
    float *cuda_logits = NULL;
#endif
    uint8_t *decoded = NULL;
    size_t prompt_count = 0U;
    size_t encoded_prompt_count = 0U;
    size_t workspace_count = 0U;
    size_t decoded_size = 0U;
    size_t vocab_size;
    size_t generated_bytes;
    NiyahStatus status;
    int exit_code = 1;

    memset(&model, 0, sizeof(model));
    memset(&optimizer_state, 0, sizeof(optimizer_state));
    memset(&optimizer_config, 0, sizeof(optimizer_config));
    memset(&cache, 0, sizeof(cache));
    memset(&generation_config, 0, sizeof(generation_config));
    memset(&generation_result, 0, sizeof(generation_result));
#ifdef NIYAH_CLI_ENABLE_CUDA
    memset(&cuda_model, 0, sizeof(cuda_model));
    memset(&cuda_decode, 0, sizeof(cuda_decode));
#endif

    if (!parse_run_options(argc, argv, &options)) {
        usage(stderr);
        return 2;
    }

    if (options.evidence_out_path != NULL &&
        path_exists(options.evidence_out_path)) {
        fprintf(
            stderr,
            "error_stage=evidence_output_exists "
            "status=NIYAH_ERR_INVALID_ARGUMENT\n");
        return 2;
    }

#ifndef NIYAH_CLI_ENABLE_CUDA
    if (options.use_cuda) {
        fprintf(stderr, "backend_unavailable=cuda\n");
        return 2;
    }
#endif

    status = niyah_tokenizer_load(
        options.tokenizer_path,
        &tokenizer);
    if (status != NIYAH_OK) {
        return fail_status("tokenizer_load", status);
    }

    status = niyah_checkpoint_load_with_tokenizer(
        options.checkpoint_path,
        tokenizer,
        &model,
        &optimizer_state,
        &optimizer_config);
    if (status != NIYAH_OK) {
        exit_code = fail_status("checkpoint_load", status);
        goto cleanup;
    }

    vocab_size = niyah_tokenizer_vocab_size(tokenizer);
    if (vocab_size == 0U ||
        vocab_size != (size_t)model.config.vocab_size) {
        exit_code = fail_status(
            "runtime_compatibility",
            NIYAH_ERR_INVALID_CONFIG);
        goto cleanup;
    }

    status = niyah_tokenizer_encode(
        tokenizer,
        (const uint8_t *)options.prompt,
        strlen(options.prompt),
        NULL,
        0U,
        &encoded_prompt_count);
    if (status != NIYAH_OK) {
        exit_code = fail_status("prompt_encode_query", status);
        goto cleanup;
    }

    if (encoded_prompt_count == 0U ||
        encoded_prompt_count == SIZE_MAX) {
        exit_code = fail_status(
            "context_capacity",
            NIYAH_ERR_INVALID_ARGUMENT);
        goto cleanup;
    }

    prompt_count = encoded_prompt_count + 1U;

    if (prompt_count > (size_t)model.config.context_length ||
        options.max_new_tokens >
            (size_t)model.config.context_length - prompt_count ||
        prompt_count > SIZE_MAX / sizeof(*prompt_tokens)) {
        exit_code = fail_status(
            "context_capacity",
            NIYAH_ERR_INVALID_ARGUMENT);
        goto cleanup;
    }

    prompt_tokens = (uint32_t *)malloc(
        prompt_count * sizeof(*prompt_tokens));
    if (prompt_tokens == NULL) {
        exit_code = fail_status(
            "prompt_allocation",
            NIYAH_ERR_OUT_OF_MEMORY);
        goto cleanup;
    }

    prompt_tokens[0] = NIYAH_TOKEN_BOS;

    status = niyah_tokenizer_encode(
        tokenizer,
        (const uint8_t *)options.prompt,
        strlen(options.prompt),
        prompt_tokens + 1U,
        encoded_prompt_count,
        &encoded_prompt_count);
    if (status != NIYAH_OK) {
        exit_code = fail_status("prompt_encode", status);
        goto cleanup;
    }

    if (encoded_prompt_count + 1U != prompt_count) {
        exit_code = fail_status(
            "prompt_encode_contract",
            NIYAH_ERR_INVALID_CONFIG);
        goto cleanup;
    }

    if (options.max_new_tokens >
        SIZE_MAX / sizeof(*generated_tokens)) {
        exit_code = fail_status(
            "output_allocation",
            NIYAH_ERR_OVERFLOW);
        goto cleanup;
    }

    generated_bytes =
        options.max_new_tokens * sizeof(*generated_tokens);
    generated_tokens = (uint32_t *)calloc(
        1U,
        generated_bytes);
    if (generated_tokens == NULL) {
        exit_code = fail_status(
            "output_allocation",
            NIYAH_ERR_OUT_OF_MEMORY);
        goto cleanup;
    }

    generation_config.max_new_tokens =
        options.max_new_tokens;
    generation_config.eos_token = NIYAH_TOKEN_EOS;
    generation_config.stop_on_eos = 1;
    generation_config.sampler.temperature =
        options.temperature;
    generation_config.sampler.seed =
        options.seed;

    if (options.use_cuda) {
#ifdef NIYAH_CLI_ENABLE_CUDA
        if (niyah_cuda_model_state_create(
                &cuda_model,
                &model) != 0) {
            fprintf(stderr, "error_stage=cuda_model_create status=CUDA_ERROR\n");
            goto cleanup;
        }

        if (niyah_cuda_decode_state_create(
                &cuda_decode,
                &cuda_model) != 0) {
            fprintf(stderr, "error_stage=cuda_decode_create status=CUDA_ERROR\n");
            goto cleanup;
        }

        if (vocab_size > SIZE_MAX / sizeof(*cuda_logits)) {
            exit_code = fail_status(
                "cuda_logits_allocation",
                NIYAH_ERR_OVERFLOW);
            goto cleanup;
        }

        cuda_logits = (float *)calloc(
            vocab_size,
            sizeof(*cuda_logits));
        if (cuda_logits == NULL) {
            exit_code = fail_status(
                "cuda_logits_allocation",
                NIYAH_ERR_OUT_OF_MEMORY);
            goto cleanup;
        }

        status = niyah_cuda_generate(
            &cuda_model,
            &cuda_decode,
            prompt_tokens,
            prompt_count,
            &generation_config,
            generated_tokens,
            options.max_new_tokens,
            &generation_result,
            cuda_logits,
            vocab_size);
        if (status != NIYAH_OK) {
            exit_code = fail_status("cuda_generation", status);
            goto cleanup;
        }
#else
        fprintf(stderr, "backend_unavailable=cuda\n");
        exit_code = 2;
        goto cleanup;
#endif
    } else {
        status = niyah_kv_cache_create(
            &cache,
            &model.config);
        if (status != NIYAH_OK) {
            exit_code = fail_status("kv_cache_create", status);
            goto cleanup;
        }

        status = niyah_generation_workspace_floats(
            &model.config,
            &workspace_count);
        if (status != NIYAH_OK) {
            exit_code = fail_status("workspace_query", status);
            goto cleanup;
        }

        if (workspace_count >
            SIZE_MAX / sizeof(*workspace)) {
            exit_code = fail_status(
                "workspace_allocation",
                NIYAH_ERR_OVERFLOW);
            goto cleanup;
        }

        workspace = (float *)calloc(
            workspace_count,
            sizeof(*workspace));
        if (workspace == NULL) {
            exit_code = fail_status(
                "workspace_allocation",
                NIYAH_ERR_OUT_OF_MEMORY);
            goto cleanup;
        }

        status = niyah_generate(
            &model,
            &cache,
            prompt_tokens,
            prompt_count,
            &generation_config,
            generated_tokens,
            options.max_new_tokens,
            &generation_result,
            workspace,
            workspace_count);
        if (status != NIYAH_OK) {
            exit_code = fail_status("generation", status);
            goto cleanup;
        }
    }

    status = niyah_tokenizer_decode(
        tokenizer,
        generated_tokens,
        generation_result.generated_tokens,
        NULL,
        0U,
        &decoded_size);
    if (status != NIYAH_OK) {
        exit_code = fail_status("decode_query", status);
        goto cleanup;
    }

    if (decoded_size == SIZE_MAX) {
        exit_code = fail_status(
            "decode_allocation",
            NIYAH_ERR_OVERFLOW);
        goto cleanup;
    }

    decoded = (uint8_t *)malloc(decoded_size + 1U);
    if (decoded == NULL) {
        exit_code = fail_status(
            "decode_allocation",
            NIYAH_ERR_OUT_OF_MEMORY);
        goto cleanup;
    }

    status = niyah_tokenizer_decode(
        tokenizer,
        generated_tokens,
        generation_result.generated_tokens,
        decoded,
        decoded_size,
        &decoded_size);
    if (status != NIYAH_OK) {
        exit_code = fail_status("decode", status);
        goto cleanup;
    }

    decoded[decoded_size] = '\0';

    if (options.execute_ir) {
        NiyahIr ir;
        int64_t result;

        if (memchr(decoded, '\0', decoded_size) != NULL) {
            exit_code = fail_status(
                "ir_text",
                NIYAH_ERR_INVALID_ARGUMENT);
            goto cleanup;
        }

        status = niyah_ir_parse(
            (const char *)decoded,
            &ir);
        if (status != NIYAH_OK) {
            exit_code = fail_status("ir_parse", status);
            goto cleanup;
        }

        status = niyah_ir_execute(&ir, &result);
        if (status != NIYAH_OK) {
            exit_code = fail_status("ir_execute", status);
            goto cleanup;
        }

        if (options.emit_receipt) {
            NiyahExecutionReceipt receipt;
            char receipt_text[256];
            size_t receipt_length = 0U;

            uint8_t receipt_hash[
                NIYAH_RECEIPT_SHA256_SIZE];

            uint8_t checkpoint_hash[
                NIYAH_CHECKPOINT_IDENTITY_SHA256_SIZE];

            uint8_t tokenizer_hash[
                NIYAH_TOKENIZER_IDENTITY_SHA256_SIZE];

            uint8_t evidence_root[
                NIYAH_EVIDENCE_ROOT_SHA256_SIZE];

            receipt.ir = ir;
            receipt.result = result;

            status = niyah_receipt_format(
                &receipt,
                receipt_text,
                sizeof(receipt_text),
                &receipt_length);

            if (status != NIYAH_OK) {
                exit_code = fail_status(
                    "receipt_format",
                    status);
                goto cleanup;
            }

            status = niyah_receipt_sha256(
                &receipt,
                receipt_hash);

            if (status != NIYAH_OK) {
                exit_code = fail_status(
                    "receipt_hash",
                    status);
                goto cleanup;
            }

            status = niyah_checkpoint_identity_sha256(
                options.checkpoint_path,
                checkpoint_hash);

            if (status != NIYAH_OK) {
                exit_code = fail_status(
                    "checkpoint_identity",
                    status);
                goto cleanup;
            }

            status = niyah_tokenizer_identity_sha256(
                tokenizer,
                tokenizer_hash);

            if (status != NIYAH_OK) {
                exit_code = fail_status(
                    "tokenizer_identity",
                    status);
                goto cleanup;
            }

            status = niyah_evidence_root_sha256(
                receipt_hash,
                checkpoint_hash,
                tokenizer_hash,
                evidence_root);

            if (status != NIYAH_OK) {
                exit_code = fail_status(
                    "evidence_root",
                    status);
                goto cleanup;
            }

            if (options.evidence_out_path != NULL) {
                status = write_evidence_file_atomic(
                    options.evidence_out_path,
                    receipt_hash,
                    checkpoint_hash,
                    tokenizer_hash,
                    evidence_root,
                    receipt_text,
                    receipt_length);

                if (status != NIYAH_OK) {
                    exit_code = fail_status(
                        "evidence_write",
                        status);
                    goto cleanup;
                }
            }

            if (!write_evidence_document(
                    stdout,
                    receipt_hash,
                    checkpoint_hash,
                    tokenizer_hash,
                    evidence_root,
                    receipt_text,
                    receipt_length)) {
                exit_code = fail_status(
                    "stdout_write",
                    NIYAH_ERR_IO);
                goto cleanup;
            }
        } else {
            if (fprintf(
                    stdout,
                    "%" PRId64 "\n",
                    result) < 0) {
                exit_code = fail_status(
                    "stdout_write",
                    NIYAH_ERR_IO);
                goto cleanup;
            }
        }
    } else {
        if (decoded_size > 0U &&
            fwrite(decoded, 1U, decoded_size, stdout) != decoded_size) {
            exit_code = fail_status("stdout_write", NIYAH_ERR_IO);
            goto cleanup;
        }

        if (fputc('\n', stdout) == EOF) {
            exit_code = fail_status("stdout_write", NIYAH_ERR_IO);
            goto cleanup;
        }
    }

    exit_code = 0;

cleanup:
    free(decoded);
#ifdef NIYAH_CLI_ENABLE_CUDA
    free(cuda_logits);
    niyah_cuda_decode_state_destroy(&cuda_decode);
    niyah_cuda_model_state_destroy(&cuda_model);
#endif
    free(workspace);
    free(generated_tokens);
    free(prompt_tokens);
    niyah_kv_cache_destroy(&cache);
    niyah_adamw_state_destroy(&optimizer_state);
    niyah_model_destroy(&model);
    niyah_tokenizer_destroy(tokenizer);
    return exit_code;
}

static int verify_evidence_command(
    int argc,
    char **argv)
{
    NiyahVerifyEvidenceOptions options;
    NiyahEvidenceDocument document;
    NiyahTokenizer *tokenizer = NULL;
    uint8_t *artifact = NULL;
    size_t artifact_size = 0U;

    uint8_t checkpoint_identity[
        NIYAH_CHECKPOINT_IDENTITY_SHA256_SIZE];

    uint8_t tokenizer_identity[
        NIYAH_TOKENIZER_IDENTITY_SHA256_SIZE];

    NiyahStatus status;
    int exit_code = 1;

    memset(&document, 0, sizeof(document));

    if (!parse_verify_evidence_options(
            argc,
            argv,
            &options)) {
        usage(stderr);
        return 2;
    }

    if (!read_file_bytes(
            options.file_path,
            &artifact,
            &artifact_size)) {
        return fail_status(
            "evidence_read",
            NIYAH_ERR_IO);
    }

    status = niyah_evidence_parse_document(
        (const char *)artifact,
        artifact_size,
        &document);

    if (status != NIYAH_OK) {
        exit_code = fail_status(
            "evidence_parse",
            status);
        goto cleanup;
    }

    status = niyah_checkpoint_identity_sha256(
        options.checkpoint_path,
        checkpoint_identity);

    if (status != NIYAH_OK) {
        exit_code = fail_status(
            "checkpoint_identity",
            status);
        goto cleanup;
    }

    status = niyah_tokenizer_load(
        options.tokenizer_path,
        &tokenizer);

    if (status != NIYAH_OK) {
        exit_code = fail_status(
            "tokenizer_load",
            status);
        goto cleanup;
    }

    status = niyah_tokenizer_identity_sha256(
        tokenizer,
        tokenizer_identity);

    if (status != NIYAH_OK) {
        exit_code = fail_status(
            "tokenizer_identity",
            status);
        goto cleanup;
    }

    status = niyah_evidence_verify_document(
        &document,
        checkpoint_identity,
        tokenizer_identity);

    if (status != NIYAH_OK) {
        exit_code = fail_status(
            "evidence_verify",
            status);
        goto cleanup;
    }

    if (fputs("VALID\n", stdout) == EOF) {
        exit_code = fail_status(
            "stdout_write",
            NIYAH_ERR_IO);
        goto cleanup;
    }

    exit_code = 0;

cleanup:
    niyah_tokenizer_destroy(tokenizer);
    free(artifact);
    return exit_code;
}

int main(int argc, char **argv)
{
    if (argc == 2 &&
        strcmp(argv[1], "--help") == 0) {
        usage(stdout);
        return 0;
    }

    if (argc >= 2 &&
        strcmp(argv[1], "prepare") == 0) {
        return prepare_command(argc, argv);
    }

    if (argc >= 2 &&
        strcmp(argv[1], "verify-evidence") == 0) {
        return verify_evidence_command(
            argc,
            argv);
    }

    if (argc >= 2 &&
        strcmp(argv[1], "run") == 0) {
        return run_command(argc, argv);
    }

    usage(stderr);
    return 2;
}
