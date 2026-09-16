#include "niyah/checkpoint.h"
#include "niyah/dataset.h"
#include "niyah/decode.h"
#include "niyah/generate.h"
#include "niyah/optimizer.h"
#include "niyah/tokenizer.h"

#ifdef NIYAH_CLI_ENABLE_CUDA
#include "niyah_cuda_matvec.h"
#endif

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct NiyahPrepareOptions {
    const char *corpus_path;
    const char *tokenizer_out;
    const char *shard_out;
    uint32_t target_vocab_size;
    uint32_t min_pair_frequency;
    size_t sequence_length;
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
} NiyahRunOptions;

static void usage(FILE *stream)
{
    fprintf(stream,
        "Usage:\n"
        "  niyah prepare --corpus FILE --tokenizer-out TOK --shard-out SHARD\n"
        "      --target-vocab N --min-pair-frequency N --sequence-length N\n"
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
           options->sequence_length > 0U;
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

static int prepare_command(int argc, char **argv)
{
    NiyahPrepareOptions options;
    NiyahTokenizerTrainConfig tokenizer_config;
    NiyahTokenizer *tokenizer = NULL;
    NiyahDatasetShard shard;
    uint8_t *corpus = NULL;
    size_t corpus_size = 0U;
    NiyahStatus status;
    int tokenizer_created = 0;
    int shard_created = 0;
    int exit_code = 1;

    memset(&tokenizer_config, 0, sizeof(tokenizer_config));
    memset(&shard, 0, sizeof(shard));

    if (!parse_prepare_options(argc, argv, &options)) {
        usage(stderr);
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

    status = niyah_dataset_shard_build_text(
        tokenizer,
        corpus,
        corpus_size,
        options.sequence_length,
        &shard);
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
        } else {
            return 0;
        }
    }

    return options->tokenizer_path != NULL &&
           options->checkpoint_path != NULL &&
           options->prompt != NULL &&
           options->prompt[0] != '\0' &&
           options->have_max_new_tokens &&
           options->max_new_tokens > 0U;
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
        &prompt_count);
    if (status != NIYAH_OK) {
        exit_code = fail_status("prompt_encode_query", status);
        goto cleanup;
    }

    if (prompt_count == 0U ||
        prompt_count > (size_t)model.config.context_length ||
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

    status = niyah_tokenizer_encode(
        tokenizer,
        (const uint8_t *)options.prompt,
        strlen(options.prompt),
        prompt_tokens,
        prompt_count,
        &prompt_count);
    if (status != NIYAH_OK) {
        exit_code = fail_status("prompt_encode", status);
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

    decoded = (uint8_t *)malloc(
        decoded_size == 0U ? 1U : decoded_size);
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

    if (decoded_size > 0U &&
        fwrite(decoded, 1U, decoded_size, stdout) != decoded_size) {
        exit_code = fail_status("stdout_write", NIYAH_ERR_IO);
        goto cleanup;
    }

    if (fputc('\n', stdout) == EOF) {
        exit_code = fail_status("stdout_write", NIYAH_ERR_IO);
        goto cleanup;
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
        strcmp(argv[1], "run") == 0) {
        return run_command(argc, argv);
    }

    usage(stderr);
    return 2;
}
