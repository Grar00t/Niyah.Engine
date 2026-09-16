#include "niyah/checkpoint.h"
#include "niyah/dataset.h"
#include "niyah/optimizer.h"
#include "niyah/tokenizer.h"
#include "niyah/training_loop.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NIYAH_TRAIN_MODE_NONE 0
#define NIYAH_TRAIN_MODE_NEW 1
#define NIYAH_TRAIN_MODE_RESUME 2

typedef struct NiyahTrainOptions {
    int mode;
    const char *tokenizer_path;
    const char **shard_paths;
    size_t shard_count;
    size_t shard_capacity;
    const char *checkpoint_in;
    const char *cursor_in;
    const char *checkpoint_out;
    const char *cursor_out;
    size_t updates;
    size_t batch_size;
    size_t accumulation_steps;
    uint64_t model_seed;
    uint64_t data_seed;
    int have_model_seed;
    int have_data_seed;
    NiyahModelConfig model_config;
    NiyahAdamWConfig optimizer_config;
    int have_context_length;
    int have_embedding_dim;
    int have_layers;
    int have_heads;
    int have_kv_heads;
    int have_ffn_hidden_dim;
    int have_rms_norm_eps;
    int have_tie_word_embeddings;
    int have_learning_rate;
    int have_beta1;
    int have_beta2;
    int have_epsilon;
    int have_weight_decay;
    int have_max_grad_norm;
} NiyahTrainOptions;

static void train_options_destroy(NiyahTrainOptions *options)
{
    if (options == NULL) return;
    free(options->shard_paths);
    memset(options, 0, sizeof(*options));
}

static void usage(FILE *stream)
{
    fprintf(stream,
        "Usage:\n"
        "  niyah-train new --tokenizer TOK --shard SHARD [--shard SHARD ...] --checkpoint-out CKPT --cursor-out CURSOR\n"
        "      --updates N --batch-size N --accumulation-steps N\n"
        "      --model-seed N --data-seed N --context-length N --embedding-dim N\n"
        "      --layers N --heads N --kv-heads N --ffn-hidden-dim N\n"
        "      --rms-norm-eps F --tie-word-embeddings 0|1\n"
        "      --learning-rate F --beta1 F --beta2 F --epsilon F\n"
        "      --weight-decay F --max-grad-norm F\n"
        "\n"
        "  niyah-train resume --tokenizer TOK --shard SHARD [--shard SHARD ...]\n"
        "      --checkpoint-in CKPT --cursor-in CURSOR\n"
        "      --checkpoint-out CKPT --cursor-out CURSOR\n"
        "      --updates N --batch-size N --accumulation-steps N\n"
        "\n"
        "Outputs are required to be new paths. Resume inputs are never overwritten.\n");
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
    fprintf(stderr, "error_stage=%s status=%s(%d)\n",
            stage, status_name(status), (int)status);
    return 1;
}

static int parse_u64(const char *text, uint64_t *out)
{
    char *end = NULL;
    unsigned long long value;
    if (text == NULL || text[0] == '\0' || text[0] == '-') return 0;
    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') return 0;
    *out = (uint64_t)value;
    return 1;
}

static int parse_size(const char *text, size_t *out)
{
    uint64_t value;
    if (!parse_u64(text, &value) || value > (uint64_t)SIZE_MAX) return 0;
    *out = (size_t)value;
    return 1;
}

static int parse_u32(const char *text, uint32_t *out)
{
    uint64_t value;
    if (!parse_u64(text, &value) || value > (uint64_t)UINT32_MAX) return 0;
    *out = (uint32_t)value;
    return 1;
}

static int parse_float_value(const char *text, float *out)
{
    char *end = NULL;
    float value;
    if (text == NULL || text[0] == '\0') return 0;
    errno = 0;
    value = strtof(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(value)) return 0;
    *out = value;
    return 1;
}

static const char *next_value(int argc, char **argv, int *index)
{
    if (*index + 1 >= argc) return NULL;
    *index += 1;
    return argv[*index];
}

static int parse_options(int argc, char **argv, NiyahTrainOptions *options)
{
    int i;
    memset(options, 0, sizeof(*options));

    if (argc == 2 && strcmp(argv[1], "--help") == 0) {
        usage(stdout);
        return 2;
    }
    if (argc < 2) return 0;
    if (strcmp(argv[1], "new") == 0) options->mode = NIYAH_TRAIN_MODE_NEW;
    else if (strcmp(argv[1], "resume") == 0) options->mode = NIYAH_TRAIN_MODE_RESUME;
    else return 0;

    options->shard_capacity = (size_t)argc;
    options->shard_paths = (const char **)calloc(
        options->shard_capacity, sizeof(*options->shard_paths));
    if (options->shard_paths == NULL) return 0;

    for (i = 2; i < argc; ++i) {
        const char *key = argv[i];
        const char *value;
        if (strcmp(key, "--tokenizer") == 0) {
            value = next_value(argc, argv, &i); if (value == NULL) return 0;
            options->tokenizer_path = value;
        } else if (strcmp(key, "--shard") == 0) {
            value = next_value(argc, argv, &i); if (value == NULL) return 0;
            if (options->shard_count >= options->shard_capacity) return 0;
            options->shard_paths[options->shard_count++] = value;
        } else if (strcmp(key, "--checkpoint-in") == 0) {
            value = next_value(argc, argv, &i); if (value == NULL) return 0;
            options->checkpoint_in = value;
        } else if (strcmp(key, "--cursor-in") == 0) {
            value = next_value(argc, argv, &i); if (value == NULL) return 0;
            options->cursor_in = value;
        } else if (strcmp(key, "--checkpoint-out") == 0) {
            value = next_value(argc, argv, &i); if (value == NULL) return 0;
            options->checkpoint_out = value;
        } else if (strcmp(key, "--cursor-out") == 0) {
            value = next_value(argc, argv, &i); if (value == NULL) return 0;
            options->cursor_out = value;
        } else if (strcmp(key, "--updates") == 0) {
            value = next_value(argc, argv, &i);
            if (value == NULL || !parse_size(value, &options->updates)) return 0;
        } else if (strcmp(key, "--batch-size") == 0) {
            value = next_value(argc, argv, &i);
            if (value == NULL || !parse_size(value, &options->batch_size)) return 0;
        } else if (strcmp(key, "--accumulation-steps") == 0) {
            value = next_value(argc, argv, &i);
            if (value == NULL || !parse_size(value, &options->accumulation_steps)) return 0;
        } else if (strcmp(key, "--model-seed") == 0) {
            value = next_value(argc, argv, &i);
            if (value == NULL || !parse_u64(value, &options->model_seed)) return 0;
            options->have_model_seed = 1;
        } else if (strcmp(key, "--data-seed") == 0) {
            value = next_value(argc, argv, &i);
            if (value == NULL || !parse_u64(value, &options->data_seed)) return 0;
            options->have_data_seed = 1;
        } else if (strcmp(key, "--context-length") == 0) {
            value = next_value(argc, argv, &i);
            if (value == NULL || !parse_u32(value, &options->model_config.context_length)) return 0;
            options->have_context_length = 1;
        } else if (strcmp(key, "--embedding-dim") == 0) {
            value = next_value(argc, argv, &i);
            if (value == NULL || !parse_u32(value, &options->model_config.embedding_dim)) return 0;
            options->have_embedding_dim = 1;
        } else if (strcmp(key, "--layers") == 0) {
            value = next_value(argc, argv, &i);
            if (value == NULL || !parse_u32(value, &options->model_config.n_layers)) return 0;
            options->have_layers = 1;
        } else if (strcmp(key, "--heads") == 0) {
            value = next_value(argc, argv, &i);
            if (value == NULL || !parse_u32(value, &options->model_config.n_heads)) return 0;
            options->have_heads = 1;
        } else if (strcmp(key, "--kv-heads") == 0) {
            value = next_value(argc, argv, &i);
            if (value == NULL || !parse_u32(value, &options->model_config.n_kv_heads)) return 0;
            options->have_kv_heads = 1;
        } else if (strcmp(key, "--ffn-hidden-dim") == 0) {
            value = next_value(argc, argv, &i);
            if (value == NULL || !parse_u32(value, &options->model_config.ffn_hidden_dim)) return 0;
            options->have_ffn_hidden_dim = 1;
        } else if (strcmp(key, "--rms-norm-eps") == 0) {
            value = next_value(argc, argv, &i);
            if (value == NULL || !parse_float_value(value, &options->model_config.rms_norm_eps)) return 0;
            options->have_rms_norm_eps = 1;
        } else if (strcmp(key, "--tie-word-embeddings") == 0) {
            uint32_t tie;
            value = next_value(argc, argv, &i);
            if (value == NULL || !parse_u32(value, &tie) || tie > 1U) return 0;
            options->model_config.tie_word_embeddings = (int)tie;
            options->have_tie_word_embeddings = 1;
        } else if (strcmp(key, "--learning-rate") == 0) {
            value = next_value(argc, argv, &i);
            if (value == NULL || !parse_float_value(value, &options->optimizer_config.learning_rate)) return 0;
            options->have_learning_rate = 1;
        } else if (strcmp(key, "--beta1") == 0) {
            value = next_value(argc, argv, &i);
            if (value == NULL || !parse_float_value(value, &options->optimizer_config.beta1)) return 0;
            options->have_beta1 = 1;
        } else if (strcmp(key, "--beta2") == 0) {
            value = next_value(argc, argv, &i);
            if (value == NULL || !parse_float_value(value, &options->optimizer_config.beta2)) return 0;
            options->have_beta2 = 1;
        } else if (strcmp(key, "--epsilon") == 0) {
            value = next_value(argc, argv, &i);
            if (value == NULL || !parse_float_value(value, &options->optimizer_config.epsilon)) return 0;
            options->have_epsilon = 1;
        } else if (strcmp(key, "--weight-decay") == 0) {
            value = next_value(argc, argv, &i);
            if (value == NULL || !parse_float_value(value, &options->optimizer_config.weight_decay)) return 0;
            options->have_weight_decay = 1;
        } else if (strcmp(key, "--max-grad-norm") == 0) {
            value = next_value(argc, argv, &i);
            if (value == NULL || !parse_float_value(value, &options->optimizer_config.max_grad_norm)) return 0;
            options->have_max_grad_norm = 1;
        } else {
            return 0;
        }
    }
    return 1;
}

static int new_fields_present(const NiyahTrainOptions *o)
{
    return o->have_model_seed || o->have_data_seed ||
           o->have_context_length || o->have_embedding_dim ||
           o->have_layers || o->have_heads || o->have_kv_heads ||
           o->have_ffn_hidden_dim || o->have_rms_norm_eps ||
           o->have_tie_word_embeddings || o->have_learning_rate ||
           o->have_beta1 || o->have_beta2 || o->have_epsilon ||
           o->have_weight_decay || o->have_max_grad_norm;
}

static int validate_options(const NiyahTrainOptions *o)
{
    if (o->tokenizer_path == NULL || o->shard_paths == NULL ||
        o->shard_count == 0U ||
        o->checkpoint_out == NULL || o->cursor_out == NULL ||
        o->updates == 0U || o->batch_size == 0U ||
        o->accumulation_steps == 0U)
        return 0;
    if (o->batch_size > SIZE_MAX / o->accumulation_steps)
        return 0;
    if (strcmp(o->checkpoint_out, o->cursor_out) == 0)
        return 0;

    if (o->mode == NIYAH_TRAIN_MODE_NEW) {
        return o->checkpoint_in == NULL && o->cursor_in == NULL &&
               o->have_model_seed && o->have_data_seed &&
               o->have_context_length && o->have_embedding_dim &&
               o->have_layers && o->have_heads && o->have_kv_heads &&
               o->have_ffn_hidden_dim && o->have_rms_norm_eps &&
               o->have_tie_word_embeddings && o->have_learning_rate &&
               o->have_beta1 && o->have_beta2 && o->have_epsilon &&
               o->have_weight_decay && o->have_max_grad_norm;
    }

    if (o->mode == NIYAH_TRAIN_MODE_RESUME) {
        if (o->checkpoint_in == NULL || o->cursor_in == NULL ||
            new_fields_present(o))
            return 0;
        if (strcmp(o->checkpoint_in, o->checkpoint_out) == 0 ||
            strcmp(o->cursor_in, o->cursor_out) == 0)
            return 0;
        return 1;
    }
    return 0;
}

static int path_exists(const char *path)
{
    FILE *file = fopen(path, "rb");
    if (file == NULL) return 0;
    (void)fclose(file);
    return 1;
}

static int validate_sample_context(const NiyahTrainingSample *samples,
                                   size_t sample_count,
                                   uint32_t context_length)
{
    size_t i;
    for (i = 0U; i < sample_count; ++i) {
        if (samples[i].token_count == 0U ||
            samples[i].token_count > (size_t)context_length)
            return 0;
    }
    return 1;
}

int main(int argc, char **argv)
{
    NiyahTrainOptions options;
    NiyahTokenizer *tokenizer = NULL;
    NiyahDatasetShard *shards = NULL;
    NiyahTrainingSample *samples = NULL;
    NiyahDatasetCursor cursor;
    NiyahModel model;
    NiyahAdamWState optimizer_state;
    NiyahAdamWConfig optimizer_config;
    size_t sample_count = 0U;
    size_t sample_bytes = 0U;
    size_t shard_index;
    size_t sample_offset = 0U;
    size_t vocab_size;
    uint8_t dataset_identity[NIYAH_DATASET_IDENTITY_SHA256_SIZE];
    float mean_loss = 0.0f;
    NiyahStatus status;
    int parsed;
    int exit_code = 1;

    memset(&cursor, 0, sizeof(cursor));
    memset(&model, 0, sizeof(model));
    memset(&optimizer_state, 0, sizeof(optimizer_state));
    memset(&optimizer_config, 0, sizeof(optimizer_config));

    parsed = parse_options(argc, argv, &options);
    if (parsed == 2) {
        train_options_destroy(&options);
        return 0;
    }
    if (parsed == 0 || !validate_options(&options)) {
        usage(stderr);
        train_options_destroy(&options);
        return 2;
    }
    if (path_exists(options.checkpoint_out) || path_exists(options.cursor_out)) {
        fprintf(stderr, "output_path_exists=1\n");
        train_options_destroy(&options);
        return 2;
    }

    status = niyah_tokenizer_load(options.tokenizer_path, &tokenizer);
    if (status != NIYAH_OK) {
        train_options_destroy(&options);
        return fail_status("tokenizer_load", status);
    }

    if (options.shard_count > SIZE_MAX / sizeof(*shards)) {
        exit_code = fail_status("shard_allocation", NIYAH_ERR_OVERFLOW);
        goto cleanup;
    }
    shards = (NiyahDatasetShard *)calloc(options.shard_count, sizeof(*shards));
    if (shards == NULL) {
        exit_code = fail_status("shard_allocation", NIYAH_ERR_OUT_OF_MEMORY);
        goto cleanup;
    }

    for (shard_index = 0U; shard_index < options.shard_count; ++shard_index) {
        status = niyah_dataset_shard_load(
            options.shard_paths[shard_index], tokenizer, &shards[shard_index]);
        if (status != NIYAH_OK) {
            exit_code = fail_status("shard_load", status);
            goto cleanup;
        }
    }

    if (options.shard_count == 1U) {
        status = niyah_dataset_shard_identity_sha256(
            &shards[0U], dataset_identity);
    } else {
        status = niyah_dataset_collection_identity_sha256(
            shards, options.shard_count, dataset_identity);
    }
    if (status != NIYAH_OK) {
        exit_code = fail_status("dataset_identity", status);
        goto cleanup;
    }

    sample_count = 0U;
    for (shard_index = 0U; shard_index < options.shard_count; ++shard_index) {
        size_t shard_samples = 0U;
        status = niyah_training_samples_from_shard(
            &shards[shard_index], NULL, 0U, &shard_samples);
        if (status != NIYAH_OK) {
            exit_code = fail_status("sample_query", status);
            goto cleanup;
        }
        if (sample_count > SIZE_MAX - shard_samples) {
            exit_code = fail_status("sample_count", NIYAH_ERR_OVERFLOW);
            goto cleanup;
        }
        sample_count += shard_samples;
    }
    if (sample_count > SIZE_MAX / sizeof(*samples)) {
        exit_code = fail_status("sample_allocation", NIYAH_ERR_OVERFLOW);
        goto cleanup;
    }
    sample_bytes = sample_count * sizeof(*samples);
    samples = (NiyahTrainingSample *)calloc(1U, sample_bytes);
    if (samples == NULL) {
        exit_code = fail_status("sample_allocation", NIYAH_ERR_OUT_OF_MEMORY);
        goto cleanup;
    }
    sample_offset = 0U;
    for (shard_index = 0U; shard_index < options.shard_count; ++shard_index) {
        size_t shard_samples = 0U;
        status = niyah_training_samples_from_shard(
            &shards[shard_index],
            samples + sample_offset,
            sample_count - sample_offset,
            &shard_samples);
        if (status != NIYAH_OK) {
            exit_code = fail_status("sample_build", status);
            goto cleanup;
        }
        sample_offset += shard_samples;
    }
    if (sample_offset != sample_count) {
        exit_code = fail_status("sample_build", NIYAH_ERR_INVALID_CONFIG);
        goto cleanup;
    }

    if (options.mode == NIYAH_TRAIN_MODE_NEW) {
        vocab_size = niyah_tokenizer_vocab_size(tokenizer);
        if (vocab_size == 0U || vocab_size > (size_t)UINT32_MAX) {
            exit_code = fail_status("vocab_size", NIYAH_ERR_INVALID_CONFIG);
            goto cleanup;
        }
        options.model_config.vocab_size = (uint32_t)vocab_size;
        status = niyah_model_create(&model, &options.model_config);
        if (status != NIYAH_OK) {
            exit_code = fail_status("model_create", status);
            goto cleanup;
        }
        if (!validate_sample_context(samples, sample_count,
                                     model.config.context_length)) {
            exit_code = fail_status("sample_context", NIYAH_ERR_INVALID_CONFIG);
            goto cleanup;
        }
        status = niyah_model_reset_parameters(&model, options.model_seed);
        if (status != NIYAH_OK) {
            exit_code = fail_status("model_init", status);
            goto cleanup;
        }
        status = niyah_adamw_config_validate(&options.optimizer_config);
        if (status != NIYAH_OK) {
            exit_code = fail_status("optimizer_config", status);
            goto cleanup;
        }
        status = niyah_adamw_state_create(&optimizer_state, &model);
        if (status != NIYAH_OK) {
            exit_code = fail_status("optimizer_create", status);
            goto cleanup;
        }
        optimizer_config = options.optimizer_config;
        status = niyah_dataset_cursor_init(
            &cursor, sample_count, options.data_seed);
        if (status != NIYAH_OK) {
            exit_code = fail_status("cursor_create", status);
            goto cleanup;
        }
        status = niyah_dataset_cursor_bind_identity(
            &cursor, dataset_identity);
        if (status != NIYAH_OK) {
            exit_code = fail_status("cursor_bind", status);
            goto cleanup;
        }
    } else {
        status = niyah_checkpoint_load_with_tokenizer(
            options.checkpoint_in, tokenizer,
            &model, &optimizer_state, &optimizer_config);
        if (status != NIYAH_OK) {
            exit_code = fail_status("checkpoint_load", status);
            goto cleanup;
        }
        status = niyah_dataset_cursor_load(options.cursor_in, &cursor);
        if (status != NIYAH_OK) {
            exit_code = fail_status("cursor_load", status);
            goto cleanup;
        }
        if (cursor.sample_count != sample_count ||
            cursor.has_dataset_identity == 0 ||
            memcmp(cursor.dataset_identity, dataset_identity,
                   NIYAH_DATASET_IDENTITY_SHA256_SIZE) != 0 ||
            !validate_sample_context(samples, sample_count,
                                     model.config.context_length)) {
            exit_code = fail_status("resume_compatibility",
                                    NIYAH_ERR_INVALID_CONFIG);
            goto cleanup;
        }
    }

    status = niyah_training_run_updates(
        &model, samples, sample_count, &cursor,
        &optimizer_state, &optimizer_config,
        options.batch_size, options.accumulation_steps,
        options.updates, &mean_loss);
    if (status != NIYAH_OK) {
        exit_code = fail_status("training", status);
        goto cleanup;
    }

    status = niyah_checkpoint_save_with_tokenizer(
        options.checkpoint_out, &model, &optimizer_state,
        &optimizer_config, tokenizer);
    if (status != NIYAH_OK) {
        exit_code = fail_status("checkpoint_save", status);
        goto cleanup;
    }
    status = niyah_dataset_cursor_save(&cursor, options.cursor_out);
    if (status != NIYAH_OK) {
        (void)remove(options.checkpoint_out);
        (void)remove(options.cursor_out);
        exit_code = fail_status("cursor_save", status);
        goto cleanup;
    }

    printf("mode=%s\n",
           options.mode == NIYAH_TRAIN_MODE_NEW ? "new" : "resume");
    printf("shards=%zu\n", options.shard_count);
    printf("samples=%zu\n", sample_count);
    printf("updates=%zu\n", options.updates);
    printf("batch_size=%zu\n", options.batch_size);
    printf("accumulation_steps=%zu\n", options.accumulation_steps);
    printf("optimizer_step=%" PRIu64 "\n", optimizer_state.step);
    printf("cursor_epoch=%" PRIu64 "\n", cursor.epoch);
    printf("cursor_position=%zu\n", cursor.position);
    printf("mean_loss=%.9g\n", (double)mean_loss);
    printf("checkpoint_out=%s\n", options.checkpoint_out);
    printf("cursor_out=%s\n", options.cursor_out);
    exit_code = 0;

cleanup:
    niyah_dataset_cursor_destroy(&cursor);
    niyah_adamw_state_destroy(&optimizer_state);
    niyah_model_destroy(&model);
    free(samples);
    if (shards != NULL) {
        for (shard_index = 0U; shard_index < options.shard_count; ++shard_index)
            niyah_dataset_shard_destroy(&shards[shard_index]);
    }
    free(shards);
    niyah_tokenizer_destroy(tokenizer);
    train_options_destroy(&options);
    return exit_code;
}
