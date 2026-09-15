#include "niyah/decode.h"
#include "niyah/transformer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expr) \
    do { \
        if (!(expr)) { \
            fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
            return 1; \
        } \
    } while (0)

static NiyahModelConfig tiny_config(void)
{
    NiyahModelConfig config;
    memset(&config, 0, sizeof(config));
    config.vocab_size = 32U;
    config.context_length = 8U;
    config.embedding_dim = 8U;
    config.n_layers = 2U;
    config.n_heads = 4U;
    config.n_kv_heads = 2U;
    config.ffn_hidden_dim = 16U;
    config.rms_norm_eps = 1.0e-5f;
    config.tie_word_embeddings = 0;
    return config;
}

static int logits_close(const float *a, const float *b, size_t n, float tolerance)
{
    size_t i;
    for (i = 0U; i < n; ++i) {
        if (!isfinite(a[i]) || !isfinite(b[i]) || fabsf(a[i] - b[i]) > tolerance) {
            fprintf(stderr,
                    "logit mismatch index=%zu a=%.9g b=%.9g diff=%.9g\n",
                    i,
                    (double)a[i],
                    (double)b[i],
                    (double)fabsf(a[i] - b[i]));
            return 0;
        }
    }
    return 1;
}

static int test_incremental_matches_full_forward(void)
{
    const uint32_t tokens[] = {3U, 5U, 7U, 9U};
    const size_t token_count = sizeof(tokens) / sizeof(tokens[0]);
    NiyahModelConfig config = tiny_config();
    NiyahModel model;
    NiyahKVCache cache;
    size_t full_workspace_count = 0U;
    size_t decode_workspace_count = 0U;
    size_t full_logits_count = token_count * (size_t)config.vocab_size;
    float *full_workspace = NULL;
    float *decode_workspace = NULL;
    float *full_logits = NULL;
    float *step_logits = NULL;
    size_t position;

    memset(&model, 0, sizeof(model));
    memset(&cache, 0, sizeof(cache));

    CHECK(niyah_model_create(&model, &config) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(&model, UINT64_C(0x12345678)) == NIYAH_OK);
    CHECK(niyah_kv_cache_create(&cache, &config) == NIYAH_OK);
    CHECK(niyah_transformer_workspace_floats(&config, token_count, &full_workspace_count) == NIYAH_OK);
    CHECK(niyah_decode_workspace_floats(&config, &decode_workspace_count) == NIYAH_OK);

    full_workspace = (float *)calloc(full_workspace_count, sizeof(float));
    decode_workspace = (float *)calloc(decode_workspace_count, sizeof(float));
    full_logits = (float *)calloc(full_logits_count, sizeof(float));
    step_logits = (float *)calloc((size_t)config.vocab_size, sizeof(float));
    CHECK(full_workspace != NULL);
    CHECK(decode_workspace != NULL);
    CHECK(full_logits != NULL);
    CHECK(step_logits != NULL);

    CHECK(niyah_transformer_forward(&model,
                                    tokens,
                                    token_count,
                                    full_logits,
                                    full_logits_count,
                                    full_workspace,
                                    full_workspace_count) == NIYAH_OK);

    for (position = 0U; position < token_count; ++position) {
        memset(step_logits, 0, (size_t)config.vocab_size * sizeof(float));
        CHECK(niyah_transformer_decode_token(&model,
                                             &cache,
                                             tokens[position],
                                             step_logits,
                                             (size_t)config.vocab_size,
                                             decode_workspace,
                                             decode_workspace_count) == NIYAH_OK);
        CHECK(niyah_kv_cache_position(&cache) == position + 1U);
        CHECK(logits_close(step_logits,
                           full_logits + position * (size_t)config.vocab_size,
                           (size_t)config.vocab_size,
                           2.0e-5f));
    }

    free(step_logits);
    free(full_logits);
    free(decode_workspace);
    free(full_workspace);
    niyah_kv_cache_destroy(&cache);
    niyah_model_destroy(&model);
    return 0;
}

static int test_reset_bounds_and_fail_closed_position(void)
{
    NiyahModelConfig config = tiny_config();
    NiyahModel model;
    NiyahKVCache cache;
    size_t workspace_count = 0U;
    float *workspace = NULL;
    float *logits = NULL;
    float first_logits[32];
    size_t i;

    memset(&model, 0, sizeof(model));
    memset(&cache, 0, sizeof(cache));
    memset(first_logits, 0, sizeof(first_logits));

    CHECK(niyah_model_create(&model, &config) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(&model, UINT64_C(77)) == NIYAH_OK);
    CHECK(niyah_kv_cache_create(&cache, &config) == NIYAH_OK);
    CHECK(niyah_decode_workspace_floats(&config, &workspace_count) == NIYAH_OK);

    workspace = (float *)calloc(workspace_count, sizeof(float));
    logits = (float *)calloc((size_t)config.vocab_size, sizeof(float));
    CHECK(workspace != NULL);
    CHECK(logits != NULL);

    CHECK(niyah_transformer_decode_token(&model,
                                         &cache,
                                         config.vocab_size,
                                         logits,
                                         (size_t)config.vocab_size,
                                         workspace,
                                         workspace_count) == NIYAH_ERR_INVALID_ARGUMENT);
    CHECK(niyah_kv_cache_position(&cache) == 0U);

    CHECK(niyah_transformer_decode_token(&model,
                                         &cache,
                                         1U,
                                         logits,
                                         (size_t)config.vocab_size,
                                         workspace,
                                         workspace_count - 1U) == NIYAH_ERR_BUFFER_TOO_SMALL);
    CHECK(niyah_kv_cache_position(&cache) == 0U);

    CHECK(niyah_transformer_decode_token(&model,
                                         &cache,
                                         1U,
                                         logits,
                                         (size_t)config.vocab_size,
                                         workspace,
                                         workspace_count) == NIYAH_OK);
    memcpy(first_logits, logits, sizeof(first_logits));
    CHECK(niyah_kv_cache_position(&cache) == 1U);

    niyah_kv_cache_reset(&cache);
    CHECK(niyah_kv_cache_position(&cache) == 0U);
    memset(logits, 0, (size_t)config.vocab_size * sizeof(float));
    CHECK(niyah_transformer_decode_token(&model,
                                         &cache,
                                         1U,
                                         logits,
                                         (size_t)config.vocab_size,
                                         workspace,
                                         workspace_count) == NIYAH_OK);
    CHECK(logits_close(first_logits, logits, (size_t)config.vocab_size, 0.0f));

    niyah_kv_cache_reset(&cache);
    for (i = 0U; i < (size_t)config.context_length; ++i) {
        CHECK(niyah_transformer_decode_token(&model,
                                             &cache,
                                             (uint32_t)(i % (size_t)config.vocab_size),
                                             logits,
                                             (size_t)config.vocab_size,
                                             workspace,
                                             workspace_count) == NIYAH_OK);
    }
    CHECK(niyah_kv_cache_position(&cache) == (size_t)config.context_length);
    CHECK(niyah_transformer_decode_token(&model,
                                         &cache,
                                         0U,
                                         logits,
                                         (size_t)config.vocab_size,
                                         workspace,
                                         workspace_count) == NIYAH_ERR_INVALID_ARGUMENT);
    CHECK(niyah_kv_cache_position(&cache) == (size_t)config.context_length);

    free(logits);
    free(workspace);
    niyah_kv_cache_destroy(&cache);
    niyah_model_destroy(&model);
    return 0;
}

static int test_cache_model_mismatch(void)
{
    NiyahModelConfig cache_config = tiny_config();
    NiyahModelConfig model_config = tiny_config();
    NiyahModel model;
    NiyahKVCache cache;
    size_t workspace_count = 0U;
    float *workspace = NULL;
    float logits[32];

    memset(&model, 0, sizeof(model));
    memset(&cache, 0, sizeof(cache));
    memset(logits, 0, sizeof(logits));

    cache_config.n_heads = 2U;
    cache_config.n_kv_heads = 2U;
    CHECK(niyah_kv_cache_create(&cache, &cache_config) == NIYAH_OK);
    CHECK(niyah_model_create(&model, &model_config) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(&model, UINT64_C(99)) == NIYAH_OK);
    CHECK(niyah_decode_workspace_floats(&model_config, &workspace_count) == NIYAH_OK);
    workspace = (float *)calloc(workspace_count, sizeof(float));
    CHECK(workspace != NULL);

    CHECK(niyah_transformer_decode_token(&model,
                                         &cache,
                                         1U,
                                         logits,
                                         32U,
                                         workspace,
                                         workspace_count) == NIYAH_ERR_INVALID_CONFIG);
    CHECK(niyah_kv_cache_position(&cache) == 0U);

    free(workspace);
    niyah_model_destroy(&model);
    niyah_kv_cache_destroy(&cache);
    return 0;
}

int main(void)
{
    CHECK(test_incremental_matches_full_forward() == 0);
    CHECK(test_reset_bounds_and_fail_closed_position() == 0);
    CHECK(test_cache_model_mismatch() == 0);

    puts("NIYAH_KV_DECODE_P3=PASS");
    return 0;
}
