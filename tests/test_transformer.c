#include "niyah/niyah.h"
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

static NiyahModelConfig forward_config(void)
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
    config.tie_word_embeddings = 1;
    return config;
}

static int all_finite(const float *values, size_t count)
{
    size_t i;
    for (i = 0U; i < count; ++i) {
        if (!isfinite(values[i])) {
            return 0;
        }
    }
    return 1;
}

static int test_workspace_contract(void)
{
    NiyahModelConfig config = forward_config();
    size_t count = 0U;

    CHECK(niyah_transformer_workspace_floats(&config, 3U, &count) == NIYAH_OK);
    CHECK(count > 0U);
    CHECK(niyah_transformer_workspace_floats(&config, 0U, &count) == NIYAH_ERR_INVALID_ARGUMENT);
    CHECK(niyah_transformer_workspace_floats(&config, 9U, &count) == NIYAH_ERR_INVALID_ARGUMENT);

    config.embedding_dim = 12U;
    config.n_heads = 4U;
    config.n_kv_heads = 2U;
    CHECK(niyah_model_config_validate(&config) == NIYAH_ERR_INVALID_CONFIG);
    CHECK(niyah_transformer_workspace_floats(&config, 3U, &count) == NIYAH_ERR_INVALID_CONFIG);
    return 0;
}

static int test_forward_determinism_and_causality(void)
{
    NiyahModelConfig config = forward_config();
    NiyahModel model;
    const uint32_t tokens_a[3] = {1U, 2U, 3U};
    const uint32_t tokens_b[3] = {1U, 2U, 4U};
    size_t workspace_count = 0U;
    size_t logits_count = 3U * (size_t)config.vocab_size;
    float *workspace = NULL;
    float *logits_a = NULL;
    float *logits_repeat = NULL;
    float *logits_b = NULL;
    size_t i;
    float final_delta = 0.0f;

    CHECK(niyah_model_create(&model, &config) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(&model, UINT64_C(20260915)) == NIYAH_OK);
    CHECK(niyah_transformer_workspace_floats(&config, 3U, &workspace_count) == NIYAH_OK);

    workspace = (float *)calloc(workspace_count, sizeof(float));
    logits_a = (float *)calloc(logits_count, sizeof(float));
    logits_repeat = (float *)calloc(logits_count, sizeof(float));
    logits_b = (float *)calloc(logits_count, sizeof(float));
    CHECK(workspace != NULL);
    CHECK(logits_a != NULL);
    CHECK(logits_repeat != NULL);
    CHECK(logits_b != NULL);

    CHECK(niyah_transformer_forward(&model,
                                    tokens_a,
                                    3U,
                                    logits_a,
                                    logits_count,
                                    workspace,
                                    workspace_count) == NIYAH_OK);
    CHECK(all_finite(logits_a, logits_count));

    memset(workspace, 0, workspace_count * sizeof(float));
    CHECK(niyah_transformer_forward(&model,
                                    tokens_a,
                                    3U,
                                    logits_repeat,
                                    logits_count,
                                    workspace,
                                    workspace_count) == NIYAH_OK);
    CHECK(memcmp(logits_a, logits_repeat, logits_count * sizeof(float)) == 0);

    memset(workspace, 0, workspace_count * sizeof(float));
    CHECK(niyah_transformer_forward(&model,
                                    tokens_b,
                                    3U,
                                    logits_b,
                                    logits_count,
                                    workspace,
                                    workspace_count) == NIYAH_OK);

    /* A future-token change must not affect logits at earlier causal positions. */
    CHECK(memcmp(logits_a,
                 logits_b,
                 2U * (size_t)config.vocab_size * sizeof(float)) == 0);

    for (i = 0U; i < (size_t)config.vocab_size; ++i) {
        final_delta += fabsf(logits_a[2U * (size_t)config.vocab_size + i] -
                             logits_b[2U * (size_t)config.vocab_size + i]);
    }
    CHECK(final_delta > 1.0e-7f);

    CHECK(niyah_transformer_forward(&model,
                                    tokens_a,
                                    3U,
                                    logits_a,
                                    logits_count - 1U,
                                    workspace,
                                    workspace_count) == NIYAH_ERR_BUFFER_TOO_SMALL);
    CHECK(niyah_transformer_forward(&model,
                                    tokens_a,
                                    3U,
                                    logits_a,
                                    logits_count,
                                    workspace,
                                    workspace_count - 1U) == NIYAH_ERR_BUFFER_TOO_SMALL);

    free(workspace);
    free(logits_a);
    free(logits_repeat);
    free(logits_b);
    niyah_model_destroy(&model);
    return 0;
}

static int test_invalid_token(void)
{
    NiyahModelConfig config = forward_config();
    NiyahModel model;
    uint32_t tokens[1];
    size_t workspace_count = 0U;
    float *workspace;
    float *logits;

    tokens[0] = config.vocab_size;
    CHECK(niyah_model_create(&model, &config) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(&model, UINT64_C(7)) == NIYAH_OK);
    CHECK(niyah_transformer_workspace_floats(&config, 1U, &workspace_count) == NIYAH_OK);
    workspace = (float *)calloc(workspace_count, sizeof(float));
    logits = (float *)calloc((size_t)config.vocab_size, sizeof(float));
    CHECK(workspace != NULL);
    CHECK(logits != NULL);

    CHECK(niyah_transformer_forward(&model,
                                    tokens,
                                    1U,
                                    logits,
                                    (size_t)config.vocab_size,
                                    workspace,
                                    workspace_count) == NIYAH_ERR_INVALID_ARGUMENT);

    free(workspace);
    free(logits);
    niyah_model_destroy(&model);
    return 0;
}

int main(void)
{
    CHECK(test_workspace_contract() == 0);
    CHECK(test_forward_determinism_and_causality() == 0);
    CHECK(test_invalid_token() == 0);

    puts("NIYAH_TRANSFORMER_FORWARD_P2=PASS");
    return 0;
}
