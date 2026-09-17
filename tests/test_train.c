#include "niyah/niyah.h"
#include "niyah/train.h"
#include "niyah/transformer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        failures += 1; \
    } \
} while (0)

static int nearly_equal(float a, float b, float tolerance)
{
    return fabsf(a - b) <= tolerance;
}

static void test_cross_entropy(void)
{
    const float logits[8] = {0.0f, 0.0f, 0.0f, 0.0f,
                             0.0f, 0.0f, 0.0f, 0.0f};
    const uint32_t targets[2] = {0U, 3U};
    float dlogits[8];
    float loss = 0.0f;
    size_t row;

    CHECK(niyah_cross_entropy_loss(logits, targets, 2U, 4U,
                                   &loss, dlogits, 8U) == NIYAH_OK);
    CHECK(nearly_equal(loss, logf(4.0f), 1.0e-6f));
    CHECK(nearly_equal(dlogits[0], -0.375f, 1.0e-6f));
    CHECK(nearly_equal(dlogits[1], 0.125f, 1.0e-6f));
    CHECK(nearly_equal(dlogits[7], -0.375f, 1.0e-6f));

    for (row = 0U; row < 2U; ++row) {
        float sum = 0.0f;
        size_t v;
        for (v = 0U; v < 4U; ++v) {
            sum += dlogits[row * 4U + v];
        }
        CHECK(nearly_equal(sum, 0.0f, 1.0e-6f));
    }

    CHECK(niyah_cross_entropy_loss(logits, targets, 2U, 4U,
                                   &loss, dlogits, 7U) == NIYAH_ERR_BUFFER_TOO_SMALL);
}

static void test_gradient_buffer(void)
{
    NiyahModelConfig config;
    NiyahModel model;
    NiyahModelGradients gradients;
    size_t i;

    memset(&config, 0, sizeof(config));
    config.vocab_size = 8U;
    config.context_length = 4U;
    config.embedding_dim = 4U;
    config.n_layers = 1U;
    config.n_heads = 2U;
    config.n_kv_heads = 1U;
    config.ffn_hidden_dim = 8U;
    config.rms_norm_eps = 1.0e-5f;
    config.tie_word_embeddings = 0;

    CHECK(niyah_model_create(&model, &config) == NIYAH_OK);
    CHECK(niyah_model_gradients_create(&gradients, &model) == NIYAH_OK);
    CHECK(gradients.count == model.weight_count);
    for (i = 0U; i < gradients.count; ++i) {
        CHECK(gradients.values[i] == 0.0f);
        gradients.values[i] = 1.0f;
    }
    niyah_model_gradients_zero(&gradients);
    for (i = 0U; i < gradients.count; ++i) {
        CHECK(gradients.values[i] == 0.0f);
    }
    niyah_model_gradients_destroy(&gradients);
    CHECK(gradients.values == NULL);
    CHECK(gradients.count == 0U);
    niyah_model_destroy(&model);
}

static void test_train_loss_uses_canonical_forward(void)
{
    NiyahModelConfig config;
    NiyahModel model;
    const uint32_t tokens[3] = {1U, 2U, 3U};
    uint32_t targets[3] = {2U, 3U, 4U};
    size_t workspace_count = 0U;
    const size_t logits_count = 3U * 8U;
    float *workspace_a;
    float *workspace_b;
    float train_logits[24];
    float reference_logits[24];
    float loss = 0.0f;
    float direct_loss = 0.0f;

    memset(&config, 0, sizeof(config));
    config.vocab_size = 8U;
    config.context_length = 4U;
    config.embedding_dim = 4U;
    config.n_layers = 1U;
    config.n_heads = 2U;
    config.n_kv_heads = 1U;
    config.ffn_hidden_dim = 8U;
    config.rms_norm_eps = 1.0e-5f;
    config.tie_word_embeddings = 0;

    CHECK(niyah_model_create(&model, &config) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(&model, UINT64_C(12345)) == NIYAH_OK);
    CHECK(niyah_transformer_workspace_floats(&config, 3U, &workspace_count) == NIYAH_OK);

    workspace_a = (float *)calloc(workspace_count, sizeof(float));
    workspace_b = (float *)calloc(workspace_count, sizeof(float));
    CHECK(workspace_a != NULL);
    CHECK(workspace_b != NULL);
    if (workspace_a == NULL || workspace_b == NULL) {
        free(workspace_a);
        free(workspace_b);
        niyah_model_destroy(&model);
        return;
    }

    CHECK(niyah_train_loss(&model, tokens, targets, 3U, &loss,
                           train_logits, logits_count,
                           workspace_a, workspace_count) == NIYAH_OK);
    CHECK(niyah_transformer_forward(&model, tokens, 3U,
                                    reference_logits, logits_count,
                                    workspace_b, workspace_count) == NIYAH_OK);
    CHECK(memcmp(train_logits, reference_logits, sizeof(train_logits)) == 0);
    CHECK(niyah_cross_entropy_loss(reference_logits, targets, 3U, 8U,
                                   &direct_loss, NULL, 0U) == NIYAH_OK);
    CHECK(nearly_equal(loss, direct_loss, 1.0e-7f));
    CHECK(isfinite(loss));

    targets[1] = 8U;
    CHECK(niyah_train_loss(&model, tokens, targets, 3U, &loss,
                           train_logits, logits_count,
                           workspace_a, workspace_count) == NIYAH_ERR_INVALID_ARGUMENT);

    free(workspace_a);
    free(workspace_b);
    niyah_model_destroy(&model);
}


static void test_masked_cross_entropy(void)
{
    static const float logits[] = {
        3.0f, 1.0f, 0.0f, -1.0f,
        0.0f, 1.0f, 4.0f, -2.0f,
        0.5f, 0.0f, -0.5f, 3.0f
    };
    static const uint32_t targets[] = {
        0U, 2U, 3U
    };

    float full_loss = 0.0f;
    float masked_loss = 0.0f;
    float full_via_mask = 0.0f;
    float d_full[12];
    float d_masked[12];
    float d_full_via_mask[12];
    size_t i;
    int changed = 0;

    CHECK(niyah_cross_entropy_loss(
              logits,
              targets,
              3U,
              4U,
              &full_loss,
              d_full,
              12U) == NIYAH_OK);

    CHECK(niyah_cross_entropy_loss_masked(
              logits,
              targets,
              3U,
              4U,
              0U,
              &full_via_mask,
              d_full_via_mask,
              12U) == NIYAH_OK);

    CHECK(full_loss == full_via_mask);

    for (i = 0U; i < 12U; ++i) {
        CHECK(d_full[i] == d_full_via_mask[i]);
    }

    CHECK(niyah_cross_entropy_loss_masked(
              logits,
              targets,
              3U,
              4U,
              2U,
              &masked_loss,
              d_masked,
              12U) == NIYAH_OK);

    CHECK(isfinite(masked_loss));

    for (i = 0U; i < 8U; ++i) {
        CHECK(d_masked[i] == 0.0f);
    }

    for (i = 8U; i < 12U; ++i) {
        if (d_masked[i] != d_full[i]) {
            changed = 1;
        }
    }

    CHECK(changed != 0);

    CHECK(niyah_cross_entropy_loss_masked(
              logits,
              targets,
              3U,
              4U,
              3U,
              &masked_loss,
              d_masked,
              12U) ==
          NIYAH_ERR_INVALID_ARGUMENT);

    puts("P8F_MASKED_CROSS_ENTROPY=PASS");
}

int main(void)
{
    test_masked_cross_entropy();
    test_cross_entropy();
    test_gradient_buffer();
    test_train_loss_uses_canonical_forward();

    if (failures != 0) {
        fprintf(stderr, "niyah_train_test: %d failure(s)\n", failures);
        return 1;
    }
    puts("niyah_train_test: PASS");
    return 0;
}
