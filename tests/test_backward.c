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

static NiyahModelConfig test_config(int tied)
{
    NiyahModelConfig c;
    memset(&c, 0, sizeof(c));
    c.vocab_size = 8U;
    c.context_length = 4U;
    c.embedding_dim = 4U;
    c.n_layers = 1U;
    c.n_heads = 2U;
    c.n_kv_heads = 1U;
    c.ffn_hidden_dim = 8U;
    c.rms_norm_eps = 1.0e-5f;
    c.tie_word_embeddings = tied;
    return c;
}

static int close_grad(float a, float b)
{
    const float scale = fmaxf(fabsf(a), fabsf(b));
    return fabsf(a - b) <= 2.5e-3f + 0.15f * scale;
}

static float canonical_loss(NiyahModel *model,
                            const uint32_t *tokens,
                            const uint32_t *targets,
                            size_t token_count)
{
    size_t ws_count = 0U;
    const size_t logits_count = token_count * (size_t)model->config.vocab_size;
    float *ws;
    float *logits;
    float loss = NAN;

    CHECK(niyah_transformer_workspace_floats(&model->config, token_count, &ws_count) == NIYAH_OK);
    ws = (float *)calloc(ws_count, sizeof(float));
    logits = (float *)calloc(logits_count, sizeof(float));
    CHECK(ws != NULL);
    CHECK(logits != NULL);
    if (ws != NULL && logits != NULL) {
        CHECK(niyah_train_loss(model, tokens, targets, token_count, &loss,
                               logits, logits_count, ws, ws_count) == NIYAH_OK);
    }
    free(ws);
    free(logits);
    return loss;
}

static void test_backward_and_finite_difference(void)
{
    const uint32_t tokens[3] = {1U, 2U, 3U};
    const uint32_t targets[3] = {2U, 3U, 4U};
    NiyahModelConfig config = test_config(0);
    NiyahModel model;
    NiyahModelGradients gradients;
    NiyahLayerLayout layer;
    size_t ws_count = 0U;
    float *ws;
    float loss = 0.0f;
    size_t samples[12];
    size_t sample_count = 0U;
    size_t i;

    CHECK(niyah_model_create(&model, &config) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(&model, UINT64_C(12345)) == NIYAH_OK);
    CHECK(niyah_model_gradients_create(&gradients, &model) == NIYAH_OK);
    CHECK(niyah_train_backward_workspace_floats(&config, 3U, &ws_count) == NIYAH_OK);
    ws = (float *)calloc(ws_count, sizeof(float));
    CHECK(ws != NULL);
    if (ws == NULL) {
        niyah_model_gradients_destroy(&gradients);
        niyah_model_destroy(&model);
        return;
    }

    CHECK(niyah_train_backward(&model, tokens, targets, 3U, &loss,
                               &gradients, ws, ws_count) == NIYAH_OK);
    CHECK(isfinite(loss));
    CHECK(fabsf(loss - canonical_loss(&model, tokens, targets, 3U)) <= 1.0e-6f);
    CHECK(niyah_model_layer_layout(&config, &model.layout, 0U, &layer) == NIYAH_OK);

    samples[sample_count++] = model.layout.token_embedding + 1U * 4U;
    samples[sample_count++] = layer.attn_norm;
    samples[sample_count++] = layer.wq;
    samples[sample_count++] = layer.wk;
    samples[sample_count++] = layer.wv;
    samples[sample_count++] = layer.wo;
    samples[sample_count++] = layer.ffn_norm;
    samples[sample_count++] = layer.w_gate;
    samples[sample_count++] = layer.w_up;
    samples[sample_count++] = layer.w_down;
    samples[sample_count++] = model.layout.final_norm;
    samples[sample_count++] = model.layout.lm_head + 2U * 4U;

    for (i = 0U; i < sample_count; ++i) {
        const size_t index = samples[i];
        const float original = model.weights[index];
        const float eps = 1.0e-3f;
        float plus;
        float minus;
        float numerical;

        model.weights[index] = original + eps;
        plus = canonical_loss(&model, tokens, targets, 3U);
        model.weights[index] = original - eps;
        minus = canonical_loss(&model, tokens, targets, 3U);
        model.weights[index] = original;
        numerical = (plus - minus) / (2.0f * eps);
        if (!close_grad(gradients.values[index], numerical)) {
            fprintf(stderr, "gradient mismatch index=%zu analytic=%g numerical=%g\n",
                    index, (double)gradients.values[index], (double)numerical);
            failures += 1;
        }
    }

    CHECK(niyah_train_backward(&model, tokens, targets, 3U, &loss,
                               &gradients, ws, ws_count - 1U) == NIYAH_ERR_BUFFER_TOO_SMALL);
    free(ws);
    niyah_model_gradients_destroy(&gradients);
    niyah_model_destroy(&model);
}

static void test_tied_embedding_accumulation(void)
{
    const uint32_t tokens[3] = {1U, 2U, 3U};
    const uint32_t targets[3] = {2U, 3U, 4U};
    NiyahModelConfig tied_config = test_config(1);
    NiyahModelConfig untied_config = test_config(0);
    NiyahModel tied;
    NiyahModel untied;
    NiyahModelGradients tied_g;
    NiyahModelGradients untied_g;
    size_t tied_ws_count = 0U;
    size_t untied_ws_count = 0U;
    float *tied_ws;
    float *untied_ws;
    float tied_loss = 0.0f;
    float untied_loss = 0.0f;
    size_t embedding_count;
    size_t i;

    CHECK(niyah_model_create(&tied, &tied_config) == NIYAH_OK);
    CHECK(niyah_model_create(&untied, &untied_config) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(&tied, UINT64_C(777)) == NIYAH_OK);
    memcpy(untied.weights, tied.weights, tied.weight_count * sizeof(float));
    embedding_count = (size_t)tied_config.vocab_size * (size_t)tied_config.embedding_dim;
    memcpy(untied.weights + untied.layout.lm_head,
           tied.weights + tied.layout.token_embedding,
           embedding_count * sizeof(float));

    CHECK(niyah_model_gradients_create(&tied_g, &tied) == NIYAH_OK);
    CHECK(niyah_model_gradients_create(&untied_g, &untied) == NIYAH_OK);
    CHECK(niyah_train_backward_workspace_floats(&tied_config, 3U, &tied_ws_count) == NIYAH_OK);
    CHECK(niyah_train_backward_workspace_floats(&untied_config, 3U, &untied_ws_count) == NIYAH_OK);
    tied_ws = (float *)calloc(tied_ws_count, sizeof(float));
    untied_ws = (float *)calloc(untied_ws_count, sizeof(float));
    CHECK(tied_ws != NULL);
    CHECK(untied_ws != NULL);

    if (tied_ws != NULL && untied_ws != NULL) {
        CHECK(niyah_train_backward(&tied, tokens, targets, 3U, &tied_loss,
                                   &tied_g, tied_ws, tied_ws_count) == NIYAH_OK);
        CHECK(niyah_train_backward(&untied, tokens, targets, 3U, &untied_loss,
                                   &untied_g, untied_ws, untied_ws_count) == NIYAH_OK);
        CHECK(fabsf(tied_loss - untied_loss) <= 1.0e-6f);
        for (i = 0U; i < embedding_count; ++i) {
            const float expected = untied_g.values[untied.layout.token_embedding + i] +
                                   untied_g.values[untied.layout.lm_head + i];
            CHECK(fabsf(tied_g.values[tied.layout.token_embedding + i] - expected) <= 2.0e-5f);
        }
    }

    free(tied_ws);
    free(untied_ws);
    niyah_model_gradients_destroy(&tied_g);
    niyah_model_gradients_destroy(&untied_g);
    niyah_model_destroy(&tied);
    niyah_model_destroy(&untied);
}

static void test_gradient_step_decreases_loss(void)
{
    const uint32_t tokens[3] = {1U, 2U, 3U};
    const uint32_t targets[3] = {2U, 3U, 4U};
    NiyahModelConfig config = test_config(0);
    NiyahModel model;
    NiyahModelGradients gradients;
    size_t ws_count = 0U;
    float *ws;
    float before = 0.0f;
    float after;
    double norm_sq = 0.0;
    float step;
    size_t i;

    CHECK(niyah_model_create(&model, &config) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(&model, UINT64_C(9981)) == NIYAH_OK);
    CHECK(niyah_model_gradients_create(&gradients, &model) == NIYAH_OK);
    CHECK(niyah_train_backward_workspace_floats(&config, 3U, &ws_count) == NIYAH_OK);
    ws = (float *)calloc(ws_count, sizeof(float));
    CHECK(ws != NULL);

    if (ws != NULL) {
        CHECK(niyah_train_backward(&model, tokens, targets, 3U, &before,
                                   &gradients, ws, ws_count) == NIYAH_OK);
        for (i = 0U; i < gradients.count; ++i) {
            norm_sq += (double)gradients.values[i] * (double)gradients.values[i];
        }
        CHECK(norm_sq > 0.0);
        step = 0.01f / (float)(sqrt(norm_sq) + 1.0e-12);
        for (i = 0U; i < model.weight_count; ++i) {
            model.weights[i] -= step * gradients.values[i];
        }
        after = canonical_loss(&model, tokens, targets, 3U);
        CHECK(isfinite(after));
        CHECK(after < before);
    }

    free(ws);
    niyah_model_gradients_destroy(&gradients);
    niyah_model_destroy(&model);
}

int main(void)
{
    test_backward_and_finite_difference();
    test_tied_embedding_accumulation();
    test_gradient_step_decreases_loss();

    if (failures != 0) {
        fprintf(stderr, "niyah_backward_test: %d failure(s)\n", failures);
        return 1;
    }
    puts("niyah_backward_test: PASS");
    return 0;
}
