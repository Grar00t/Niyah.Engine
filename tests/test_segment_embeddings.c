#include "niyah/checkpoint.h"
#include "niyah/decode.h"
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

static NiyahModelConfig test_config(uint32_t n_segments)
{
    NiyahModelConfig c;
    memset(&c, 0, sizeof(c));
    c.vocab_size = 16U;
    c.context_length = 8U;
    c.embedding_dim = 8U;
    c.n_layers = 2U;
    c.n_heads = 2U;
    c.n_kv_heads = 1U;
    c.ffn_hidden_dim = 16U;
    c.rms_norm_eps = 1.0e-5f;
    c.tie_word_embeddings = 1;
    c.n_segments = n_segments;
    return c;
}

static void test_layout_offsets(void)
{
    NiyahModelConfig disabled = test_config(0U);
    NiyahModelConfig enabled = test_config(3U);
    NiyahModelLayout layout_disabled;
    NiyahModelLayout layout_enabled;

    CHECK(niyah_model_layout_compute(&disabled, &layout_disabled) == NIYAH_OK);
    CHECK(niyah_model_layout_compute(&enabled, &layout_enabled) == NIYAH_OK);

    /* disabled: segment region is empty, placed right after token_embedding */
    CHECK(layout_disabled.segment_embedding ==
          layout_disabled.token_embedding + (size_t)disabled.vocab_size * disabled.embedding_dim);
    CHECK(layout_disabled.layers == layout_disabled.segment_embedding);

    /* enabled: segment region holds n_segments * embedding_dim floats */
    CHECK(layout_enabled.segment_embedding ==
          layout_enabled.token_embedding + (size_t)enabled.vocab_size * enabled.embedding_dim);
    CHECK(layout_enabled.layers ==
          layout_enabled.segment_embedding + (size_t)enabled.n_segments * enabled.embedding_dim);
    CHECK(layout_enabled.total_floats ==
          layout_disabled.total_floats + (size_t)enabled.n_segments * enabled.embedding_dim);
}

static void test_forward_rejections(void)
{
    NiyahModelConfig config = test_config(0U);
    NiyahModel model;
    const uint32_t tokens[2] = {1U, 2U};
    const uint32_t segments[2] = {0U, 0U};
    size_t ws_count = 0U;
    size_t logits_count = 2U * config.vocab_size;
    float *ws;
    float *logits;

    memset(&model, 0, sizeof(model));
    CHECK(niyah_model_create(&model, &config) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(&model, UINT64_C(1)) == NIYAH_OK);
    CHECK(niyah_transformer_workspace_floats(&config, 2U, &ws_count) == NIYAH_OK);
    ws = (float *)calloc(ws_count, sizeof(float));
    logits = (float *)calloc(logits_count, sizeof(float));
    CHECK(ws != NULL && logits != NULL);
    if (ws != NULL && logits != NULL) {
        /* n_segments == 0: any segment usage must be rejected */
        CHECK(niyah_transformer_forward_with_segments(
                  &model, tokens, 2U, segments, logits, logits_count, ws, ws_count) ==
              NIYAH_ERR_INVALID_ARGUMENT);
        CHECK(niyah_transformer_forward_with_segments(
                  &model, tokens, 2U, NULL, logits, logits_count, ws, ws_count) ==
              NIYAH_ERR_INVALID_ARGUMENT);
    }
    free(logits);
    free(ws);
    niyah_model_destroy(&model);
}

static void test_forward_changes_with_segment_id(void)
{
    NiyahModelConfig config = test_config(2U);
    NiyahModel model;
    const uint32_t tokens[3] = {1U, 2U, 3U};
    const uint32_t segments_a[3] = {0U, 0U, 0U};
    const uint32_t segments_b[3] = {1U, 1U, 1U};
    const uint32_t segments_bad[3] = {0U, 2U, 0U};
    size_t ws_count = 0U;
    size_t logits_count = 3U * config.vocab_size;
    float *ws;
    float *logits_a;
    float *logits_b;
    int differs = 0;
    size_t i;

    memset(&model, 0, sizeof(model));
    CHECK(niyah_model_create(&model, &config) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(&model, UINT64_C(2)) == NIYAH_OK);
    CHECK(niyah_transformer_workspace_floats(&config, 3U, &ws_count) == NIYAH_OK);
    ws = (float *)calloc(ws_count, sizeof(float));
    logits_a = (float *)calloc(logits_count, sizeof(float));
    logits_b = (float *)calloc(logits_count, sizeof(float));
    CHECK(ws != NULL && logits_a != NULL && logits_b != NULL);
    if (ws != NULL && logits_a != NULL && logits_b != NULL) {
        CHECK(niyah_transformer_forward_with_segments(
                  &model, tokens, 3U, segments_a, logits_a, logits_count, ws, ws_count) ==
              NIYAH_OK);
        memset(ws, 0, ws_count * sizeof(float));
        CHECK(niyah_transformer_forward_with_segments(
                  &model, tokens, 3U, segments_b, logits_b, logits_count, ws, ws_count) ==
              NIYAH_OK);
        for (i = 0U; i < logits_count; ++i) {
            if (logits_a[i] != logits_b[i]) {
                differs = 1;
                break;
            }
        }
        CHECK(differs != 0);

        CHECK(niyah_transformer_forward_with_segments(
                  &model, tokens, 3U, segments_bad, logits_a, logits_count, ws, ws_count) ==
              NIYAH_ERR_INVALID_ARGUMENT);
    }
    free(logits_b);
    free(logits_a);
    free(ws);
    niyah_model_destroy(&model);
}

static void test_decode_matches_forward(void)
{
    NiyahModelConfig config = test_config(2U);
    NiyahModel model;
    NiyahKVCache cache;
    const uint32_t token = 5U;
    const uint32_t segment_id = 1U;
    const uint32_t segment_ids[1] = {1U};
    size_t forward_ws_count = 0U;
    size_t decode_ws_count = 0U;
    size_t logits_count = config.vocab_size;
    float *forward_ws;
    float *decode_ws;
    float *forward_logits;
    float *decode_logits;
    size_t i;

    memset(&model, 0, sizeof(model));
    memset(&cache, 0, sizeof(cache));
    CHECK(niyah_model_create(&model, &config) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(&model, UINT64_C(3)) == NIYAH_OK);
    CHECK(niyah_kv_cache_create(&cache, &config) == NIYAH_OK);
    CHECK(niyah_transformer_workspace_floats(&config, 1U, &forward_ws_count) == NIYAH_OK);
    CHECK(niyah_decode_workspace_floats(&config, &decode_ws_count) == NIYAH_OK);

    forward_ws = (float *)calloc(forward_ws_count, sizeof(float));
    decode_ws = (float *)calloc(decode_ws_count, sizeof(float));
    forward_logits = (float *)calloc(logits_count, sizeof(float));
    decode_logits = (float *)calloc(logits_count, sizeof(float));
    CHECK(forward_ws != NULL && decode_ws != NULL &&
          forward_logits != NULL && decode_logits != NULL);
    if (forward_ws != NULL && decode_ws != NULL &&
        forward_logits != NULL && decode_logits != NULL) {
        CHECK(niyah_transformer_forward_with_segments(
                  &model, &token, 1U, segment_ids,
                  forward_logits, logits_count, forward_ws, forward_ws_count) == NIYAH_OK);
        CHECK(niyah_transformer_decode_token_with_segment(
                  &model, &cache, token, segment_id,
                  decode_logits, logits_count, decode_ws, decode_ws_count) == NIYAH_OK);
        for (i = 0U; i < logits_count; ++i) {
            CHECK(fabsf(forward_logits[i] - decode_logits[i]) <= 1.0e-4f);
        }
    }

    free(decode_logits);
    free(forward_logits);
    free(decode_ws);
    free(forward_ws);
    niyah_kv_cache_destroy(&cache);
    niyah_model_destroy(&model);
}

static float loss_for_segment_weight(NiyahModel *model,
                                     const uint32_t *tokens,
                                     const uint32_t *targets,
                                     const uint32_t *segment_ids,
                                     size_t token_count,
                                     size_t weight_index,
                                     float delta)
{
    size_t ws_count = 0U;
    size_t logits_count = token_count * model->config.vocab_size;
    float *ws;
    float *logits;
    float loss = NAN;
    float original = model->weights[weight_index];

    model->weights[weight_index] = original + delta;
    if (niyah_transformer_workspace_floats(&model->config, token_count, &ws_count) == NIYAH_OK) {
        ws = (float *)calloc(ws_count, sizeof(float));
        logits = (float *)calloc(logits_count, sizeof(float));
        if (ws != NULL && logits != NULL) {
            if (niyah_transformer_forward_with_segments(
                    model, tokens, token_count, segment_ids,
                    logits, logits_count, ws, ws_count) == NIYAH_OK) {
                float direct_loss = NAN;
                (void)niyah_cross_entropy_loss(logits, targets, token_count,
                                               model->config.vocab_size,
                                               &direct_loss, NULL, 0U);
                loss = direct_loss;
            }
        }
        free(logits);
        free(ws);
    }
    model->weights[weight_index] = original;
    return loss;
}

static void test_segment_gradient_finite_difference(void)
{
    NiyahModelConfig config = test_config(2U);
    NiyahModel model;
    NiyahModelGradients gradients;
    const uint32_t tokens[3] = {1U, 2U, 3U};
    const uint32_t targets[3] = {2U, 3U, 4U};
    const uint32_t segment_ids[3] = {0U, 1U, 1U};
    size_t ws_count = 0U;
    float *ws;
    float loss = 0.0f;
    size_t index;
    const float eps = 1.0e-3f;

    memset(&model, 0, sizeof(model));
    memset(&gradients, 0, sizeof(gradients));
    CHECK(niyah_model_create(&model, &config) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(&model, UINT64_C(4)) == NIYAH_OK);
    CHECK(niyah_model_gradients_create(&gradients, &model) == NIYAH_OK);
    CHECK(niyah_train_backward_workspace_floats(&config, 3U, &ws_count) == NIYAH_OK);
    ws = (float *)calloc(ws_count, sizeof(float));
    CHECK(ws != NULL);
    if (ws != NULL) {
        CHECK(niyah_train_backward_masked_with_segments(
                  &model, tokens, targets, 3U, 0U, segment_ids,
                  &loss, &gradients, ws, ws_count) == NIYAH_OK);
        CHECK(isfinite(loss));

        /* sample one float inside the segment_embedding region (segment 1, dim 3) */
        index = model.layout.segment_embedding + 1U * config.embedding_dim + 3U;
        {
            const float plus = loss_for_segment_weight(
                &model, tokens, targets, segment_ids, 3U, index, eps);
            const float minus = loss_for_segment_weight(
                &model, tokens, targets, segment_ids, 3U, index, -eps);
            const float numerical = (plus - minus) / (2.0f * eps);
            const float analytic = gradients.values[index];
            const float scale = fmaxf(fabsf(analytic), fabsf(numerical));
            CHECK(isfinite(plus) && isfinite(minus));
            CHECK(fabsf(analytic - numerical) <= 2.5e-3f + 0.15f * scale);
        }
    }

    free(ws);
    niyah_model_gradients_destroy(&gradients);
    niyah_model_destroy(&model);
}

static void test_backward_rejects_disabled_segments(void)
{
    NiyahModelConfig config = test_config(0U);
    NiyahModel model;
    NiyahModelGradients gradients;
    const uint32_t tokens[3] = {1U, 2U, 3U};
    const uint32_t targets[3] = {2U, 3U, 4U};
    const uint32_t segment_ids[3] = {0U, 0U, 0U};
    size_t ws_count = 0U;
    float *ws;
    float *snapshot;
    float loss = 0.0f;

    memset(&model, 0, sizeof(model));
    memset(&gradients, 0, sizeof(gradients));
    CHECK(niyah_model_create(&model, &config) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(&model, UINT64_C(5)) == NIYAH_OK);
    CHECK(niyah_model_gradients_create(&gradients, &model) == NIYAH_OK);
    CHECK(niyah_train_backward_workspace_floats(&config, 3U, &ws_count) == NIYAH_OK);
    ws = (float *)calloc(ws_count, sizeof(float));
    snapshot = (float *)malloc(gradients.count * sizeof(float));
    CHECK(ws != NULL && snapshot != NULL);
    if (ws != NULL && snapshot != NULL) {
        memcpy(snapshot, gradients.values, gradients.count * sizeof(float));
        CHECK(niyah_train_backward_masked_with_segments(
                  &model, tokens, targets, 3U, 0U, segment_ids,
                  &loss, &gradients, ws, ws_count) == NIYAH_ERR_INVALID_ARGUMENT);
        CHECK(memcmp(snapshot, gradients.values, gradients.count * sizeof(float)) == 0);
        CHECK(niyah_train_backward_masked_with_segments(
                  &model, tokens, targets, 3U, 0U, NULL,
                  &loss, &gradients, ws, ws_count) == NIYAH_ERR_INVALID_ARGUMENT);
    }

    free(snapshot);
    free(ws);
    niyah_model_gradients_destroy(&gradients);
    niyah_model_destroy(&model);
}

static void test_checkpoint_round_trip_with_segments(void)
{
    const char *path = "niyah_segment_checkpoint.bin";
    NiyahModelConfig config = test_config(3U);
    NiyahModel model;
    NiyahAdamWState state;
    NiyahAdamWConfig opt;
    NiyahModel loaded;
    NiyahAdamWState loaded_state;
    NiyahAdamWConfig loaded_config;

    memset(&model, 0, sizeof(model));
    memset(&state, 0, sizeof(state));
    memset(&loaded, 0, sizeof(loaded));
    memset(&loaded_state, 0, sizeof(loaded_state));
    memset(&loaded_config, 0, sizeof(loaded_config));

    opt.learning_rate = 1.0e-3f;
    opt.beta1 = 0.9f;
    opt.beta2 = 0.999f;
    opt.epsilon = 1.0e-8f;
    opt.weight_decay = 0.0f;
    opt.max_grad_norm = 1.0f;

    (void)remove(path);
    CHECK(niyah_model_create(&model, &config) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(&model, UINT64_C(6)) == NIYAH_OK);
    CHECK(niyah_adamw_state_create(&state, &model) == NIYAH_OK);

    CHECK(niyah_checkpoint_save(path, &model, &state, &opt) == NIYAH_OK);
    CHECK(niyah_checkpoint_load(path, &loaded, &loaded_state, &loaded_config) == NIYAH_OK);

    if (loaded.weights != NULL) {
        CHECK(loaded.config.n_segments == config.n_segments);
        CHECK(loaded.layout.segment_embedding == model.layout.segment_embedding);
        CHECK(loaded.weight_count == model.weight_count);
        CHECK(memcmp(loaded.weights, model.weights,
                     model.weight_count * sizeof(float)) == 0);
    }

    niyah_adamw_state_destroy(&loaded_state);
    niyah_model_destroy(&loaded);
    niyah_adamw_state_destroy(&state);
    niyah_model_destroy(&model);
    (void)remove(path);
}

int main(void)
{
    test_layout_offsets();
    test_forward_rejections();
    test_forward_changes_with_segment_id();
    test_decode_matches_forward();
    test_segment_gradient_finite_difference();
    test_backward_rejects_disabled_segments();
    test_checkpoint_round_trip_with_segments();

    if (failures != 0) {
        fprintf(stderr, "niyah_segment_embeddings_test: %d failure(s)\n", failures);
        return 1;
    }
    puts("NIYAH_SEGMENT_EMBEDDINGS=PASS");
    return 0;
}
