#include "niyah/training_loop.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", \
                __FILE__, __LINE__, #expr); \
        failures += 1; \
    } \
} while (0)

static NiyahModelConfig test_config(void)
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
    c.tie_word_embeddings = 1;
    return c;
}

static NiyahAdamWConfig optimizer_config(void)
{
    NiyahAdamWConfig c;
    c.learning_rate = 5.0e-3f;
    c.beta1 = 0.9f;
    c.beta2 = 0.999f;
    c.epsilon = 1.0e-8f;
    c.weight_decay = 0.0f;
    c.max_grad_norm = 1.0f;
    return c;
}

static void test_deterministic_run(void)
{
    static const uint32_t t0[] = {1U, 2U, 3U};
    static const uint32_t y0[] = {2U, 3U, 4U};
    static const uint32_t t1[] = {2U, 3U, 4U};
    static const uint32_t y1[] = {3U, 4U, 5U};
    static const uint32_t t2[] = {3U, 4U, 5U};
    static const uint32_t y2[] = {4U, 5U, 6U};
    const NiyahTrainingSample samples[] = {
        {t0, y0, 3U}, {t1, y1, 3U}, {t2, y2, 3U}
    };
    NiyahModelConfig config = test_config();
    NiyahAdamWConfig opt = optimizer_config();
    NiyahModel a, b;
    NiyahAdamWState sa, sb;
    NiyahDatasetCursor ca, cb;
    float mean_a = NAN, mean_b = NAN;

    memset(&a, 0, sizeof(a)); memset(&b, 0, sizeof(b));
    memset(&sa, 0, sizeof(sa)); memset(&sb, 0, sizeof(sb));
    memset(&ca, 0, sizeof(ca)); memset(&cb, 0, sizeof(cb));

    CHECK(niyah_model_create(&a, &config) == NIYAH_OK);
    CHECK(niyah_model_create(&b, &config) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(&a, UINT64_C(20260916)) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(&b, UINT64_C(20260916)) == NIYAH_OK);
    CHECK(niyah_adamw_state_create(&sa, &a) == NIYAH_OK);
    CHECK(niyah_adamw_state_create(&sb, &b) == NIYAH_OK);
    CHECK(niyah_dataset_cursor_init(&ca, 3U, UINT64_C(44)) == NIYAH_OK);
    CHECK(niyah_dataset_cursor_init(&cb, 3U, UINT64_C(44)) == NIYAH_OK);

    CHECK(niyah_training_run_steps(
              &a, samples, 3U, &ca, &sa, &opt, 18U, &mean_a) == NIYAH_OK);
    CHECK(niyah_training_run_steps(
              &b, samples, 3U, &cb, &sb, &opt, 18U, &mean_b) == NIYAH_OK);

    CHECK(mean_a == mean_b);
    CHECK(sa.step == UINT64_C(18));
    CHECK(sb.step == sa.step);
    CHECK(ca.epoch == cb.epoch);
    CHECK(ca.position == cb.position);
    CHECK(memcmp(a.weights, b.weights,
                 a.weight_count * sizeof(float)) == 0);
    CHECK(memcmp(sa.m, sb.m, sa.count * sizeof(float)) == 0);
    CHECK(memcmp(sa.v, sb.v, sa.count * sizeof(float)) == 0);

    niyah_dataset_cursor_destroy(&cb);
    niyah_dataset_cursor_destroy(&ca);
    niyah_adamw_state_destroy(&sb);
    niyah_adamw_state_destroy(&sa);
    niyah_model_destroy(&b);
    niyah_model_destroy(&a);
}

static void test_cursor_rollback_on_backward_failure(void)
{
    static const uint32_t tokens[] = {1U, 2U, 3U};
    static const uint32_t bad_targets[] = {2U, 3U, 99U};
    const NiyahTrainingSample sample = {tokens, bad_targets, 3U};
    NiyahModelConfig config = test_config();
    NiyahAdamWConfig opt = optimizer_config();
    NiyahModel model;
    NiyahModelGradients gradients;
    NiyahAdamWState state;
    NiyahDatasetCursor cursor;
    float *workspace = NULL;
    size_t workspace_count = 0U;
    size_t sample_index = 999U;
    float loss = NAN;

    memset(&model, 0, sizeof(model));
    memset(&gradients, 0, sizeof(gradients));
    memset(&state, 0, sizeof(state));
    memset(&cursor, 0, sizeof(cursor));

    CHECK(niyah_model_create(&model, &config) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(&model, UINT64_C(5)) == NIYAH_OK);
    CHECK(niyah_model_gradients_create(&gradients, &model) == NIYAH_OK);
    CHECK(niyah_adamw_state_create(&state, &model) == NIYAH_OK);
    CHECK(niyah_dataset_cursor_init(&cursor, 1U, UINT64_C(7)) == NIYAH_OK);
    CHECK(niyah_train_backward_workspace_floats(
              &config, 3U, &workspace_count) == NIYAH_OK);

    workspace = (float *)calloc(workspace_count, sizeof(float));
    CHECK(workspace != NULL);
    if (workspace != NULL) {
        CHECK(niyah_training_step(
                  &model, &sample, 1U, &cursor,
                  &gradients, workspace, workspace_count,
                  &state, &opt, &sample_index, &loss) ==
              NIYAH_ERR_INVALID_ARGUMENT);
        CHECK(cursor.epoch == UINT64_C(0));
        CHECK(cursor.position == 0U);
        CHECK(state.step == UINT64_C(0));
        CHECK(sample_index == 999U);
        CHECK(isnan(loss));
    }

    free(workspace);
    niyah_dataset_cursor_destroy(&cursor);
    niyah_adamw_state_destroy(&state);
    niyah_model_gradients_destroy(&gradients);
    niyah_model_destroy(&model);
}

static void test_loss_decreases(void)
{
    static const uint32_t tokens[] = {1U, 2U, 3U};
    static const uint32_t targets[] = {2U, 3U, 4U};
    const NiyahTrainingSample sample = {tokens, targets, 3U};
    NiyahModelConfig config = test_config();
    NiyahAdamWConfig opt = optimizer_config();
    NiyahModel model;
    NiyahModelGradients gradients;
    NiyahAdamWState state;
    NiyahDatasetCursor cursor;
    size_t workspace_count = 0U;
    float *workspace = NULL;
    float before = NAN;
    float after = NAN;
    float mean = NAN;

    memset(&model, 0, sizeof(model));
    memset(&gradients, 0, sizeof(gradients));
    memset(&state, 0, sizeof(state));
    memset(&cursor, 0, sizeof(cursor));

    CHECK(niyah_model_create(&model, &config) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(
              &model, UINT64_C(20260916)) == NIYAH_OK);
    CHECK(niyah_model_gradients_create(&gradients, &model) == NIYAH_OK);
    CHECK(niyah_adamw_state_create(&state, &model) == NIYAH_OK);
    CHECK(niyah_dataset_cursor_init(
              &cursor, 1U, UINT64_C(0)) == NIYAH_OK);
    CHECK(niyah_train_backward_workspace_floats(
              &config, 3U, &workspace_count) == NIYAH_OK);

    workspace = (float *)calloc(workspace_count, sizeof(float));
    CHECK(workspace != NULL);

    if (workspace != NULL) {
        CHECK(niyah_train_backward(
                  &model, tokens, targets, 3U, &before,
                  &gradients, workspace, workspace_count) == NIYAH_OK);

        CHECK(niyah_training_run_steps(
                  &model, &sample, 1U, &cursor,
                  &state, &opt, 32U, &mean) == NIYAH_OK);

        CHECK(niyah_train_backward(
                  &model, tokens, targets, 3U, &after,
                  &gradients, workspace, workspace_count) == NIYAH_OK);

        CHECK(isfinite(mean));
        CHECK(after < before);
        CHECK(state.step == UINT64_C(32));
    }

    free(workspace);
    niyah_dataset_cursor_destroy(&cursor);
    niyah_adamw_state_destroy(&state);
    niyah_model_gradients_destroy(&gradients);
    niyah_model_destroy(&model);
}

int main(void)
{
    test_deterministic_run();
    test_cursor_rollback_on_backward_failure();
    test_loss_decreases();

    if (failures != 0) {
        fprintf(stderr, "niyah_training_loop_test: %d failure(s)\n", failures);
        return 1;
    }

    puts("NIYAH_TRAINING_LOOP_P6_E=PASS");
    return 0;
}
