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
        {t0, y0, 3U, 0U}, {t1, y1, 3U, 0U}, {t2, y2, 3U, 0U}
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
    const NiyahTrainingSample sample = {tokens, bad_targets, 3U, 0U};
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
    const NiyahTrainingSample sample = {tokens, targets, 3U, 0U};
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


static void test_accumulated_step_matches_single_step(void)
{
    static const uint32_t tokens[] = {1U, 2U, 3U};
    static const uint32_t targets[] = {2U, 3U, 4U};
    const NiyahTrainingSample sample = {tokens, targets, 3U, 0U};
    NiyahModelConfig config = test_config();
    NiyahAdamWConfig opt = optimizer_config();
    NiyahModel a, b;
    NiyahAdamWState sa, sb;
    NiyahDatasetCursor ca, cb;
    NiyahModelGradients ga, gb_sample, gb_accum;
    float *wa = NULL;
    float *wb = NULL;
    size_t workspace_count = 0U;
    size_t index = 999U;
    size_t consumed = 999U;
    float loss_a = NAN;
    float loss_b = NAN;

    memset(&a, 0, sizeof(a)); memset(&b, 0, sizeof(b));
    memset(&sa, 0, sizeof(sa)); memset(&sb, 0, sizeof(sb));
    memset(&ca, 0, sizeof(ca)); memset(&cb, 0, sizeof(cb));
    memset(&ga, 0, sizeof(ga));
    memset(&gb_sample, 0, sizeof(gb_sample));
    memset(&gb_accum, 0, sizeof(gb_accum));

    CHECK(niyah_model_create(&a, &config) == NIYAH_OK);
    CHECK(niyah_model_create(&b, &config) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(&a, UINT64_C(81)) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(&b, UINT64_C(81)) == NIYAH_OK);
    CHECK(niyah_adamw_state_create(&sa, &a) == NIYAH_OK);
    CHECK(niyah_adamw_state_create(&sb, &b) == NIYAH_OK);
    CHECK(niyah_dataset_cursor_init(&ca, 1U, UINT64_C(3)) == NIYAH_OK);
    CHECK(niyah_dataset_cursor_init(&cb, 1U, UINT64_C(3)) == NIYAH_OK);
    CHECK(niyah_model_gradients_create(&ga, &a) == NIYAH_OK);
    CHECK(niyah_model_gradients_create(&gb_sample, &b) == NIYAH_OK);
    CHECK(niyah_model_gradients_create(&gb_accum, &b) == NIYAH_OK);
    CHECK(niyah_train_backward_workspace_floats(
              &config, 3U, &workspace_count) == NIYAH_OK);

    wa = (float *)calloc(workspace_count, sizeof(float));
    wb = (float *)calloc(workspace_count, sizeof(float));
    CHECK(wa != NULL);
    CHECK(wb != NULL);

    if (wa != NULL && wb != NULL) {
        CHECK(niyah_training_step(
                  &a, &sample, 1U, &ca,
                  &ga, wa, workspace_count,
                  &sa, &opt, &index, &loss_a) == NIYAH_OK);

        CHECK(niyah_training_accumulated_step(
                  &b, &sample, 1U, &cb,
                  &gb_sample, &gb_accum, wb, workspace_count,
                  &sb, &opt, 1U, 1U,
                  &consumed, &loss_b) == NIYAH_OK);

        CHECK(index == 0U);
        CHECK(consumed == 1U);
        CHECK(loss_a == loss_b);
        CHECK(sa.step == UINT64_C(1));
        CHECK(sb.step == UINT64_C(1));
        CHECK(memcmp(a.weights, b.weights,
                     a.weight_count * sizeof(float)) == 0);
        CHECK(memcmp(sa.m, sb.m, sa.count * sizeof(float)) == 0);
        CHECK(memcmp(sa.v, sb.v, sa.count * sizeof(float)) == 0);
    }

    free(wb);
    free(wa);
    niyah_model_gradients_destroy(&gb_accum);
    niyah_model_gradients_destroy(&gb_sample);
    niyah_model_gradients_destroy(&ga);
    niyah_dataset_cursor_destroy(&cb);
    niyah_dataset_cursor_destroy(&ca);
    niyah_adamw_state_destroy(&sb);
    niyah_adamw_state_destroy(&sa);
    niyah_model_destroy(&b);
    niyah_model_destroy(&a);
}

static void test_minibatch_accumulation_deterministic(void)
{
    static const uint32_t t0[] = {1U, 2U, 3U};
    static const uint32_t y0[] = {2U, 3U, 4U};
    static const uint32_t t1[] = {2U, 3U, 4U};
    static const uint32_t y1[] = {3U, 4U, 5U};
    static const uint32_t t2[] = {3U, 4U, 5U};
    static const uint32_t y2[] = {4U, 5U, 6U};
    const NiyahTrainingSample samples[] = {
        {t0, y0, 3U, 0U}, {t1, y1, 3U, 0U}, {t2, y2, 3U, 0U}
    };
    NiyahModelConfig config = test_config();
    NiyahAdamWConfig opt = optimizer_config();
    NiyahModel a, b;
    NiyahAdamWState sa, sb;
    NiyahDatasetCursor ca, cb;
    float mean_a = NAN;
    float mean_b = NAN;

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

    CHECK(niyah_training_run_updates(
              &a, samples, 3U, &ca, &sa, &opt,
              2U, 3U, 5U, &mean_a) == NIYAH_OK);
    CHECK(niyah_training_run_updates(
              &b, samples, 3U, &cb, &sb, &opt,
              2U, 3U, 5U, &mean_b) == NIYAH_OK);

    CHECK(isfinite(mean_a));
    CHECK(mean_a == mean_b);
    CHECK(sa.step == UINT64_C(5));
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

static void test_accumulated_step_token_weighted(void)
{
    static const uint32_t short_tokens[] = {1U};
    static const uint32_t short_targets[] = {2U};
    static const uint32_t long_tokens[] = {1U, 2U, 3U};
    static const uint32_t long_targets[] = {2U, 3U, 4U};

    const NiyahTrainingSample samples[] = {
        {short_tokens, short_targets, 1U, 0U},
        {long_tokens, long_targets, 3U, 0U}
    };

    NiyahModelConfig config = test_config();
    NiyahAdamWConfig opt = optimizer_config();

    NiyahModel actual, expected;
    NiyahAdamWState actual_state, expected_state;
    NiyahDatasetCursor cursor;

    NiyahModelGradients sample_actual;
    NiyahModelGradients accum_actual;
    NiyahModelGradients sample_expected;
    NiyahModelGradients accum_expected;

    float *workspace = NULL;
    size_t workspace_count = 0U;
    size_t consumed = 0U;
    size_t order[2];
    size_t total_supervised = 0U;
    double weighted_loss = 0.0;
    float actual_loss = NAN;
    float expected_loss;
    size_t k;
    size_t i;

    memset(&actual, 0, sizeof(actual));
    memset(&expected, 0, sizeof(expected));
    memset(&actual_state, 0, sizeof(actual_state));
    memset(&expected_state, 0, sizeof(expected_state));
    memset(&cursor, 0, sizeof(cursor));
    memset(&sample_actual, 0, sizeof(sample_actual));
    memset(&accum_actual, 0, sizeof(accum_actual));
    memset(&sample_expected, 0, sizeof(sample_expected));
    memset(&accum_expected, 0, sizeof(accum_expected));

    CHECK(niyah_model_create(&actual, &config) == NIYAH_OK);
    CHECK(niyah_model_create(&expected, &config) == NIYAH_OK);

    CHECK(niyah_model_reset_parameters(
              &actual, UINT64_C(20260923)) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(
              &expected, UINT64_C(20260923)) == NIYAH_OK);

    CHECK(niyah_adamw_state_create(
              &actual_state, &actual) == NIYAH_OK);
    CHECK(niyah_adamw_state_create(
              &expected_state, &expected) == NIYAH_OK);

    CHECK(niyah_dataset_cursor_init(
              &cursor, 2U, UINT64_C(42)) == NIYAH_OK);

    CHECK(niyah_model_gradients_create(
              &sample_actual, &actual) == NIYAH_OK);
    CHECK(niyah_model_gradients_create(
              &accum_actual, &actual) == NIYAH_OK);
    CHECK(niyah_model_gradients_create(
              &sample_expected, &expected) == NIYAH_OK);
    CHECK(niyah_model_gradients_create(
              &accum_expected, &expected) == NIYAH_OK);

    CHECK(niyah_train_backward_workspace_floats(
              &config, 3U, &workspace_count) == NIYAH_OK);

    workspace = (float *)calloc(
        workspace_count, sizeof(float));
    CHECK(workspace != NULL);

    if (workspace != NULL) {
        order[0] = cursor.order[0];
        order[1] = cursor.order[1];

        niyah_model_gradients_zero(&accum_expected);

        for (k = 0U; k < 2U; ++k) {
            const NiyahTrainingSample *sample =
                &samples[order[k]];
            const size_t supervised =
                sample->token_count - sample->loss_start;
            float sample_loss = NAN;

            CHECK(niyah_train_backward_masked(
                      &expected,
                      sample->tokens,
                      sample->targets,
                      sample->token_count,
                      sample->loss_start,
                      &sample_loss,
                      &sample_expected,
                      workspace,
                      workspace_count) == NIYAH_OK);

            for (i = 0U;
                 i < sample_expected.count;
                 ++i) {
                sample_expected.values[i] *=
                    (float)supervised;
            }

            for (i = 0U;
                 i < accum_expected.count;
                 ++i) {
                accum_expected.values[i] +=
                    sample_expected.values[i];
            }

            total_supervised += supervised;
            weighted_loss +=
                (double)sample_loss *
                (double)supervised;
        }

        for (i = 0U;
             i < accum_expected.count;
             ++i) {
            accum_expected.values[i] /=
                (float)total_supervised;
        }

        expected_loss =
            (float)(
                weighted_loss /
                (double)total_supervised);

        CHECK(niyah_adamw_step(
                  &expected,
                  &accum_expected,
                  &expected_state,
                  &opt) == NIYAH_OK);

        CHECK(niyah_training_accumulated_step(
                  &actual,
                  samples,
                  2U,
                  &cursor,
                  &sample_actual,
                  &accum_actual,
                  workspace,
                  workspace_count,
                  &actual_state,
                  &opt,
                  2U,
                  1U,
                  &consumed,
                  &actual_loss) == NIYAH_OK);

        CHECK(consumed == 2U);
        CHECK(total_supervised == 4U);
        CHECK(actual_loss == expected_loss);

        CHECK(memcmp(
                  actual.weights,
                  expected.weights,
                  actual.weight_count *
                      sizeof(float)) == 0);

        CHECK(memcmp(
                  actual_state.m,
                  expected_state.m,
                  actual_state.count *
                      sizeof(float)) == 0);

        CHECK(memcmp(
                  actual_state.v,
                  expected_state.v,
                  actual_state.count *
                      sizeof(float)) == 0);
    }

    free(workspace);

    niyah_model_gradients_destroy(&accum_expected);
    niyah_model_gradients_destroy(&sample_expected);
    niyah_model_gradients_destroy(&accum_actual);
    niyah_model_gradients_destroy(&sample_actual);

    niyah_dataset_cursor_destroy(&cursor);

    niyah_adamw_state_destroy(&expected_state);
    niyah_adamw_state_destroy(&actual_state);

    niyah_model_destroy(&expected);
    niyah_model_destroy(&actual);
}

static void test_accumulated_step_rolls_back_whole_group(void)
{
    static const uint32_t good_tokens[] = {1U, 2U, 3U};
    static const uint32_t good_targets[] = {2U, 3U, 4U};
    static const uint32_t bad_targets[] = {2U, 3U, 99U};
    NiyahTrainingSample samples[] = {
        {good_tokens, good_targets, 3U, 0U},
        {good_tokens, bad_targets, 3U, 0U}
    };
    NiyahModelConfig config = test_config();
    NiyahAdamWConfig opt = optimizer_config();
    NiyahModel model;
    NiyahAdamWState state;
    NiyahDatasetCursor cursor;
    NiyahModelGradients sample_grad;
    NiyahModelGradients accum_grad;
    float *workspace = NULL;
    float *weights_before = NULL;
    size_t workspace_count = 0U;
    size_t consumed = 999U;
    float loss = NAN;

    memset(&model, 0, sizeof(model));
    memset(&state, 0, sizeof(state));
    memset(&cursor, 0, sizeof(cursor));
    memset(&sample_grad, 0, sizeof(sample_grad));
    memset(&accum_grad, 0, sizeof(accum_grad));

    CHECK(niyah_model_create(&model, &config) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(&model, UINT64_C(9)) == NIYAH_OK);
    CHECK(niyah_adamw_state_create(&state, &model) == NIYAH_OK);
    CHECK(niyah_dataset_cursor_init(&cursor, 2U, UINT64_C(0)) == NIYAH_OK);
    CHECK(niyah_model_gradients_create(&sample_grad, &model) == NIYAH_OK);
    CHECK(niyah_model_gradients_create(&accum_grad, &model) == NIYAH_OK);
    CHECK(niyah_train_backward_workspace_floats(
              &config, 3U, &workspace_count) == NIYAH_OK);

    workspace = (float *)calloc(workspace_count, sizeof(float));
    weights_before = (float *)malloc(model.weight_count * sizeof(float));
    CHECK(workspace != NULL);
    CHECK(weights_before != NULL);

    if (workspace != NULL && weights_before != NULL) {
        if (cursor.order[0U] == 1U) {
            NiyahTrainingSample tmp = samples[0U];
            samples[0U] = samples[1U];
            samples[1U] = tmp;
        }

        memcpy(weights_before, model.weights,
               model.weight_count * sizeof(float));

        CHECK(niyah_training_accumulated_step(
                  &model, samples, 2U, &cursor,
                  &sample_grad, &accum_grad,
                  workspace, workspace_count,
                  &state, &opt, 2U, 1U,
                  &consumed, &loss) == NIYAH_ERR_INVALID_ARGUMENT);

        CHECK(cursor.epoch == UINT64_C(0));
        CHECK(cursor.position == 0U);
        CHECK(state.step == UINT64_C(0));
        CHECK(consumed == 999U);
        CHECK(isnan(loss));
        CHECK(memcmp(model.weights, weights_before,
                     model.weight_count * sizeof(float)) == 0);
    }

    free(weights_before);
    free(workspace);
    niyah_model_gradients_destroy(&accum_grad);
    niyah_model_gradients_destroy(&sample_grad);
    niyah_dataset_cursor_destroy(&cursor);
    niyah_adamw_state_destroy(&state);
    niyah_model_destroy(&model);
}

static void test_training_samples_from_shard(void)
{
    static uint32_t tokens[] = {
        NIYAH_TOKEN_BOS, 1U, 2U, 3U, NIYAH_TOKEN_EOS
    };
    NiyahDatasetShard shard;
    NiyahTrainingSample samples[2];
    NiyahTrainingSample sentinel[1];
    size_t count = 999U;

    memset(&shard, 0, sizeof(shard));
    memset(samples, 0, sizeof(samples));
    memset(sentinel, 0, sizeof(sentinel));

    shard.tokens = tokens;
    shard.token_count = sizeof(tokens) / sizeof(tokens[0]);
    shard.sequence_length = 2U;
    shard.sample_count = 2U;

    CHECK(niyah_training_samples_from_shard(
              &shard, NULL, 0U, &count) == NIYAH_OK);
    CHECK(count == 2U);

    count = 999U;
    CHECK(niyah_training_samples_from_shard(
              &shard, sentinel, 1U, &count) ==
          NIYAH_ERR_BUFFER_TOO_SMALL);
    CHECK(count == 2U);
    CHECK(sentinel[0].tokens == NULL);
    CHECK(sentinel[0].targets == NULL);
    CHECK(sentinel[0].token_count == 0U);

    CHECK(niyah_training_samples_from_shard(
              &shard, samples, 2U, &count) == NIYAH_OK);
    CHECK(count == 2U);
    CHECK(samples[0].tokens == &tokens[0]);
    CHECK(samples[0].targets == &tokens[1]);
    CHECK(samples[0].token_count == 2U);
    CHECK(samples[1].tokens == &tokens[2]);
    CHECK(samples[1].targets == &tokens[3]);
    CHECK(samples[1].token_count == 2U);

    CHECK(niyah_training_samples_from_shard(
              &shard, NULL, 1U, &count) ==
          NIYAH_ERR_INVALID_ARGUMENT);

    shard.sample_count = 0U;
    CHECK(niyah_training_samples_from_shard(
              &shard, NULL, 0U, &count) ==
          NIYAH_ERR_INVALID_CONFIG);
}


typedef struct TestSampleProviderContext {
    const NiyahTrainingSample *samples;
    size_t sample_count;
    size_t calls;
    int fail;
    NiyahStatus failure_status;
    int invalidate_sample;
} TestSampleProviderContext;

static NiyahStatus test_sample_provider(
    size_t sample_index,
    NiyahTrainingSample *out_sample,
    void *user_data)
{
    TestSampleProviderContext *context =
        (TestSampleProviderContext *)user_data;

    if (context == NULL || out_sample == NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;

    if (sample_index >= context->sample_count)
        return NIYAH_ERR_INVALID_ARGUMENT;

    context->calls += 1U;

    if (context->fail)
        return context->failure_status;

    *out_sample = context->samples[sample_index];

    if (context->invalidate_sample)
        out_sample->token_count = 0U;

    return NIYAH_OK;
}

typedef struct TestBackwardContext {
    size_t calls;
} TestBackwardContext;

static NiyahStatus test_passthrough_backward(
    const NiyahModel *model,
    const NiyahTrainingSample *sample,
    NiyahModelGradients *gradients,
    float *workspace,
    size_t workspace_count,
    float *out_loss,
    void *user_data)
{
    TestBackwardContext *context =
        (TestBackwardContext *)user_data;

    if (context == NULL || sample == NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;

    context->calls += 1U;

    return niyah_train_backward_masked(
        model,
        sample->tokens,
        sample->targets,
        sample->token_count,
        sample->loss_start,
        out_loss,
        gradients,
        workspace,
        workspace_count);
}

static void test_provider_success_matches_array(void)
{
    static const uint32_t t0[] = {1U, 2U, 3U};
    static const uint32_t y0[] = {2U, 3U, 4U};
    static const uint32_t t1[] = {2U, 3U, 4U};
    static const uint32_t y1[] = {3U, 4U, 5U};
    static const uint32_t t2[] = {3U, 4U, 5U};
    static const uint32_t y2[] = {4U, 5U, 6U};

    const NiyahTrainingSample samples[] = {
        {t0, y0, 3U, 0U},
        {t1, y1, 3U, 0U},
        {t2, y2, 3U, 0U}
    };

    NiyahModelConfig config = test_config();
    NiyahAdamWConfig opt = optimizer_config();
    NiyahModel array_model, provider_model;
    NiyahAdamWState array_state, provider_state;
    NiyahDatasetCursor array_cursor, provider_cursor;
    TestSampleProviderContext provider;
    float array_loss = NAN;
    float provider_loss = NAN;

    memset(&array_model, 0, sizeof(array_model));
    memset(&provider_model, 0, sizeof(provider_model));
    memset(&array_state, 0, sizeof(array_state));
    memset(&provider_state, 0, sizeof(provider_state));
    memset(&array_cursor, 0, sizeof(array_cursor));
    memset(&provider_cursor, 0, sizeof(provider_cursor));
    memset(&provider, 0, sizeof(provider));

    provider.samples = samples;
    provider.sample_count = 3U;

    CHECK(niyah_model_create(
              &array_model, &config) == NIYAH_OK);
    CHECK(niyah_model_create(
              &provider_model, &config) == NIYAH_OK);

    CHECK(niyah_model_reset_parameters(
              &array_model,
              UINT64_C(20260924)) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(
              &provider_model,
              UINT64_C(20260924)) == NIYAH_OK);

    CHECK(niyah_adamw_state_create(
              &array_state, &array_model) == NIYAH_OK);
    CHECK(niyah_adamw_state_create(
              &provider_state, &provider_model) == NIYAH_OK);

    CHECK(niyah_dataset_cursor_init(
              &array_cursor, 3U,
              UINT64_C(77)) == NIYAH_OK);
    CHECK(niyah_dataset_cursor_init(
              &provider_cursor, 3U,
              UINT64_C(77)) == NIYAH_OK);

    CHECK(niyah_training_run_updates_with_progress(
              &array_model,
              samples,
              3U,
              &array_cursor,
              &array_state,
              &opt,
              2U,
              2U,
              5U,
              NULL,
              NULL,
              &array_loss) == NIYAH_OK);

    CHECK(
        niyah_training_run_updates_with_progress_with_provider(
            &provider_model,
            3U,
            3U,
            test_sample_provider,
            &provider,
            &provider_cursor,
            &provider_state,
            &opt,
            2U,
            2U,
            5U,
            NULL,
            NULL,
            NULL,
            NULL,
            &provider_loss) == NIYAH_OK);

    CHECK(provider.calls == 20U);
    CHECK(array_loss == provider_loss);

    CHECK(array_state.step == UINT64_C(5));
    CHECK(provider_state.step == array_state.step);

    CHECK(array_cursor.epoch ==
          provider_cursor.epoch);
    CHECK(array_cursor.position ==
          provider_cursor.position);

    CHECK(memcmp(
              array_model.weights,
              provider_model.weights,
              array_model.weight_count *
                  sizeof(float)) == 0);

    CHECK(memcmp(
              array_state.m,
              provider_state.m,
              array_state.count *
                  sizeof(float)) == 0);

    CHECK(memcmp(
              array_state.v,
              provider_state.v,
              array_state.count *
                  sizeof(float)) == 0);

    niyah_dataset_cursor_destroy(&provider_cursor);
    niyah_dataset_cursor_destroy(&array_cursor);

    niyah_adamw_state_destroy(&provider_state);
    niyah_adamw_state_destroy(&array_state);

    niyah_model_destroy(&provider_model);
    niyah_model_destroy(&array_model);
}

static void test_provider_failure_rolls_back_cursor(void)
{
    static const uint32_t tokens[] = {1U, 2U, 3U};
    static const uint32_t targets[] = {2U, 3U, 4U};

    const NiyahTrainingSample samples[] = {
        {tokens, targets, 3U, 0U},
        {tokens, targets, 3U, 0U}
    };

    NiyahModelConfig config = test_config();
    NiyahAdamWConfig opt = optimizer_config();
    NiyahModel model;
    NiyahAdamWState state;
    NiyahDatasetCursor cursor;
    TestSampleProviderContext provider;
    float *weights_before = NULL;
    float loss = NAN;

    memset(&model, 0, sizeof(model));
    memset(&state, 0, sizeof(state));
    memset(&cursor, 0, sizeof(cursor));
    memset(&provider, 0, sizeof(provider));

    provider.samples = samples;
    provider.sample_count = 2U;
    provider.fail = 1;
    provider.failure_status = NIYAH_ERR_IO;

    CHECK(niyah_model_create(
              &model, &config) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(
              &model, UINT64_C(901)) == NIYAH_OK);
    CHECK(niyah_adamw_state_create(
              &state, &model) == NIYAH_OK);
    CHECK(niyah_dataset_cursor_init(
              &cursor, 2U, UINT64_C(11)) == NIYAH_OK);

    weights_before =
        (float *)malloc(
            model.weight_count * sizeof(float));
    CHECK(weights_before != NULL);

    if (weights_before != NULL) {
        memcpy(
            weights_before,
            model.weights,
            model.weight_count * sizeof(float));

        CHECK(
            niyah_training_run_updates_with_progress_with_provider(
                &model,
                2U,
                3U,
                test_sample_provider,
                &provider,
                &cursor,
                &state,
                &opt,
                2U,
                1U,
                1U,
                NULL,
                NULL,
                NULL,
                NULL,
                &loss) == NIYAH_ERR_IO);

        CHECK(provider.calls == 1U);
        CHECK(cursor.epoch == UINT64_C(0));
        CHECK(cursor.position == 0U);
        CHECK(state.step == UINT64_C(0));
        CHECK(isnan(loss));

        CHECK(memcmp(
                  model.weights,
                  weights_before,
                  model.weight_count *
                      sizeof(float)) == 0);
    }

    free(weights_before);
    niyah_dataset_cursor_destroy(&cursor);
    niyah_adamw_state_destroy(&state);
    niyah_model_destroy(&model);
}

static void test_provider_invalid_sample_rejected(void)
{
    static const uint32_t tokens[] = {1U, 2U, 3U};
    static const uint32_t targets[] = {2U, 3U, 4U};

    const NiyahTrainingSample sample = {
        tokens, targets, 3U, 0U
    };

    NiyahModelConfig config = test_config();
    NiyahAdamWConfig opt = optimizer_config();
    NiyahModel model;
    NiyahAdamWState state;
    NiyahDatasetCursor cursor;
    TestSampleProviderContext provider;
    float loss = NAN;

    memset(&model, 0, sizeof(model));
    memset(&state, 0, sizeof(state));
    memset(&cursor, 0, sizeof(cursor));
    memset(&provider, 0, sizeof(provider));

    provider.samples = &sample;
    provider.sample_count = 1U;
    provider.invalidate_sample = 1;

    CHECK(niyah_model_create(
              &model, &config) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(
              &model, UINT64_C(902)) == NIYAH_OK);
    CHECK(niyah_adamw_state_create(
              &state, &model) == NIYAH_OK);
    CHECK(niyah_dataset_cursor_init(
              &cursor, 1U, UINT64_C(12)) == NIYAH_OK);

    CHECK(
        niyah_training_run_updates_with_progress_with_provider(
            &model,
            1U,
            3U,
            test_sample_provider,
            &provider,
            &cursor,
            &state,
            &opt,
            1U,
            1U,
            1U,
            NULL,
            NULL,
            NULL,
            NULL,
            &loss) == NIYAH_ERR_INVALID_ARGUMENT);

    CHECK(provider.calls == 1U);
    CHECK(cursor.epoch == UINT64_C(0));
    CHECK(cursor.position == 0U);
    CHECK(state.step == UINT64_C(0));
    CHECK(isnan(loss));

    niyah_dataset_cursor_destroy(&cursor);
    niyah_adamw_state_destroy(&state);
    niyah_model_destroy(&model);
}

static void test_provider_max_token_guard(void)
{
    static const uint32_t tokens[] = {
        1U, 2U, 3U, 4U
    };
    static const uint32_t targets[] = {
        2U, 3U, 4U, 5U
    };

    const NiyahTrainingSample sample = {
        tokens, targets, 4U, 0U
    };

    NiyahModelConfig config = test_config();
    NiyahAdamWConfig opt = optimizer_config();
    NiyahModel model;
    NiyahAdamWState state;
    NiyahDatasetCursor cursor;
    TestSampleProviderContext provider;
    float loss = NAN;

    memset(&model, 0, sizeof(model));
    memset(&state, 0, sizeof(state));
    memset(&cursor, 0, sizeof(cursor));
    memset(&provider, 0, sizeof(provider));

    provider.samples = &sample;
    provider.sample_count = 1U;

    CHECK(niyah_model_create(
              &model, &config) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(
              &model, UINT64_C(903)) == NIYAH_OK);
    CHECK(niyah_adamw_state_create(
              &state, &model) == NIYAH_OK);
    CHECK(niyah_dataset_cursor_init(
              &cursor, 1U, UINT64_C(13)) == NIYAH_OK);

    CHECK(
        niyah_training_run_updates_with_progress_with_provider(
            &model,
            1U,
            3U,
            test_sample_provider,
            &provider,
            &cursor,
            &state,
            &opt,
            1U,
            1U,
            1U,
            NULL,
            NULL,
            NULL,
            NULL,
            &loss) == NIYAH_ERR_INVALID_CONFIG);

    CHECK(provider.calls == 1U);
    CHECK(cursor.epoch == UINT64_C(0));
    CHECK(cursor.position == 0U);
    CHECK(state.step == UINT64_C(0));
    CHECK(isnan(loss));

    niyah_dataset_cursor_destroy(&cursor);
    niyah_adamw_state_destroy(&state);
    niyah_model_destroy(&model);
}

static void test_provider_custom_backward_dispatch(void)
{
    static const uint32_t t0[] = {1U, 2U, 3U};
    static const uint32_t y0[] = {2U, 3U, 4U};
    static const uint32_t t1[] = {2U, 3U, 4U};
    static const uint32_t y1[] = {3U, 4U, 5U};

    const NiyahTrainingSample samples[] = {
        {t0, y0, 3U, 0U},
        {t1, y1, 3U, 0U}
    };

    NiyahModelConfig config = test_config();
    NiyahAdamWConfig opt = optimizer_config();

    NiyahModel cpu_model, callback_model;
    NiyahAdamWState cpu_state, callback_state;
    NiyahDatasetCursor cpu_cursor, callback_cursor;

    TestSampleProviderContext cpu_provider;
    TestSampleProviderContext callback_provider;
    TestBackwardContext backward_context;

    float cpu_loss = NAN;
    float callback_loss = NAN;

    memset(&cpu_model, 0, sizeof(cpu_model));
    memset(&callback_model, 0, sizeof(callback_model));
    memset(&cpu_state, 0, sizeof(cpu_state));
    memset(&callback_state, 0, sizeof(callback_state));
    memset(&cpu_cursor, 0, sizeof(cpu_cursor));
    memset(&callback_cursor, 0, sizeof(callback_cursor));
    memset(&cpu_provider, 0, sizeof(cpu_provider));
    memset(&callback_provider, 0, sizeof(callback_provider));
    memset(&backward_context, 0, sizeof(backward_context));

    cpu_provider.samples = samples;
    cpu_provider.sample_count = 2U;

    callback_provider.samples = samples;
    callback_provider.sample_count = 2U;

    CHECK(niyah_model_create(
              &cpu_model, &config) == NIYAH_OK);
    CHECK(niyah_model_create(
              &callback_model, &config) == NIYAH_OK);

    CHECK(niyah_model_reset_parameters(
              &cpu_model,
              UINT64_C(904)) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(
              &callback_model,
              UINT64_C(904)) == NIYAH_OK);

    CHECK(niyah_adamw_state_create(
              &cpu_state, &cpu_model) == NIYAH_OK);
    CHECK(niyah_adamw_state_create(
              &callback_state,
              &callback_model) == NIYAH_OK);

    CHECK(niyah_dataset_cursor_init(
              &cpu_cursor,
              2U,
              UINT64_C(14)) == NIYAH_OK);
    CHECK(niyah_dataset_cursor_init(
              &callback_cursor,
              2U,
              UINT64_C(14)) == NIYAH_OK);

    CHECK(
        niyah_training_run_updates_with_progress_with_provider(
            &cpu_model,
            2U,
            3U,
            test_sample_provider,
            &cpu_provider,
            &cpu_cursor,
            &cpu_state,
            &opt,
            2U,
            1U,
            3U,
            NULL,
            NULL,
            NULL,
            NULL,
            &cpu_loss) == NIYAH_OK);

    CHECK(
        niyah_training_run_updates_with_progress_with_provider(
            &callback_model,
            2U,
            3U,
            test_sample_provider,
            &callback_provider,
            &callback_cursor,
            &callback_state,
            &opt,
            2U,
            1U,
            3U,
            test_passthrough_backward,
            &backward_context,
            NULL,
            NULL,
            &callback_loss) == NIYAH_OK);

    CHECK(cpu_provider.calls == 6U);
    CHECK(callback_provider.calls == 6U);
    CHECK(backward_context.calls == 6U);

    CHECK(cpu_loss == callback_loss);
    CHECK(cpu_state.step == UINT64_C(3));
    CHECK(callback_state.step == cpu_state.step);

    CHECK(cpu_cursor.epoch ==
          callback_cursor.epoch);
    CHECK(cpu_cursor.position ==
          callback_cursor.position);

    CHECK(memcmp(
              cpu_model.weights,
              callback_model.weights,
              cpu_model.weight_count *
                  sizeof(float)) == 0);

    CHECK(memcmp(
              cpu_state.m,
              callback_state.m,
              cpu_state.count *
                  sizeof(float)) == 0);

    CHECK(memcmp(
              cpu_state.v,
              callback_state.v,
              cpu_state.count *
                  sizeof(float)) == 0);

    niyah_dataset_cursor_destroy(&callback_cursor);
    niyah_dataset_cursor_destroy(&cpu_cursor);

    niyah_adamw_state_destroy(&callback_state);
    niyah_adamw_state_destroy(&cpu_state);

    niyah_model_destroy(&callback_model);
    niyah_model_destroy(&cpu_model);
}

typedef struct ProgressLog {
    size_t indices[8];
    size_t updates[8];
    float losses[8];
    size_t count;
} ProgressLog;

static void record_progress(size_t update_index, size_t updates,
                             float loss, void *user_data)
{
    ProgressLog *log = (ProgressLog *)user_data;
    CHECK(log->count < 8U);
    if (log->count < 8U) {
        log->indices[log->count] = update_index;
        log->updates[log->count] = updates;
        log->losses[log->count] = loss;
        log->count += 1U;
    }
}

static void test_run_updates_with_progress_invokes_callback(void)
{
    static const uint32_t t0[] = {1U, 2U, 3U};
    static const uint32_t y0[] = {2U, 3U, 4U};
    static const uint32_t t1[] = {2U, 3U, 4U};
    static const uint32_t y1[] = {3U, 4U, 5U};
    const NiyahTrainingSample samples[] = {
        {t0, y0, 3U, 0U}, {t1, y1, 3U, 0U}
    };
    NiyahModelConfig config = test_config();
    NiyahAdamWConfig opt = optimizer_config();
    NiyahModel model;
    NiyahAdamWState state;
    NiyahDatasetCursor cursor;
    ProgressLog log;
    float mean_loss = NAN;
    size_t i;

    memset(&model, 0, sizeof(model));
    memset(&state, 0, sizeof(state));
    memset(&cursor, 0, sizeof(cursor));
    memset(&log, 0, sizeof(log));

    CHECK(niyah_model_create(&model, &config) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(&model, UINT64_C(2026)) == NIYAH_OK);
    CHECK(niyah_adamw_state_create(&state, &model) == NIYAH_OK);
    CHECK(niyah_dataset_cursor_init(&cursor, 2U, UINT64_C(7)) == NIYAH_OK);

    CHECK(niyah_training_run_updates_with_progress(
              &model, samples, 2U, &cursor, &state, &opt,
              1U, 1U, 3U, record_progress, &log, &mean_loss) == NIYAH_OK);

    CHECK(isfinite(mean_loss));
    CHECK(log.count == 3U);
    for (i = 0U; i < log.count; ++i) {
        CHECK(log.indices[i] == i);
        CHECK(log.updates[i] == 3U);
        CHECK(isfinite(log.losses[i]));
    }

    niyah_dataset_cursor_destroy(&cursor);
    niyah_adamw_state_destroy(&state);
    niyah_model_destroy(&model);
}

int main(void)
{
    test_training_samples_from_shard();
    test_deterministic_run();
    test_cursor_rollback_on_backward_failure();
    test_loss_decreases();
    test_accumulated_step_matches_single_step();
    test_accumulated_step_token_weighted();
    test_minibatch_accumulation_deterministic();
    test_accumulated_step_rolls_back_whole_group();
    test_provider_success_matches_array();
    test_provider_failure_rolls_back_cursor();
    test_provider_invalid_sample_rejected();
    test_provider_max_token_guard();
    test_provider_custom_backward_dispatch();
    test_run_updates_with_progress_invokes_callback();

    if (failures != 0) {
        fprintf(stderr, "niyah_training_loop_test: %d failure(s)\n", failures);
        return 1;
    }

    puts("NIYAH_TRAINING_LOOP_P6_E=PASS");
    return 0;
}
