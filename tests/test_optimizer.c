#include "niyah/niyah.h"
#include "niyah/optimizer.h"
#include "niyah/train.h"

#include <float.h>
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

typedef struct StateSnapshot {
    float *weights;
    float *m;
    float *v;
    size_t bytes;
    uint64_t step;
} StateSnapshot;

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

static NiyahAdamWConfig default_optimizer_config(void)
{
    NiyahAdamWConfig c;
    c.learning_rate = 1.0e-2f;
    c.beta1 = 0.9f;
    c.beta2 = 0.999f;
    c.epsilon = 1.0e-8f;
    c.weight_decay = 0.0f;
    c.max_grad_norm = 1.0f;
    return c;
}

static int close_float(float actual, double expected, double tolerance)
{
    return fabs((double)actual - expected) <= tolerance;
}

static int all_exact_zero(const float *values, size_t count)
{
    size_t i;
    for (i = 0U; i < count; ++i) {
        if (values[i] != 0.0f) {
            return 0;
        }
    }
    return 1;
}

static int snapshot_take(const NiyahModel *model,
                         const NiyahAdamWState *state,
                         StateSnapshot *snapshot)
{
    snapshot->bytes = model->weight_count * sizeof(float);
    snapshot->weights = (float *)malloc(snapshot->bytes);
    snapshot->m = (float *)malloc(snapshot->bytes);
    snapshot->v = (float *)malloc(snapshot->bytes);
    if (snapshot->weights == NULL || snapshot->m == NULL || snapshot->v == NULL) {
        free(snapshot->weights);
        free(snapshot->m);
        free(snapshot->v);
        memset(snapshot, 0, sizeof(*snapshot));
        return 0;
    }
    memcpy(snapshot->weights, model->weights, snapshot->bytes);
    memcpy(snapshot->m, state->m, snapshot->bytes);
    memcpy(snapshot->v, state->v, snapshot->bytes);
    snapshot->step = state->step;
    return 1;
}

static int snapshot_unchanged(const NiyahModel *model,
                              const NiyahAdamWState *state,
                              const StateSnapshot *snapshot)
{
    return state->step == snapshot->step &&
           memcmp(model->weights, snapshot->weights, snapshot->bytes) == 0 &&
           memcmp(state->m, snapshot->m, snapshot->bytes) == 0 &&
           memcmp(state->v, snapshot->v, snapshot->bytes) == 0;
}

static void snapshot_destroy(StateSnapshot *snapshot)
{
    free(snapshot->weights);
    free(snapshot->m);
    free(snapshot->v);
    memset(snapshot, 0, sizeof(*snapshot));
}

static void test_state_initialization_and_identity(void)
{
    NiyahModelConfig config = test_config(0);
    NiyahModel a;
    NiyahModel b;
    NiyahModelGradients ga;
    NiyahModelGradients gb;
    NiyahAdamWState state;
    NiyahAdamWConfig opt = default_optimizer_config();
    size_t i;

    memset(&a, 0, sizeof(a));
    memset(&b, 0, sizeof(b));
    memset(&ga, 0, sizeof(ga));
    memset(&gb, 0, sizeof(gb));
    memset(&state, 0, sizeof(state));

    CHECK(niyah_model_create(&a, &config) == NIYAH_OK);
    CHECK(niyah_model_create(&b, &config) == NIYAH_OK);
    CHECK(niyah_model_gradients_create(&ga, &a) == NIYAH_OK);
    CHECK(niyah_model_gradients_create(&gb, &b) == NIYAH_OK);
    CHECK(niyah_adamw_state_create(&state, &a) == NIYAH_OK);
    CHECK(state.count == a.weight_count);
    CHECK(state.step == 0U);
    CHECK(state.bound_model == &a);
    CHECK(state.bound_weights == a.weights);
    for (i = 0U; i < state.count; ++i) {
        CHECK(state.m[i] == 0.0f);
        CHECK(state.v[i] == 0.0f);
    }

    CHECK(niyah_adamw_step(&b, &gb, &state, &opt) == NIYAH_ERR_INVALID_ARGUMENT);
    CHECK(state.step == 0U);
    CHECK(all_exact_zero(state.m, state.count));
    CHECK(all_exact_zero(state.v, state.count));

    niyah_adamw_state_destroy(&state);
    niyah_model_gradients_destroy(&gb);
    niyah_model_gradients_destroy(&ga);
    niyah_model_destroy(&b);
    niyah_model_destroy(&a);
}

static void test_structurally_different_equal_count_rejected(void)
{
    NiyahModelConfig ca;
    NiyahModelConfig cb;
    NiyahModel a;
    NiyahModel b;
    NiyahModelGradients gb;
    NiyahAdamWState state;
    NiyahAdamWConfig opt = default_optimizer_config();

    memset(&ca, 0, sizeof(ca));
    ca.vocab_size = 2U;
    ca.context_length = 4U;
    ca.embedding_dim = 4U;
    ca.n_layers = 1U;
    ca.n_heads = 2U;
    ca.n_kv_heads = 1U;
    ca.ffn_hidden_dim = 4U;
    ca.rms_norm_eps = 1.0e-5f;
    ca.tie_word_embeddings = 0;

    cb = ca;
    cb.vocab_size = 4U;
    cb.tie_word_embeddings = 1;

    memset(&a, 0, sizeof(a));
    memset(&b, 0, sizeof(b));
    memset(&gb, 0, sizeof(gb));
    memset(&state, 0, sizeof(state));

    CHECK(niyah_model_create(&a, &ca) == NIYAH_OK);
    CHECK(niyah_model_create(&b, &cb) == NIYAH_OK);
    CHECK(a.weight_count == b.weight_count);
    CHECK(a.config.vocab_size != b.config.vocab_size);
    CHECK(a.config.tie_word_embeddings != b.config.tie_word_embeddings);
    CHECK(niyah_model_gradients_create(&gb, &b) == NIYAH_OK);
    CHECK(niyah_adamw_state_create(&state, &a) == NIYAH_OK);
    CHECK(niyah_adamw_step(&b, &gb, &state, &opt) == NIYAH_ERR_INVALID_ARGUMENT);
    CHECK(state.step == 0U);

    niyah_adamw_state_destroy(&state);
    niyah_model_gradients_destroy(&gb);
    niyah_model_destroy(&b);
    niyah_model_destroy(&a);
}

static void expect_invalid_hyperparameter(NiyahModel *model,
                                          NiyahModelGradients *gradients,
                                          NiyahAdamWState *state,
                                          const NiyahAdamWConfig *config)
{
    CHECK(niyah_adamw_step(model, gradients, state, config) == NIYAH_ERR_INVALID_CONFIG);
    CHECK(state->step == 0U);
}

static void test_hyperparameter_validation(void)
{
    NiyahModelConfig config = test_config(0);
    NiyahModel model;
    NiyahModelGradients gradients;
    NiyahAdamWState state;
    NiyahAdamWConfig base = default_optimizer_config();
    NiyahAdamWConfig c;

    memset(&model, 0, sizeof(model));
    memset(&gradients, 0, sizeof(gradients));
    memset(&state, 0, sizeof(state));
    CHECK(niyah_model_create(&model, &config) == NIYAH_OK);
    CHECK(niyah_model_gradients_create(&gradients, &model) == NIYAH_OK);
    CHECK(niyah_adamw_state_create(&state, &model) == NIYAH_OK);

    c = base; c.learning_rate = 0.0f; expect_invalid_hyperparameter(&model, &gradients, &state, &c);
    c = base; c.learning_rate = -1.0f; expect_invalid_hyperparameter(&model, &gradients, &state, &c);
    c = base; c.learning_rate = NAN; expect_invalid_hyperparameter(&model, &gradients, &state, &c);
    c = base; c.learning_rate = INFINITY; expect_invalid_hyperparameter(&model, &gradients, &state, &c);

    c = base; c.beta1 = -0.01f; expect_invalid_hyperparameter(&model, &gradients, &state, &c);
    c = base; c.beta1 = 1.0f; expect_invalid_hyperparameter(&model, &gradients, &state, &c);
    c = base; c.beta1 = NAN; expect_invalid_hyperparameter(&model, &gradients, &state, &c);
    c = base; c.beta1 = INFINITY; expect_invalid_hyperparameter(&model, &gradients, &state, &c);

    c = base; c.beta2 = -0.01f; expect_invalid_hyperparameter(&model, &gradients, &state, &c);
    c = base; c.beta2 = 1.0f; expect_invalid_hyperparameter(&model, &gradients, &state, &c);
    c = base; c.beta2 = NAN; expect_invalid_hyperparameter(&model, &gradients, &state, &c);
    c = base; c.beta2 = INFINITY; expect_invalid_hyperparameter(&model, &gradients, &state, &c);

    c = base; c.epsilon = 0.0f; expect_invalid_hyperparameter(&model, &gradients, &state, &c);
    c = base; c.epsilon = -1.0f; expect_invalid_hyperparameter(&model, &gradients, &state, &c);
    c = base; c.epsilon = NAN; expect_invalid_hyperparameter(&model, &gradients, &state, &c);
    c = base; c.epsilon = INFINITY; expect_invalid_hyperparameter(&model, &gradients, &state, &c);

    c = base; c.weight_decay = -0.01f; expect_invalid_hyperparameter(&model, &gradients, &state, &c);
    c = base; c.weight_decay = NAN; expect_invalid_hyperparameter(&model, &gradients, &state, &c);
    c = base; c.weight_decay = INFINITY; expect_invalid_hyperparameter(&model, &gradients, &state, &c);

    c = base; c.max_grad_norm = 0.0f; expect_invalid_hyperparameter(&model, &gradients, &state, &c);
    c = base; c.max_grad_norm = -1.0f; expect_invalid_hyperparameter(&model, &gradients, &state, &c);
    c = base; c.max_grad_norm = NAN; expect_invalid_hyperparameter(&model, &gradients, &state, &c);
    c = base; c.max_grad_norm = INFINITY; expect_invalid_hyperparameter(&model, &gradients, &state, &c);

    CHECK(all_exact_zero(model.weights, model.weight_count));
    CHECK(all_exact_zero(state.m, state.count));
    CHECK(all_exact_zero(state.v, state.count));

    niyah_adamw_state_destroy(&state);
    niyah_model_gradients_destroy(&gradients);
    niyah_model_destroy(&model);
}

static void test_global_norm_and_clipping(void)
{
    NiyahModelConfig config = test_config(0);
    NiyahModel model;
    NiyahModelGradients gradients;
    NiyahAdamWState state;
    NiyahAdamWConfig opt = default_optimizer_config();
    double norm = -1.0;
    float *gradient_copy;
    size_t bytes;
    float *weight_copy;

    memset(&model, 0, sizeof(model));
    memset(&gradients, 0, sizeof(gradients));
    memset(&state, 0, sizeof(state));
    CHECK(niyah_model_create(&model, &config) == NIYAH_OK);
    CHECK(niyah_model_gradients_create(&gradients, &model) == NIYAH_OK);
    CHECK(niyah_adamw_state_create(&state, &model) == NIYAH_OK);

    gradients.values[0] = 3.0f;
    gradients.values[1] = 4.0f;
    CHECK(niyah_model_gradients_global_l2_norm(&gradients, &norm) == NIYAH_OK);
    CHECK(fabs(norm - 5.0) <= 1.0e-12);

    gradients.values[0] = FLT_MAX;
    gradients.values[1] = FLT_MAX;
    CHECK(niyah_model_gradients_global_l2_norm(&gradients, &norm) == NIYAH_OK);
    CHECK(isfinite(norm));
    CHECK(norm > (double)FLT_MAX);

    gradients.values[0] = 3.0f;
    gradients.values[1] = 4.0f;
    bytes = gradients.count * sizeof(float);
    gradient_copy = (float *)malloc(bytes);
    CHECK(gradient_copy != NULL);
    if (gradient_copy != NULL) {
        memcpy(gradient_copy, gradients.values, bytes);
    }

    opt.learning_rate = 1.0e-2f;
    opt.beta1 = 0.0f;
    opt.beta2 = 0.0f;
    opt.epsilon = 1.0f;
    opt.weight_decay = 0.0f;
    opt.max_grad_norm = 2.5f;
    CHECK(niyah_adamw_step(&model, &gradients, &state, &opt) == NIYAH_OK);
    CHECK(state.step == 1U);
    CHECK(close_float(state.m[0], 1.5, 1.0e-6));
    CHECK(close_float(state.m[1], 2.0, 1.0e-6));
    CHECK(close_float(state.v[0], 2.25, 1.0e-6));
    CHECK(close_float(state.v[1], 4.0, 1.0e-6));
    if (gradient_copy != NULL) {
        CHECK(memcmp(gradient_copy, gradients.values, bytes) == 0);
    }

    memset(gradients.values, 0, bytes);
    CHECK(niyah_model_gradients_global_l2_norm(&gradients, &norm) == NIYAH_OK);
    CHECK(norm == 0.0);
    weight_copy = (float *)malloc(bytes);
    CHECK(weight_copy != NULL);
    if (weight_copy != NULL) {
        memcpy(weight_copy, model.weights, bytes);
    }
    CHECK(niyah_adamw_step(&model, &gradients, &state, &opt) == NIYAH_OK);
    CHECK(state.step == 2U);
    if (weight_copy != NULL) {
        CHECK(memcmp(weight_copy, model.weights, bytes) == 0);
    }

    free(weight_copy);
    free(gradient_copy);
    niyah_adamw_state_destroy(&state);
    niyah_model_gradients_destroy(&gradients);
    niyah_model_destroy(&model);
}

static void test_hand_computed_adamw(void)
{
    NiyahModelConfig config = test_config(0);
    NiyahModel model;
    NiyahModelGradients gradients;
    NiyahAdamWState state;
    NiyahAdamWConfig opt = default_optimizer_config();
    NiyahLayerLayout layer;
    size_t first;
    size_t second;

    memset(&model, 0, sizeof(model));
    memset(&gradients, 0, sizeof(gradients));
    memset(&state, 0, sizeof(state));
    CHECK(niyah_model_create(&model, &config) == NIYAH_OK);
    CHECK(niyah_model_gradients_create(&gradients, &model) == NIYAH_OK);
    CHECK(niyah_adamw_state_create(&state, &model) == NIYAH_OK);
    CHECK(niyah_model_layer_layout(&model.config, &model.layout, 0U, &layer) == NIYAH_OK);

    first = layer.wq;
    second = layer.wk;
    model.weights[first] = 1.0f;
    model.weights[second] = -2.0f;
    gradients.values[first] = 2.0f;
    gradients.values[second] = -1.0f;

    opt.learning_rate = 0.1f;
    opt.beta1 = 0.9f;
    opt.beta2 = 0.999f;
    opt.epsilon = 1.0e-8f;
    opt.weight_decay = 0.0f;
    opt.max_grad_norm = 10.0f;

    CHECK(niyah_adamw_step(&model, &gradients, &state, &opt) == NIYAH_OK);
    CHECK(state.step == 1U);
    CHECK(close_float(state.m[first], 0.2, 2.0e-7));
    CHECK(close_float(state.v[first], 0.004, 2.0e-8));
    CHECK(close_float(model.weights[first], 0.9000000005, 2.0e-6));
    CHECK(close_float(state.m[second], -0.1, 2.0e-7));
    CHECK(close_float(state.v[second], 0.001, 2.0e-8));
    CHECK(close_float(model.weights[second], -1.900000001, 2.0e-6));

    memset(gradients.values, 0, gradients.count * sizeof(float));
    gradients.values[first] = 1.0f;
    CHECK(niyah_adamw_step(&model, &gradients, &state, &opt) == NIYAH_OK);
    CHECK(state.step == 2U);
    CHECK(close_float(state.m[first], 0.28, 2.0e-6));
    CHECK(close_float(state.v[first], 0.004996, 2.0e-7));
    CHECK(close_float(model.weights[first], 0.8067820372, 3.0e-6));

    niyah_adamw_state_destroy(&state);
    niyah_model_gradients_destroy(&gradients);
    niyah_model_destroy(&model);
}

static void test_decay_policy(int tied)
{
    NiyahModelConfig config = test_config(tied);
    NiyahModel model;
    NiyahModelGradients gradients;
    NiyahAdamWState state;
    NiyahAdamWConfig opt = default_optimizer_config();
    NiyahLayerLayout layer;
    size_t i;
    const double expected_decay = 1.998;

    memset(&model, 0, sizeof(model));
    memset(&gradients, 0, sizeof(gradients));
    memset(&state, 0, sizeof(state));
    CHECK(niyah_model_create(&model, &config) == NIYAH_OK);
    CHECK(niyah_model_gradients_create(&gradients, &model) == NIYAH_OK);
    CHECK(niyah_adamw_state_create(&state, &model) == NIYAH_OK);
    CHECK(niyah_model_layer_layout(&model.config, &model.layout, 0U, &layer) == NIYAH_OK);

    for (i = 0U; i < model.weight_count; ++i) {
        model.weights[i] = 2.0f;
    }
    opt.learning_rate = 0.01f;
    opt.weight_decay = 0.1f;
    opt.max_grad_norm = 1.0f;

    CHECK(niyah_adamw_step(&model, &gradients, &state, &opt) == NIYAH_OK);
    CHECK(close_float(model.weights[model.layout.token_embedding], expected_decay, 2.0e-6));
    CHECK(close_float(model.weights[layer.wq], expected_decay, 2.0e-6));
    CHECK(close_float(model.weights[layer.wk], expected_decay, 2.0e-6));
    CHECK(close_float(model.weights[layer.wv], expected_decay, 2.0e-6));
    CHECK(close_float(model.weights[layer.wo], expected_decay, 2.0e-6));
    CHECK(close_float(model.weights[layer.w_gate], expected_decay, 2.0e-6));
    CHECK(close_float(model.weights[layer.w_up], expected_decay, 2.0e-6));
    CHECK(close_float(model.weights[layer.w_down], expected_decay, 2.0e-6));
    CHECK(model.weights[layer.attn_norm] == 2.0f);
    CHECK(model.weights[layer.ffn_norm] == 2.0f);
    CHECK(model.weights[model.layout.final_norm] == 2.0f);
    CHECK(all_exact_zero(state.m, state.count));
    CHECK(all_exact_zero(state.v, state.count));

    if (tied != 0) {
        CHECK(model.layout.lm_head == model.layout.token_embedding);
        CHECK(close_float(model.weights[model.layout.lm_head], expected_decay, 2.0e-6));
    } else {
        CHECK(model.layout.lm_head != model.layout.token_embedding);
        CHECK(close_float(model.weights[model.layout.lm_head], expected_decay, 2.0e-6));
    }

    niyah_adamw_state_destroy(&state);
    niyah_model_gradients_destroy(&gradients);
    niyah_model_destroy(&model);
}

static void test_zero_decay_zero_gradient(void)
{
    NiyahModelConfig config = test_config(0);
    NiyahModel model;
    NiyahModelGradients gradients;
    NiyahAdamWState state;
    NiyahAdamWConfig opt = default_optimizer_config();
    NiyahLayerLayout layer;

    memset(&model, 0, sizeof(model));
    memset(&gradients, 0, sizeof(gradients));
    memset(&state, 0, sizeof(state));
    CHECK(niyah_model_create(&model, &config) == NIYAH_OK);
    CHECK(niyah_model_gradients_create(&gradients, &model) == NIYAH_OK);
    CHECK(niyah_adamw_state_create(&state, &model) == NIYAH_OK);
    CHECK(niyah_model_layer_layout(&model.config, &model.layout, 0U, &layer) == NIYAH_OK);
    model.weights[layer.wq] = 2.0f;
    opt.weight_decay = 0.0f;
    CHECK(niyah_adamw_step(&model, &gradients, &state, &opt) == NIYAH_OK);
    CHECK(model.weights[layer.wq] == 2.0f);
    CHECK(all_exact_zero(state.m, state.count));
    CHECK(all_exact_zero(state.v, state.count));

    niyah_adamw_state_destroy(&state);
    niyah_model_gradients_destroy(&gradients);
    niyah_model_destroy(&model);
}

static void check_rejection_preserves(NiyahModel *model,
                                      NiyahModelGradients *gradients,
                                      NiyahAdamWState *state,
                                      const NiyahAdamWConfig *opt,
                                      NiyahStatus expected)
{
    StateSnapshot snapshot;
    memset(&snapshot, 0, sizeof(snapshot));
    CHECK(snapshot_take(model, state, &snapshot));
    if (snapshot.weights != NULL) {
        CHECK(niyah_adamw_step(model, gradients, state, opt) == expected);
        CHECK(snapshot_unchanged(model, state, &snapshot));
    }
    snapshot_destroy(&snapshot);
}

static void test_failure_atomicity_and_corrupt_state(void)
{
    NiyahModelConfig config = test_config(0);
    NiyahModel model;
    NiyahModelGradients gradients;
    NiyahAdamWState state;
    NiyahAdamWConfig opt = default_optimizer_config();
    float *saved_gradient_values;
    float *saved_m;
    size_t i;

    memset(&model, 0, sizeof(model));
    memset(&gradients, 0, sizeof(gradients));
    memset(&state, 0, sizeof(state));
    CHECK(niyah_model_create(&model, &config) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(&model, UINT64_C(1234)) == NIYAH_OK);
    CHECK(niyah_model_gradients_create(&gradients, &model) == NIYAH_OK);
    CHECK(niyah_adamw_state_create(&state, &model) == NIYAH_OK);
    for (i = 0U; i < state.count; ++i) {
        state.m[i] = 0.01f;
        state.v[i] = 0.02f;
    }

    gradients.values[0] = NAN;
    check_rejection_preserves(&model, &gradients, &state, &opt, NIYAH_ERR_INVALID_ARGUMENT);
    gradients.values[0] = INFINITY;
    check_rejection_preserves(&model, &gradients, &state, &opt, NIYAH_ERR_INVALID_ARGUMENT);
    gradients.values[0] = -INFINITY;
    check_rejection_preserves(&model, &gradients, &state, &opt, NIYAH_ERR_INVALID_ARGUMENT);
    gradients.values[0] = 0.0f;

    state.step = UINT64_MAX;
    check_rejection_preserves(&model, &gradients, &state, &opt, NIYAH_ERR_OVERFLOW);
    state.step = 0U;

    state.m[0] = NAN;
    check_rejection_preserves(&model, &gradients, &state, &opt, NIYAH_ERR_INVALID_CONFIG);
    state.m[0] = 0.01f;
    state.v[0] = -1.0f;
    check_rejection_preserves(&model, &gradients, &state, &opt, NIYAH_ERR_INVALID_CONFIG);
    state.v[0] = INFINITY;
    check_rejection_preserves(&model, &gradients, &state, &opt, NIYAH_ERR_INVALID_CONFIG);
    state.v[0] = 0.02f;

    model.weights[0] = INFINITY;
    check_rejection_preserves(&model, &gradients, &state, &opt, NIYAH_ERR_INVALID_CONFIG);
    model.weights[0] = 1.0f;

    model.weights[0] = FLT_MAX;
    opt.learning_rate = FLT_MAX;
    opt.weight_decay = FLT_MAX;
    check_rejection_preserves(&model, &gradients, &state, &opt, NIYAH_ERR_OVERFLOW);
    model.weights[0] = 1.0f;
    opt = default_optimizer_config();

    gradients.values[0] = FLT_MAX;
    opt.beta1 = 0.0f;
    opt.beta2 = 0.0f;
    opt.max_grad_norm = FLT_MAX;
    check_rejection_preserves(&model, &gradients, &state, &opt, NIYAH_ERR_OVERFLOW);
    gradients.values[0] = 0.0f;
    opt = default_optimizer_config();

    saved_gradient_values = gradients.values;
    gradients.values = model.weights;
    CHECK(niyah_adamw_step(&model, &gradients, &state, &opt) == NIYAH_ERR_INVALID_ARGUMENT);
    gradients.values = saved_gradient_values;

    saved_m = state.m;
    state.m = model.weights;
    CHECK(niyah_adamw_step(&model, &gradients, &state, &opt) == NIYAH_ERR_INVALID_ARGUMENT);
    state.m = saved_m;

    niyah_adamw_state_destroy(&state);
    niyah_model_gradients_destroy(&gradients);
    niyah_model_destroy(&model);
}

static void run_training_chain(int tied)
{
    const uint32_t tokens[3] = {1U, 2U, 3U};
    const uint32_t targets[3] = {2U, 3U, 4U};
    NiyahModelConfig config = test_config(tied);
    NiyahModel a;
    NiyahModel b;
    NiyahModelGradients ga;
    NiyahModelGradients gb;
    NiyahAdamWState sa;
    NiyahAdamWState sb;
    NiyahAdamWConfig opt = default_optimizer_config();
    size_t workspace_count = 0U;
    float *wa = NULL;
    float *wb = NULL;
    float initial_loss = NAN;
    float loss_a = NAN;
    float loss_b = NAN;
    float final_loss = NAN;
    size_t step;

    memset(&a, 0, sizeof(a));
    memset(&b, 0, sizeof(b));
    memset(&ga, 0, sizeof(ga));
    memset(&gb, 0, sizeof(gb));
    memset(&sa, 0, sizeof(sa));
    memset(&sb, 0, sizeof(sb));

    CHECK(niyah_model_create(&a, &config) == NIYAH_OK);
    CHECK(niyah_model_create(&b, &config) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(&a, UINT64_C(20260916)) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(&b, UINT64_C(20260916)) == NIYAH_OK);
    CHECK(niyah_model_gradients_create(&ga, &a) == NIYAH_OK);
    CHECK(niyah_model_gradients_create(&gb, &b) == NIYAH_OK);
    CHECK(niyah_adamw_state_create(&sa, &a) == NIYAH_OK);
    CHECK(niyah_adamw_state_create(&sb, &b) == NIYAH_OK);
    CHECK(niyah_train_backward_workspace_floats(&config, 3U, &workspace_count) == NIYAH_OK);
    wa = (float *)calloc(workspace_count, sizeof(float));
    wb = (float *)calloc(workspace_count, sizeof(float));
    CHECK(wa != NULL);
    CHECK(wb != NULL);

    opt.learning_rate = 5.0e-3f;
    opt.beta1 = 0.9f;
    opt.beta2 = 0.999f;
    opt.epsilon = 1.0e-8f;
    opt.weight_decay = 0.0f;
    opt.max_grad_norm = 1.0f;

    if (wa != NULL && wb != NULL) {
        for (step = 0U; step < 32U; ++step) {
            CHECK(niyah_train_backward(&a, tokens, targets, 3U, &loss_a,
                                       &ga, wa, workspace_count) == NIYAH_OK);
            CHECK(niyah_train_backward(&b, tokens, targets, 3U, &loss_b,
                                       &gb, wb, workspace_count) == NIYAH_OK);
            CHECK(isfinite(loss_a));
            CHECK(isfinite(loss_b));
            CHECK(loss_a == loss_b);
            CHECK(memcmp(ga.values, gb.values, ga.count * sizeof(float)) == 0);
            if (step == 0U) {
                initial_loss = loss_a;
            }
            CHECK(niyah_adamw_step(&a, &ga, &sa, &opt) == NIYAH_OK);
            CHECK(niyah_adamw_step(&b, &gb, &sb, &opt) == NIYAH_OK);
            CHECK(sa.step == (uint64_t)(step + 1U));
            CHECK(sb.step == sa.step);
            CHECK(memcmp(a.weights, b.weights, a.weight_count * sizeof(float)) == 0);
            CHECK(memcmp(sa.m, sb.m, sa.count * sizeof(float)) == 0);
            CHECK(memcmp(sa.v, sb.v, sa.count * sizeof(float)) == 0);
        }
        CHECK(niyah_train_backward(&a, tokens, targets, 3U, &final_loss,
                                   &ga, wa, workspace_count) == NIYAH_OK);
        CHECK(isfinite(initial_loss));
        CHECK(isfinite(final_loss));
        CHECK(final_loss < initial_loss * 0.98f);
        CHECK(sa.step == 32U);
        CHECK(sb.step == 32U);
    }

    free(wb);
    free(wa);
    niyah_adamw_state_destroy(&sb);
    niyah_adamw_state_destroy(&sa);
    niyah_model_gradients_destroy(&gb);
    niyah_model_gradients_destroy(&ga);
    niyah_model_destroy(&b);
    niyah_model_destroy(&a);
}

int main(void)
{
    test_state_initialization_and_identity();
    test_structurally_different_equal_count_rejected();
    test_hyperparameter_validation();
    test_global_norm_and_clipping();
    test_hand_computed_adamw();
    test_decay_policy(1);
    test_decay_policy(0);
    test_zero_decay_zero_gradient();
    test_failure_atomicity_and_corrupt_state();
    run_training_chain(1);
    run_training_chain(0);

    if (failures != 0) {
        fprintf(stderr, "niyah_optimizer_test: %d failure(s)\n", failures);
        return 1;
    }
    puts("NIYAH_ADAMW_P6_A=PASS");
    return 0;
}
