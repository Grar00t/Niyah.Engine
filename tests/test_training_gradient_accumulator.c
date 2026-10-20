#include "test_numeric_compare.h"
#include "niyah/training_loop.h"

#include <math.h>
#include <stdint.h>
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


typedef struct SampleProviderContext {
    const NiyahTrainingSample *samples;
    size_t count;
} SampleProviderContext;


typedef struct FakeGradientBackend {
    const float *current;
    float *accumulated;
    size_t count;
    size_t begin_calls;
    size_t accumulate_calls;
    size_t scale_calls;
    size_t finalize_calls;
    int fail_phase;
    size_t fail_call;
} FakeGradientBackend;


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


static NiyahStatus sample_provider(
    size_t sample_index,
    NiyahTrainingSample *out_sample,
    void *user_data)
{
    SampleProviderContext *context =
        (SampleProviderContext *)user_data;

    if (context == NULL ||
        out_sample == NULL ||
        sample_index >= context->count) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    *out_sample = context->samples[sample_index];
    return NIYAH_OK;
}


static NiyahStatus fake_backward(
    const NiyahModel *model,
    const NiyahTrainingSample *sample,
    NiyahModelGradients *gradients,
    float *workspace,
    size_t workspace_count,
    float *out_loss,
    void *user_data)
{
    FakeGradientBackend *context =
        (FakeGradientBackend *)user_data;
    NiyahStatus status;

    if (context == NULL ||
        gradients == NULL ||
        gradients->values == NULL ||
        gradients->count != context->count) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    status = niyah_train_backward_masked(
        model,
        sample->tokens,
        sample->targets,
        sample->token_count,
        sample->loss_start,
        out_loss,
        gradients,
        workspace,
        workspace_count);

    if (status == NIYAH_OK)
        context->current = gradients->values;

    return status;
}


static NiyahStatus fake_begin(void *user_data)
{
    FakeGradientBackend *context =
        (FakeGradientBackend *)user_data;

    if (context == NULL ||
        context->accumulated == NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;

    memset(
        context->accumulated,
        0,
        context->count * sizeof(float));

    context->current = NULL;
    context->begin_calls += 1U;
    if (context->fail_phase == 1) return NIYAH_ERR_IO;

    return NIYAH_OK;
}


static NiyahStatus fake_accumulate(
    float scale,
    void *user_data)
{
    FakeGradientBackend *context =
        (FakeGradientBackend *)user_data;
    size_t i;

    if (context == NULL ||
        context->current == NULL ||
        context->accumulated == NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;

    for (i = 0U; i < context->count; ++i) {
        const float scaled =
            context->current[i] * scale;

        context->accumulated[i] =
            context->accumulated[i] + scaled;
    }

    context->accumulate_calls += 1U;
    if (context->fail_phase == 2 && context->accumulate_calls == context->fail_call)
        return NIYAH_ERR_IO;
    return NIYAH_OK;
}


static NiyahStatus fake_scale(
    float scale,
    void *user_data)
{
    FakeGradientBackend *context =
        (FakeGradientBackend *)user_data;
    size_t i;

    if (context == NULL ||
        context->accumulated == NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;

    for (i = 0U; i < context->count; ++i)
        context->accumulated[i] *= scale;

    context->scale_calls += 1U;
    if (context->fail_phase == 3 && context->scale_calls == context->fail_call)
        return NIYAH_ERR_IO;
    return NIYAH_OK;
}


static NiyahStatus fake_finalize(
    NiyahModelGradients *out_gradients,
    void *user_data)
{
    FakeGradientBackend *context =
        (FakeGradientBackend *)user_data;

    if (context == NULL ||
        out_gradients == NULL ||
        out_gradients->values == NULL ||
        out_gradients->count != context->count)
        return NIYAH_ERR_INVALID_ARGUMENT;

    memcpy(
        out_gradients->values,
        context->accumulated,
        context->count * sizeof(float));

    context->finalize_calls += 1U;
    if (context->fail_phase == 4) return NIYAH_ERR_IO;
    return NIYAH_OK;
}


static float max_abs_diff(
    const float *a,
    const float *b,
    size_t count)
{
    return niyah_test_max_abs_diff(a, b, count);
}



static void test_nonfinite_comparisons(void)
{
    const float normal[2] = {0.0f, 1.0f};
    float bad[2] = {0.0f, NAN};
    CHECK(isinf(max_abs_diff(normal, bad, 2U)));
    CHECK(isinf(max_abs_diff(bad, bad, 2U)));
    bad[1] = INFINITY;
    CHECK(isinf(max_abs_diff(normal, bad, 2U)));
    CHECK(isinf(max_abs_diff(bad, bad, 2U)));
    bad[1] = -INFINITY;
    CHECK(isinf(max_abs_diff(normal, bad, 2U)));
    CHECK(max_abs_diff(normal, normal, 2U) == 0.0f);
}

static NiyahStatus run_backend(NiyahModel *model, NiyahAdamWState *state,
    NiyahDatasetCursor *cursor, SampleProviderContext *provider,
    FakeGradientBackend *backend, const NiyahAdamWConfig *opt, float *loss)
{
    const NiyahTrainingGradientAccumulatorOps ops = {
        fake_begin, fake_accumulate, fake_scale, fake_finalize
    };
    return niyah_training_run_updates_with_progress_with_provider_and_accumulator(
        model, 2U, 3U, sample_provider, provider, cursor, state, opt,
        2U, 1U, 1U, fake_backward, backend, &ops, NULL, NULL, loss);
}


/* Synthetic mean gradients are an independent weighting oracle, not CUDA
 * evidence. For supervised lengths 1 and 2: (-g + 2*(2*g))/3 == g,
 * and (2 + 2*5)/3 == 4. Both masks are nonzero. */
static NiyahStatus analytical_backward(const NiyahModel *model,
    const NiyahTrainingSample *sample, NiyahModelGradients *gradients,
    float *workspace, size_t workspace_count, float *loss, void *user_data)
{
    FakeGradientBackend *backend = (FakeGradientBackend *)user_data;
    size_t i;
    const float factor = sample->loss_start == 2U ? -1.0f : 2.0f;
    (void)model; (void)workspace; (void)workspace_count;
    for (i = 0U; i < gradients->count; ++i)
        gradients->values[i] = factor * (i % 2U == 0U ? 0.125f : -0.25f);
    *loss = sample->loss_start == 2U ? 2.0f : 5.0f;
    backend->current = gradients->values;
    return NIYAH_OK;
}

static void test_independent_three_target_weighting(void)
{
    const uint32_t tokens[3] = {1U, 2U, 3U}, targets[3] = {2U, 3U, 4U};
    const NiyahTrainingSample samples[2] = {
        {tokens, targets, 3U, 2U}, {tokens, targets, 3U, 1U}
    };
    const NiyahTrainingGradientAccumulatorOps ops = {
        fake_begin, fake_accumulate, fake_scale, fake_finalize
    };
    NiyahModelConfig config = test_config();
    NiyahAdamWConfig opt = optimizer_config();
    NiyahModel actual = {0}, expected = {0};
    NiyahModelGradients expected_gradient = {0};
    NiyahAdamWState state = {0}, expected_state = {0};
    NiyahDatasetCursor cursor = {0};
    SampleProviderContext provider = {samples, 2U};
    FakeGradientBackend backend = {0};
    float loss = NAN;
    size_t i;
    CHECK(niyah_model_create(&actual, &config) == NIYAH_OK);
    CHECK(niyah_model_create(&expected, &config) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(&actual, 992U) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(&expected, 992U) == NIYAH_OK);
    CHECK(niyah_adamw_state_create(&state, &actual) == NIYAH_OK);
    CHECK(niyah_adamw_state_create(&expected_state, &expected) == NIYAH_OK);
    CHECK(niyah_model_gradients_create(&expected_gradient, &expected) == NIYAH_OK);
    CHECK(niyah_dataset_cursor_init(&cursor, 2U, 993U) == NIYAH_OK);
    backend.count = actual.weight_count;
    backend.accumulated = (float *)calloc(backend.count, sizeof(float));
    CHECK(backend.accumulated != NULL);
    if (backend.accumulated != NULL) {
        for (i = 0U; i < expected_gradient.count; ++i)
            expected_gradient.values[i] = i % 2U == 0U ? 0.125f : -0.25f;
        CHECK(niyah_adamw_step(&expected, &expected_gradient, &expected_state, &opt) == NIYAH_OK);
        CHECK(niyah_training_run_updates_with_progress_with_provider_and_accumulator(
            &actual, 2U, 3U, sample_provider, &provider, &cursor, &state, &opt,
            2U, 1U, 1U, analytical_backward, &backend, &ops,
            NULL, NULL, &loss) == NIYAH_OK);
        CHECK(loss == 4.0f);
        CHECK(memcmp(backend.accumulated, expected_gradient.values,
                     expected_gradient.count * sizeof(float)) == 0);
        CHECK(memcmp(actual.weights, expected.weights, actual.weight_count * sizeof(float)) == 0);
        CHECK(memcmp(state.m, expected_state.m, state.count * sizeof(float)) == 0);
        CHECK(memcmp(state.v, expected_state.v, state.count * sizeof(float)) == 0);
    }
    free(backend.accumulated);
    niyah_dataset_cursor_destroy(&cursor);
    niyah_model_gradients_destroy(&expected_gradient);
    niyah_adamw_state_destroy(&state); niyah_adamw_state_destroy(&expected_state);
    niyah_model_destroy(&actual); niyah_model_destroy(&expected);
}

static void test_callback_failures_and_recovery(void)
{
    const uint32_t tokens[3] = {1U, 2U, 3U}, targets[3] = {2U, 3U, 4U};
    const NiyahTrainingSample samples[2] = {
        {tokens, targets, 3U, 2U}, {tokens, targets, 3U, 1U}
    };
    const int phases[5] = {1, 2, 3, 3, 4};
    const size_t fail_calls[5] = {1U, 2U, 1U, 2U, 1U};
    size_t scenario;
    for (scenario = 0U; scenario < 5U; ++scenario) {
        NiyahModelConfig config = test_config();
        NiyahAdamWConfig opt = optimizer_config();
        NiyahModel model = {0}, reference = {0};
        NiyahAdamWState state = {0}, ref_state = {0};
        NiyahDatasetCursor cursor = {0}, ref_cursor = {0}, before_cursor;
        FakeGradientBackend backend = {0};
        SampleProviderContext provider = {samples, 2U};
        float *snapshot;
        size_t bytes, before_order[2];
        uint64_t before_step;
        float loss = -99.0f, reference_loss = NAN;
        uint8_t identity[32] = {1U};
        CHECK(niyah_model_create(&model, &config) == NIYAH_OK);
        CHECK(niyah_model_create(&reference, &config) == NIYAH_OK);
        CHECK(niyah_model_reset_parameters(&model, 76U) == NIYAH_OK);
        CHECK(niyah_model_reset_parameters(&reference, 76U) == NIYAH_OK);
        CHECK(niyah_adamw_state_create(&state, &model) == NIYAH_OK);
        CHECK(niyah_adamw_state_create(&ref_state, &reference) == NIYAH_OK);
        CHECK(niyah_dataset_cursor_init(&cursor, 2U, 77U) == NIYAH_OK);
        CHECK(niyah_dataset_cursor_init(&ref_cursor, 2U, 77U) == NIYAH_OK);
        CHECK(niyah_dataset_cursor_bind_identity(&cursor, identity) == NIYAH_OK);
        CHECK(niyah_dataset_cursor_bind_checkpoint_identity(&cursor, identity) == NIYAH_OK);
        CHECK(niyah_dataset_cursor_bind_identity(&ref_cursor, identity) == NIYAH_OK);
        CHECK(niyah_dataset_cursor_bind_checkpoint_identity(&ref_cursor, identity) == NIYAH_OK);
        backend.count = model.weight_count;
        backend.accumulated = (float *)calloc(backend.count, sizeof(float));
        bytes = model.weight_count * sizeof(float);
        snapshot = (float *)malloc(3U * bytes);
        CHECK(backend.accumulated != NULL && snapshot != NULL);
        if (backend.accumulated != NULL && snapshot != NULL) {
            /* A completed group creates nonzero optimizer moments. */
            CHECK(run_backend(&model, &state, &cursor, &provider, &backend, &opt, &loss) == NIYAH_OK);
            CHECK(niyah_training_run_updates(&reference, samples, 2U, &ref_cursor,
                  &ref_state, &opt, 2U, 1U, 1U, &reference_loss) == NIYAH_OK);
            CHECK(niyah_dataset_cursor_seek(&cursor, 4U, 2U) == NIYAH_OK);
            CHECK(niyah_dataset_cursor_seek(&ref_cursor, 4U, 2U) == NIYAH_OK);
            before_cursor = cursor;
            memcpy(before_order, cursor.order, sizeof(before_order));
            before_step = state.step;
            memcpy(snapshot, model.weights, bytes);
            memcpy(snapshot + model.weight_count, state.m, bytes);
            memcpy(snapshot + 2U * model.weight_count, state.v, bytes);
            backend.begin_calls = backend.accumulate_calls = 0U;
            backend.scale_calls = backend.finalize_calls = 0U;
            backend.fail_phase = phases[scenario];
            backend.fail_call = fail_calls[scenario];
            loss = -99.0f;
            CHECK(run_backend(&model, &state, &cursor, &provider, &backend, &opt, &loss) == NIYAH_ERR_IO);
            CHECK(memcmp(&before_cursor, &cursor, sizeof(cursor)) == 0);
            CHECK(memcmp(before_order, cursor.order, sizeof(before_order)) == 0);
            CHECK(memcmp(snapshot, model.weights, bytes) == 0);
            CHECK(memcmp(snapshot + model.weight_count, state.m, bytes) == 0);
            CHECK(memcmp(snapshot + 2U * model.weight_count, state.v, bytes) == 0);
            CHECK(state.step == before_step && loss == -99.0f);
            backend.fail_phase = 0;
            CHECK(run_backend(&model, &state, &cursor, &provider, &backend, &opt, &loss) == NIYAH_OK);
            CHECK(niyah_training_run_updates(&reference, samples, 2U, &ref_cursor,
                  &ref_state, &opt, 2U, 1U, 1U, &reference_loss) == NIYAH_OK);
            CHECK(max_abs_diff(model.weights, reference.weights, model.weight_count) <= 1.0e-6f);
            CHECK(max_abs_diff(state.m, ref_state.m, state.count) <= 1.0e-6f);
            CHECK(max_abs_diff(state.v, ref_state.v, state.count) <= 1.0e-6f);
            CHECK(isfinite(loss) && loss == reference_loss);
            CHECK(cursor.epoch == ref_cursor.epoch && cursor.position == ref_cursor.position);
            CHECK(memcmp(cursor.order, ref_cursor.order, 2U * sizeof(size_t)) == 0);
        }
        free(snapshot); free(backend.accumulated);
        niyah_dataset_cursor_destroy(&cursor); niyah_dataset_cursor_destroy(&ref_cursor);
        niyah_adamw_state_destroy(&state); niyah_adamw_state_destroy(&ref_state);
        niyah_model_destroy(&model); niyah_model_destroy(&reference);
    }
}

int main(void)
{
    static const uint32_t short_tokens[] = {1U};
    static const uint32_t short_targets[] = {2U};

    static const uint32_t long_tokens[] = {
        1U, 2U, 3U
    };
    static const uint32_t long_targets[] = {
        2U, 3U, 4U
    };

    const NiyahTrainingSample samples[] = {
        {
            short_tokens,
            short_targets,
            1U,
            0U
        },
        {
            long_tokens,
            long_targets,
            3U,
            0U
        }
    };

    const NiyahTrainingGradientAccumulatorOps ops = {
        fake_begin,
        fake_accumulate,
        fake_scale,
        fake_finalize
    };

    NiyahModelConfig config = test_config();
    NiyahAdamWConfig opt = optimizer_config();

    NiyahModel expected;
    NiyahModel actual;

    NiyahAdamWState expected_state;
    NiyahAdamWState actual_state;

    NiyahDatasetCursor expected_cursor;
    NiyahDatasetCursor actual_cursor;

    SampleProviderContext provider_context;
    FakeGradientBackend backend;

    float expected_loss = NAN;
    float actual_loss = NAN;

    float weight_error;
    float m_error;
    float v_error;

    memset(&expected, 0, sizeof(expected));
    memset(&actual, 0, sizeof(actual));
    memset(&expected_state, 0, sizeof(expected_state));
    memset(&actual_state, 0, sizeof(actual_state));
    memset(&expected_cursor, 0, sizeof(expected_cursor));
    memset(&actual_cursor, 0, sizeof(actual_cursor));
    memset(&backend, 0, sizeof(backend));

    test_nonfinite_comparisons();
    test_callback_failures_and_recovery();
    test_independent_three_target_weighting();

    provider_context.samples = samples;
    provider_context.count = 2U;

    CHECK(
        niyah_model_create(
            &expected,
            &config) == NIYAH_OK);

    CHECK(
        niyah_model_create(
            &actual,
            &config) == NIYAH_OK);

    CHECK(
        niyah_model_reset_parameters(
            &expected,
            UINT64_C(20260925)) == NIYAH_OK);

    CHECK(
        niyah_model_reset_parameters(
            &actual,
            UINT64_C(20260925)) == NIYAH_OK);

    CHECK(
        niyah_adamw_state_create(
            &expected_state,
            &expected) == NIYAH_OK);

    CHECK(
        niyah_adamw_state_create(
            &actual_state,
            &actual) == NIYAH_OK);

    CHECK(
        niyah_dataset_cursor_init(
            &expected_cursor,
            2U,
            UINT64_C(42)) == NIYAH_OK);

    CHECK(
        niyah_dataset_cursor_init(
            &actual_cursor,
            2U,
            UINT64_C(42)) == NIYAH_OK);

    backend.count = actual.weight_count;
    backend.accumulated = (float *)calloc(
        backend.count,
        sizeof(float));

    CHECK(backend.accumulated != NULL);

    if (backend.accumulated != NULL) {
        CHECK(
            niyah_training_run_updates(
                &expected,
                samples,
                2U,
                &expected_cursor,
                &expected_state,
                &opt,
                2U,
                1U,
                1U,
                &expected_loss) == NIYAH_OK);

        CHECK(
            niyah_training_run_updates_with_progress_with_provider_and_accumulator(
                &actual,
                2U,
                3U,
                sample_provider,
                &provider_context,
                &actual_cursor,
                &actual_state,
                &opt,
                2U,
                1U,
                1U,
                fake_backward,
                &backend,
                &ops,
                NULL,
                NULL,
                &actual_loss) == NIYAH_OK);

        weight_error = max_abs_diff(
            expected.weights,
            actual.weights,
            actual.weight_count);

        m_error = max_abs_diff(
            expected_state.m,
            actual_state.m,
            actual_state.count);

        v_error = max_abs_diff(
            expected_state.v,
            actual_state.v,
            actual_state.count);

        printf(
            "gradient_accumulator_weight_max_abs=%.9g\n",
            weight_error);

        printf(
            "gradient_accumulator_m_max_abs=%.9g\n",
            m_error);

        printf(
            "gradient_accumulator_v_max_abs=%.9g\n",
            v_error);

        CHECK(isfinite(expected_loss));
        CHECK(actual_loss == expected_loss);

        CHECK(expected_state.step == UINT64_C(1));
        CHECK(actual_state.step == expected_state.step);

        CHECK(
            expected_cursor.epoch ==
            actual_cursor.epoch);

        CHECK(
            expected_cursor.position ==
            actual_cursor.position);

        CHECK(backend.begin_calls == 1U);
        CHECK(backend.accumulate_calls == 2U);

        /*
         * Mixed supervised lengths cause:
         * 1. one retroactive weighting scale
         * 2. one final normalization scale
         */
        CHECK(backend.scale_calls == 2U);
        CHECK(backend.finalize_calls == 1U);

        CHECK(weight_error <= 1.0e-6f);
        CHECK(m_error <= 1.0e-6f);
        CHECK(v_error <= 1.0e-6f);
    }

    free(backend.accumulated);

    niyah_dataset_cursor_destroy(
        &actual_cursor);
    niyah_dataset_cursor_destroy(
        &expected_cursor);

    niyah_adamw_state_destroy(
        &actual_state);
    niyah_adamw_state_destroy(
        &expected_state);

    niyah_model_destroy(&actual);
    niyah_model_destroy(&expected);

    if (failures != 0) {
        fprintf(
            stderr,
            "gradient_accumulator_failures=%d\n",
            failures);
        return 1;
    }

    puts("TRAINING_GRADIENT_ACCUMULATOR=PASS");
    return 0;
}
