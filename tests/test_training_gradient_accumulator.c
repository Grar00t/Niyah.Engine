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
    return NIYAH_OK;
}


static float max_abs_diff(
    const float *a,
    const float *b,
    size_t count)
{
    float result = 0.0f;
    size_t i;

    for (i = 0U; i < count; ++i) {
        const float d = fabsf(a[i] - b[i]);

        if (d > result)
            result = d;
    }

    return result;
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
