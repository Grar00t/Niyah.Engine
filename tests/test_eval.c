#include "niyah/eval.h"
#include "niyah/train.h"
#include "niyah/transformer.h"

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

static int nearly_equal(double a, double b, double tolerance)
{
    return fabs(a - b) <= tolerance;
}

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

static void test_uniform_model_perplexity(void)
{
    static const uint32_t t0[] = {1U};
    static const uint32_t y0[] = {2U};
    static const uint32_t t1[] = {1U, 2U, 3U};
    static const uint32_t y1[] = {2U, 3U, 4U};
    const NiyahEvaluationSample samples[] = {
        {t0, y0, 1U, 0U},
        {t1, y1, 3U, 0U}
    };
    NiyahModelConfig config = test_config();
    NiyahModel model;
    NiyahEvaluationMetrics metrics;

    memset(&model, 0, sizeof(model));
    memset(&metrics, 0, sizeof(metrics));

    CHECK(niyah_model_create(&model, &config) == NIYAH_OK);
    if (model.weights != NULL) {
        memset(model.weights, 0, model.weight_count * sizeof(float));
        CHECK(niyah_evaluate(
                  &model, samples, 2U, &metrics) == NIYAH_OK);
        CHECK(metrics.sample_count == 2U);
        CHECK(metrics.token_count == 4U);
        CHECK(nearly_equal(metrics.mean_loss, log(8.0), 1.0e-6));
        CHECK(nearly_equal(metrics.perplexity, 8.0, 1.0e-5));
    }

    niyah_model_destroy(&model);
}

static void test_weighted_mean_matches_direct_losses(void)
{
    static const uint32_t t0[] = {1U};
    static const uint32_t y0[] = {2U};
    static const uint32_t t1[] = {1U, 2U, 3U};
    static const uint32_t y1[] = {2U, 3U, 4U};
    const NiyahEvaluationSample samples[] = {
        {t0, y0, 1U, 0U},
        {t1, y1, 3U, 0U}
    };
    NiyahModelConfig config = test_config();
    NiyahModel model;
    NiyahEvaluationMetrics metrics;
    size_t workspace_count = 0U;
    float *workspace = NULL;
    float logits[24];
    float loss0 = NAN;
    float loss1 = NAN;
    double expected;

    memset(&model, 0, sizeof(model));
    memset(&metrics, 0, sizeof(metrics));

    CHECK(niyah_model_create(&model, &config) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(
              &model, UINT64_C(20260916)) == NIYAH_OK);
    CHECK(niyah_transformer_workspace_floats(
              &config, 3U, &workspace_count) == NIYAH_OK);

    workspace = (float *)calloc(workspace_count, sizeof(float));
    CHECK(workspace != NULL);

    if (workspace != NULL) {
        CHECK(niyah_train_loss(
                  &model, t0, y0, 1U, &loss0,
                  logits, 8U, workspace, workspace_count) == NIYAH_OK);
        CHECK(niyah_train_loss(
                  &model, t1, y1, 3U, &loss1,
                  logits, 24U, workspace, workspace_count) == NIYAH_OK);

        expected = ((double)loss0 + 3.0 * (double)loss1) / 4.0;

        CHECK(niyah_evaluate(
                  &model, samples, 2U, &metrics) == NIYAH_OK);
        CHECK(nearly_equal(metrics.mean_loss, expected, 1.0e-7));
        CHECK(nearly_equal(metrics.perplexity,
                           exp(expected), 1.0e-6));
    }

    free(workspace);
    niyah_model_destroy(&model);
}

static void test_masked_targets_and_token_count(void)
{
    static const uint32_t tokens[] = {1U, 2U, 3U, 4U};
    static const uint32_t targets_a[] = {2U, 3U, 4U, 5U};
    static const uint32_t targets_prompt_changed[] = {7U, 0U, 4U, 5U};
    static const uint32_t targets_response_changed[] = {2U, 3U, 6U, 5U};
    const NiyahEvaluationSample sample_a = {
        tokens, targets_a, 4U, 2U
    };
    const NiyahEvaluationSample sample_prompt_changed = {
        tokens, targets_prompt_changed, 4U, 2U
    };
    const NiyahEvaluationSample sample_response_changed = {
        tokens, targets_response_changed, 4U, 2U
    };
    NiyahModelConfig config = test_config();
    NiyahModel model;
    NiyahEvaluationMetrics a;
    NiyahEvaluationMetrics prompt_changed;
    NiyahEvaluationMetrics response_changed;

    memset(&model, 0, sizeof(model));
    memset(&a, 0, sizeof(a));
    memset(&prompt_changed, 0, sizeof(prompt_changed));
    memset(&response_changed, 0, sizeof(response_changed));

    CHECK(niyah_model_create(&model, &config) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(
              &model, UINT64_C(20260921)) == NIYAH_OK);

    CHECK(niyah_evaluate(
              &model, &sample_a, 1U, &a) == NIYAH_OK);
    CHECK(niyah_evaluate(
              &model, &sample_prompt_changed, 1U,
              &prompt_changed) == NIYAH_OK);
    CHECK(niyah_evaluate(
              &model, &sample_response_changed, 1U,
              &response_changed) == NIYAH_OK);

    CHECK(a.token_count == 2U);
    CHECK(prompt_changed.token_count == 2U);
    CHECK(response_changed.token_count == 2U);
    CHECK(a.mean_loss == prompt_changed.mean_loss);
    CHECK(a.perplexity == prompt_changed.perplexity);
    CHECK(fabs(a.mean_loss - response_changed.mean_loss) > 1.0e-8);

    niyah_model_destroy(&model);
}

static void test_evaluation_is_read_only_and_deterministic(void)
{
    static const uint32_t tokens[] = {1U, 2U, 3U};
    static const uint32_t targets[] = {2U, 3U, 4U};
    const NiyahEvaluationSample sample = {tokens, targets, 3U, 0U};
    NiyahModelConfig config = test_config();
    NiyahModel model;
    NiyahEvaluationMetrics a;
    NiyahEvaluationMetrics b;
    float *before = NULL;

    memset(&model, 0, sizeof(model));
    memset(&a, 0, sizeof(a));
    memset(&b, 0, sizeof(b));

    CHECK(niyah_model_create(&model, &config) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(
              &model, UINT64_C(91)) == NIYAH_OK);

    before = (float *)malloc(model.weight_count * sizeof(float));
    CHECK(before != NULL);
    if (before != NULL) {
        memcpy(before, model.weights,
               model.weight_count * sizeof(float));

        CHECK(niyah_evaluate(
                  &model, &sample, 1U, &a) == NIYAH_OK);
        CHECK(niyah_evaluate(
                  &model, &sample, 1U, &b) == NIYAH_OK);

        CHECK(a.sample_count == b.sample_count);
        CHECK(a.token_count == b.token_count);
        CHECK(a.mean_loss == b.mean_loss);
        CHECK(a.perplexity == b.perplexity);
        CHECK(memcmp(before, model.weights,
                     model.weight_count * sizeof(float)) == 0);
    }

    free(before);
    niyah_model_destroy(&model);
}

static void test_failure_does_not_publish_metrics(void)
{
    static const uint32_t tokens[] = {1U, 2U, 3U};
    static const uint32_t bad_targets[] = {2U, 3U, 99U};
    const NiyahEvaluationSample sample = {
        tokens, bad_targets, 3U, 0U
    };
    const NiyahEvaluationSample bad_mask = {
        tokens, bad_targets, 3U, 3U
    };
    NiyahModelConfig config = test_config();
    NiyahModel model;
    NiyahEvaluationMetrics metrics;

    memset(&model, 0, sizeof(model));
    metrics.sample_count = 111U;
    metrics.token_count = 222U;
    metrics.mean_loss = 333.0;
    metrics.perplexity = 444.0;

    CHECK(niyah_model_create(&model, &config) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(
              &model, UINT64_C(7)) == NIYAH_OK);

    CHECK(niyah_evaluate(
              &model, &sample, 1U, &metrics) ==
          NIYAH_ERR_INVALID_ARGUMENT);
    CHECK(metrics.sample_count == 111U);
    CHECK(metrics.token_count == 222U);
    CHECK(metrics.mean_loss == 333.0);
    CHECK(metrics.perplexity == 444.0);

    CHECK(niyah_evaluate(
              &model, &bad_mask, 1U, &metrics) ==
          NIYAH_ERR_INVALID_CONFIG);
    CHECK(metrics.sample_count == 111U);
    CHECK(metrics.token_count == 222U);
    CHECK(metrics.mean_loss == 333.0);
    CHECK(metrics.perplexity == 444.0);

    niyah_model_destroy(&model);
}

int main(void)
{
    test_uniform_model_perplexity();
    test_weighted_mean_matches_direct_losses();
    test_masked_targets_and_token_count();
    test_evaluation_is_read_only_and_deterministic();
    test_failure_does_not_publish_metrics();

    if (failures != 0) {
        fprintf(stderr, "niyah_eval_test: %d failure(s)\n", failures);
        return 1;
    }

    puts("NIYAH_EVAL_P6_G=PASS");
    return 0;
}
