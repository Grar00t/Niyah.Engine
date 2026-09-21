#include "niyah/eval.h"
#include "niyah/train.h"
#include "niyah/transformer.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

static int niyah_size_add_ok(size_t a, size_t b, size_t *out)
{
    if (out == NULL || a > SIZE_MAX - b) {
        return 0;
    }
    *out = a + b;
    return 1;
}

static int niyah_size_mul_ok(size_t a, size_t b, size_t *out)
{
    if (out == NULL || (a != 0U && b > SIZE_MAX / a)) {
        return 0;
    }
    *out = a * b;
    return 1;
}

NiyahStatus niyah_evaluate(
    const NiyahModel *model,
    const NiyahEvaluationSample *samples,
    size_t sample_count,
    NiyahEvaluationMetrics *out_metrics)
{
    size_t i;
    size_t max_tokens = 0U;
    size_t total_tokens = 0U;
    size_t logits_count = 0U;
    size_t workspace_count = 0U;
    float *logits = NULL;
    float *workspace = NULL;
    double weighted_loss = 0.0;
    double mean_loss;
    double perplexity;
    NiyahEvaluationMetrics metrics;
    NiyahStatus status;

    if (model == NULL || model->weights == NULL ||
        samples == NULL || out_metrics == NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;
    if (sample_count == 0U)
        return NIYAH_ERR_INVALID_CONFIG;

    status = niyah_model_config_validate(&model->config);
    if (status != NIYAH_OK) return status;

    for (i = 0U; i < sample_count; ++i) {
        size_t scored_tokens;

        if (samples[i].tokens == NULL ||
            samples[i].targets == NULL ||
            samples[i].token_count == 0U)
            return NIYAH_ERR_INVALID_ARGUMENT;
        if (samples[i].token_count > (size_t)model->config.context_length ||
            samples[i].loss_start >= samples[i].token_count)
            return NIYAH_ERR_INVALID_CONFIG;

        scored_tokens =
            samples[i].token_count - samples[i].loss_start;
        if (!niyah_size_add_ok(
                total_tokens, scored_tokens, &total_tokens))
            return NIYAH_ERR_OVERFLOW;
        if (samples[i].token_count > max_tokens)
            max_tokens = samples[i].token_count;
    }

    if (!niyah_size_mul_ok(
            max_tokens, (size_t)model->config.vocab_size, &logits_count))
        return NIYAH_ERR_OVERFLOW;

    status = niyah_transformer_workspace_floats(
        &model->config, max_tokens, &workspace_count);
    if (status != NIYAH_OK) return status;

    if (logits_count > SIZE_MAX / sizeof(float) ||
        workspace_count > SIZE_MAX / sizeof(float))
        return NIYAH_ERR_OVERFLOW;

    logits = (float *)malloc(logits_count * sizeof(float));
    if (logits == NULL)
        return NIYAH_ERR_OUT_OF_MEMORY;

    workspace = (float *)malloc(workspace_count * sizeof(float));
    if (workspace == NULL) {
        free(logits);
        return NIYAH_ERR_OUT_OF_MEMORY;
    }

    for (i = 0U; i < sample_count; ++i) {
        float sample_loss = 0.0f;
        size_t sample_logits_count = 0U;
        size_t scored_tokens =
            samples[i].token_count - samples[i].loss_start;
        double contribution;

        if (!niyah_size_mul_ok(
                samples[i].token_count,
                (size_t)model->config.vocab_size,
                &sample_logits_count)) {
            free(workspace);
            free(logits);
            return NIYAH_ERR_OVERFLOW;
        }

        status = niyah_transformer_forward(
            model,
            samples[i].tokens,
            samples[i].token_count,
            logits,
            sample_logits_count,
            workspace,
            workspace_count);
        if (status != NIYAH_OK) {
            free(workspace);
            free(logits);
            return status;
        }

        status = niyah_cross_entropy_loss_masked(
            logits,
            samples[i].targets,
            samples[i].token_count,
            (size_t)model->config.vocab_size,
            samples[i].loss_start,
            &sample_loss,
            NULL,
            0U);
        if (status != NIYAH_OK) {
            free(workspace);
            free(logits);
            return status;
        }

        contribution =
            (double)sample_loss * (double)scored_tokens;
        if (!isfinite(contribution) ||
            !isfinite(weighted_loss + contribution)) {
            free(workspace);
            free(logits);
            return NIYAH_ERR_OVERFLOW;
        }
        weighted_loss += contribution;
    }

    free(workspace);
    free(logits);

    mean_loss = weighted_loss / (double)total_tokens;
    if (!isfinite(mean_loss))
        return NIYAH_ERR_OVERFLOW;

    perplexity = exp(mean_loss);
    if (!isfinite(perplexity))
        return NIYAH_ERR_OVERFLOW;

    metrics.sample_count = sample_count;
    metrics.token_count = total_tokens;
    metrics.mean_loss = mean_loss;
    metrics.perplexity = perplexity;
    *out_metrics = metrics;
    return NIYAH_OK;
}
