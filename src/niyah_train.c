#include "niyah/train.h"
#include "niyah/transformer.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int niyah_size_mul_ok(size_t a, size_t b, size_t *out)
{
    if (out == NULL || (a != 0U && b > SIZE_MAX / a)) {
        return 0;
    }
    *out = a * b;
    return 1;
}

NiyahStatus niyah_model_gradients_create(NiyahModelGradients *gradients,
                                         const NiyahModel *model)
{
    size_t bytes;

    if (gradients == NULL || model == NULL || model->weights == NULL ||
        model->weight_count == 0U) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    memset(gradients, 0, sizeof(*gradients));
    if (!niyah_size_mul_ok(model->weight_count, sizeof(float), &bytes)) {
        return NIYAH_ERR_OVERFLOW;
    }
    gradients->values = (float *)calloc(1U, bytes);
    if (gradients->values == NULL) {
        return NIYAH_ERR_OUT_OF_MEMORY;
    }
    gradients->count = model->weight_count;
    return NIYAH_OK;
}

void niyah_model_gradients_destroy(NiyahModelGradients *gradients)
{
    if (gradients == NULL) {
        return;
    }
    free(gradients->values);
    memset(gradients, 0, sizeof(*gradients));
}

void niyah_model_gradients_zero(NiyahModelGradients *gradients)
{
    if (gradients == NULL || gradients->values == NULL || gradients->count == 0U) {
        return;
    }
    memset(gradients->values, 0, gradients->count * sizeof(float));
}

NiyahStatus niyah_cross_entropy_loss(const float *logits,
                                     const uint32_t *targets,
                                     size_t token_count,
                                     size_t vocab_size,
                                     float *out_loss,
                                     float *dlogits,
                                     size_t dlogits_count)
{
    size_t required = 0U;
    size_t t;
    double total_loss = 0.0;

    if (logits == NULL || targets == NULL || out_loss == NULL ||
        token_count == 0U || vocab_size == 0U) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    if (!niyah_size_mul_ok(token_count, vocab_size, &required)) {
        return NIYAH_ERR_OVERFLOW;
    }
    if (dlogits != NULL && dlogits_count < required) {
        return NIYAH_ERR_BUFFER_TOO_SMALL;
    }

    for (t = 0U; t < token_count; ++t) {
        const float *row = logits + t * vocab_size;
        float max_logit = row[0];
        double sum_exp = 0.0;
        size_t v;

        if ((size_t)targets[t] >= vocab_size) {
            return NIYAH_ERR_INVALID_ARGUMENT;
        }
        for (v = 1U; v < vocab_size; ++v) {
            if (row[v] > max_logit) {
                max_logit = row[v];
            }
        }
        if (!isfinite(max_logit)) {
            return NIYAH_ERR_INVALID_ARGUMENT;
        }
        for (v = 0U; v < vocab_size; ++v) {
            const double e = exp((double)row[v] - (double)max_logit);
            if (!isfinite(e)) {
                return NIYAH_ERR_INVALID_ARGUMENT;
            }
            sum_exp += e;
        }
        if (!(sum_exp > 0.0) || !isfinite(sum_exp)) {
            return NIYAH_ERR_INVALID_ARGUMENT;
        }

        total_loss += log(sum_exp) + (double)max_logit - (double)row[targets[t]];

        if (dlogits != NULL) {
            const double inv_tokens = 1.0 / (double)token_count;
            for (v = 0U; v < vocab_size; ++v) {
                double p = exp((double)row[v] - (double)max_logit) / sum_exp;
                if (v == (size_t)targets[t]) {
                    p -= 1.0;
                }
                dlogits[t * vocab_size + v] = (float)(p * inv_tokens);
            }
        }
    }

    *out_loss = (float)(total_loss / (double)token_count);
    return isfinite(*out_loss) ? NIYAH_OK : NIYAH_ERR_INVALID_ARGUMENT;
}

NiyahStatus niyah_train_loss(const NiyahModel *model,
                             const uint32_t *tokens,
                             const uint32_t *targets,
                             size_t token_count,
                             float *out_loss,
                             float *logits,
                             size_t logits_count,
                             float *workspace,
                             size_t workspace_count)
{
    size_t required_logits = 0U;
    NiyahStatus status;

    if (model == NULL || tokens == NULL || targets == NULL || out_loss == NULL ||
        logits == NULL || workspace == NULL || token_count == 0U) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    if (!niyah_size_mul_ok(token_count, (size_t)model->config.vocab_size,
                           &required_logits)) {
        return NIYAH_ERR_OVERFLOW;
    }
    if (logits_count < required_logits) {
        return NIYAH_ERR_BUFFER_TOO_SMALL;
    }

    status = niyah_transformer_forward(model,
                                       tokens,
                                       token_count,
                                       logits,
                                       logits_count,
                                       workspace,
                                       workspace_count);
    if (status != NIYAH_OK) {
        return status;
    }

    return niyah_cross_entropy_loss(logits,
                                    targets,
                                    token_count,
                                    (size_t)model->config.vocab_size,
                                    out_loss,
                                    NULL,
                                    0U);
}
