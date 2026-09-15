#ifndef NIYAH_TRAIN_H
#define NIYAH_TRAIN_H

#include "niyah/niyah.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NiyahModelGradients {
    float *values;
    size_t count;
} NiyahModelGradients;

NiyahStatus niyah_model_gradients_create(NiyahModelGradients *gradients,
                                         const NiyahModel *model);
void niyah_model_gradients_destroy(NiyahModelGradients *gradients);
void niyah_model_gradients_zero(NiyahModelGradients *gradients);

/* Stable mean cross-entropy over [token][vocab] logits.
 * When dlogits is non-NULL it receives d(mean loss)/d(logits).
 */
NiyahStatus niyah_cross_entropy_loss(const float *logits,
                                     const uint32_t *targets,
                                     size_t token_count,
                                     size_t vocab_size,
                                     float *out_loss,
                                     float *dlogits,
                                     size_t dlogits_count);

/* P5 training objective entry point. It deliberately reuses the canonical
 * inference forward path so training and inference cannot drift at logits.
 * Explicit Transformer backward is added on this same branch after this gate.
 */
NiyahStatus niyah_train_loss(const NiyahModel *model,
                             const uint32_t *tokens,
                             const uint32_t *targets,
                             size_t token_count,
                             float *out_loss,
                             float *logits,
                             size_t logits_count,
                             float *workspace,
                             size_t workspace_count);

#ifdef __cplusplus
}
#endif

#endif
