#ifndef NIYAH_OPTIMIZER_H
#define NIYAH_OPTIMIZER_H

#include "niyah/train.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NiyahAdamWConfig {
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    float max_grad_norm;
} NiyahAdamWConfig;

typedef struct NiyahAdamWState {
    float *m;
    float *v;
    size_t count;
    uint64_t step;

    /* In-process compatibility binding. Pointer identity is not a persisted
     * checkpoint identity and must be re-established after future loading.
     */
    const NiyahModel *bound_model;
    const float *bound_weights;
    NiyahModelConfig model_config;
    NiyahModelLayout model_layout;
} NiyahAdamWState;

NiyahStatus niyah_adamw_state_create(NiyahAdamWState *state,
                                     const NiyahModel *model);
void niyah_adamw_state_destroy(NiyahAdamWState *state);

/* Robust global L2 norm over the complete canonical gradient vector.
 * The input gradient array is read-only.
 */
NiyahStatus niyah_model_gradients_global_l2_norm(const NiyahModelGradients *gradients,
                                                 double *out_norm);

/* Reference CPU AdamW with mandatory global gradient clipping.
 * Weight decay is decoupled from moments and excludes RMSNorm scales.
 * Validation and candidate preflight complete before weights, moments, or step
 * are mutated. The caller-owned gradient array is never modified.
 */
NiyahStatus niyah_adamw_step(NiyahModel *model,
                             const NiyahModelGradients *gradients,
                             NiyahAdamWState *state,
                             const NiyahAdamWConfig *config);

#ifdef __cplusplus
}
#endif

#endif
