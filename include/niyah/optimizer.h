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

    /* Persisted LR schedule state. Zero selects the constant schedule. */
    uint64_t warmup_steps;

    /* In-process compatibility binding. Pointer identity is not a persisted
     * checkpoint identity and must be re-established after future loading.
     */
    const NiyahModel *bound_model;
    const float *bound_weights;
    NiyahModelConfig model_config;
    NiyahModelLayout model_layout;
} NiyahAdamWState;

/* Compute the effective learning rate for a one-based optimizer step.
 *
 * warmup_steps == 0 selects the constant schedule.
 * During warmup, step N uses base_learning_rate * N / warmup_steps.
 * At and after warmup_steps, the effective rate equals base_learning_rate.
 *
 * This function is pure: it does not mutate optimizer state or configuration.
 */
NiyahStatus niyah_adamw_linear_warmup_learning_rate(
    float base_learning_rate,
    uint64_t optimizer_step,
    uint64_t warmup_steps,
    float *out_learning_rate);

NiyahStatus niyah_adamw_config_validate(const NiyahAdamWConfig *config);
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
