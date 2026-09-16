#ifndef NIYAH_TRAINING_LOOP_H
#define NIYAH_TRAINING_LOOP_H

#include "niyah/dataset.h"
#include "niyah/optimizer.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NiyahTrainingSample {
    const uint32_t *tokens;
    const uint32_t *targets;
    size_t token_count;
} NiyahTrainingSample;

NiyahStatus niyah_training_step(
    NiyahModel *model,
    const NiyahTrainingSample *samples,
    size_t sample_count,
    NiyahDatasetCursor *cursor,
    NiyahModelGradients *gradients,
    float *workspace,
    size_t workspace_count,
    NiyahAdamWState *optimizer_state,
    const NiyahAdamWConfig *optimizer_config,
    size_t *out_sample_index,
    float *out_loss);

NiyahStatus niyah_training_run_steps(
    NiyahModel *model,
    const NiyahTrainingSample *samples,
    size_t sample_count,
    NiyahDatasetCursor *cursor,
    NiyahAdamWState *optimizer_state,
    const NiyahAdamWConfig *optimizer_config,
    size_t steps,
    float *out_mean_loss);

/* One optimizer update over batch_size * accumulation_steps samples.
 *
 * Each sample backward produces a mean-loss gradient. Gradients are averaged
 * equally across all consumed samples before the single AdamW update.
 * The dataset cursor is restored to its pre-call epoch/position if selection,
 * backward, accumulation, or optimizer update fails.
 */
NiyahStatus niyah_training_accumulated_step(
    NiyahModel *model,
    const NiyahTrainingSample *samples,
    size_t sample_count,
    NiyahDatasetCursor *cursor,
    NiyahModelGradients *sample_gradients,
    NiyahModelGradients *accumulated_gradients,
    float *workspace,
    size_t workspace_count,
    NiyahAdamWState *optimizer_state,
    const NiyahAdamWConfig *optimizer_config,
    size_t batch_size,
    size_t accumulation_steps,
    size_t *out_samples_consumed,
    float *out_mean_loss);

NiyahStatus niyah_training_run_updates(
    NiyahModel *model,
    const NiyahTrainingSample *samples,
    size_t sample_count,
    NiyahDatasetCursor *cursor,
    NiyahAdamWState *optimizer_state,
    const NiyahAdamWConfig *optimizer_config,
    size_t batch_size,
    size_t accumulation_steps,
    size_t updates,
    float *out_mean_loss);

#ifdef __cplusplus
}
#endif

#endif
