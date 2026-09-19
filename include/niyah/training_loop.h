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
    size_t loss_start;
} NiyahTrainingSample;

/* Build zero-copy training sample descriptors over one loaded dataset shard.
 *
 * Query mode: samples == NULL and sample_capacity == 0 returns the required
 * descriptor count in out_sample_count. The returned descriptors borrow token
 * storage from shard; shard must outlive their use.
 */
NiyahStatus niyah_training_samples_from_shard(
    const NiyahDatasetShard *shard,
    NiyahTrainingSample *samples,
    size_t sample_capacity,
    size_t *out_sample_count);

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

/* Invoked after each successfully completed update with the zero-based
 * update index, total update count, and that update's mean loss.
 */
typedef void (*NiyahTrainingProgressFn)(
    size_t update_index, size_t updates, float loss, void *user_data);

/* Same contract as niyah_training_run_updates, additionally invoking
 * progress_fn (if non-NULL) after each update completes. */
NiyahStatus niyah_training_run_updates_with_progress(
    NiyahModel *model,
    const NiyahTrainingSample *samples,
    size_t sample_count,
    NiyahDatasetCursor *cursor,
    NiyahAdamWState *optimizer_state,
    const NiyahAdamWConfig *optimizer_config,
    size_t batch_size,
    size_t accumulation_steps,
    size_t updates,
    NiyahTrainingProgressFn progress_fn,
    void *progress_user_data,
    float *out_mean_loss);

#ifdef __cplusplus
}
#endif

#endif
