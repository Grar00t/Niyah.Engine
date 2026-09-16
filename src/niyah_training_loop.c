#include "niyah/training_loop.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>

static NiyahStatus validate_samples(const NiyahModel *model,
                                    const NiyahTrainingSample *samples,
                                    size_t sample_count,
                                    const NiyahDatasetCursor *cursor)
{
    size_t i;

    if (model == NULL || samples == NULL || cursor == NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;
    if (sample_count == 0U ||
        cursor->sample_count != sample_count ||
        cursor->order == NULL)
        return NIYAH_ERR_INVALID_CONFIG;

    for (i = 0U; i < sample_count; ++i) {
        if (samples[i].tokens == NULL ||
            samples[i].targets == NULL ||
            samples[i].token_count == 0U)
            return NIYAH_ERR_INVALID_ARGUMENT;
        if (samples[i].token_count > model->config.context_length)
            return NIYAH_ERR_INVALID_CONFIG;
    }

    return NIYAH_OK;
}

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
    float *out_loss)
{
    uint64_t old_epoch;
    size_t old_position;
    size_t sample_index;
    float loss;
    NiyahStatus status;
    NiyahStatus rollback_status;

    if (gradients == NULL || workspace == NULL ||
        optimizer_state == NULL || optimizer_config == NULL ||
        out_sample_index == NULL || out_loss == NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;

    status = validate_samples(model, samples, sample_count, cursor);
    if (status != NIYAH_OK) return status;

    old_epoch = cursor->epoch;
    old_position = cursor->position;

    status = niyah_dataset_cursor_next(cursor, &sample_index);
    if (status != NIYAH_OK) return status;

    if (sample_index >= sample_count) {
        rollback_status = niyah_dataset_cursor_seek(
            cursor, old_epoch, old_position);
        return rollback_status == NIYAH_OK
            ? NIYAH_ERR_INVALID_CONFIG
            : rollback_status;
    }

    status = niyah_train_backward(
        model,
        samples[sample_index].tokens,
        samples[sample_index].targets,
        samples[sample_index].token_count,
        &loss,
        gradients,
        workspace,
        workspace_count);
    if (status != NIYAH_OK) {
        rollback_status = niyah_dataset_cursor_seek(
            cursor, old_epoch, old_position);
        return rollback_status == NIYAH_OK ? status : rollback_status;
    }

    status = niyah_adamw_step(
        model, gradients, optimizer_state, optimizer_config);
    if (status != NIYAH_OK) {
        rollback_status = niyah_dataset_cursor_seek(
            cursor, old_epoch, old_position);
        return rollback_status == NIYAH_OK ? status : rollback_status;
    }

    *out_sample_index = sample_index;
    *out_loss = loss;
    return NIYAH_OK;
}

NiyahStatus niyah_training_run_steps(
    NiyahModel *model,
    const NiyahTrainingSample *samples,
    size_t sample_count,
    NiyahDatasetCursor *cursor,
    NiyahAdamWState *optimizer_state,
    const NiyahAdamWConfig *optimizer_config,
    size_t steps,
    float *out_mean_loss)
{
    NiyahModelGradients gradients;
    float *workspace = NULL;
    size_t workspace_count = 0U;
    size_t required = 0U;
    size_t i;
    size_t step;
    size_t sample_index = 0U;
    float loss = 0.0f;
    double loss_sum = 0.0;
    NiyahStatus status;

    if (optimizer_state == NULL || optimizer_config == NULL ||
        out_mean_loss == NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;
    if (steps == 0U) return NIYAH_ERR_INVALID_CONFIG;

    status = validate_samples(model, samples, sample_count, cursor);
    if (status != NIYAH_OK) return status;

    for (i = 0U; i < sample_count; ++i) {
        status = niyah_train_backward_workspace_floats(
            &model->config, samples[i].token_count, &required);
        if (status != NIYAH_OK) return status;
        if (required > workspace_count) workspace_count = required;
    }

    gradients.values = NULL;
    gradients.count = 0U;
    status = niyah_model_gradients_create(&gradients, model);
    if (status != NIYAH_OK) return status;

    if (workspace_count > SIZE_MAX / sizeof(float)) {
        niyah_model_gradients_destroy(&gradients);
        return NIYAH_ERR_OVERFLOW;
    }

    workspace = (float *)calloc(workspace_count, sizeof(float));
    if (workspace == NULL) {
        niyah_model_gradients_destroy(&gradients);
        return NIYAH_ERR_OUT_OF_MEMORY;
    }

    for (step = 0U; step < steps; ++step) {
        status = niyah_training_step(
            model, samples, sample_count, cursor,
            &gradients, workspace, workspace_count,
            optimizer_state, optimizer_config,
            &sample_index, &loss);
        if (status != NIYAH_OK) {
            free(workspace);
            niyah_model_gradients_destroy(&gradients);
            return status;
        }
        loss_sum += (double)loss;
    }

    free(workspace);
    niyah_model_gradients_destroy(&gradients);

    loss_sum /= (double)steps;
    if (!isfinite(loss_sum) || loss_sum > (double)FLT_MAX)
        return NIYAH_ERR_OVERFLOW;

    *out_mean_loss = (float)loss_sum;
    return NIYAH_OK;
}
