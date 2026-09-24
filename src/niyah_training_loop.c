#include "niyah/training_loop.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>

NiyahStatus niyah_training_samples_from_shard(
    const NiyahDatasetShard *shard,
    NiyahTrainingSample *samples,
    size_t sample_capacity,
    size_t *out_sample_count)
{
    const uint32_t *tokens = NULL;
    const uint32_t *targets = NULL;
    size_t token_count = 0U;
    size_t loss_start = 0U;
    size_t i;
    NiyahStatus status;

    if (shard == NULL || out_sample_count == NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;
    if (samples == NULL && sample_capacity != 0U)
        return NIYAH_ERR_INVALID_ARGUMENT;
    if (shard->sample_count == 0U)
        return NIYAH_ERR_INVALID_CONFIG;

    status = niyah_dataset_shard_sample_with_loss(
        shard, 0U, &tokens, &targets,
        &token_count, &loss_start);
    if (status != NIYAH_OK)
        return status;

    *out_sample_count = shard->sample_count;

    if (samples == NULL)
        return NIYAH_OK;
    if (sample_capacity < shard->sample_count)
        return NIYAH_ERR_BUFFER_TOO_SMALL;

    samples[0U].tokens = tokens;
    samples[0U].targets = targets;
    samples[0U].token_count = token_count;
    samples[0U].loss_start = loss_start;

    if (shard->has_explicit_samples != 0) {
        /*
         * Sample 0 above already validated the complete explicit
         * geometry through niyah_dataset_shard_sample_with_loss().
         * Do not repeat the O(sample_count) geometry validation for
         * every descriptor.
         */
        for (i = 1U;
             i < shard->sample_count;
             ++i) {
            const size_t start =
                shard->sample_offsets[i];

            samples[i].tokens =
                shard->tokens + start;
            samples[i].targets =
                shard->tokens + start + 1U;
            samples[i].token_count =
                shard->sample_lengths[i];
            samples[i].loss_start =
                shard->has_loss_starts != 0
                    ? shard->sample_loss_starts[i]
                    : 0U;
        }
    } else {
        for (i = 1U;
             i < shard->sample_count;
             ++i) {
            status =
                niyah_dataset_shard_sample_with_loss(
                    shard,
                    i,
                    &tokens,
                    &targets,
                    &token_count,
                    &loss_start);
            if (status != NIYAH_OK)
                return status;

            samples[i].tokens =
                tokens;
            samples[i].targets =
                targets;
            samples[i].token_count =
                token_count;
            samples[i].loss_start =
                loss_start;
        }
    }

    return NIYAH_OK;
}

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
        if (samples[i].token_count >
                model->config.context_length ||
            samples[i].loss_start >=
                samples[i].token_count)
            return NIYAH_ERR_INVALID_CONFIG;
    }

    return NIYAH_OK;
}

static NiyahStatus validate_training_sample(
    const NiyahModel *model,
    const NiyahTrainingSample *sample)
{
    if (model == NULL || sample == NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;

    if (sample->tokens == NULL ||
        sample->targets == NULL ||
        sample->token_count == 0U)
        return NIYAH_ERR_INVALID_ARGUMENT;

    if (sample->token_count >
            model->config.context_length ||
        sample->loss_start >= sample->token_count)
        return NIYAH_ERR_INVALID_CONFIG;

    return NIYAH_OK;
}

static NiyahStatus validate_training_source(
    const NiyahModel *model,
    const NiyahTrainingSample *samples,
    size_t sample_count,
    const NiyahDatasetCursor *cursor,
    NiyahTrainingSampleProviderFn sample_provider)
{
    if (sample_provider == NULL)
        return validate_samples(
            model, samples, sample_count, cursor);

    if (model == NULL || cursor == NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;

    if (samples != NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;

    if (sample_count == 0U ||
        cursor->sample_count != sample_count ||
        cursor->order == NULL)
        return NIYAH_ERR_INVALID_CONFIG;

    return NIYAH_OK;
}

static NiyahStatus resolve_training_sample(
    const NiyahModel *model,
    const NiyahTrainingSample *samples,
    size_t sample_count,
    size_t sample_index,
    NiyahTrainingSampleProviderFn sample_provider,
    void *sample_provider_user_data,
    NiyahTrainingSample *out_sample)
{
    NiyahStatus status;

    if (out_sample == NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;

    if (sample_index >= sample_count)
        return NIYAH_ERR_INVALID_CONFIG;

    if (sample_provider != NULL) {
        status = sample_provider(
            sample_index,
            out_sample,
            sample_provider_user_data);
        if (status != NIYAH_OK)
            return status;

        return validate_training_sample(
            model, out_sample);
    }

    *out_sample = samples[sample_index];
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

    status = niyah_train_backward_masked(
        model,
        samples[sample_index].tokens,
        samples[sample_index].targets,
        samples[sample_index].token_count,
        samples[sample_index].loss_start,
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


static NiyahStatus rollback_cursor(NiyahDatasetCursor *cursor,
                                   uint64_t epoch,
                                   size_t position,
                                   NiyahStatus original_status)
{
    NiyahStatus rollback_status =
        niyah_dataset_cursor_seek(cursor, epoch, position);
    return rollback_status == NIYAH_OK ? original_status : rollback_status;
}

static NiyahStatus gradients_accumulate(NiyahModelGradients *dst,
                                        const NiyahModelGradients *src)
{
    size_t i;

    if (dst == NULL || src == NULL ||
        dst->values == NULL || src->values == NULL ||
        dst->count != src->count)
        return NIYAH_ERR_INVALID_ARGUMENT;

    for (i = 0U; i < dst->count; ++i) {
        float value = dst->values[i] + src->values[i];
        if (!isfinite(value)) return NIYAH_ERR_OVERFLOW;
        dst->values[i] = value;
    }
    return NIYAH_OK;
}

static NiyahStatus gradients_scale(NiyahModelGradients *gradients,
                                   float scale)
{
    size_t i;

    if (gradients == NULL || gradients->values == NULL || !isfinite(scale))
        return NIYAH_ERR_INVALID_ARGUMENT;

    for (i = 0U; i < gradients->count; ++i) {
        float value = gradients->values[i] * scale;
        if (!isfinite(value)) return NIYAH_ERR_OVERFLOW;
        gradients->values[i] = value;
    }
    return NIYAH_OK;
}

static NiyahStatus niyah_training_accumulated_step_impl(
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
    float *out_mean_loss,
    size_t max_token_count,
    NiyahTrainingSampleProviderFn sample_provider,
    void *sample_provider_user_data,
    NiyahTrainingBackwardFn backward_fn,
    void *backward_user_data)
{
    uint64_t old_epoch;
    size_t old_position;
    size_t total_samples;
    size_t consumed;
    size_t total_supervised_targets = 0U;
    size_t common_supervised_targets = 0U;
    int token_weighting_active = 0;
    double loss_sum = 0.0;
    NiyahStatus status;

    if (sample_gradients == NULL || accumulated_gradients == NULL ||
        workspace == NULL || optimizer_state == NULL ||
        optimizer_config == NULL || out_samples_consumed == NULL ||
        out_mean_loss == NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;

    status = validate_training_source(
        model, samples, sample_count, cursor, sample_provider);
    if (status != NIYAH_OK)
        return status;

    if (batch_size == 0U || accumulation_steps == 0U)
        return NIYAH_ERR_INVALID_CONFIG;
    if (batch_size > SIZE_MAX / accumulation_steps)
        return NIYAH_ERR_OVERFLOW;
    total_samples = batch_size * accumulation_steps;

    if (sample_gradients->values == NULL ||
        accumulated_gradients->values == NULL ||
        sample_gradients->count != model->weight_count ||
        accumulated_gradients->count != model->weight_count)
        return NIYAH_ERR_INVALID_ARGUMENT;

    old_epoch = cursor->epoch;
    old_position = cursor->position;
    niyah_model_gradients_zero(accumulated_gradients);

    for (consumed = 0U; consumed < total_samples; ++consumed) {
        size_t sample_index;
        NiyahTrainingSample sample;
        float loss;

        status = niyah_dataset_cursor_next(cursor, &sample_index);
        if (status != NIYAH_OK)
            return rollback_cursor(cursor, old_epoch, old_position, status);
        if (sample_index >= sample_count)
            return rollback_cursor(
                cursor, old_epoch, old_position, NIYAH_ERR_INVALID_CONFIG);

        status = resolve_training_sample(
            model,
            samples,
            sample_count,
            sample_index,
            sample_provider,
            sample_provider_user_data,
            &sample);
        if (status != NIYAH_OK)
            return rollback_cursor(
                cursor, old_epoch, old_position, status);

        if (sample_provider != NULL &&
            sample.token_count > max_token_count)
            return rollback_cursor(
                cursor, old_epoch, old_position,
                NIYAH_ERR_INVALID_CONFIG);

        if (backward_fn != NULL) {
            status = backward_fn(
                model,
                &sample,
                sample_gradients,
                workspace,
                workspace_count,
                &loss,
                backward_user_data);
        } else {
            status = niyah_train_backward_masked(
                model,
                sample.tokens,
                sample.targets,
                sample.token_count,
                sample.loss_start,
                &loss,
                sample_gradients,
                workspace,
                workspace_count);
        }
        if (status != NIYAH_OK)
            return rollback_cursor(cursor, old_epoch, old_position, status);

        {
            const size_t supervised_count =
                sample.token_count -
                sample.loss_start;

            if (supervised_count >
                    SIZE_MAX - total_supervised_targets)
                return rollback_cursor(
                    cursor, old_epoch, old_position,
                    NIYAH_ERR_OVERFLOW);

            total_supervised_targets += supervised_count;

            /*
             * Preserve the historical byte-identical averaging path while
             * every sample has the same supervised length. If a different
             * length appears, convert the already accumulated equal-length
             * means into token sums exactly once, then continue token
             * weighted for the remainder of this optimizer update.
             */
            if (consumed == 0U) {
                common_supervised_targets = supervised_count;
            } else if (!token_weighting_active &&
                       supervised_count != common_supervised_targets) {
                status = gradients_scale(
                    accumulated_gradients,
                    (float)common_supervised_targets);
                if (status != NIYAH_OK)
                    return rollback_cursor(
                        cursor, old_epoch, old_position, status);

                loss_sum *= (double)common_supervised_targets;
                if (!isfinite(loss_sum))
                    return rollback_cursor(
                        cursor, old_epoch, old_position,
                        NIYAH_ERR_OVERFLOW);

                token_weighting_active = 1;
            }

            if (token_weighting_active) {
                status = gradients_scale(
                    sample_gradients, (float)supervised_count);
                if (status != NIYAH_OK)
                    return rollback_cursor(
                        cursor, old_epoch, old_position, status);

                loss_sum +=
                    (double)loss * (double)supervised_count;
            } else {
                loss_sum += (double)loss;
            }

            if (!isfinite(loss_sum))
                return rollback_cursor(
                    cursor, old_epoch, old_position,
                    NIYAH_ERR_OVERFLOW);
        }

        status = gradients_accumulate(
            accumulated_gradients, sample_gradients);
        if (status != NIYAH_OK)
            return rollback_cursor(cursor, old_epoch, old_position, status);
    }

    if (total_supervised_targets == 0U)
        return rollback_cursor(
            cursor, old_epoch, old_position,
            NIYAH_ERR_INVALID_CONFIG);

    status = gradients_scale(
        accumulated_gradients,
        token_weighting_active
            ? 1.0f / (float)total_supervised_targets
            : 1.0f / (float)total_samples);
    if (status != NIYAH_OK)
        return rollback_cursor(cursor, old_epoch, old_position, status);

    status = niyah_adamw_step(
        model, accumulated_gradients, optimizer_state, optimizer_config);
    if (status != NIYAH_OK)
        return rollback_cursor(cursor, old_epoch, old_position, status);

    loss_sum /= token_weighting_active
        ? (double)total_supervised_targets
        : (double)total_samples;
    if (!isfinite(loss_sum) || loss_sum > (double)FLT_MAX)
        return NIYAH_ERR_OVERFLOW;

    *out_samples_consumed = total_samples;
    *out_mean_loss = (float)loss_sum;
    return NIYAH_OK;
}

static NiyahStatus niyah_training_run_updates_with_progress_impl(
    NiyahModel *model,
    const NiyahTrainingSample *samples,
    size_t sample_count,
    NiyahDatasetCursor *cursor,
    NiyahAdamWState *optimizer_state,
    const NiyahAdamWConfig *optimizer_config,
    size_t batch_size,
    size_t accumulation_steps,
    size_t updates,
    size_t max_token_count,
    NiyahTrainingSampleProviderFn sample_provider,
    void *sample_provider_user_data,
    NiyahTrainingBackwardFn backward_fn,
    void *backward_user_data,
    NiyahTrainingProgressFn progress_fn,
    void *progress_user_data,
    float *out_mean_loss)
{
    NiyahModelGradients sample_gradients;
    NiyahModelGradients accumulated_gradients;
    float *workspace = NULL;
    size_t workspace_count = 0U;
    size_t required = 0U;
    size_t i;
    size_t update;
    double loss_sum = 0.0;
    NiyahStatus status;

    if (optimizer_state == NULL || optimizer_config == NULL ||
        out_mean_loss == NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;
    if (updates == 0U || batch_size == 0U || accumulation_steps == 0U)
        return NIYAH_ERR_INVALID_CONFIG;
    if (batch_size > SIZE_MAX / accumulation_steps)
        return NIYAH_ERR_OVERFLOW;

    status = validate_training_source(
        model, samples, sample_count, cursor, sample_provider);
    if (status != NIYAH_OK)
        return status;

    if (sample_provider != NULL) {
        if (max_token_count == 0U ||
            max_token_count > model->config.context_length)
            return NIYAH_ERR_INVALID_CONFIG;

        status = niyah_train_backward_workspace_floats(
            &model->config,
            max_token_count,
            &workspace_count);
        if (status != NIYAH_OK)
            return status;
    } else {
        for (i = 0U; i < sample_count; ++i) {
            status = niyah_train_backward_workspace_floats(
                &model->config,
                samples[i].token_count,
                &required);
            if (status != NIYAH_OK)
                return status;

            if (required > workspace_count)
                workspace_count = required;
        }
    }

    sample_gradients.values = NULL;
    sample_gradients.count = 0U;
    accumulated_gradients.values = NULL;
    accumulated_gradients.count = 0U;

    status = niyah_model_gradients_create(&sample_gradients, model);
    if (status != NIYAH_OK) return status;

    status = niyah_model_gradients_create(&accumulated_gradients, model);
    if (status != NIYAH_OK) {
        niyah_model_gradients_destroy(&sample_gradients);
        return status;
    }

    if (workspace_count > SIZE_MAX / sizeof(float)) {
        niyah_model_gradients_destroy(&accumulated_gradients);
        niyah_model_gradients_destroy(&sample_gradients);
        return NIYAH_ERR_OVERFLOW;
    }

    workspace = (float *)calloc(workspace_count, sizeof(float));
    if (workspace == NULL) {
        niyah_model_gradients_destroy(&accumulated_gradients);
        niyah_model_gradients_destroy(&sample_gradients);
        return NIYAH_ERR_OUT_OF_MEMORY;
    }

    for (update = 0U; update < updates; ++update) {
        size_t consumed = 0U;
        float mean_loss = 0.0f;

        status = niyah_training_accumulated_step_impl(
            model, samples, sample_count, cursor,
            &sample_gradients, &accumulated_gradients,
            workspace, workspace_count,
            optimizer_state, optimizer_config,
            batch_size, accumulation_steps,
            &consumed, &mean_loss,
            max_token_count,
            sample_provider,
            sample_provider_user_data,
            backward_fn, backward_user_data);
        if (status != NIYAH_OK) {
            free(workspace);
            niyah_model_gradients_destroy(&accumulated_gradients);
            niyah_model_gradients_destroy(&sample_gradients);
            return status;
        }

        if (consumed != batch_size * accumulation_steps) {
            free(workspace);
            niyah_model_gradients_destroy(&accumulated_gradients);
            niyah_model_gradients_destroy(&sample_gradients);
            return NIYAH_ERR_INVALID_CONFIG;
        }

        loss_sum += (double)mean_loss;
        if (!isfinite(loss_sum)) {
            free(workspace);
            niyah_model_gradients_destroy(&accumulated_gradients);
            niyah_model_gradients_destroy(&sample_gradients);
            return NIYAH_ERR_OVERFLOW;
        }

        if (progress_fn != NULL) {
            progress_fn(update, updates, mean_loss, progress_user_data);
        }
    }

    free(workspace);
    niyah_model_gradients_destroy(&accumulated_gradients);
    niyah_model_gradients_destroy(&sample_gradients);

    loss_sum /= (double)updates;
    if (!isfinite(loss_sum) || loss_sum > (double)FLT_MAX)
        return NIYAH_ERR_OVERFLOW;

    *out_mean_loss = (float)loss_sum;
    return NIYAH_OK;
}

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
    float *out_mean_loss)
{
    return niyah_training_run_updates_with_progress(
        model, samples, sample_count, cursor,
        optimizer_state, optimizer_config,
        batch_size, accumulation_steps, updates,
        NULL, NULL, out_mean_loss);
}

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
    float *out_mean_loss)
{
    return niyah_training_accumulated_step_impl(
        model,
        samples,
        sample_count,
        cursor,
        sample_gradients,
        accumulated_gradients,
        workspace,
        workspace_count,
        optimizer_state,
        optimizer_config,
        batch_size,
        accumulation_steps,
        out_samples_consumed,
        out_mean_loss,
        0U,
        NULL,
        NULL,
        NULL,
        NULL);
}

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
    float *out_mean_loss)
{
    return niyah_training_run_updates_with_progress_impl(
        model,
        samples,
        sample_count,
        cursor,
        optimizer_state,
        optimizer_config,
        batch_size,
        accumulation_steps,
        updates,
        0U,
        NULL,
        NULL,
        NULL,
        NULL,
        progress_fn,
        progress_user_data,
        out_mean_loss);
}

NiyahStatus niyah_training_run_updates_with_progress_with_backward(
    NiyahModel *model,
    const NiyahTrainingSample *samples,
    size_t sample_count,
    NiyahDatasetCursor *cursor,
    NiyahAdamWState *optimizer_state,
    const NiyahAdamWConfig *optimizer_config,
    size_t batch_size,
    size_t accumulation_steps,
    size_t updates,
    NiyahTrainingBackwardFn backward_fn,
    void *backward_user_data,
    NiyahTrainingProgressFn progress_fn,
    void *progress_user_data,
    float *out_mean_loss)
{
    if (backward_fn == NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;

    return niyah_training_run_updates_with_progress_impl(
        model,
        samples,
        sample_count,
        cursor,
        optimizer_state,
        optimizer_config,
        batch_size,
        accumulation_steps,
        updates,
        0U,
        NULL,
        NULL,
        backward_fn,
        backward_user_data,
        progress_fn,
        progress_user_data,
        out_mean_loss);
}

NiyahStatus niyah_training_run_updates_with_progress_with_provider(
    NiyahModel *model,
    size_t sample_count,
    size_t max_token_count,
    NiyahTrainingSampleProviderFn sample_provider,
    void *sample_provider_user_data,
    NiyahDatasetCursor *cursor,
    NiyahAdamWState *optimizer_state,
    const NiyahAdamWConfig *optimizer_config,
    size_t batch_size,
    size_t accumulation_steps,
    size_t updates,
    NiyahTrainingBackwardFn backward_fn,
    void *backward_user_data,
    NiyahTrainingProgressFn progress_fn,
    void *progress_user_data,
    float *out_mean_loss)
{
    if (sample_provider == NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;

    return niyah_training_run_updates_with_progress_impl(
        model,
        NULL,
        sample_count,
        cursor,
        optimizer_state,
        optimizer_config,
        batch_size,
        accumulation_steps,
        updates,
        max_token_count,
        sample_provider,
        sample_provider_user_data,
        backward_fn,
        backward_user_data,
        progress_fn,
        progress_user_data,
        out_mean_loss);
}
