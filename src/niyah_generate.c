#include "niyah/generate.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>



#define NIYAH_SAME_TOKEN_LIMIT 8U
#define NIYAH_ENTROPY_FLOOR 0.05f

static NiyahStatus niyah_generation_entropy(const float *logits, size_t count,
                                            float *out_entropy)
{
    size_t i;
    float max_logit;
    double sum = 0.0;
    double weighted = 0.0;

    if (logits == NULL || count == 0U) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    max_logit = logits[0];
    for (i = 0U; i < count; ++i) {
        if (!isfinite(logits[i])) return NIYAH_ERR_INVALID_CONFIG;
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }
    for (i = 0U; i < count; ++i) {
        const double e = exp((double)logits[i] - (double)max_logit);
        sum += e;
        weighted += e * ((double)logits[i] - (double)max_logit);
    }
    if (!(sum > 0.0) || !isfinite(sum) || !isfinite(weighted)) {
        return NIYAH_ERR_INVALID_CONFIG;
    }
    *out_entropy = (float)(log(sum) - weighted / sum);
    return isfinite(*out_entropy) ? NIYAH_OK : NIYAH_ERR_INVALID_CONFIG;
}

static int niyah_size_add_ok(size_t a, size_t b, size_t *out)
{
    if (out == NULL || a > SIZE_MAX - b) {
        return 0;
    }
    *out = a + b;
    return 1;
}

NiyahStatus niyah_generation_workspace_floats(const NiyahModelConfig *config,
                                              size_t *out_floats)
{
    size_t decode_floats = 0U;
    size_t total = 0U;
    NiyahStatus status;

    if (out_floats == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    status = niyah_decode_workspace_floats(config, &decode_floats);
    if (status != NIYAH_OK) {
        return status;
    }
    if (!niyah_size_add_ok(decode_floats, (size_t)config->vocab_size, &total)) {
        return NIYAH_ERR_OVERFLOW;
    }
    *out_floats = total;
    return NIYAH_OK;
}

static NiyahStatus niyah_generation_fail(NiyahKVCache *cache,
                                         NiyahGenerationResult *result,
                                         NiyahStatus status)
{
    if (cache != NULL) {
        niyah_kv_cache_reset(cache);
    }
    if (result != NULL) {
        memset(result, 0, sizeof(*result));
    }
    return status;
}

static NiyahStatus niyah_generate_impl(const NiyahModel *model,
                           NiyahKVCache *cache,
                           const uint32_t *prompt_tokens,
                           size_t prompt_count,
                           const NiyahGenerationConfig *config,
                           uint32_t *output_tokens,
                           size_t output_capacity,
                           NiyahGenerationResult *result,
                           float *workspace,
                           size_t workspace_count,
    NiyahGenerationStopReport *stop_report)
{
    size_t required_workspace = 0U;
    size_t decode_workspace_count = 0U;
    size_t required_context = 0U;
    float *decode_workspace;
    float *logits;
    size_t i;
    size_t generated = 0U;
    NiyahGuard shared_guard;
    NiyahSampler sampler;
    NiyahStatus status;

    if (model == NULL || model->weights == NULL || cache == NULL ||
        prompt_tokens == NULL || prompt_count == 0U || config == NULL ||
        output_tokens == NULL || result == NULL || workspace == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    memset(result, 0, sizeof(*result));

    if (config->max_new_tokens == 0U ||
        output_capacity < config->max_new_tokens ||
        prompt_count > (size_t)model->config.context_length ||
        !niyah_size_add_ok(prompt_count, config->max_new_tokens, &required_context) ||
        required_context > (size_t)model->config.context_length) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    if (config->stop_on_eos != 0 &&
        (size_t)config->eos_token >= (size_t)model->config.vocab_size) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    if (niyah_kv_cache_position(cache) != 0U) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    status = niyah_generation_workspace_floats(&model->config, &required_workspace);
    if (status != NIYAH_OK) {
        return status;
    }
    if (workspace_count < required_workspace) {
        return NIYAH_ERR_BUFFER_TOO_SMALL;
    }
    status = niyah_decode_workspace_floats(&model->config, &decode_workspace_count);
    if (status != NIYAH_OK) {
        return status;
    }

    status = niyah_sampler_init(&sampler, &config->sampler);
    if (status != NIYAH_OK) {
        return status;
    }

    decode_workspace = workspace;
    logits = workspace + decode_workspace_count;

    for (i = 0U; i < prompt_count; ++i) {
        status = niyah_transformer_decode_token(model,
                                                cache,
                                                prompt_tokens[i],
                                                logits,
                                                (size_t)model->config.vocab_size,
                                                decode_workspace,
                                                decode_workspace_count);
        if (status != NIYAH_OK) {
            return niyah_generation_fail(cache, result, status);
        }
    }

    {
        NiyahGuardConfig gc;
        memset(&gc, 0, sizeof(gc));
        gc.max_tokens = config->max_new_tokens;
        gc.same_limit = 8U;
        gc.max_period = 8U;
        gc.cycles = 4U;
        gc.entropy_floor = 0.0f;
        gc.eos_token = config->eos_token;
        gc.stop_on_eos = config->stop_on_eos;
        (void)niyah_guard_init(&shared_guard, &gc);
    }
    result->prompt_tokens = prompt_count;

    while (generated < config->max_new_tokens) {
        uint32_t token = 0U;
        float entropy;
        status = niyah_generation_entropy(logits,
            (size_t)model->config.vocab_size, &entropy);
        if (status != NIYAH_OK && stop_report != NULL)
            stop_report->reason = NIYAH_STOP_NUMERIC_ERROR;
        if (status != NIYAH_OK) {
            return niyah_generation_fail(cache, result, status);
        }

        /* Low entropy is confidence, not proof of a loop. Finite-logit
         * validation above remains mandatory; generated-token guards follow. */

        status = niyah_sampler_sample(&sampler,
                                      logits,
                                      (size_t)model->config.vocab_size,
                                      &token);
        if (status != NIYAH_OK) {
            return niyah_generation_fail(cache, result, status);
        }

        output_tokens[generated] = token;
        generated += 1U;
        result->generated_tokens = generated;


        if (config->stop_on_eos != 0 && token == config->eos_token) {
            result->stopped_on_eos = 1;
            break;
        }
        if (niyah_guard_token(&shared_guard, token) == NIYAH_STOP_SINGLE_TOKEN_LOOP ||
            shared_guard.reason == NIYAH_STOP_MULTI_TOKEN_LOOP) {
            break;
        }

        status = niyah_transformer_decode_token(model,
                                                cache,
                                                token,
                                                logits,
                                                (size_t)model->config.vocab_size,
                                                decode_workspace,
                                                decode_workspace_count);
        if (status != NIYAH_OK) {
            return niyah_generation_fail(cache, result, status);
        }
    }

    if (stop_report != NULL) {
        stop_report->reason = result->stopped_on_eos ? NIYAH_STOP_EOS :
            (shared_guard.reason == NIYAH_STOP_NONE ?
                NIYAH_STOP_TOKEN_BUDGET : shared_guard.reason);
        stop_report->detected_period = shared_guard.detected_period;
    }
    return NIYAH_OK;
}

NiyahStatus niyah_generate(const NiyahModel *model,
                           NiyahKVCache *cache,
                           const uint32_t *prompt_tokens,
                           size_t prompt_count,
                           const NiyahGenerationConfig *config,
                           uint32_t *output_tokens,
                           size_t output_capacity,
                           NiyahGenerationResult *result,
                           float *workspace,
                           size_t workspace_count)
{
    return niyah_generate_impl(model, cache, prompt_tokens, prompt_count, config, output_tokens, output_capacity, result, workspace, workspace_count, NULL);
}

NiyahStatus niyah_generate_ex(const NiyahModel *model,
                           NiyahKVCache *cache,
                           const uint32_t *prompt_tokens,
                           size_t prompt_count,
                           const NiyahGenerationConfig *config,
                           uint32_t *output_tokens,
                           size_t output_capacity,
                           NiyahGenerationResult *result,
                           float *workspace,
                           size_t workspace_count,
    NiyahGenerationStopReport *stop_report)
{
    NiyahStatus status;
    if (stop_report != NULL) memset(stop_report, 0, sizeof(*stop_report));
    status = niyah_generate_impl(model, cache, prompt_tokens, prompt_count, config, output_tokens, output_capacity, result, workspace, workspace_count, stop_report);
    if (status != NIYAH_OK && stop_report != NULL &&
        stop_report->reason == NIYAH_STOP_NONE)
        stop_report->reason = NIYAH_STOP_BACKEND_ERROR;
    return status;
}



NiyahStatus niyah_generation_workspace_floats_with_repetition(
    const NiyahModelConfig *config,
    const NiyahRepetitionConfig *repetition,
    size_t *out_floats)
{
    size_t total = 0U;
    NiyahStatus status;
    if (out_floats == NULL) return NIYAH_ERR_INVALID_ARGUMENT;
    *out_floats = 0U;
    if (config == NULL || repetition == NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;
    if (!isfinite(repetition->penalty) || repetition->penalty < 1.0f ||
        repetition->window == 0U)
        return NIYAH_ERR_INVALID_CONFIG;
    status = niyah_generation_workspace_floats(config, &total);
    if (status != NIYAH_OK) return status;
    if (repetition->penalty != 1.0f) {
        if (!niyah_size_add_ok(total, (size_t)config->vocab_size, &total) ||
            total > SIZE_MAX / sizeof(float) ||
            total > (size_t)PTRDIFF_MAX / sizeof(float))
            return NIYAH_ERR_OVERFLOW;
    }
    *out_floats = total;
    return NIYAH_OK;
}

static NiyahStatus niyah_generate_with_repetition_impl(const NiyahModel *model,
                           NiyahKVCache *cache,
                           const uint32_t *prompt_tokens,
                           size_t prompt_count,
                           const NiyahGenerationConfig *config,
                           const NiyahRepetitionConfig *repetition,
                           uint32_t *output_tokens,
                           size_t output_capacity,
                           NiyahGenerationResult *result,
                           float *workspace,
                           size_t workspace_count,
    NiyahGenerationStopReport *stop_report)
{
    size_t required_workspace = 0U;
    size_t decode_workspace_count = 0U;
    size_t required_context = 0U;
    float *decode_workspace;
    float *logits;
    float *sampling_scratch;
    size_t i;
    size_t generated = 0U;
    NiyahGuard shared_guard;
    NiyahSampler sampler;
    NiyahStatus status;

    if (repetition == NULL)
        return niyah_generation_fail(cache, result, NIYAH_ERR_INVALID_ARGUMENT);
    if (!isfinite(repetition->penalty) || repetition->penalty < 1.0f ||
        repetition->window == 0U)
        return niyah_generation_fail(cache, result, NIYAH_ERR_INVALID_CONFIG);
    if (repetition->penalty == 1.0f) {
        if (stop_report != NULL) return niyah_generate_ex(model, cache, prompt_tokens, prompt_count, config,
                              output_tokens, output_capacity, result,
                              workspace, workspace_count, stop_report);
        return niyah_generate(model, cache, prompt_tokens, prompt_count, config,
                              output_tokens, output_capacity, result,
                              workspace, workspace_count);
    }

    if (model == NULL || model->weights == NULL || cache == NULL ||
        prompt_tokens == NULL || prompt_count == 0U || config == NULL ||
        output_tokens == NULL || result == NULL || workspace == NULL) {
        return niyah_generation_fail(cache, result, NIYAH_ERR_INVALID_ARGUMENT);
    }

    memset(result, 0, sizeof(*result));

    if (config->max_new_tokens == 0U ||
        output_capacity < config->max_new_tokens ||
        prompt_count > (size_t)model->config.context_length ||
        !niyah_size_add_ok(prompt_count, config->max_new_tokens, &required_context) ||
        required_context > (size_t)model->config.context_length) {
        return niyah_generation_fail(cache, result, NIYAH_ERR_INVALID_ARGUMENT);
    }
    if (config->stop_on_eos != 0 &&
        (size_t)config->eos_token >= (size_t)model->config.vocab_size) {
        return niyah_generation_fail(cache, result, NIYAH_ERR_INVALID_ARGUMENT);
    }
    if (niyah_kv_cache_position(cache) != 0U) {
        return niyah_generation_fail(cache, result, NIYAH_ERR_INVALID_ARGUMENT);
    }

    status = niyah_generation_workspace_floats_with_repetition(
        &model->config, repetition, &required_workspace);
    if (status != NIYAH_OK) {
        return niyah_generation_fail(cache, result, status);
    }
    if (workspace_count < required_workspace) {
        return niyah_generation_fail(cache, result, NIYAH_ERR_BUFFER_TOO_SMALL);
    }
    status = niyah_decode_workspace_floats(&model->config, &decode_workspace_count);
    if (status != NIYAH_OK) {
        return niyah_generation_fail(cache, result, status);
    }

    status = niyah_sampler_init(&sampler, &config->sampler);
    if (status != NIYAH_OK) {
        return niyah_generation_fail(cache, result, status);
    }

    decode_workspace = workspace;
    logits = workspace + decode_workspace_count;
    sampling_scratch = logits + (size_t)model->config.vocab_size;

    for (i = 0U; i < prompt_count; ++i) {
        status = niyah_transformer_decode_token(model,
                                                cache,
                                                prompt_tokens[i],
                                                logits,
                                                (size_t)model->config.vocab_size,
                                                decode_workspace,
                                                decode_workspace_count);
        if (status != NIYAH_OK) {
            return niyah_generation_fail(cache, result, status);
        }
    }

    {
        NiyahGuardConfig gc;
        memset(&gc, 0, sizeof(gc));
        gc.max_tokens = config->max_new_tokens;
        gc.same_limit = 8U;
        gc.max_period = 8U;
        gc.cycles = 4U;
        gc.entropy_floor = 0.0f;
        gc.eos_token = config->eos_token;
        gc.stop_on_eos = config->stop_on_eos;
        (void)niyah_guard_init(&shared_guard, &gc);
    }
    result->prompt_tokens = prompt_count;

    while (generated < config->max_new_tokens) {
        uint32_t token = 0U;
        float entropy;
        status = niyah_generation_entropy(logits,
            (size_t)model->config.vocab_size, &entropy);
        if (status != NIYAH_OK && stop_report != NULL)
            stop_report->reason = NIYAH_STOP_NUMERIC_ERROR;
        if (status != NIYAH_OK) {
            return niyah_generation_fail(cache, result, status);
        }

        /* Low entropy is confidence, not proof of a loop. Finite-logit
         * validation above remains mandatory; generated-token guards follow. */

        {
            const size_t history_count = generated < repetition->window
                ? generated : repetition->window;
            const uint32_t *history = output_tokens + (generated - history_count);
            status = niyah_sampler_sample_with_history(
                &sampler, logits, (size_t)model->config.vocab_size,
                history, history_count, repetition->penalty,
                sampling_scratch, (size_t)model->config.vocab_size, &token);
        }
        if (status != NIYAH_OK) {
            return niyah_generation_fail(cache, result, status);
        }

        output_tokens[generated] = token;
        generated += 1U;
        result->generated_tokens = generated;


        if (config->stop_on_eos != 0 && token == config->eos_token) {
            result->stopped_on_eos = 1;
            break;
        }
        if (niyah_guard_token(&shared_guard, token) == NIYAH_STOP_SINGLE_TOKEN_LOOP ||
            shared_guard.reason == NIYAH_STOP_MULTI_TOKEN_LOOP) {
            break;
        }

        status = niyah_transformer_decode_token(model,
                                                cache,
                                                token,
                                                logits,
                                                (size_t)model->config.vocab_size,
                                                decode_workspace,
                                                decode_workspace_count);
        if (status != NIYAH_OK) {
            return niyah_generation_fail(cache, result, status);
        }
    }

    if (stop_report != NULL) {
        stop_report->reason = result->stopped_on_eos ? NIYAH_STOP_EOS :
            (shared_guard.reason == NIYAH_STOP_NONE ?
                NIYAH_STOP_TOKEN_BUDGET : shared_guard.reason);
        stop_report->detected_period = shared_guard.detected_period;
    }
    return NIYAH_OK;
}

NiyahStatus niyah_generate_with_repetition(const NiyahModel *model,
                           NiyahKVCache *cache,
                           const uint32_t *prompt_tokens,
                           size_t prompt_count,
                           const NiyahGenerationConfig *config,
                           const NiyahRepetitionConfig *repetition,
                           uint32_t *output_tokens,
                           size_t output_capacity,
                           NiyahGenerationResult *result,
                           float *workspace,
                           size_t workspace_count)
{
    return niyah_generate_with_repetition_impl(model, cache, prompt_tokens, prompt_count, config, repetition, output_tokens, output_capacity, result, workspace, workspace_count, NULL);
}

NiyahStatus niyah_generate_with_repetition_ex(const NiyahModel *model,
                           NiyahKVCache *cache,
                           const uint32_t *prompt_tokens,
                           size_t prompt_count,
                           const NiyahGenerationConfig *config,
                           const NiyahRepetitionConfig *repetition,
                           uint32_t *output_tokens,
                           size_t output_capacity,
                           NiyahGenerationResult *result,
                           float *workspace,
                           size_t workspace_count,
    NiyahGenerationStopReport *stop_report)
{
    NiyahStatus status;
    if (stop_report != NULL) memset(stop_report, 0, sizeof(*stop_report));
    status = niyah_generate_with_repetition_impl(model, cache, prompt_tokens, prompt_count, config, repetition, output_tokens, output_capacity, result, workspace, workspace_count, stop_report);
    if (status != NIYAH_OK && stop_report != NULL &&
        stop_report->reason == NIYAH_STOP_NONE)
        stop_report->reason = NIYAH_STOP_BACKEND_ERROR;
    return status;
}
