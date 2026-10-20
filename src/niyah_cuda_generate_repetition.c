#include "niyah_cuda_generate_repetition.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

static NiyahStatus niyah_cuda_repetition_fail(
    NiyahCudaDecodeState *decode_state,
    NiyahGenerationResult *result,
    NiyahStatus status)
{
    int reset_failed = 0;
    if (decode_state != NULL)
        reset_failed = niyah_cuda_decode_state_reset(decode_state) != 0;
    if (result != NULL) memset(result, 0, sizeof(*result));
    return reset_failed ? NIYAH_ERR_IO : status;
}

static NiyahStatus niyah_cuda_repetition_entropy(
    const float *logits, size_t count, float *out_entropy)
{
    size_t i;
    float max_logit;
    double sum = 0.0;
    double weighted = 0.0;
    if (logits == NULL || count == 0U) return NIYAH_ERR_INVALID_ARGUMENT;
    max_logit = logits[0];
    for (i = 0U; i < count; ++i) {
        if (!isfinite(logits[i])) return NIYAH_ERR_INVALID_CONFIG;
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    for (i = 0U; i < count; ++i) {
        const double shifted = (double)logits[i] - (double)max_logit;
        const double e = exp(shifted);
        sum += e;
        weighted += e * shifted;
    }
    if (!(sum > 0.0) || !isfinite(sum) || !isfinite(weighted))
        return NIYAH_ERR_INVALID_CONFIG;
    *out_entropy = (float)(log(sum) - weighted / sum);
    return isfinite(*out_entropy) ? NIYAH_OK : NIYAH_ERR_INVALID_CONFIG;
}

NiyahStatus niyah_cuda_generation_workspace_floats_with_repetition(
    const NiyahModelConfig *config,
    const NiyahRepetitionConfig *repetition,
    size_t *out_floats)
{
    size_t total;
    NiyahStatus status;
    if (out_floats == NULL) return NIYAH_ERR_INVALID_ARGUMENT;
    *out_floats = 0U;
    if (config == NULL || repetition == NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;
    if (!isfinite(repetition->penalty) || repetition->penalty < 1.0f ||
        repetition->window == 0U)
        return NIYAH_ERR_INVALID_CONFIG;
    status = niyah_model_config_validate(config);
    if (status != NIYAH_OK) return status;
    total = (size_t)config->vocab_size;
    if (repetition->penalty != 1.0f) {
        if (total > SIZE_MAX - total) return NIYAH_ERR_OVERFLOW;
        total += total;
    }
    if (total > SIZE_MAX / sizeof(float) ||
        total > (size_t)PTRDIFF_MAX / sizeof(float))
        return NIYAH_ERR_OVERFLOW;
    *out_floats = total;
    return NIYAH_OK;
}

static NiyahStatus niyah_cuda_generate_with_repetition_impl(
    const NiyahCudaModelState *model_state,
    NiyahCudaDecodeState *decode_state,
    const uint32_t *prompt_tokens,
    size_t prompt_count,
    const NiyahGenerationConfig *config,
    const NiyahRepetitionConfig *repetition,
    uint32_t *output_tokens,
    size_t output_capacity,
    NiyahGenerationResult *result,
    float *host_workspace,
    size_t host_workspace_count,
    NiyahGenerationStopReport *stop_report)
{
    size_t required_workspace = 0U;
    size_t vocab;
    size_t generated = 0U;
    size_t i;
    float *logits;
    float *sampling_scratch;
    NiyahGuard shared_guard;
    NiyahSampler sampler;
    NiyahStatus status;

    if (repetition == NULL)
        return niyah_cuda_repetition_fail(decode_state, result,
                                          NIYAH_ERR_INVALID_ARGUMENT);
    if (!isfinite(repetition->penalty) || repetition->penalty < 1.0f ||
        repetition->window == 0U)
        return niyah_cuda_repetition_fail(decode_state, result,
                                          NIYAH_ERR_INVALID_CONFIG);
    if (repetition->penalty == 1.0f) {
        if (stop_report != NULL) return niyah_cuda_generate_ex(model_state, decode_state, prompt_tokens,
            prompt_count, config, output_tokens, output_capacity, result,
            host_workspace, host_workspace_count, stop_report);
        return niyah_cuda_generate(model_state, decode_state, prompt_tokens,
            prompt_count, config, output_tokens, output_capacity, result,
            host_workspace, host_workspace_count);
    }

    if (model_state == NULL || decode_state == NULL || prompt_tokens == NULL ||
        prompt_count == 0U || config == NULL || output_tokens == NULL ||
        result == NULL || host_workspace == NULL)
        return niyah_cuda_repetition_fail(decode_state, result,
                                          NIYAH_ERR_INVALID_ARGUMENT);
    memset(result, 0, sizeof(*result));
    vocab = (size_t)model_state->config.vocab_size;
    if (config->max_new_tokens == 0U ||
        output_capacity < config->max_new_tokens ||
        prompt_count > (size_t)model_state->config.context_length ||
        config->max_new_tokens >
            (size_t)model_state->config.context_length - prompt_count ||
        decode_state->next_position != 0U ||
        (config->stop_on_eos != 0 && (size_t)config->eos_token >= vocab))
        return niyah_cuda_repetition_fail(decode_state, result,
                                          NIYAH_ERR_INVALID_ARGUMENT);
    status = niyah_cuda_generation_workspace_floats_with_repetition(
        &model_state->config, repetition, &required_workspace);
    if (status != NIYAH_OK)
        return niyah_cuda_repetition_fail(decode_state, result, status);
    if (host_workspace_count < required_workspace)
        return niyah_cuda_repetition_fail(decode_state, result,
                                          NIYAH_ERR_BUFFER_TOO_SMALL);
    for (i = 0U; i < prompt_count; ++i) {
        if ((size_t)prompt_tokens[i] >= vocab)
            return niyah_cuda_repetition_fail(decode_state, result,
                                              NIYAH_ERR_INVALID_ARGUMENT);
    }
    status = niyah_sampler_init(&sampler, &config->sampler);
    if (status != NIYAH_OK)
        return niyah_cuda_repetition_fail(decode_state, result, status);

    logits = host_workspace;
    sampling_scratch = logits + vocab;
    for (i = 0U; i < prompt_count; ++i) {
        if (niyah_cuda_decode_token(model_state, decode_state, prompt_tokens[i],
                                    logits, vocab) != 0)
            return niyah_cuda_repetition_fail(decode_state, result, NIYAH_ERR_IO);
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
        const size_t history_count = generated < repetition->window
            ? generated : repetition->window;
        const uint32_t *history = output_tokens + (generated - history_count);
        status = niyah_cuda_repetition_entropy(logits, vocab, &entropy);
        if (status != NIYAH_OK && stop_report != NULL)
            stop_report->reason = NIYAH_STOP_NUMERIC_ERROR;
        if (status != NIYAH_OK)
            return niyah_cuda_repetition_fail(decode_state, result, status);
        /* Low entropy is confidence, not proof of a loop. Finite-logit
         * validation above remains mandatory; generated-token guards follow. */
        status = niyah_sampler_sample_with_history(&sampler, logits, vocab,
            history, history_count, repetition->penalty, sampling_scratch,
            vocab, &token);
        if (status != NIYAH_OK)
            return niyah_cuda_repetition_fail(decode_state, result, status);
        output_tokens[generated++] = token;
        result->generated_tokens = generated;

        if (config->stop_on_eos != 0 && token == config->eos_token) {
            result->stopped_on_eos = 1;
            break;
        }
        if (niyah_guard_token(&shared_guard, token) == NIYAH_STOP_SINGLE_TOKEN_LOOP ||
            shared_guard.reason == NIYAH_STOP_MULTI_TOKEN_LOOP) {
            break;
        }
        if (niyah_cuda_decode_token(model_state, decode_state, token,
                                    logits, vocab) != 0)
            return niyah_cuda_repetition_fail(decode_state, result, NIYAH_ERR_IO);
    }
    if (stop_report != NULL) {
        stop_report->reason = result->stopped_on_eos ? NIYAH_STOP_EOS :
            (shared_guard.reason == NIYAH_STOP_NONE ?
                NIYAH_STOP_TOKEN_BUDGET : shared_guard.reason);
        stop_report->detected_period = shared_guard.detected_period;
    }
    return NIYAH_OK;
}

NiyahStatus niyah_cuda_generate_with_repetition(
    const NiyahCudaModelState *model_state,
    NiyahCudaDecodeState *decode_state,
    const uint32_t *prompt_tokens,
    size_t prompt_count,
    const NiyahGenerationConfig *config,
    const NiyahRepetitionConfig *repetition,
    uint32_t *output_tokens,
    size_t output_capacity,
    NiyahGenerationResult *result,
    float *host_workspace,
    size_t host_workspace_count)
{
    return niyah_cuda_generate_with_repetition_impl(model_state, decode_state, prompt_tokens, prompt_count, config, repetition, output_tokens, output_capacity, result, host_workspace, host_workspace_count, NULL);
}

NiyahStatus niyah_cuda_generate_with_repetition_ex(
    const NiyahCudaModelState *model_state,
    NiyahCudaDecodeState *decode_state,
    const uint32_t *prompt_tokens,
    size_t prompt_count,
    const NiyahGenerationConfig *config,
    const NiyahRepetitionConfig *repetition,
    uint32_t *output_tokens,
    size_t output_capacity,
    NiyahGenerationResult *result,
    float *host_workspace,
    size_t host_workspace_count,
    NiyahGenerationStopReport *stop_report)
{
    NiyahStatus status;
    if (stop_report != NULL) memset(stop_report, 0, sizeof(*stop_report));
    status = niyah_cuda_generate_with_repetition_impl(model_state, decode_state, prompt_tokens, prompt_count, config, repetition, output_tokens, output_capacity, result, host_workspace, host_workspace_count, stop_report);
    if (status != NIYAH_OK && stop_report != NULL &&
        stop_report->reason == NIYAH_STOP_NONE)
        stop_report->reason = NIYAH_STOP_BACKEND_ERROR;
    return status;
}
