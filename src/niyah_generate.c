#include "niyah/generate.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

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
    size_t required_workspace = 0U;
    size_t decode_workspace_count = 0U;
    size_t required_context = 0U;
    float *decode_workspace;
    float *logits;
    size_t i;
    size_t generated = 0U;
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

    result->prompt_tokens = prompt_count;

    while (generated < config->max_new_tokens) {
        uint32_t token = 0U;

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

    return NIYAH_OK;
}
