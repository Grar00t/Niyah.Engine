#include "niyah/generate.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>



#define NIYAH_SAME_TOKEN_LIMIT 8U
#define NIYAH_ENTROPY_FLOOR 0.05f

static float niyah_generation_entropy(const float *logits, size_t count)
{
    size_t i;
    float max_logit;
    double sum = 0.0;
    double weighted = 0.0;

    if (logits == NULL || count == 0U) {
        return 0.0f;
    }
    max_logit = logits[0];
    for (i = 1U; i < count; ++i) {
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
        return 0.0f;
    }
    return (float)(log(sum) - weighted / sum);
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
    uint32_t last_token = 0U;
    size_t same_token_run = 0U;
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
        const float entropy =
            niyah_generation_entropy(logits, (size_t)model->config.vocab_size);

        if (entropy < NIYAH_ENTROPY_FLOOR) {
            break;
        }

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

        if (generated == 1U || token != last_token) {
            last_token = token;
            same_token_run = 1U;
        } else {
            same_token_run += 1U;
        }

        if (config->stop_on_eos != 0 && token == config->eos_token) {
            result->stopped_on_eos = 1;
            break;
        }
        if (same_token_run >= NIYAH_SAME_TOKEN_LIMIT) {
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
