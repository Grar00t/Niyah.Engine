#include "niyah/sampler.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

static uint64_t niyah_sampler_next_u64(uint64_t *state)
{
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * UINT64_C(2685821657736338717);
}

NiyahStatus niyah_sampler_init(NiyahSampler *sampler,
                               const NiyahSamplerConfig *config)
{
    if (sampler == NULL || config == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    if (!isfinite(config->temperature) || config->temperature < 0.0f) {
        return NIYAH_ERR_INVALID_CONFIG;
    }

    sampler->config = *config;
    sampler->state = config->seed != 0U
        ? config->seed
        : UINT64_C(0x4e495941485f534d);
    return NIYAH_OK;
}

NiyahStatus niyah_sampler_sample(NiyahSampler *sampler,
                                 const float *logits,
                                 size_t vocab_size,
                                 uint32_t *out_token)
{
    size_t i;

    if (sampler == NULL || logits == NULL || out_token == NULL ||
        vocab_size == 0U || vocab_size > (size_t)UINT32_MAX) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    for (i = 0U; i < vocab_size; ++i) {
        if (!isfinite(logits[i])) {
            return NIYAH_ERR_INVALID_CONFIG;
        }
    }

    if (sampler->config.temperature == 0.0f) {
        size_t best_index = 0U;
        float best_value = logits[0];
        for (i = 1U; i < vocab_size; ++i) {
            if (logits[i] > best_value) {
                best_value = logits[i];
                best_index = i;
            }
        }
        *out_token = (uint32_t)best_index;
        return NIYAH_OK;
    }

    {
        float max_logit = -FLT_MAX;
        double total = 0.0;
        double threshold;
        double cumulative = 0.0;
        const double temperature = (double)sampler->config.temperature;
        const uint64_t random_bits = niyah_sampler_next_u64(&sampler->state) >> 11U;
        const double unit = (double)random_bits * (1.0 / 9007199254740992.0);

        for (i = 0U; i < vocab_size; ++i) {
            if (logits[i] > max_logit) {
                max_logit = logits[i];
            }
        }
        for (i = 0U; i < vocab_size; ++i) {
            total += exp(((double)logits[i] - (double)max_logit) / temperature);
        }
        if (!(total > 0.0) || !isfinite(total)) {
            return NIYAH_ERR_INVALID_CONFIG;
        }

        threshold = unit * total;
        for (i = 0U; i < vocab_size; ++i) {
            cumulative += exp(((double)logits[i] - (double)max_logit) / temperature);
            if (threshold < cumulative) {
                *out_token = (uint32_t)i;
                return NIYAH_OK;
            }
        }

        *out_token = (uint32_t)(vocab_size - 1U);
    }
    return NIYAH_OK;
}
