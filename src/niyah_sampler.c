#include "niyah/sampler.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>


/* NIYAH_SAMPLER_TRACE_V1 */
#include <stdio.h>
#include <stdlib.h>
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

static int niyah_sampler_trace_enabled(void)
{
#if defined(_WIN32)
    return GetEnvironmentVariableW(
        L"NIYAH_SAMPLER_TRACE", NULL, 0) > 0U;
#else
    return getenv("NIYAH_SAMPLER_TRACE") != NULL;
#endif
}

static void niyah_sampler_trace_logits(
    const float *logits, size_t count)
{
    size_t i;
    if (!niyah_sampler_trace_enabled()) return;
    fprintf(stderr, "NTRACE_LOGITS");
    for (i = 0U; i < count; ++i) {
        fprintf(stderr, " %.9g", (double)logits[i]);
    }
    fprintf(stderr, "\n");
}

static void niyah_sampler_trace_token(uint32_t token)
{
    if (!niyah_sampler_trace_enabled()) return;
    fprintf(stderr, "NTRACE_TOKEN %u\n", token);
}

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

    niyah_sampler_trace_logits(logits, vocab_size);

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
        niyah_sampler_trace_token(*out_token);
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
                niyah_sampler_trace_token(*out_token);
                return NIYAH_OK;
            }
        }

        *out_token = (uint32_t)(vocab_size - 1U);
        niyah_sampler_trace_token(*out_token);
    }
    return NIYAH_OK;
}

/* NIYAH_HISTORY_PENALTY_API_V1 */
NiyahStatus niyah_sampler_sample_with_history(
    NiyahSampler *sampler, const float *logits, size_t vocab_size,
    const uint32_t *history, size_t history_count, float penalty,
    float *scratch, size_t scratch_count, uint32_t *out_token)
{
    size_t i, j;
    if (!sampler || !logits || !out_token || !vocab_size ||
        vocab_size > (size_t)UINT32_MAX || (history_count && !history))
        return NIYAH_ERR_INVALID_ARGUMENT;
    if (!isfinite(penalty) || penalty < 1.0f ||
        !isfinite(sampler->config.temperature) ||
        sampler->config.temperature < 0.0f)
        return NIYAH_ERR_INVALID_CONFIG;
    for (i = 0; i < vocab_size; ++i)
        if (!isfinite(logits[i])) return NIYAH_ERR_INVALID_CONFIG;
    for (j = 0; j < history_count; ++j)
        if ((size_t)history[j] >= vocab_size)
            return NIYAH_ERR_INVALID_ARGUMENT;
    if (penalty == 1.0f || !history_count)
        return niyah_sampler_sample(sampler, logits, vocab_size, out_token);
    if (!scratch) return NIYAH_ERR_INVALID_ARGUMENT;
    if (scratch_count < vocab_size) return NIYAH_ERR_BUFFER_TOO_SMALL;
    if (vocab_size > SIZE_MAX / sizeof(float)) return NIYAH_ERR_OVERFLOW;
    {
        uintptr_t a = (uintptr_t)logits, b = (uintptr_t)scratch;
        size_t bytes = vocab_size * sizeof(float);
        if ((a <= b && b - a < bytes) || (b < a && a - b < bytes))
            return NIYAH_ERR_INVALID_ARGUMENT;
    }
    for (i = 0; i < vocab_size; ++i) {
        double value = (double)logits[i];
        for (j = 0; j < history_count; ++j) {
            if ((size_t)history[j] == i) {
                value = value > 0.0 ? value / (double)penalty
                                    : value * (double)penalty;
                break;
            }
        }
        if (!isfinite(value) || value > FLT_MAX || value < -FLT_MAX)
            return NIYAH_ERR_INVALID_CONFIG;
        scratch[i] = (float)value;
    }
    return niyah_sampler_sample(sampler, scratch, vocab_size, out_token);
}
