#ifndef NIYAH_SAMPLER_H
#define NIYAH_SAMPLER_H

#include "niyah/niyah.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NiyahSamplerConfig {
    /* 0.0f selects greedy argmax. Values > 0 enable seeded categorical sampling. */
    float temperature;
    uint64_t seed;
} NiyahSamplerConfig;

typedef struct NiyahSampler {
    NiyahSamplerConfig config;
    uint64_t state;
} NiyahSampler;

NiyahStatus niyah_sampler_init(NiyahSampler *sampler,
                               const NiyahSamplerConfig *config);

NiyahStatus niyah_sampler_sample(NiyahSampler *sampler,
                                 const float *logits,
                                 size_t vocab_size,
                                 uint32_t *out_token);

#ifdef __cplusplus
}
#endif

#endif
