#ifndef NIYAH_SAMPLER_INTERNAL_H
#define NIYAH_SAMPLER_INTERNAL_H

#include "niyah.h"

typedef struct {
    float   prob;
    int32_t index;
} NiyahSamplerCandidate;

int32_t niyah_sample_with_scratch(const float* logits,
                                  int32_t n_vocab,
                                  const NiyahSamplerConfig* config,
                                  float* probs,
                                  NiyahSamplerCandidate* pool,
                                  int32_t pool_capacity);

#endif
