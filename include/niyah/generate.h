#ifndef NIYAH_GENERATE_H
#define NIYAH_GENERATE_H

#include "niyah/decode.h"
#include "niyah/sampler.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NiyahGenerationConfig {
    size_t max_new_tokens;
    uint32_t eos_token;
    int stop_on_eos;
    NiyahSamplerConfig sampler;
} NiyahGenerationConfig;

typedef struct NiyahGenerationResult {
    size_t prompt_tokens;
    size_t generated_tokens;
    int stopped_on_eos;
} NiyahGenerationResult;

NiyahStatus niyah_generation_workspace_floats(const NiyahModelConfig *config,
                                              size_t *out_floats);

/*
 * Starts from an empty KV cache, prefills prompt_tokens with the canonical
 * incremental decoder, then samples and decodes new tokens autoregressively.
 * output_tokens contains generated tokens only (not the prompt).
 *
 * On failure the cache is reset and result is zeroed. The caller owns all
 * output/workspace memory; no allocation occurs in the generation hot path.
 */
NiyahStatus niyah_generate(const NiyahModel *model,
                           NiyahKVCache *cache,
                           const uint32_t *prompt_tokens,
                           size_t prompt_count,
                           const NiyahGenerationConfig *config,
                           uint32_t *output_tokens,
                           size_t output_capacity,
                           NiyahGenerationResult *result,
                           float *workspace,
                           size_t workspace_count);

#ifdef __cplusplus
}
#endif

#endif
