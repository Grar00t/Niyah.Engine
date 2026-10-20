#ifndef NIYAH_GENERATE_H
#define NIYAH_GENERATE_H

#include "niyah/decode.h"
#include "niyah/sampler.h"
#include "niyah/shared_guard.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* NIYAH_SHARED_GUARD_INTEGRATION_V1: ABI-additive stop report. */
typedef struct NiyahGenerationStopReport {
    NiyahStopReason reason;
    unsigned detected_period;
} NiyahGenerationStopReport;

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

/* Repetition applies once per distinct token in the trailing window of
 * generated output only. Prompt tokens are not included. penalty == 1.0f
 * delegates to the legacy API, including its workspace and RNG behavior.
 * penalty must be finite and >= 1.0f; window must be nonzero. */
typedef struct NiyahRepetitionConfig {
    float penalty;
    size_t window;
} NiyahRepetitionConfig;

/* Enabled layout: [decode workspace][raw logits: V][sampling scratch: V].
 * Disabled layout is exactly the legacy workspace. Counts are in floats. */
NiyahStatus niyah_generation_workspace_floats_with_repetition(
    const NiyahModelConfig *config,
    const NiyahRepetitionConfig *repetition,
    size_t *out_floats);

/* All arrays are caller-owned and must not overlap each other or model/cache
 * storage. output_tokens doubles as generated-token history; no extra history
 * allocation and no allocation/free per token. Raw logits remain unmodified
 * by the penalty. Finite-logit validation and generated-token loop guards remain.
 * Low entropy alone is not a stop condition; a confident prediction can be correct.
 * An enabled-path failure resets the cache and zeroes result; output contents
 * after a failure are unspecified and must not be consumed. */
NiyahStatus niyah_generate_with_repetition(
    const NiyahModel *model,
    NiyahKVCache *cache,
    const uint32_t *prompt_tokens,
    size_t prompt_count,
    const NiyahGenerationConfig *config,
    const NiyahRepetitionConfig *repetition,
    uint32_t *output_tokens,
    size_t output_capacity,
    NiyahGenerationResult *result,
    float *workspace,
    size_t workspace_count);

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
    NiyahGenerationStopReport *stop_report);

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
    NiyahGenerationStopReport *stop_report);

#ifdef __cplusplus
}
#endif

#endif
