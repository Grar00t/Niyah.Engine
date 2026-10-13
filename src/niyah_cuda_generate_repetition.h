#ifndef NIYAH_CUDA_GENERATE_REPETITION_H
#define NIYAH_CUDA_GENERATE_REPETITION_H

#include "niyah_cuda_matvec.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Decode workspace remains in the existing persistent device decode state.
 * Host layout when enabled: [raw logits: V][sampling scratch: V].
 * When disabled, returns V and generation delegates to niyah_cuda_generate.
 * All host arrays must be disjoint and remain caller-owned. */
NiyahStatus niyah_cuda_generation_workspace_floats_with_repetition(
    const NiyahModelConfig *config,
    const NiyahRepetitionConfig *repetition,
    size_t *out_floats);

/* Enabled path uses the CPU generation stop policy: raw entropy < 0.05,
 * eight identical emitted tokens, optional EOS, or max_new_tokens.
 * No per-token host allocation. GPU decode is performed by the existing
 * niyah_cuda_decode_token API; this is not a CPU fallback.
 * If cache reset fails, returns NIYAH_ERR_IO and the state must be destroyed. */
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
    size_t host_workspace_count);

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
    NiyahGenerationStopReport *stop_report);

#ifdef __cplusplus
}
#endif

#endif
