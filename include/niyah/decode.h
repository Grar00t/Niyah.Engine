#ifndef NIYAH_DECODE_H
#define NIYAH_DECODE_H

#include "niyah/niyah.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NiyahKVCache {
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    size_t context_length;
    size_t head_dim;
    size_t kv_dim;
    size_t next_position;
    size_t values_per_tensor;
    float *keys;
    float *values;
} NiyahKVCache;

NiyahStatus niyah_kv_cache_create(NiyahKVCache *cache,
                                  const NiyahModelConfig *config);
void niyah_kv_cache_destroy(NiyahKVCache *cache);
void niyah_kv_cache_reset(NiyahKVCache *cache);
size_t niyah_kv_cache_position(const NiyahKVCache *cache);

NiyahStatus niyah_decode_workspace_floats(const NiyahModelConfig *config,
                                          size_t *out_floats);

/*
 * Incrementally decodes exactly one token at cache->next_position.
 * The cache stores per-layer K/V vectors after RoPE and advances only after a
 * complete successful token decode. The caller owns logits and workspace.
 */
NiyahStatus niyah_transformer_decode_token(const NiyahModel *model,
                                           NiyahKVCache *cache,
                                           uint32_t token,
                                           float *logits,
                                           size_t logits_count,
                                           float *workspace,
                                           size_t workspace_count);

#ifdef __cplusplus
}
#endif

#endif
