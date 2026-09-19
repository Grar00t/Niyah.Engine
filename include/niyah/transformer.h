#ifndef NIYAH_TRANSFORMER_H
#define NIYAH_TRANSFORMER_H

#include "niyah/niyah.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Reference CPU forward path for the canonical NiyahModel.
 *
 * The caller owns logits and workspace. No allocation occurs inside forward.
 * logits contains token_count * vocab_size floats in row-major [token][vocab]
 * order, which keeps this path usable by both inference and future training.
 */
NiyahStatus niyah_transformer_workspace_floats(const NiyahModelConfig *config,
                                               size_t token_count,
                                               size_t *out_floats);

NiyahStatus niyah_transformer_forward(const NiyahModel *model,
                                      const uint32_t *tokens,
                                      size_t token_count,
                                      float *logits,
                                      size_t logits_count,
                                      float *workspace,
                                      size_t workspace_count);

/* Same as niyah_transformer_forward, but adds model->weights[layout.segment_embedding
 * + segment_ids[t] * embedding_dim] into token t's embedding before layer 0.
 * Requires config->n_segments > 0 and segment_ids[t] < config->n_segments for all t.
 * This is a training signal for distinguishing input spans (e.g. instruction vs.
 * untrusted data); it is not a security boundary. */
NiyahStatus niyah_transformer_forward_with_segments(const NiyahModel *model,
                                                    const uint32_t *tokens,
                                                    size_t token_count,
                                                    const uint32_t *segment_ids,
                                                    float *logits,
                                                    size_t logits_count,
                                                    float *workspace,
                                                    size_t workspace_count);

#ifdef __cplusplus
}
#endif

#endif
