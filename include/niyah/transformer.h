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

#ifdef __cplusplus
}
#endif

#endif
