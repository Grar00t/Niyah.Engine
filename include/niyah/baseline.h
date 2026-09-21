#ifndef NIYAH_BASELINE_H
#define NIYAH_BASELINE_H

#include "niyah/eval.h"
#include "niyah/niyah.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

NiyahStatus niyah_add1_bigram_mean_nll(
    const uint32_t *tokens,
    size_t token_count,
    size_t vocab_size,
    double *out_mean_nll_nats);

/* Add-1 bigram reference over the exact scored target geometry used by
 * NiyahEvaluationSample. Full sample context is retained, but only
 * transitions in [loss_start, token_count) contribute counts and NLL.
 */
NiyahStatus niyah_add1_bigram_samples_mean_nll(
    const NiyahEvaluationSample *samples,
    size_t sample_count,
    size_t vocab_size,
    double *out_mean_nll_nats);

#ifdef __cplusplus
}
#endif

#endif
