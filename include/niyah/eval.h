#ifndef NIYAH_EVAL_H
#define NIYAH_EVAL_H

#include "niyah/niyah.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NiyahEvaluationSample {
    const uint32_t *tokens;
    const uint32_t *targets;
    size_t token_count;
    size_t loss_start;
} NiyahEvaluationSample;

typedef struct NiyahEvaluationMetrics {
    size_t sample_count;
    size_t token_count;
    double mean_loss;
    double perplexity;
} NiyahEvaluationMetrics;

/* Read-only held-out evaluation over caller-owned token/target samples.
 *
 * Each sample keeps its full causal context. Direct objective scoring begins
 * at loss_start and covers [loss_start, token_count). loss_start == 0 is the
 * canonical all-token objective.
 *
 * mean_loss is token-weighted mean cross entropy over scored targets, not an
 * unweighted mean of per-sample losses. perplexity is exp(mean_loss).
 */
NiyahStatus niyah_evaluate(
    const NiyahModel *model,
    const NiyahEvaluationSample *samples,
    size_t sample_count,
    NiyahEvaluationMetrics *out_metrics);

#ifdef __cplusplus
}
#endif

#endif
