#ifndef NIYAH_EVAL_H
#define NIYAH_EVAL_H

#include "niyah/common.h"
#include "niyah/dataset.h"
#include "niyah/model.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct niyah_eval_result {
    uint64_t transitions;
    double nll;
    double avg_nll;
    double perplexity;
} niyah_eval_result;

niyah_status niyah_eval_model(
    const niyah_model *model,
    const niyah_dataset *dataset,
    niyah_eval_result *out);
niyah_status niyah_eval_add1_bigram_baseline(
    const niyah_dataset *dataset,
    niyah_eval_result *out);

#ifdef __cplusplus
}
#endif

#endif
