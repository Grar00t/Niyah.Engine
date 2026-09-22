#ifndef NIYAH_MODEL_H
#define NIYAH_MODEL_H

#include "niyah/common.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NIYAH_MODEL_VOCAB 256u

typedef struct niyah_model {
    uint64_t counts[NIYAH_MODEL_VOCAB][NIYAH_MODEL_VOCAB];
    uint64_t transitions_seen;
} niyah_model;

void niyah_model_init(niyah_model *model);
niyah_status niyah_model_observe(niyah_model *model, uint32_t prev, uint32_t next);
uint32_t niyah_model_greedy_next(const niyah_model *model, uint32_t prev);
double niyah_model_logprob_add1(const niyah_model *model, uint32_t prev, uint32_t next);

#ifdef __cplusplus
}
#endif

#endif
