#include "niyah/model.h"
#include <math.h>
#include <string.h>

void niyah_model_init(niyah_model *model) {
    if (!model) return;
    memset(model, 0, sizeof(*model));
}

niyah_status niyah_model_observe(niyah_model *model, uint32_t prev, uint32_t next) {
    if (!model || prev >= NIYAH_MODEL_VOCAB || next >= NIYAH_MODEL_VOCAB) return NIYAH_ERR_INVALID;
    if (model->counts[prev][next] == UINT64_MAX || model->transitions_seen == UINT64_MAX) return NIYAH_ERR_STATE;
    model->counts[prev][next]++;
    model->transitions_seen++;
    return NIYAH_OK;
}

uint32_t niyah_model_greedy_next(const niyah_model *model, uint32_t prev) {
    if (!model || prev >= NIYAH_MODEL_VOCAB) return 0u;
    uint32_t best = 0u;
    uint64_t best_count = model->counts[prev][0];
    for (uint32_t t = 1u; t < NIYAH_MODEL_VOCAB; ++t) {
        if (model->counts[prev][t] > best_count) {
            best_count = model->counts[prev][t];
            best = t;
        }
    }
    return best;
}

double niyah_model_logprob_add1(const niyah_model *model, uint32_t prev, uint32_t next) {
    if (!model || prev >= NIYAH_MODEL_VOCAB || next >= NIYAH_MODEL_VOCAB) return -INFINITY;
    uint64_t row_total = 0u;
    for (uint32_t t = 0u; t < NIYAH_MODEL_VOCAB; ++t) row_total += model->counts[prev][t];
    const double num = (double)model->counts[prev][next] + 1.0;
    const double den = (double)row_total + (double)NIYAH_MODEL_VOCAB;
    return log(num / den);
}
