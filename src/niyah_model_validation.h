#ifndef NIYAH_MODEL_VALIDATION_H
#define NIYAH_MODEL_VALIDATION_H

#include "niyah/niyah.h"

/* Internal preflight: never trust caller-editable offsets for gradient writes. */
static NiyahStatus niyah_training_model_validate(const NiyahModel *model)
{
    NiyahModelLayout c;
    const NiyahModelLayout *l;
    NiyahStatus status;
    if (model == NULL || model->weights == NULL || model->weight_count == 0U)
        return NIYAH_ERR_INVALID_ARGUMENT;
    status = niyah_model_layout_compute(&model->config, &c);
    if (status != NIYAH_OK) return status;
    l = &model->layout;
    if (model->weight_count != c.total_floats ||
        l->token_embedding != c.token_embedding ||
        l->segment_embedding != c.segment_embedding ||
        l->layers != c.layers || l->layer_stride != c.layer_stride ||
        l->final_norm != c.final_norm || l->lm_head != c.lm_head ||
        l->total_floats != c.total_floats || l->head_dim != c.head_dim ||
        l->kv_dim != c.kv_dim)
        return NIYAH_ERR_INVALID_CONFIG;
    return NIYAH_OK;
}

#endif
