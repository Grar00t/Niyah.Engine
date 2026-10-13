#ifndef NIYAH_MODEL_INTERNAL_H
#define NIYAH_MODEL_INTERNAL_H

#include "niyah/niyah.h"

typedef struct NiyahLayerWeightsView {
    const float *attn_norm;
    const float *wq;
    const float *wk;
    const float *wv;
    const float *wo;
    const float *ffn_norm;
    const float *w_gate;
    const float *w_up;
    const float *w_down;
} NiyahLayerWeightsView;

int niyah_model_is_read_only(const NiyahModel *model);
const float *niyah_model_token_embedding_weights(const NiyahModel *model);
const float *niyah_model_segment_embedding_weights(const NiyahModel *model);
const float *niyah_model_final_norm_weights(const NiyahModel *model);
const float *niyah_model_lm_head_weights(const NiyahModel *model);
NiyahStatus niyah_model_layer_weights_view(const NiyahModel *model,
                                           uint32_t layer_index,
                                           NiyahLayerWeightsView *out);

#endif
