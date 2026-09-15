#ifndef NIYAH_NIYAH_H
#define NIYAH_NIYAH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum NiyahStatus {
    NIYAH_OK = 0,
    NIYAH_ERR_INVALID_ARGUMENT = 1,
    NIYAH_ERR_INVALID_CONFIG = 2,
    NIYAH_ERR_OVERFLOW = 3,
    NIYAH_ERR_OUT_OF_MEMORY = 4
} NiyahStatus;

typedef struct NiyahModelConfig {
    uint32_t vocab_size;
    uint32_t context_length;
    uint32_t embedding_dim;
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t ffn_hidden_dim;
    float rms_norm_eps;
    int tie_word_embeddings;
} NiyahModelConfig;

typedef struct NiyahLayerLayout {
    size_t attn_norm;
    size_t wq;
    size_t wk;
    size_t wv;
    size_t wo;
    size_t ffn_norm;
    size_t w_gate;
    size_t w_up;
    size_t w_down;
} NiyahLayerLayout;

typedef struct NiyahModelLayout {
    size_t token_embedding;
    size_t layers;
    size_t layer_stride;
    size_t final_norm;
    size_t lm_head;
    size_t total_floats;
    size_t head_dim;
    size_t kv_dim;
} NiyahModelLayout;

typedef struct NiyahModel {
    NiyahModelConfig config;
    NiyahModelLayout layout;
    float *weights;
    size_t weight_count;
} NiyahModel;

NiyahStatus niyah_model_config_validate(const NiyahModelConfig *config);
NiyahStatus niyah_model_layout_compute(const NiyahModelConfig *config,
                                       NiyahModelLayout *layout);
NiyahStatus niyah_model_layer_layout(const NiyahModelConfig *config,
                                     const NiyahModelLayout *layout,
                                     uint32_t layer_index,
                                     NiyahLayerLayout *layer);
NiyahStatus niyah_model_create(NiyahModel *model, const NiyahModelConfig *config);
void niyah_model_destroy(NiyahModel *model);
NiyahStatus niyah_model_reset_parameters(NiyahModel *model, uint64_t seed);

void niyah_matvec(float *out,
                  const float *matrix,
                  const float *x,
                  size_t rows,
                  size_t cols);

NiyahStatus niyah_rmsnorm(float *out,
                          const float *x,
                          const float *weight,
                          size_t n,
                          float eps);

#ifdef __cplusplus
}
#endif

#endif
