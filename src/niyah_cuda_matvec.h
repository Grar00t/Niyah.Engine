#ifndef NIYAH_CUDA_MATVEC_H
#define NIYAH_CUDA_MATVEC_H

#include "niyah/niyah.h"
#include "niyah/generate.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int niyah_cuda_matvec(float *out,
                      const float *matrix,
                      const float *x,
                      size_t rows,
                      size_t cols);

typedef struct NiyahCudaModelState {
    void *device_weights;
    void *device_input;
    void *device_output;
    size_t weight_count;
    size_t input_capacity;
    size_t output_capacity;
    NiyahModelConfig config;
    NiyahModelLayout layout;
} NiyahCudaModelState;

int niyah_cuda_model_state_create(NiyahCudaModelState *state,
                                  const NiyahModel *model);
int niyah_cuda_model_state_sync(NiyahCudaModelState *state,
                                const NiyahModel *model);
void niyah_cuda_model_state_destroy(NiyahCudaModelState *state);
int niyah_cuda_model_state_matvec(NiyahCudaModelState *state,
                                  size_t weight_offset,
                                  const float *x,
                                  float *out,
                                  size_t rows,
                                  size_t cols);

int niyah_cuda_model_state_matvec_device(
    const NiyahCudaModelState *state,
    size_t weight_offset,
    const void *device_x,
    void *device_out,
    size_t rows,
    size_t cols);


typedef struct NiyahCudaTrainState {
    void *device_gradients;
    void *device_workspace;
    void *device_tokens;
    void *device_targets;
    size_t gradient_capacity;
    size_t workspace_capacity;
    size_t token_capacity;
} NiyahCudaTrainState;

int niyah_cuda_train_state_create(
    NiyahCudaTrainState *state,
    const NiyahModel *model,
    size_t max_tokens,
    size_t workspace_floats);

void niyah_cuda_train_state_destroy(
    NiyahCudaTrainState *state);

int niyah_cuda_train_state_zero_gradients(
    NiyahCudaTrainState *state);

int niyah_cuda_train_state_copy_gradients_to_host(
    const NiyahCudaTrainState *state,
    float *host_gradients,
    size_t gradient_count);


int niyah_cuda_train_state_copy_workspace_from_host(
    NiyahCudaTrainState *state,
    size_t workspace_offset,
    const float *host_values,
    size_t value_count);

int niyah_cuda_train_state_copy_workspace_to_host(
    const NiyahCudaTrainState *state,
    size_t workspace_offset,
    float *host_values,
    size_t value_count);

int niyah_cuda_train_state_zero_workspace(
    NiyahCudaTrainState *state,
    size_t workspace_offset,
    size_t value_count);

int niyah_cuda_train_state_copy_gradient_range_to_host(
    const NiyahCudaTrainState *state,
    size_t gradient_offset,
    float *host_values,
    size_t value_count);

int niyah_cuda_train_linear_forward(
    const NiyahCudaModelState *model_state,
    NiyahCudaTrainState *train_state,
    size_t weight_offset,
    size_t x_workspace_offset,
    size_t y_workspace_offset,
    size_t token_count,
    size_t rows,
    size_t cols);

int niyah_cuda_train_linear_backward(
    const NiyahCudaModelState *model_state,
    NiyahCudaTrainState *train_state,
    size_t weight_offset,
    size_t x_workspace_offset,
    size_t dy_workspace_offset,
    size_t dx_workspace_offset,
    size_t token_count,
    size_t rows,
    size_t cols);

typedef struct NiyahCudaDecodeState {
    void *device_keys;
    void *device_values;
    void *device_workspace;
    void *device_logits;
    size_t context_length;
    size_t head_dim;
    size_t kv_dim;
    size_t values_per_tensor;
    size_t workspace_floats;
    size_t logits_capacity;
    size_t next_position;
    NiyahModelConfig config;
} NiyahCudaDecodeState;

int niyah_cuda_decode_state_create(
    NiyahCudaDecodeState *state,
    const NiyahCudaModelState *model_state);
int niyah_cuda_decode_state_reset(NiyahCudaDecodeState *state);
void niyah_cuda_decode_state_destroy(NiyahCudaDecodeState *state);

int niyah_cuda_decode_token(
    const NiyahCudaModelState *model_state,
    NiyahCudaDecodeState *decode_state,
    uint32_t token,
    float *logits,
    size_t logits_count);

NiyahStatus niyah_cuda_generate(
    const NiyahCudaModelState *model_state,
    NiyahCudaDecodeState *decode_state,
    const uint32_t *prompt_tokens,
    size_t prompt_count,
    const NiyahGenerationConfig *config,
    uint32_t *output_tokens,
    size_t output_capacity,
    NiyahGenerationResult *result,
    float *host_logits,
    size_t host_logits_count);

#ifdef __cplusplus
}
#endif

#endif
