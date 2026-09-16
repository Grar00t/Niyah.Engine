#ifndef NIYAH_CUDA_MATVEC_H
#define NIYAH_CUDA_MATVEC_H

#include "niyah/niyah.h"

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

#ifdef __cplusplus
}
#endif

#endif
