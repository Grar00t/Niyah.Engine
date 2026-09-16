#ifndef NIYAH_CUDA_MATVEC_H
#define NIYAH_CUDA_MATVEC_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int niyah_cuda_matvec(float *out,
                      const float *matrix,
                      const float *x,
                      size_t rows,
                      size_t cols);

#ifdef __cplusplus
}
#endif

#endif
