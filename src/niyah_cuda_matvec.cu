#include "niyah_cuda_matvec.h"

#include <cuda_runtime.h>

#include <stddef.h>

__global__ static void niyah_cuda_matvec_kernel(float *out,
                                                const float *matrix,
                                                const float *x,
                                                size_t rows,
                                                size_t cols)
{
    const size_t row =
        (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;

    if (row < rows) {
        const float *matrix_row = matrix + row * cols;
        float sum = 0.0f;
        size_t col;

        for (col = 0U; col < cols; ++col) {
            sum += matrix_row[col] * x[col];
        }
        out[row] = sum;
    }
}

extern "C" int niyah_cuda_matvec(float *out,
                                 const float *matrix,
                                 const float *x,
                                 size_t rows,
                                 size_t cols)
{
    float *device_out = NULL;
    float *device_matrix = NULL;
    float *device_x = NULL;
    size_t matrix_count;
    size_t matrix_bytes;
    size_t out_bytes;
    size_t x_bytes;
    cudaError_t error;
    int result = 1;

    if (out == NULL || matrix == NULL || x == NULL ||
        rows == 0U || cols == 0U) {
        return 1;
    }

    if (rows > ((size_t)-1) / cols) {
        return 1;
    }
    matrix_count = rows * cols;

    if (matrix_count > ((size_t)-1) / sizeof(float) ||
        rows > ((size_t)-1) / sizeof(float) ||
        cols > ((size_t)-1) / sizeof(float)) {
        return 1;
    }

    matrix_bytes = matrix_count * sizeof(float);
    out_bytes = rows * sizeof(float);
    x_bytes = cols * sizeof(float);

    error = cudaMalloc((void **)&device_matrix, matrix_bytes);
    if (error != cudaSuccess) goto cleanup;

    error = cudaMalloc((void **)&device_x, x_bytes);
    if (error != cudaSuccess) goto cleanup;

    error = cudaMalloc((void **)&device_out, out_bytes);
    if (error != cudaSuccess) goto cleanup;

    error = cudaMemcpy(device_matrix, matrix, matrix_bytes,
                       cudaMemcpyHostToDevice);
    if (error != cudaSuccess) goto cleanup;

    error = cudaMemcpy(device_x, x, x_bytes, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) goto cleanup;

    {
        const unsigned int threads = 128U;
        const unsigned int blocks =
            (unsigned int)((rows + (size_t)threads - 1U) /
                           (size_t)threads);

        niyah_cuda_matvec_kernel<<<blocks, threads>>>(
            device_out, device_matrix, device_x, rows, cols);
    }

    error = cudaGetLastError();
    if (error != cudaSuccess) goto cleanup;

    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) goto cleanup;

    error = cudaMemcpy(out, device_out, out_bytes,
                       cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) goto cleanup;

    result = 0;

cleanup:
    if (device_out != NULL) (void)cudaFree(device_out);
    if (device_x != NULL) (void)cudaFree(device_x);
    if (device_matrix != NULL) (void)cudaFree(device_matrix);
    return result;
}
