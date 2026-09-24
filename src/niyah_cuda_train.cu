#include "niyah_cuda_matvec.h"

#include <cuda_runtime.h>

#include <limits.h>
#include <stddef.h>

static int range_ok(size_t offset, size_t count, size_t capacity)
{
    return offset <= capacity && count <= capacity - offset;
}

static int overlap(size_t a, size_t an, size_t b, size_t bn)
{
    if (an == 0U || bn == 0U) return 0;
    return a < b + bn && b < a + an;
}

extern "C" int niyah_cuda_train_state_copy_workspace_from_host(
    NiyahCudaTrainState *state,
    size_t offset,
    const float *host,
    size_t count)
{
    if (state == NULL || state->device_workspace == NULL ||
        host == NULL || count == 0U ||
        !range_ok(offset, count, state->workspace_capacity) ||
        count > ((size_t)-1) / sizeof(float))
        return 1;

    return cudaMemcpy(
        (float *)state->device_workspace + offset,
        host,
        count * sizeof(float),
        cudaMemcpyHostToDevice) == cudaSuccess ? 0 : 1;
}

extern "C" int niyah_cuda_train_state_copy_workspace_to_host(
    const NiyahCudaTrainState *state,
    size_t offset,
    float *host,
    size_t count)
{
    if (state == NULL || state->device_workspace == NULL ||
        host == NULL || count == 0U ||
        !range_ok(offset, count, state->workspace_capacity) ||
        count > ((size_t)-1) / sizeof(float))
        return 1;

    return cudaMemcpy(
        host,
        (const float *)state->device_workspace + offset,
        count * sizeof(float),
        cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : 1;
}

extern "C" int niyah_cuda_train_state_zero_workspace(
    NiyahCudaTrainState *state,
    size_t offset,
    size_t count)
{
    if (state == NULL || state->device_workspace == NULL ||
        count == 0U ||
        !range_ok(offset, count, state->workspace_capacity) ||
        count > ((size_t)-1) / sizeof(float))
        return 1;

    return cudaMemset(
        (float *)state->device_workspace + offset,
        0,
        count * sizeof(float)) == cudaSuccess ? 0 : 1;
}

extern "C" int niyah_cuda_train_state_copy_gradient_range_to_host(
    const NiyahCudaTrainState *state,
    size_t offset,
    float *host,
    size_t count)
{
    if (state == NULL || state->device_gradients == NULL ||
        host == NULL || count == 0U ||
        !range_ok(offset, count, state->gradient_capacity) ||
        count > ((size_t)-1) / sizeof(float))
        return 1;

    return cudaMemcpy(
        host,
        (const float *)state->device_gradients + offset,
        count * sizeof(float),
        cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : 1;
}

__global__ static void linear_forward_kernel(
    float *y,
    const float *w,
    const float *x,
    size_t tokens,
    size_t rows,
    size_t cols)
{
    const size_t index =
        (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = tokens * rows;

    if (index < total) {
        const size_t t = index / rows;
        const size_t r = index % rows;
        float sum = 0.0f;

        for (size_t c = 0U; c < cols; ++c)
            sum += w[r * cols + c] * x[t * cols + c];

        y[index] = sum;
    }
}

__global__ static void linear_dweight_kernel(
    float *dw,
    const float *x,
    const float *dy,
    size_t tokens,
    size_t rows,
    size_t cols)
{
    const size_t index =
        (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = rows * cols;

    if (index < total) {
        const size_t r = index / cols;
        const size_t c = index % cols;
        float value = dw[index];

        for (size_t t = 0U; t < tokens; ++t)
            value += dy[t * rows + r] * x[t * cols + c];

        dw[index] = value;
    }
}

__global__ static void linear_dx_kernel(
    float *dx,
    const float *w,
    const float *dy,
    size_t tokens,
    size_t rows,
    size_t cols)
{
    const size_t index =
        (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = tokens * cols;

    if (index < total) {
        const size_t t = index / cols;
        const size_t c = index % cols;
        float value = dx[index];

        for (size_t r = 0U; r < rows; ++r)
            value += w[r * cols + c] * dy[t * rows + r];

        dx[index] = value;
    }
}

static int shape_ok(
    const NiyahCudaModelState *ms,
    const NiyahCudaTrainState *ts,
    size_t weight_offset,
    size_t tokens,
    size_t rows,
    size_t cols,
    size_t *matrix,
    size_t *x_count,
    size_t *y_count)
{
    if (ms == NULL || ts == NULL ||
        ms->device_weights == NULL ||
        ts->device_gradients == NULL ||
        ts->device_workspace == NULL ||
        tokens == 0U || rows == 0U || cols == 0U ||
        rows > ((size_t)-1) / cols ||
        tokens > ((size_t)-1) / cols ||
        tokens > ((size_t)-1) / rows)
        return 0;

    *matrix = rows * cols;
    *x_count = tokens * cols;
    *y_count = tokens * rows;

    return weight_offset <= ms->weight_count &&
           *matrix <= ms->weight_count - weight_offset &&
           weight_offset <= ts->gradient_capacity &&
           *matrix <= ts->gradient_capacity - weight_offset;
}

extern "C" int niyah_cuda_train_linear_forward(
    const NiyahCudaModelState *ms,
    NiyahCudaTrainState *ts,
    size_t weight_offset,
    size_t x_offset,
    size_t y_offset,
    size_t tokens,
    size_t rows,
    size_t cols)
{
    size_t matrix, x_count, y_count;
    const unsigned int threads = 256U;

    if (!shape_ok(ms, ts, weight_offset, tokens, rows, cols,
                  &matrix, &x_count, &y_count) ||
        !range_ok(x_offset, x_count, ts->workspace_capacity) ||
        !range_ok(y_offset, y_count, ts->workspace_capacity) ||
        overlap(x_offset, x_count, y_offset, y_count) ||
        y_count > (size_t)UINT_MAX * threads)
        return 1;

    (void)matrix;

    const unsigned int blocks =
        (unsigned int)((y_count + threads - 1U) / threads);

    linear_forward_kernel<<<blocks, threads>>>(
        (float *)ts->device_workspace + y_offset,
        (const float *)ms->device_weights + weight_offset,
        (const float *)ts->device_workspace + x_offset,
        tokens, rows, cols);

    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

extern "C" int niyah_cuda_train_linear_backward(
    const NiyahCudaModelState *ms,
    NiyahCudaTrainState *ts,
    size_t weight_offset,
    size_t x_offset,
    size_t dy_offset,
    size_t dx_offset,
    size_t tokens,
    size_t rows,
    size_t cols)
{
    size_t matrix, x_count, dy_count;
    const unsigned int threads = 256U;

    if (!shape_ok(ms, ts, weight_offset, tokens, rows, cols,
                  &matrix, &x_count, &dy_count) ||
        !range_ok(x_offset, x_count, ts->workspace_capacity) ||
        !range_ok(dy_offset, dy_count, ts->workspace_capacity) ||
        !range_ok(dx_offset, x_count, ts->workspace_capacity) ||
        overlap(x_offset, x_count, dy_offset, dy_count) ||
        overlap(x_offset, x_count, dx_offset, x_count) ||
        overlap(dy_offset, dy_count, dx_offset, x_count) ||
        matrix > (size_t)UINT_MAX * threads ||
        x_count > (size_t)UINT_MAX * threads)
        return 1;

    unsigned int blocks =
        (unsigned int)((matrix + threads - 1U) / threads);

    linear_dweight_kernel<<<blocks, threads>>>(
        (float *)ts->device_gradients + weight_offset,
        (const float *)ts->device_workspace + x_offset,
        (const float *)ts->device_workspace + dy_offset,
        tokens, rows, cols);

    if (cudaGetLastError() != cudaSuccess)
        return 1;

    blocks =
        (unsigned int)((x_count + threads - 1U) / threads);

    linear_dx_kernel<<<blocks, threads>>>(
        (float *)ts->device_workspace + dx_offset,
        (const float *)ms->device_weights + weight_offset,
        (const float *)ts->device_workspace + dy_offset,
        tokens, rows, cols);

    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
