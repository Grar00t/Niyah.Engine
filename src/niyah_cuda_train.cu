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


__global__ static void niyah_cuda_gradient_accumulate_kernel(
    float *accumulated,
    const float *gradient,
    float scale,
    size_t count)
{
    size_t i =
        (size_t)blockIdx.x * (size_t)blockDim.x +
        (size_t)threadIdx.x;

    const size_t stride =
        (size_t)blockDim.x * (size_t)gridDim.x;

    for (; i < count; i += stride) {
        accumulated[i] += gradient[i] * scale;
    }
}


extern "C" int niyah_cuda_train_state_zero_accumulated_gradients(
    NiyahCudaTrainState *state)
{
    if (state == NULL ||
        state->device_accumulated_gradients == NULL ||
        state->gradient_capacity == 0U ||
        state->gradient_capacity > ((size_t)-1) / sizeof(float)) {
        return 1;
    }

    return cudaMemset(
               state->device_accumulated_gradients,
               0,
               state->gradient_capacity * sizeof(float)) == cudaSuccess
        ? 0
        : 1;
}


extern "C" int niyah_cuda_train_state_accumulate_gradients(
    NiyahCudaTrainState *state,
    float scale)
{
    const unsigned int threads = 256U;
    unsigned int blocks;

    if (state == NULL ||
        state->device_gradients == NULL ||
        state->device_accumulated_gradients == NULL ||
        state->gradient_capacity == 0U) {
        return 1;
    }

    blocks = (unsigned int)(
        (state->gradient_capacity + (size_t)threads - 1U) /
        (size_t)threads);

    if (blocks == 0U) {
        return 1;
    }

    if (blocks > 65535U) {
        blocks = 65535U;
    }

    niyah_cuda_gradient_accumulate_kernel<<<blocks, threads>>>(
        (float *)state->device_accumulated_gradients,
        (const float *)state->device_gradients,
        scale,
        state->gradient_capacity);

    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}



__global__ static void niyah_cuda_gradient_scale_kernel(
    float *values,
    float scale,
    size_t count)
{
    size_t i =
        (size_t)blockIdx.x * (size_t)blockDim.x +
        (size_t)threadIdx.x;

    const size_t stride =
        (size_t)blockDim.x * (size_t)gridDim.x;

    for (; i < count; i += stride) {
        values[i] *= scale;
    }
}


extern "C" int niyah_cuda_train_state_scale_accumulated_gradients(
    NiyahCudaTrainState *state,
    float scale)
{
    const unsigned int threads = 256U;
    unsigned int blocks;

    if (state == NULL ||
        state->device_accumulated_gradients == NULL ||
        state->gradient_capacity == 0U) {
        return 1;
    }

    blocks = (unsigned int)(
        (state->gradient_capacity + (size_t)threads - 1U) /
        (size_t)threads);

    if (blocks == 0U) {
        return 1;
    }

    if (blocks > 65535U) {
        blocks = 65535U;
    }

    niyah_cuda_gradient_scale_kernel<<<blocks, threads>>>(
        (float *)state->device_accumulated_gradients,
        scale,
        state->gradient_capacity);

    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}


extern "C" int
niyah_cuda_train_state_copy_accumulated_gradients_to_host(
    const NiyahCudaTrainState *state,
    float *host_gradients,
    size_t gradient_count)
{
    if (state == NULL ||
        state->device_accumulated_gradients == NULL ||
        host_gradients == NULL ||
        gradient_count == 0U ||
        gradient_count > state->gradient_capacity ||
        gradient_count > ((size_t)-1) / sizeof(float)) {
        return 1;
    }

    return cudaMemcpy(
               host_gradients,
               state->device_accumulated_gradients,
               gradient_count * sizeof(float),
               cudaMemcpyDeviceToHost) == cudaSuccess
        ? 0
        : 1;
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

__global__ static void rmsnorm_dweight_kernel(
    float *dw,
    const float *x,
    const float *dy,
    size_t tokens,
    size_t width,
    float eps)
{
    const size_t c =
        (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (c < width) {
        float value = dw[c];

        for (size_t t = 0U; t < tokens; ++t) {
            double sum_sq = 0.0;

            for (size_t i = 0U; i < width; ++i) {
                const double xv = (double)x[t * width + i];
                sum_sq += xv * xv;
            }

            const float inv =
                1.0f / sqrtf((float)(sum_sq / (double)width) + eps);

            value += dy[t * width + c] *
                     x[t * width + c] *
                     inv;
        }

        dw[c] = value;
    }
}

__global__ static void rmsnorm_dx_kernel(
    float *dx,
    const float *x,
    const float *dy,
    const float *weight,
    size_t tokens,
    size_t width,
    float eps)
{
    const size_t t =
        (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (t < tokens) {
        double sum_sq = 0.0;
        double dot = 0.0;

        for (size_t i = 0U; i < width; ++i) {
            const double xv = (double)x[t * width + i];
            sum_sq += xv * xv;
        }

        const float inv =
            1.0f / sqrtf((float)(sum_sq / (double)width) + eps);

        for (size_t i = 0U; i < width; ++i) {
            dot += (double)dy[t * width + i] *
                   (double)weight[i] *
                   (double)x[t * width + i];
        }

        const float coeff =
            inv * inv * inv * (float)(dot / (double)width);

        for (size_t i = 0U; i < width; ++i) {
            dx[t * width + i] =
                dy[t * width + i] * weight[i] * inv -
                x[t * width + i] * coeff;
        }
    }
}

__global__ static void attention_dq_kernel(
    float *dq,
    const float *da,
    const float *q,
    const float *k,
    const float *v,
    size_t tokens,
    size_t n_heads,
    size_t n_kv_heads,
    size_t head_dim,
    size_t dim,
    size_t kv_dim)
{
    const size_t item =
        (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t item_count = tokens * n_heads;

    if (item < item_count) {
        const size_t t = item / n_heads;
        const size_t h = item % n_heads;
        const size_t group_size = n_heads / n_kv_heads;
        const size_t kh = h / group_size;
        const float scale = 1.0f / sqrtf((float)head_dim);
        const float *qh =
            q + t * dim + h * head_dim;
        const float *dah =
            da + t * dim + h * head_dim;

        float max_score = -3.402823466e+38F;

        for (size_t src = 0U; src <= t; ++src) {
            const float *khv =
                k + src * kv_dim + kh * head_dim;
            float score = 0.0f;

            for (size_t d = 0U; d < head_dim; ++d)
                score += qh[d] * khv[d];

            score *= scale;
            if (score > max_score)
                max_score = score;
        }

        float sum_exp = 0.0f;

        for (size_t src = 0U; src <= t; ++src) {
            const float *khv =
                k + src * kv_dim + kh * head_dim;
            float score = 0.0f;

            for (size_t d = 0U; d < head_dim; ++d)
                score += qh[d] * khv[d];

            sum_exp +=
                expf(score * scale - max_score);
        }

        if (!(sum_exp > 0.0f) || !isfinite(sum_exp))
            return;

        float mean_dp = 0.0f;

        for (size_t src = 0U; src <= t; ++src) {
            const float *khv =
                k + src * kv_dim + kh * head_dim;
            const float *vh =
                v + src * kv_dim + kh * head_dim;
            float score = 0.0f;
            float dprob = 0.0f;

            for (size_t d = 0U; d < head_dim; ++d) {
                score += qh[d] * khv[d];
                dprob += dah[d] * vh[d];
            }

            const float probability =
                expf(score * scale - max_score) /
                sum_exp;

            mean_dp += probability * dprob;
        }

        for (size_t d = 0U; d < head_dim; ++d) {
            float value = 0.0f;

            for (size_t src = 0U; src <= t; ++src) {
                const float *khv =
                    k + src * kv_dim + kh * head_dim;
                const float *vh =
                    v + src * kv_dim + kh * head_dim;
                float score = 0.0f;
                float dprob = 0.0f;

                for (size_t j = 0U; j < head_dim; ++j) {
                    score += qh[j] * khv[j];
                    dprob += dah[j] * vh[j];
                }

                const float probability =
                    expf(score * scale - max_score) /
                    sum_exp;

                const float ds =
                    probability * (dprob - mean_dp);

                value += ds * scale * khv[d];
            }

            dq[t * dim + h * head_dim + d] = value;
        }
    }
}

__global__ static void attention_dkdv_kernel(
    float *dk,
    float *dv,
    const float *da,
    const float *q,
    const float *k,
    const float *v,
    size_t tokens,
    size_t n_heads,
    size_t n_kv_heads,
    size_t head_dim,
    size_t dim,
    size_t kv_dim)
{
    const size_t item =
        (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t item_count =
        tokens * n_kv_heads * head_dim;

    if (item < item_count) {
        const size_t per_token =
            n_kv_heads * head_dim;
        const size_t src =
            item / per_token;
        const size_t rem =
            item % per_token;
        const size_t kh =
            rem / head_dim;
        const size_t d =
            rem % head_dim;

        const size_t group_size =
            n_heads / n_kv_heads;
        const size_t first_head =
            kh * group_size;
        const size_t last_head =
            first_head + group_size;
        const float scale =
            1.0f / sqrtf((float)head_dim);

        float dk_value = 0.0f;
        float dv_value = 0.0f;

        for (size_t t = src; t < tokens; ++t) {
            for (size_t h = first_head;
                 h < last_head;
                 ++h) {
                const float *qh =
                    q + t * dim + h * head_dim;
                const float *dah =
                    da + t * dim + h * head_dim;

                float max_score =
                    -3.402823466e+38F;

                for (size_t s = 0U; s <= t; ++s) {
                    const float *khv =
                        k + s * kv_dim +
                        kh * head_dim;
                    float score = 0.0f;

                    for (size_t j = 0U;
                         j < head_dim;
                         ++j) {
                        score += qh[j] * khv[j];
                    }

                    score *= scale;
                    if (score > max_score)
                        max_score = score;
                }

                float sum_exp = 0.0f;

                for (size_t s = 0U; s <= t; ++s) {
                    const float *khv =
                        k + s * kv_dim +
                        kh * head_dim;
                    float score = 0.0f;

                    for (size_t j = 0U;
                         j < head_dim;
                         ++j) {
                        score += qh[j] * khv[j];
                    }

                    sum_exp +=
                        expf(
                            score * scale -
                            max_score);
                }

                if (!(sum_exp > 0.0f) ||
                    !isfinite(sum_exp)) {
                    continue;
                }

                float mean_dp = 0.0f;

                for (size_t s = 0U; s <= t; ++s) {
                    const float *khv =
                        k + s * kv_dim +
                        kh * head_dim;
                    const float *vh =
                        v + s * kv_dim +
                        kh * head_dim;
                    float score = 0.0f;
                    float dprob = 0.0f;

                    for (size_t j = 0U;
                         j < head_dim;
                         ++j) {
                        score += qh[j] * khv[j];
                        dprob += dah[j] * vh[j];
                    }

                    const float probability =
                        expf(
                            score * scale -
                            max_score) /
                        sum_exp;

                    mean_dp +=
                        probability * dprob;
                }

                const float *src_k =
                    k + src * kv_dim +
                    kh * head_dim;
                const float *src_v =
                    v + src * kv_dim +
                    kh * head_dim;

                float src_score = 0.0f;
                float src_dprob = 0.0f;

                for (size_t j = 0U;
                     j < head_dim;
                     ++j) {
                    src_score +=
                        qh[j] * src_k[j];
                    src_dprob +=
                        dah[j] * src_v[j];
                }

                const float probability =
                    expf(
                        src_score * scale -
                        max_score) /
                    sum_exp;

                const float ds =
                    probability *
                    (src_dprob - mean_dp);

                dk_value +=
                    ds * scale * qh[d];

                dv_value +=
                    probability * dah[d];
            }
        }

        dk[src * kv_dim + kh * head_dim + d] =
            dk_value;
        dv[src * kv_dim + kh * head_dim + d] =
            dv_value;
    }
}

extern "C" int niyah_cuda_train_attention_backward(
    NiyahCudaTrainState *ts,
    size_t q_offset,
    size_t k_offset,
    size_t v_offset,
    size_t da_offset,
    size_t dq_offset,
    size_t dk_offset,
    size_t dv_offset,
    size_t token_count,
    size_t n_heads,
    size_t n_kv_heads,
    size_t head_dim)
{
    const unsigned int threads = 128U;
    size_t dim;
    size_t kv_dim;
    size_t q_count;
    size_t kv_count;
    size_t dq_items;
    size_t dkdv_items;

    if (ts == NULL ||
        ts->device_workspace == NULL ||
        token_count == 0U ||
        n_heads == 0U ||
        n_kv_heads == 0U ||
        head_dim == 0U ||
        (n_heads % n_kv_heads) != 0U ||
        n_heads > ((size_t)-1) / head_dim ||
        n_kv_heads > ((size_t)-1) / head_dim) {
        return 1;
    }

    dim = n_heads * head_dim;
    kv_dim = n_kv_heads * head_dim;

    if (token_count > ((size_t)-1) / dim ||
        token_count > ((size_t)-1) / kv_dim ||
        token_count > ((size_t)-1) / n_heads) {
        return 1;
    }

    q_count = token_count * dim;
    kv_count = token_count * kv_dim;
    dq_items = token_count * n_heads;

    if (kv_count > ((size_t)-1) / head_dim)
        return 1;

    dkdv_items =
        token_count * n_kv_heads * head_dim;

    if (!range_ok(
            q_offset, q_count,
            ts->workspace_capacity) ||
        !range_ok(
            k_offset, kv_count,
            ts->workspace_capacity) ||
        !range_ok(
            v_offset, kv_count,
            ts->workspace_capacity) ||
        !range_ok(
            da_offset, q_count,
            ts->workspace_capacity) ||
        !range_ok(
            dq_offset, q_count,
            ts->workspace_capacity) ||
        !range_ok(
            dk_offset, kv_count,
            ts->workspace_capacity) ||
        !range_ok(
            dv_offset, kv_count,
            ts->workspace_capacity)) {
        return 1;
    }

    if (overlap(q_offset, q_count, dq_offset, q_count) ||
        overlap(q_offset, q_count, dk_offset, kv_count) ||
        overlap(q_offset, q_count, dv_offset, kv_count) ||
        overlap(k_offset, kv_count, dq_offset, q_count) ||
        overlap(k_offset, kv_count, dk_offset, kv_count) ||
        overlap(k_offset, kv_count, dv_offset, kv_count) ||
        overlap(v_offset, kv_count, dq_offset, q_count) ||
        overlap(v_offset, kv_count, dk_offset, kv_count) ||
        overlap(v_offset, kv_count, dv_offset, kv_count) ||
        overlap(da_offset, q_count, dq_offset, q_count) ||
        overlap(da_offset, q_count, dk_offset, kv_count) ||
        overlap(da_offset, q_count, dv_offset, kv_count) ||
        overlap(dq_offset, q_count, dk_offset, kv_count) ||
        overlap(dq_offset, q_count, dv_offset, kv_count) ||
        overlap(dk_offset, kv_count, dv_offset, kv_count)) {
        return 1;
    }

    if (dq_items >
            (size_t)UINT_MAX * threads ||
        dkdv_items >
            (size_t)UINT_MAX * threads) {
        return 1;
    }

    float *workspace =
        (float *)ts->device_workspace;

    if (cudaMemset(
            workspace + dq_offset,
            0,
            q_count * sizeof(float)) != cudaSuccess ||
        cudaMemset(
            workspace + dk_offset,
            0,
            kv_count * sizeof(float)) != cudaSuccess ||
        cudaMemset(
            workspace + dv_offset,
            0,
            kv_count * sizeof(float)) != cudaSuccess) {
        return 1;
    }

    unsigned int blocks =
        (unsigned int)(
            (dq_items + threads - 1U) /
            threads);

    attention_dq_kernel<<<blocks, threads>>>(
        workspace + dq_offset,
        workspace + da_offset,
        workspace + q_offset,
        workspace + k_offset,
        workspace + v_offset,
        token_count,
        n_heads,
        n_kv_heads,
        head_dim,
        dim,
        kv_dim);

    if (cudaGetLastError() != cudaSuccess)
        return 1;

    blocks =
        (unsigned int)(
            (dkdv_items + threads - 1U) /
            threads);

    attention_dkdv_kernel<<<blocks, threads>>>(
        workspace + dk_offset,
        workspace + dv_offset,
        workspace + da_offset,
        workspace + q_offset,
        workspace + k_offset,
        workspace + v_offset,
        token_count,
        n_heads,
        n_kv_heads,
        head_dim,
        dim,
        kv_dim);

    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

__global__ static void silu_mul_backward_kernel(
    float *dgate,
    float *dup,
    const float *gate,
    const float *up,
    const float *dact,
    size_t n)
{
    const size_t i =
        (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        const float x = gate[i];
        const float s =
            1.0f / (1.0f + expf(-x));
        const float silu = x * s;
        const float dsilu =
            s + x * s * (1.0f - s);
        const float da = dact[i];

        dgate[i] = da * up[i] * dsilu;
        dup[i] = da * silu;
    }
}

extern "C" int niyah_cuda_train_silu_mul_backward(
    NiyahCudaTrainState *ts,
    size_t gate_offset,
    size_t up_offset,
    size_t dact_offset,
    size_t dgate_offset,
    size_t dup_offset,
    size_t value_count)
{
    const unsigned int threads = 256U;

    if (ts == NULL ||
        ts->device_workspace == NULL ||
        value_count == 0U ||
        !range_ok(
            gate_offset, value_count,
            ts->workspace_capacity) ||
        !range_ok(
            up_offset, value_count,
            ts->workspace_capacity) ||
        !range_ok(
            dact_offset, value_count,
            ts->workspace_capacity) ||
        !range_ok(
            dgate_offset, value_count,
            ts->workspace_capacity) ||
        !range_ok(
            dup_offset, value_count,
            ts->workspace_capacity) ||
        overlap(
            gate_offset, value_count,
            up_offset, value_count) ||
        overlap(
            gate_offset, value_count,
            dact_offset, value_count) ||
        overlap(
            gate_offset, value_count,
            dgate_offset, value_count) ||
        overlap(
            gate_offset, value_count,
            dup_offset, value_count) ||
        overlap(
            up_offset, value_count,
            dact_offset, value_count) ||
        overlap(
            up_offset, value_count,
            dgate_offset, value_count) ||
        overlap(
            up_offset, value_count,
            dup_offset, value_count) ||
        overlap(
            dact_offset, value_count,
            dgate_offset, value_count) ||
        overlap(
            dact_offset, value_count,
            dup_offset, value_count) ||
        overlap(
            dgate_offset, value_count,
            dup_offset, value_count) ||
        value_count > (size_t)UINT_MAX * threads) {
        return 1;
    }

    const unsigned int blocks =
        (unsigned int)(
            (value_count + threads - 1U) /
            threads);

    silu_mul_backward_kernel<<<blocks, threads>>>(
        (float *)ts->device_workspace +
            dgate_offset,
        (float *)ts->device_workspace +
            dup_offset,
        (const float *)ts->device_workspace +
            gate_offset,
        (const float *)ts->device_workspace +
            up_offset,
        (const float *)ts->device_workspace +
            dact_offset,
        value_count);

    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

extern "C" int niyah_cuda_train_rmsnorm_backward(
    const NiyahCudaModelState *ms,
    NiyahCudaTrainState *ts,
    size_t weight_offset,
    size_t x_offset,
    size_t dy_offset,
    size_t dx_offset,
    size_t tokens,
    size_t width,
    float eps)
{
    const unsigned int threads = 256U;
    size_t value_count;

    if (ms == NULL || ts == NULL ||
        ms->device_weights == NULL ||
        ts->device_gradients == NULL ||
        ts->device_workspace == NULL ||
        tokens == 0U || width == 0U ||
        !(eps > 0.0f) ||
        tokens > ((size_t)-1) / width)
        return 1;

    value_count = tokens * width;

    if (weight_offset > ms->weight_count ||
        width > ms->weight_count - weight_offset ||
        weight_offset > ts->gradient_capacity ||
        width > ts->gradient_capacity - weight_offset ||
        !range_ok(x_offset, value_count, ts->workspace_capacity) ||
        !range_ok(dy_offset, value_count, ts->workspace_capacity) ||
        !range_ok(dx_offset, value_count, ts->workspace_capacity) ||
        overlap(x_offset, value_count, dy_offset, value_count) ||
        overlap(x_offset, value_count, dx_offset, value_count) ||
        overlap(dy_offset, value_count, dx_offset, value_count) ||
        width > (size_t)UINT_MAX * threads ||
        tokens > (size_t)UINT_MAX * threads)
        return 1;

    unsigned int blocks =
        (unsigned int)((width + threads - 1U) / threads);

    rmsnorm_dweight_kernel<<<blocks, threads>>>(
        (float *)ts->device_gradients + weight_offset,
        (const float *)ts->device_workspace + x_offset,
        (const float *)ts->device_workspace + dy_offset,
        tokens, width, eps);

    if (cudaGetLastError() != cudaSuccess)
        return 1;

    blocks =
        (unsigned int)((tokens + threads - 1U) / threads);

    rmsnorm_dx_kernel<<<blocks, threads>>>(
        (float *)ts->device_workspace + dx_offset,
        (const float *)ts->device_workspace + x_offset,
        (const float *)ts->device_workspace + dy_offset,
        (const float *)ms->device_weights + weight_offset,
        tokens, width, eps);

    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
