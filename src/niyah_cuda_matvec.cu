#include "niyah_cuda_matvec.h"

#include <cuda_runtime.h>

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

__global__ static void niyah_cuda_matvec_kernel(float *out,
                                                const float *matrix,
                                                const float *x,
                                                size_t rows,
                                                size_t cols)
{
    __shared__ float partial[128];
    const size_t row = (size_t)blockIdx.x;
    const unsigned int lane = threadIdx.x;
    float sum = 0.0f;
    size_t col;
    unsigned int stride;

    if (row >= rows) {
        return;
    }

    for (col = (size_t)lane;
         col < cols;
         col += (size_t)blockDim.x) {
        sum += matrix[row * cols + col] * x[col];
    }

    partial[lane] = sum;
    __syncthreads();

    for (stride = blockDim.x / 2U;
         stride > 0U;
         stride >>= 1U) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        __syncthreads();
    }

    if (lane == 0U) {
        out[row] = partial[0];
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
        rows == 0U || cols == 0U ||
        rows > (size_t)UINT_MAX) {
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
        const unsigned int blocks = (unsigned int)rows;

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

static size_t niyah_cuda_max_size(size_t a, size_t b)
{
    return a > b ? a : b;
}

static int niyah_cuda_config_equal(const NiyahModelConfig *a,
                                   const NiyahModelConfig *b)
{
    return a->vocab_size == b->vocab_size &&
           a->context_length == b->context_length &&
           a->embedding_dim == b->embedding_dim &&
           a->n_layers == b->n_layers &&
           a->n_heads == b->n_heads &&
           a->n_kv_heads == b->n_kv_heads &&
           a->ffn_hidden_dim == b->ffn_hidden_dim &&
           a->rms_norm_eps == b->rms_norm_eps &&
           (a->tie_word_embeddings != 0) ==
               (b->tie_word_embeddings != 0) &&
           a->n_segments == b->n_segments;
}

static int niyah_cuda_layout_equal(const NiyahModelLayout *a,
                                   const NiyahModelLayout *b)
{
    return a->token_embedding == b->token_embedding &&
           a->layers == b->layers &&
           a->layer_stride == b->layer_stride &&
           a->final_norm == b->final_norm &&
           a->lm_head == b->lm_head &&
           a->total_floats == b->total_floats &&
           a->head_dim == b->head_dim &&
           a->kv_dim == b->kv_dim;
}

static int niyah_cuda_model_state_matches(
    const NiyahCudaModelState *state,
    const NiyahModel *model)
{
    return state != NULL &&
           model != NULL &&
           model->weights != NULL &&
           state->device_weights != NULL &&
           state->device_input != NULL &&
           state->device_output != NULL &&
           state->weight_count == model->weight_count &&
           niyah_cuda_config_equal(&state->config, &model->config) &&
           niyah_cuda_layout_equal(&state->layout, &model->layout);
}

extern "C" void niyah_cuda_model_state_destroy(NiyahCudaModelState *state)
{
    if (state == NULL) {
        return;
    }

    if (state->device_output != NULL) {
        (void)cudaFree(state->device_output);
    }
    if (state->device_input != NULL) {
        (void)cudaFree(state->device_input);
    }
    if (state->device_weights != NULL) {
        (void)cudaFree(state->device_weights);
    }

    memset(state, 0, sizeof(*state));
}

extern "C" int niyah_cuda_model_state_create(
    NiyahCudaModelState *state,
    const NiyahModel *model)
{
    size_t weight_bytes;
    size_t input_bytes;
    size_t output_bytes;
    size_t input_capacity;
    size_t output_capacity;
    const size_t dim =
        model != NULL ? (size_t)model->config.embedding_dim : 0U;
    const size_t ffn =
        model != NULL ? (size_t)model->config.ffn_hidden_dim : 0U;
    const size_t vocab =
        model != NULL ? (size_t)model->config.vocab_size : 0U;
    cudaError_t error;

    if (state == NULL || model == NULL ||
        model->weights == NULL || model->weight_count == 0U) {
        return 1;
    }

    memset(state, 0, sizeof(*state));

    if (model->weight_count > ((size_t)-1) / sizeof(float)) {
        return 1;
    }

    input_capacity = niyah_cuda_max_size(dim, ffn);
    output_capacity = niyah_cuda_max_size(vocab, dim);
    output_capacity =
        niyah_cuda_max_size(output_capacity, model->layout.kv_dim);
    output_capacity =
        niyah_cuda_max_size(output_capacity, ffn);

    if (input_capacity == 0U || output_capacity == 0U ||
        input_capacity > ((size_t)-1) / sizeof(float) ||
        output_capacity > ((size_t)-1) / sizeof(float)) {
        return 1;
    }

    weight_bytes = model->weight_count * sizeof(float);
    input_bytes = input_capacity * sizeof(float);
    output_bytes = output_capacity * sizeof(float);

    error = cudaMalloc(&state->device_weights, weight_bytes);
    if (error != cudaSuccess) {
        goto fail;
    }

    error = cudaMalloc(&state->device_input, input_bytes);
    if (error != cudaSuccess) {
        goto fail;
    }

    error = cudaMalloc(&state->device_output, output_bytes);
    if (error != cudaSuccess) {
        goto fail;
    }

    error = cudaMemcpy(state->device_weights,
                       model->weights,
                       weight_bytes,
                       cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        goto fail;
    }

    state->weight_count = model->weight_count;
    state->input_capacity = input_capacity;
    state->output_capacity = output_capacity;
    state->config = model->config;
    state->layout = model->layout;
    return 0;

fail:
    niyah_cuda_model_state_destroy(state);
    return 1;
}

extern "C" int niyah_cuda_model_state_sync(
    NiyahCudaModelState *state,
    const NiyahModel *model)
{
    size_t bytes;

    if (!niyah_cuda_model_state_matches(state, model) ||
        state->weight_count > ((size_t)-1) / sizeof(float)) {
        return 1;
    }

    bytes = state->weight_count * sizeof(float);
    return cudaMemcpy(state->device_weights,
                      model->weights,
                      bytes,
                      cudaMemcpyHostToDevice) == cudaSuccess
               ? 0
               : 1;
}

extern "C" int niyah_cuda_model_state_matvec(
    NiyahCudaModelState *state,
    size_t weight_offset,
    const float *x,
    float *out,
    size_t rows,
    size_t cols)
{
    size_t matrix_count;
    size_t input_bytes;
    size_t output_bytes;
    unsigned int blocks;
    const unsigned int threads = 128U;
    cudaError_t error;

    if (state == NULL ||
        state->device_weights == NULL ||
        state->device_input == NULL ||
        state->device_output == NULL ||
        x == NULL || out == NULL ||
        rows == 0U || cols == 0U ||
        cols > state->input_capacity ||
        rows > state->output_capacity ||
        rows > ((size_t)-1) / cols ||
        rows > (size_t)UINT_MAX) {
        return 1;
    }

    matrix_count = rows * cols;

    if (weight_offset > state->weight_count ||
        matrix_count > state->weight_count - weight_offset ||
        cols > ((size_t)-1) / sizeof(float) ||
        rows > ((size_t)-1) / sizeof(float)) {
        return 1;
    }

    input_bytes = cols * sizeof(float);
    output_bytes = rows * sizeof(float);

    error = cudaMemcpy(state->device_input,
                       x,
                       input_bytes,
                       cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        return 1;
    }

    blocks = (unsigned int)rows;

    niyah_cuda_matvec_kernel<<<blocks, threads>>>(
        (float *)state->device_output,
        (const float *)state->device_weights + weight_offset,
        (const float *)state->device_input,
        rows,
        cols);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        return 1;
    }

    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        return 1;
    }

    error = cudaMemcpy(out,
                       state->device_output,
                       output_bytes,
                       cudaMemcpyDeviceToHost);
    return error == cudaSuccess ? 0 : 1;
}

static int niyah_cuda_size_add_ok(size_t a, size_t b, size_t *out)
{
    if (out == NULL || a > ((size_t)-1) - b) {
        return 0;
    }

    *out = a + b;
    return 1;
}

static int niyah_cuda_size_mul_ok(size_t a, size_t b, size_t *out)
{
    if (out == NULL || (a != 0U && b > ((size_t)-1) / a)) {
        return 0;
    }

    *out = a * b;
    return 1;
}

extern "C" void niyah_cuda_decode_state_destroy(
    NiyahCudaDecodeState *state)
{
    if (state == NULL) {
        return;
    }

    if (state->device_logits != NULL) {
        (void)cudaFree(state->device_logits);
    }
    if (state->device_workspace != NULL) {
        (void)cudaFree(state->device_workspace);
    }
    if (state->device_values != NULL) {
        (void)cudaFree(state->device_values);
    }
    if (state->device_keys != NULL) {
        (void)cudaFree(state->device_keys);
    }

    memset(state, 0, sizeof(*state));
}

extern "C" int niyah_cuda_decode_state_reset(
    NiyahCudaDecodeState *state)
{
    size_t kv_bytes;
    size_t workspace_bytes;
    size_t logits_bytes;

    if (state == NULL ||
        state->device_keys == NULL ||
        state->device_values == NULL ||
        state->device_workspace == NULL ||
        state->device_logits == NULL ||
        state->values_per_tensor == 0U ||
        state->workspace_floats == 0U ||
        state->logits_capacity == 0U ||
        !niyah_cuda_size_mul_ok(
            state->values_per_tensor, sizeof(float), &kv_bytes) ||
        !niyah_cuda_size_mul_ok(
            state->workspace_floats, sizeof(float), &workspace_bytes) ||
        !niyah_cuda_size_mul_ok(
            state->logits_capacity, sizeof(float), &logits_bytes)) {
        return 1;
    }

    if (cudaMemset(state->device_keys, 0, kv_bytes) != cudaSuccess ||
        cudaMemset(state->device_values, 0, kv_bytes) != cudaSuccess ||
        cudaMemset(state->device_workspace, 0, workspace_bytes) != cudaSuccess ||
        cudaMemset(state->device_logits, 0, logits_bytes) != cudaSuccess) {
        return 1;
    }

    state->next_position = 0U;
    return 0;
}

extern "C" int niyah_cuda_decode_state_create(
    NiyahCudaDecodeState *state,
    const NiyahCudaModelState *model_state)
{
    const NiyahModelConfig *config;
    size_t dim;
    size_t ffn;
    size_t head_dim;
    size_t kv_dim;
    size_t per_layer;
    size_t values_per_tensor;
    size_t workspace_floats;
    size_t term;
    size_t kv_bytes;
    size_t workspace_bytes;
    size_t logits_bytes;

    if (state == NULL || model_state == NULL ||
        model_state->device_weights == NULL ||
        model_state->weight_count == 0U) {
        return 1;
    }

    memset(state, 0, sizeof(*state));
    config = &model_state->config;

    if (config->vocab_size < 2U ||
        config->context_length == 0U ||
        config->embedding_dim == 0U ||
        config->n_layers == 0U ||
        config->n_heads == 0U ||
        config->n_kv_heads == 0U ||
        config->ffn_hidden_dim == 0U ||
        (config->embedding_dim % config->n_heads) != 0U ||
        (config->n_heads % config->n_kv_heads) != 0U) {
        return 1;
    }

    dim = (size_t)config->embedding_dim;
    ffn = (size_t)config->ffn_hidden_dim;
    head_dim = dim / (size_t)config->n_heads;

    if (head_dim < 2U || (head_dim % 2U) != 0U ||
        !niyah_cuda_size_mul_ok(
            head_dim, (size_t)config->n_kv_heads, &kv_dim) ||
        kv_dim != model_state->layout.kv_dim ||
        model_state->layout.total_floats != model_state->weight_count) {
        return 1;
    }

    if (!niyah_cuda_size_mul_ok(
            (size_t)config->context_length, kv_dim, &per_layer) ||
        !niyah_cuda_size_mul_ok(
            (size_t)config->n_layers,
            per_layer,
            &values_per_tensor)) {
        return 1;
    }

    /*
     * CUDA attention keeps one context-length score slice per head:
     * 5*dim + 2*kv_dim + 2*ffn + n_heads*context_length.
     * The public CPU decode workspace contract remains unchanged.
     */
    if (!niyah_cuda_size_mul_ok(5U, dim, &workspace_floats) ||
        !niyah_cuda_size_mul_ok(2U, kv_dim, &term) ||
        !niyah_cuda_size_add_ok(
            workspace_floats, term, &workspace_floats) ||
        !niyah_cuda_size_mul_ok(2U, ffn, &term) ||
        !niyah_cuda_size_add_ok(
            workspace_floats, term, &workspace_floats) ||
        !niyah_cuda_size_mul_ok(
            (size_t)config->n_heads,
            (size_t)config->context_length,
            &term) ||
        !niyah_cuda_size_add_ok(
            workspace_floats,
            term,
            &workspace_floats)) {
        return 1;
    }

    if (!niyah_cuda_size_mul_ok(
            values_per_tensor, sizeof(float), &kv_bytes) ||
        !niyah_cuda_size_mul_ok(
            workspace_floats, sizeof(float), &workspace_bytes) ||
        !niyah_cuda_size_mul_ok(
            (size_t)config->vocab_size,
            sizeof(float),
            &logits_bytes)) {
        return 1;
    }

    if (cudaMalloc(&state->device_keys, kv_bytes) != cudaSuccess) {
        goto fail;
    }
    if (cudaMalloc(&state->device_values, kv_bytes) != cudaSuccess) {
        goto fail;
    }
    if (cudaMalloc(&state->device_workspace, workspace_bytes) != cudaSuccess) {
        goto fail;
    }
    if (cudaMalloc(&state->device_logits, logits_bytes) != cudaSuccess) {
        goto fail;
    }

    state->context_length = (size_t)config->context_length;
    state->head_dim = head_dim;
    state->kv_dim = kv_dim;
    state->values_per_tensor = values_per_tensor;
    state->workspace_floats = workspace_floats;
    state->logits_capacity = (size_t)config->vocab_size;
    state->config = *config;

    if (niyah_cuda_decode_state_reset(state) != 0) {
        goto fail;
    }

    return 0;

fail:
    niyah_cuda_decode_state_destroy(state);
    return 1;
}

static int niyah_cuda_model_state_matvec_device_launch(
    const NiyahCudaModelState *state,
    size_t weight_offset,
    const void *device_x,
    void *device_out,
    size_t rows,
    size_t cols)
{
    size_t matrix_count;
    unsigned int blocks;
    const unsigned int threads = 128U;

    if (state == NULL ||
        state->device_weights == NULL ||
        device_x == NULL ||
        device_out == NULL ||
        device_x == device_out ||
        rows == 0U ||
        cols == 0U ||
        !niyah_cuda_size_mul_ok(
            rows, cols, &matrix_count) ||
        weight_offset > state->weight_count ||
        matrix_count >
            state->weight_count - weight_offset ||
        rows > (size_t)UINT_MAX) {
        return 1;
    }

    blocks = (unsigned int)rows;

    niyah_cuda_matvec_kernel<<<blocks, threads>>>(
        (float *)device_out,
        (const float *)state->device_weights +
            weight_offset,
        (const float *)device_x,
        rows,
        cols);

    return cudaGetLastError() == cudaSuccess
        ? 0
        : 1;
}

extern "C" int niyah_cuda_model_state_matvec_device(
    const NiyahCudaModelState *state,
    size_t weight_offset,
    const void *device_x,
    void *device_out,
    size_t rows,
    size_t cols)
{
    if (niyah_cuda_model_state_matvec_device_launch(
            state,
            weight_offset,
            device_x,
            device_out,
            rows,
            cols) != 0) {
        return 1;
    }

    /*
     * Preserve the P7-D public primitive contract: completion and
     * asynchronous execution errors are observed before return.
     */
    return cudaDeviceSynchronize() == cudaSuccess
        ? 0
        : 1;
}

__global__ static void niyah_cuda_rmsnorm_kernel(
    float *out,
    const float *x,
    const float *weight,
    size_t n,
    float eps)
{
    __shared__ float shared_inv_rms;

    if (threadIdx.x == 0U) {
        double sum_sq = 0.0;
        size_t i;

        for (i = 0U; i < n; ++i) {
            const double v = (double)x[i];
            sum_sq += v * v;
        }

        shared_inv_rms =
            1.0f / sqrtf((float)(sum_sq / (double)n) + eps);
    }

    __syncthreads();

    {
        size_t i;

        for (i = (size_t)threadIdx.x;
             i < n;
             i += (size_t)blockDim.x) {
            out[i] =
                x[i] * shared_inv_rms * weight[i];
        }
    }
}

__global__ static void niyah_cuda_rope_kernel(
    float *vector,
    size_t total_pairs,
    size_t head_dim,
    size_t position)
{
    const size_t pair =
        (size_t)blockIdx.x * (size_t)blockDim.x +
        (size_t)threadIdx.x;

    if (pair < total_pairs) {
        const size_t pairs_per_head = head_dim / 2U;
        const size_t head = pair / pairs_per_head;
        const size_t pair_in_head = pair % pairs_per_head;
        const size_t i = pair_in_head * 2U;
        float *head_vector = vector + head * head_dim;
        const float pos = (float)position;
        const float exponent =
            -((float)i / (float)head_dim);
        const float inv_freq =
            powf(10000.0f, exponent);
        const float angle = pos * inv_freq;
        const float c = cosf(angle);
        const float sn = sinf(angle);
        const float x0 = head_vector[i];
        const float x1 = head_vector[i + 1U];

        head_vector[i] = x0 * c - x1 * sn;
        head_vector[i + 1U] = x0 * sn + x1 * c;
    }
}

__global__ static void niyah_cuda_attention_one_kernel(
    float *out,
    const float *q,
    const float *keys,
    const float *values,
    float *scores,
    size_t score_stride,
    size_t layer_base,
    size_t position,
    size_t n_heads,
    size_t n_kv_heads,
    size_t head_dim,
    size_t kv_dim)
{
    const size_t head = (size_t)blockIdx.x;

    if (head < n_heads && threadIdx.x == 0U) {
        const size_t group_size = n_heads / n_kv_heads;
        const size_t kv_head = head / group_size;
        const float *q_head = q + head * head_dim;
        float *head_scores = scores + head * score_stride;
        const float scale = 1.0f / sqrtf((float)head_dim);
        float max_score = -FLT_MAX;
        float normalizer = 0.0f;
        size_t source;
        size_t d;

        for (source = 0U; source <= position; ++source) {
            const float *k_head =
                keys +
                layer_base +
                source * kv_dim +
                kv_head * head_dim;
            float dot = 0.0f;

            for (d = 0U; d < head_dim; ++d) {
                dot += q_head[d] * k_head[d];
            }

            head_scores[source] = dot * scale;
            if (head_scores[source] > max_score) {
                max_score = head_scores[source];
            }
        }

        for (source = 0U; source <= position; ++source) {
            head_scores[source] =
                expf(head_scores[source] - max_score);
            normalizer += head_scores[source];
        }

        for (d = 0U; d < head_dim; ++d) {
            float value = 0.0f;

            for (source = 0U;
                 source <= position;
                 ++source) {
                const float *v_head =
                    values +
                    layer_base +
                    source * kv_dim +
                    kv_head * head_dim;

                value +=
                    (head_scores[source] / normalizer) *
                    v_head[d];
            }

            out[head * head_dim + d] = value;
        }
    }
}

__global__ static void niyah_cuda_add_kernel(
    float *dst,
    const float *src,
    size_t n)
{
    const size_t i =
        (size_t)blockIdx.x * (size_t)blockDim.x +
        (size_t)threadIdx.x;

    if (i < n) {
        dst[i] += src[i];
    }
}

__global__ static void niyah_cuda_silu_mul_kernel(
    float *gate,
    const float *up,
    size_t n)
{
    const size_t i =
        (size_t)blockIdx.x * (size_t)blockDim.x +
        (size_t)threadIdx.x;

    if (i < n) {
        const float x = gate[i];
        gate[i] =
            (x / (1.0f + expf(-x))) * up[i];
    }
}

static int niyah_cuda_check_launch(void)
{
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

static int niyah_cuda_blocks_for(
    size_t n,
    unsigned int threads,
    unsigned int *out_blocks)
{
    size_t blocks;

    if (n == 0U || threads == 0U || out_blocks == NULL ||
        n > ((size_t)-1) - ((size_t)threads - 1U)) {
        return 1;
    }

    blocks =
        (n + (size_t)threads - 1U) /
        (size_t)threads;

    if (blocks == 0U || blocks > (size_t)UINT_MAX) {
        return 1;
    }

    *out_blocks = (unsigned int)blocks;
    return 0;
}

static int niyah_cuda_decode_state_matches_model(
    const NiyahCudaDecodeState *decode_state,
    const NiyahCudaModelState *model_state)
{
    const NiyahModelConfig *config;
    size_t expected_values;
    size_t per_layer;
    size_t expected_workspace;
    size_t term;
    size_t dim;
    size_t ffn;

    if (decode_state == NULL ||
        model_state == NULL ||
        decode_state->device_keys == NULL ||
        decode_state->device_values == NULL ||
        decode_state->device_workspace == NULL ||
        decode_state->device_logits == NULL ||
        model_state->device_weights == NULL ||
        !niyah_cuda_config_equal(
            &decode_state->config,
            &model_state->config)) {
        return 0;
    }

    config = &model_state->config;
    dim = (size_t)config->embedding_dim;
    ffn = (size_t)config->ffn_hidden_dim;

    if (decode_state->context_length !=
            (size_t)config->context_length ||
        decode_state->head_dim !=
            model_state->layout.head_dim ||
        decode_state->kv_dim !=
            model_state->layout.kv_dim ||
        decode_state->logits_capacity !=
            (size_t)config->vocab_size ||
        model_state->layout.total_floats !=
            model_state->weight_count ||
        decode_state->next_position >
            decode_state->context_length) {
        return 0;
    }

    if (!niyah_cuda_size_mul_ok(
            decode_state->context_length,
            decode_state->kv_dim,
            &per_layer) ||
        !niyah_cuda_size_mul_ok(
            (size_t)config->n_layers,
            per_layer,
            &expected_values) ||
        expected_values !=
            decode_state->values_per_tensor) {
        return 0;
    }

    if (!niyah_cuda_size_mul_ok(
            5U, dim, &expected_workspace) ||
        !niyah_cuda_size_mul_ok(
            2U, decode_state->kv_dim, &term) ||
        !niyah_cuda_size_add_ok(
            expected_workspace,
            term,
            &expected_workspace) ||
        !niyah_cuda_size_mul_ok(
            2U, ffn, &term) ||
        !niyah_cuda_size_add_ok(
            expected_workspace,
            term,
            &expected_workspace) ||
        !niyah_cuda_size_mul_ok(
            (size_t)config->n_heads,
            decode_state->context_length,
            &term) ||
        !niyah_cuda_size_add_ok(
            expected_workspace,
            term,
            &expected_workspace)) {
        return 0;
    }

    return expected_workspace ==
        decode_state->workspace_floats;
}

static int niyah_cuda_layer_layout(
    const NiyahCudaModelState *state,
    uint32_t layer_index,
    NiyahLayerLayout *layer)
{
    const NiyahModelConfig *config;
    size_t base;
    size_t delta;
    size_t cursor;
    size_t count;
    size_t dim;
    size_t kv_dim;
    size_t ffn;
    size_t expected_end;

    if (state == NULL || layer == NULL ||
        layer_index >= state->config.n_layers) {
        return 1;
    }

    config = &state->config;
    dim = (size_t)config->embedding_dim;
    kv_dim = state->layout.kv_dim;
    ffn = (size_t)config->ffn_hidden_dim;

    if (!niyah_cuda_size_mul_ok(
            (size_t)layer_index,
            state->layout.layer_stride,
            &delta) ||
        !niyah_cuda_size_add_ok(
            state->layout.layers,
            delta,
            &base)) {
        return 1;
    }

    cursor = base;
    layer->attn_norm = cursor;

    if (!niyah_cuda_size_add_ok(cursor, dim, &cursor)) {
        return 1;
    }

    layer->wq = cursor;
    if (!niyah_cuda_size_mul_ok(dim, dim, &count) ||
        !niyah_cuda_size_add_ok(cursor, count, &cursor)) {
        return 1;
    }

    layer->wk = cursor;
    if (!niyah_cuda_size_mul_ok(kv_dim, dim, &count) ||
        !niyah_cuda_size_add_ok(cursor, count, &cursor)) {
        return 1;
    }

    layer->wv = cursor;
    if (!niyah_cuda_size_add_ok(cursor, count, &cursor)) {
        return 1;
    }

    layer->wo = cursor;
    if (!niyah_cuda_size_mul_ok(dim, dim, &count) ||
        !niyah_cuda_size_add_ok(cursor, count, &cursor)) {
        return 1;
    }

    layer->ffn_norm = cursor;
    if (!niyah_cuda_size_add_ok(cursor, dim, &cursor)) {
        return 1;
    }

    layer->w_gate = cursor;
    if (!niyah_cuda_size_mul_ok(ffn, dim, &count) ||
        !niyah_cuda_size_add_ok(cursor, count, &cursor)) {
        return 1;
    }

    layer->w_up = cursor;
    if (!niyah_cuda_size_add_ok(cursor, count, &cursor)) {
        return 1;
    }

    layer->w_down = cursor;
    if (!niyah_cuda_size_mul_ok(dim, ffn, &count) ||
        !niyah_cuda_size_add_ok(cursor, count, &cursor) ||
        !niyah_cuda_size_add_ok(
            base,
            state->layout.layer_stride,
            &expected_end)) {
        return 1;
    }

    return cursor == expected_end &&
           cursor <= state->weight_count
               ? 0
               : 1;
}

static int niyah_cuda_rmsnorm_device(
    const NiyahCudaModelState *model_state,
    float *out,
    const float *x,
    size_t weight_offset,
    size_t n)
{
    if (model_state == NULL ||
        out == NULL ||
        x == NULL ||
        n == 0U ||
        weight_offset > model_state->weight_count ||
        n > model_state->weight_count - weight_offset) {
        return 1;
    }

    {
        const unsigned int threads = 128U;

        niyah_cuda_rmsnorm_kernel<<<1U, threads>>>(
            out,
            x,
            (const float *)model_state->device_weights +
                weight_offset,
            n,
            model_state->config.rms_norm_eps);
    }

    return niyah_cuda_check_launch();
}

static int niyah_cuda_rope_device(
    float *vector,
    size_t n_heads,
    size_t head_dim,
    size_t position)
{
    const unsigned int threads = 128U;
    size_t pairs_per_head;
    size_t total_pairs;
    unsigned int blocks;

    if (vector == NULL ||
        n_heads == 0U ||
        head_dim < 2U ||
        (head_dim % 2U) != 0U) {
        return 1;
    }

    pairs_per_head = head_dim / 2U;

    if (!niyah_cuda_size_mul_ok(
            n_heads,
            pairs_per_head,
            &total_pairs) ||
        niyah_cuda_blocks_for(
            total_pairs,
            threads,
            &blocks) != 0) {
        return 1;
    }

    niyah_cuda_rope_kernel<<<blocks, threads>>>(
        vector,
        total_pairs,
        head_dim,
        position);

    return niyah_cuda_check_launch();
}

static int niyah_cuda_attention_one_device(
    float *out,
    const float *q,
    const NiyahCudaDecodeState *decode_state,
    uint32_t layer_index,
    size_t position,
    float *scores)
{
    const size_t n_heads =
        (size_t)decode_state->config.n_heads;
    const size_t n_kv_heads =
        (size_t)decode_state->config.n_kv_heads;
    size_t layer_base;
    size_t per_layer;
    unsigned int blocks;

    if (out == NULL ||
        q == NULL ||
        decode_state == NULL ||
        scores == NULL ||
        n_heads == 0U ||
        n_kv_heads == 0U ||
        (n_heads % n_kv_heads) != 0U ||
        layer_index >= decode_state->config.n_layers ||
        position >= decode_state->context_length ||
        !niyah_cuda_size_mul_ok(
            decode_state->context_length,
            decode_state->kv_dim,
            &per_layer) ||
        !niyah_cuda_size_mul_ok(
            (size_t)layer_index,
            per_layer,
            &layer_base) ||
        niyah_cuda_blocks_for(
            n_heads, 1U, &blocks) != 0) {
        return 1;
    }

    niyah_cuda_attention_one_kernel<<<blocks, 1U>>>(
        out,
        q,
        (const float *)decode_state->device_keys,
        (const float *)decode_state->device_values,
        scores,
        decode_state->context_length,
        layer_base,
        position,
        n_heads,
        n_kv_heads,
        decode_state->head_dim,
        decode_state->kv_dim);

    return niyah_cuda_check_launch();
}

static int niyah_cuda_add_device(
    float *dst,
    const float *src,
    size_t n)
{
    const unsigned int threads = 128U;
    unsigned int blocks;

    if (dst == NULL || src == NULL || dst == src ||
        niyah_cuda_blocks_for(
            n, threads, &blocks) != 0) {
        return 1;
    }

    niyah_cuda_add_kernel<<<blocks, threads>>>(
        dst, src, n);

    return niyah_cuda_check_launch();
}

static int niyah_cuda_silu_mul_device(
    float *gate,
    const float *up,
    size_t n)
{
    const unsigned int threads = 128U;
    unsigned int blocks;

    if (gate == NULL || up == NULL || gate == up ||
        niyah_cuda_blocks_for(
            n, threads, &blocks) != 0) {
        return 1;
    }

    niyah_cuda_silu_mul_kernel<<<blocks, threads>>>(
        gate, up, n);

    return niyah_cuda_check_launch();
}

extern "C" int niyah_cuda_decode_token(
    const NiyahCudaModelState *model_state,
    NiyahCudaDecodeState *decode_state,
    uint32_t token,
    float *logits,
    size_t logits_count)
{
    const NiyahModelConfig *config;
    float *workspace;
    float *hidden;
    float *norm;
    float *q;
    float *k;
    float *v;
    float *attn;
    float *proj;
    float *gate;
    float *up;
    float *scores;
    size_t dim;
    size_t ffn;
    size_t kv_dim;
    size_t vocab;
    size_t position;
    size_t embedding_delta;
    size_t embedding_offset;
    size_t embedding_bytes;
    size_t kv_bytes;
    size_t score_floats;
    uint32_t layer_index;

    if (!niyah_cuda_decode_state_matches_model(
            decode_state, model_state) ||
        logits == NULL) {
        return 1;
    }

    config = &model_state->config;
    dim = (size_t)config->embedding_dim;
    ffn = (size_t)config->ffn_hidden_dim;
    kv_dim = decode_state->kv_dim;
    vocab = (size_t)config->vocab_size;
    position = decode_state->next_position;

    if ((size_t)token >= vocab ||
        logits_count < vocab ||
        position >= decode_state->context_length ||
        !niyah_cuda_size_mul_ok(
            (size_t)token, dim, &embedding_delta) ||
        !niyah_cuda_size_add_ok(
            model_state->layout.token_embedding,
            embedding_delta,
            &embedding_offset) ||
        embedding_offset > model_state->weight_count ||
        dim > model_state->weight_count - embedding_offset ||
        !niyah_cuda_size_mul_ok(
            dim, sizeof(float), &embedding_bytes) ||
        !niyah_cuda_size_mul_ok(
            kv_dim, sizeof(float), &kv_bytes) ||
        !niyah_cuda_size_mul_ok(
            (size_t)config->n_heads,
            decode_state->context_length,
            &score_floats)) {
        return 1;
    }

    workspace =
        (float *)decode_state->device_workspace;

    hidden = workspace;
    norm = hidden + dim;
    q = norm + dim;
    k = q + dim;
    v = k + kv_dim;
    attn = v + kv_dim;
    proj = attn + dim;
    gate = proj + dim;
    up = gate + ffn;
    scores = up + ffn;

    if ((size_t)(scores - workspace) >
            decode_state->workspace_floats ||
        score_floats >
            decode_state->workspace_floats -
                (size_t)(scores - workspace)) {
        return 1;
    }

    if (cudaMemcpy(
            hidden,
            (const float *)model_state->device_weights +
                embedding_offset,
            embedding_bytes,
            cudaMemcpyDeviceToDevice) != cudaSuccess) {
        return 1;
    }

    for (layer_index = 0U;
         layer_index < config->n_layers;
         ++layer_index) {
        NiyahLayerLayout layer;
        size_t per_layer;
        size_t layer_base;
        size_t position_delta;
        size_t cache_offset;

        if (niyah_cuda_layer_layout(
                model_state,
                layer_index,
                &layer) != 0 ||
            !niyah_cuda_size_mul_ok(
                decode_state->context_length,
                kv_dim,
                &per_layer) ||
            !niyah_cuda_size_mul_ok(
                (size_t)layer_index,
                per_layer,
                &layer_base) ||
            !niyah_cuda_size_mul_ok(
                position,
                kv_dim,
                &position_delta) ||
            !niyah_cuda_size_add_ok(
                layer_base,
                position_delta,
                &cache_offset) ||
            cache_offset >
                decode_state->values_per_tensor ||
            kv_dim >
                decode_state->values_per_tensor -
                    cache_offset) {
            return 1;
        }

        if (niyah_cuda_rmsnorm_device(
                model_state,
                norm,
                hidden,
                layer.attn_norm,
                dim) != 0) {
            return 1;
        }

        if (niyah_cuda_model_state_matvec_device_launch(
                model_state,
                layer.wq,
                norm,
                q,
                dim,
                dim) != 0 ||
            niyah_cuda_model_state_matvec_device_launch(
                model_state,
                layer.wk,
                norm,
                k,
                kv_dim,
                dim) != 0 ||
            niyah_cuda_model_state_matvec_device_launch(
                model_state,
                layer.wv,
                norm,
                v,
                kv_dim,
                dim) != 0) {
            return 1;
        }

        if (niyah_cuda_rope_device(
                q,
                (size_t)config->n_heads,
                decode_state->head_dim,
                position) != 0 ||
            niyah_cuda_rope_device(
                k,
                (size_t)config->n_kv_heads,
                decode_state->head_dim,
                position) != 0) {
            return 1;
        }

        if (cudaMemcpy(
                (float *)decode_state->device_keys +
                    cache_offset,
                k,
                kv_bytes,
                cudaMemcpyDeviceToDevice) != cudaSuccess ||
            cudaMemcpy(
                (float *)decode_state->device_values +
                    cache_offset,
                v,
                kv_bytes,
                cudaMemcpyDeviceToDevice) != cudaSuccess) {
            return 1;
        }

        if (niyah_cuda_attention_one_device(
                attn,
                q,
                decode_state,
                layer_index,
                position,
                scores) != 0) {
            return 1;
        }

        if (niyah_cuda_model_state_matvec_device_launch(
                model_state,
                layer.wo,
                attn,
                proj,
                dim,
                dim) != 0 ||
            niyah_cuda_add_device(
                hidden,
                proj,
                dim) != 0) {
            return 1;
        }

        if (niyah_cuda_rmsnorm_device(
                model_state,
                norm,
                hidden,
                layer.ffn_norm,
                dim) != 0) {
            return 1;
        }

        if (niyah_cuda_model_state_matvec_device_launch(
                model_state,
                layer.w_gate,
                norm,
                gate,
                ffn,
                dim) != 0 ||
            niyah_cuda_model_state_matvec_device_launch(
                model_state,
                layer.w_up,
                norm,
                up,
                ffn,
                dim) != 0 ||
            niyah_cuda_silu_mul_device(
                gate,
                up,
                ffn) != 0 ||
            niyah_cuda_model_state_matvec_device_launch(
                model_state,
                layer.w_down,
                gate,
                proj,
                dim,
                ffn) != 0 ||
            niyah_cuda_add_device(
                hidden,
                proj,
                dim) != 0) {
            return 1;
        }
    }

    if (niyah_cuda_rmsnorm_device(
            model_state,
            norm,
            hidden,
            model_state->layout.final_norm,
            dim) != 0 ||
        niyah_cuda_model_state_matvec_device_launch(
            model_state,
            model_state->layout.lm_head,
            norm,
            decode_state->device_logits,
            vocab,
            dim) != 0) {
        return 1;
    }

    if (cudaMemcpy(
            logits,
            decode_state->device_logits,
            vocab * sizeof(float),
            cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 1;
    }

    decode_state->next_position = position + 1U;
    return 0;
}

static NiyahStatus niyah_cuda_generation_fail(
    NiyahCudaDecodeState *decode_state,
    NiyahGenerationResult *result,
    NiyahStatus status)
{
    if (decode_state != NULL) {
        (void)niyah_cuda_decode_state_reset(decode_state);
    }

    if (result != NULL) {
        memset(result, 0, sizeof(*result));
    }

    return status;
}

extern "C" NiyahStatus niyah_cuda_generate(
    const NiyahCudaModelState *model_state,
    NiyahCudaDecodeState *decode_state,
    const uint32_t *prompt_tokens,
    size_t prompt_count,
    const NiyahGenerationConfig *config,
    uint32_t *output_tokens,
    size_t output_capacity,
    NiyahGenerationResult *result,
    float *host_logits,
    size_t host_logits_count)
{
    const NiyahModelConfig *model_config;
    size_t required_context;
    size_t vocab;
    size_t i;
    size_t generated = 0U;
    NiyahSampler sampler;
    NiyahStatus status;

    if (model_state == NULL ||
        decode_state == NULL ||
        prompt_tokens == NULL ||
        prompt_count == 0U ||
        config == NULL ||
        output_tokens == NULL ||
        result == NULL ||
        host_logits == NULL ||
        !niyah_cuda_decode_state_matches_model(
            decode_state,
            model_state)) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    memset(result, 0, sizeof(*result));

    model_config = &model_state->config;
    vocab = (size_t)model_config->vocab_size;

    if (config->max_new_tokens == 0U ||
        output_capacity < config->max_new_tokens ||
        prompt_count >
            (size_t)model_config->context_length ||
        !niyah_cuda_size_add_ok(
            prompt_count,
            config->max_new_tokens,
            &required_context) ||
        required_context >
            (size_t)model_config->context_length ||
        host_logits_count < vocab ||
        decode_state->next_position != 0U) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    if (config->stop_on_eos != 0 &&
        (size_t)config->eos_token >= vocab) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    /*
     * Validate the complete prompt before mutating CUDA KV state.
     * The CPU path reaches the same final externally visible state on an
     * invalid prompt because its failure path resets the cache/result.
     */
    for (i = 0U; i < prompt_count; ++i) {
        if ((size_t)prompt_tokens[i] >= vocab) {
            return niyah_cuda_generation_fail(
                decode_state,
                result,
                NIYAH_ERR_INVALID_ARGUMENT);
        }
    }

    status = niyah_sampler_init(
        &sampler,
        &config->sampler);
    if (status != NIYAH_OK) {
        return niyah_cuda_generation_fail(
            decode_state,
            result,
            status);
    }

    for (i = 0U; i < prompt_count; ++i) {
        if (niyah_cuda_decode_token(
                model_state,
                decode_state,
                prompt_tokens[i],
                host_logits,
                host_logits_count) != 0) {
            return niyah_cuda_generation_fail(
                decode_state,
                result,
                NIYAH_ERR_IO);
        }
    }

    result->prompt_tokens = prompt_count;

    while (generated < config->max_new_tokens) {
        uint32_t token = 0U;

        status = niyah_sampler_sample(
            &sampler,
            host_logits,
            vocab,
            &token);
        if (status != NIYAH_OK) {
            return niyah_cuda_generation_fail(
                decode_state,
                result,
                status);
        }

        output_tokens[generated] = token;
        generated += 1U;
        result->generated_tokens = generated;

        if (config->stop_on_eos != 0 &&
            token == config->eos_token) {
            result->stopped_on_eos = 1;
            break;
        }

        if (niyah_cuda_decode_token(
                model_state,
                decode_state,
                token,
                host_logits,
                host_logits_count) != 0) {
            return niyah_cuda_generation_fail(
                decode_state,
                result,
                NIYAH_ERR_IO);
        }
    }

    return NIYAH_OK;
}
