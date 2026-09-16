#include "niyah_cuda_matvec.h"

#include <cuda_runtime.h>

#include <stddef.h>
#include <string.h>

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
               (b->tie_word_embeddings != 0);
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
        rows > ((size_t)-1) / cols) {
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

    blocks = (unsigned int)(
        (rows + (size_t)threads - 1U) / (size_t)threads);

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
     * Matches niyah_decode_workspace_floats():
     * 5*dim + 2*kv_dim + 2*ffn + context_length.
     */
    if (!niyah_cuda_size_mul_ok(5U, dim, &workspace_floats) ||
        !niyah_cuda_size_mul_ok(2U, kv_dim, &term) ||
        !niyah_cuda_size_add_ok(
            workspace_floats, term, &workspace_floats) ||
        !niyah_cuda_size_mul_ok(2U, ffn, &term) ||
        !niyah_cuda_size_add_ok(
            workspace_floats, term, &workspace_floats) ||
        !niyah_cuda_size_add_ok(
            workspace_floats,
            (size_t)config->context_length,
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
