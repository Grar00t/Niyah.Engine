#include "niyah/niyah.h"
#include "niyah_cuda_matvec.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static float max_abs(const float *a, const float *b, size_t n)
{
    float m = 0.0f;
    for (size_t i = 0U; i < n; ++i) {
        const float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(void)
{
    const size_t tokens = 4U;
    const size_t rows = 16U;
    const size_t cols = 8U;

    NiyahModelConfig cfg = {0};
    NiyahModel model = {0};
    NiyahLayerLayout layer = {0};
    NiyahCudaModelState ms = {0};
    NiyahCudaTrainState ts = {0};

    cfg.vocab_size = 16U;
    cfg.context_length = 8U;
    cfg.embedding_dim = 8U;
    cfg.n_layers = 1U;
    cfg.n_heads = 2U;
    cfg.n_kv_heads = 1U;
    cfg.ffn_hidden_dim = 16U;
    cfg.rms_norm_eps = 1.0e-5f;

    if (niyah_model_create(&model, &cfg) != NIYAH_OK ||
        niyah_model_reset_parameters(&model, 12345U) != NIYAH_OK ||
        niyah_model_layer_layout(
            &cfg, &model.layout, 0U, &layer) != NIYAH_OK)
        return 1;

    const size_t matrix_count = rows * cols;
    const size_t x_count = tokens * cols;
    const size_t y_count = tokens * rows;

    const size_t x_off = 0U;
    const size_t y_off = x_off + x_count;
    const size_t dy_off = y_off + y_count;
    const size_t dx_off = dy_off + y_count;
    const size_t workspace_count = dx_off + x_count;

    float *x = calloc(x_count, sizeof(float));
    float *dy = calloc(y_count, sizeof(float));
    float *cpu_y = calloc(y_count, sizeof(float));
    float *gpu_y = calloc(y_count, sizeof(float));
    float *cpu_dw = calloc(matrix_count, sizeof(float));
    float *gpu_dw = calloc(matrix_count, sizeof(float));
    float *cpu_dx = calloc(x_count, sizeof(float));
    float *gpu_dx = calloc(x_count, sizeof(float));

    if (!x || !dy || !cpu_y || !gpu_y ||
        !cpu_dw || !gpu_dw || !cpu_dx || !gpu_dx)
        return 1;

    for (size_t i = 0U; i < x_count; ++i)
        x[i] = ((int)(i % 17U) - 8) * 0.015625f;

    for (size_t i = 0U; i < y_count; ++i)
        dy[i] = ((int)(i % 13U) - 6) * 0.0078125f;

    const float *w = model.weights + layer.w_gate;

    for (size_t t = 0U; t < tokens; ++t) {
        for (size_t r = 0U; r < rows; ++r) {
            float sum = 0.0f;
            for (size_t c = 0U; c < cols; ++c)
                sum += w[r * cols + c] * x[t * cols + c];
            cpu_y[t * rows + r] = sum;
        }

        for (size_t r = 0U; r < rows; ++r) {
            const float d = dy[t * rows + r];
            for (size_t c = 0U; c < cols; ++c)
                cpu_dw[r * cols + c] += d * x[t * cols + c];
        }

        for (size_t c = 0U; c < cols; ++c)
            for (size_t r = 0U; r < rows; ++r)
                cpu_dx[t * cols + c] +=
                    w[r * cols + c] * dy[t * rows + r];
    }

    if (niyah_cuda_model_state_create(&ms, &model) != 0 ||
        niyah_cuda_train_state_create(
            &ts, &model, tokens, workspace_count) != 0 ||
        niyah_cuda_train_state_copy_workspace_from_host(
            &ts, x_off, x, x_count) != 0 ||
        niyah_cuda_train_state_copy_workspace_from_host(
            &ts, dy_off, dy, y_count) != 0 ||
        niyah_cuda_train_state_zero_workspace(
            &ts, dx_off, x_count) != 0 ||
        niyah_cuda_train_state_zero_gradients(&ts) != 0 ||
        niyah_cuda_train_linear_forward(
            &ms, &ts, layer.w_gate,
            x_off, y_off, tokens, rows, cols) != 0 ||
        niyah_cuda_train_linear_backward(
            &ms, &ts, layer.w_gate,
            x_off, dy_off, dx_off,
            tokens, rows, cols) != 0 ||
        niyah_cuda_train_state_copy_workspace_to_host(
            &ts, y_off, gpu_y, y_count) != 0 ||
        niyah_cuda_train_state_copy_workspace_to_host(
            &ts, dx_off, gpu_dx, x_count) != 0 ||
        niyah_cuda_train_state_copy_gradient_range_to_host(
            &ts, layer.w_gate, gpu_dw, matrix_count) != 0)
        return 1;

    const float ey = max_abs(cpu_y, gpu_y, y_count);
    const float ew = max_abs(cpu_dw, gpu_dw, matrix_count);
    const float ex = max_abs(cpu_dx, gpu_dx, x_count);

    printf("forward_max_abs=%.9g\n", ey);
    printf("dweight_max_abs=%.9g\n", ew);
    printf("dx_max_abs=%.9g\n", ex);

    niyah_cuda_train_state_destroy(&ts);
    niyah_cuda_model_state_destroy(&ms);
    niyah_model_destroy(&model);

    free(x); free(dy); free(cpu_y); free(gpu_y);
    free(cpu_dw); free(gpu_dw); free(cpu_dx); free(gpu_dx);

    return ey <= 2.0e-4f &&
           ew <= 2.0e-4f &&
           ex <= 2.0e-4f ? 0 : 1;
}
