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
    const size_t width = 8U;
    const size_t count = tokens * width;

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

    const size_t x_off = 0U;
    const size_t dy_off = x_off + count;
    const size_t dx_off = dy_off + count;
    const size_t workspace_count = dx_off + count;

    float *x = calloc(count, sizeof(float));
    float *dy = calloc(count, sizeof(float));
    float *cpu_dx = calloc(count, sizeof(float));
    float *gpu_dx = calloc(count, sizeof(float));
    float *cpu_dw = calloc(width, sizeof(float));
    float *gpu_dw = calloc(width, sizeof(float));

    if (!x || !dy || !cpu_dx || !gpu_dx || !cpu_dw || !gpu_dw)
        return 1;

    for (size_t i = 0U; i < count; ++i) {
        x[i] = ((int)(i % 17U) - 8) * 0.015625f;
        dy[i] = ((int)(i % 13U) - 6) * 0.0078125f;
    }

    const float *weight = model.weights + layer.attn_norm;

    for (size_t t = 0U; t < tokens; ++t) {
        double sum_sq = 0.0;
        double dot = 0.0;

        for (size_t i = 0U; i < width; ++i) {
            const double xv = (double)x[t * width + i];
            sum_sq += xv * xv;
        }

        const float inv =
            1.0f / sqrtf((float)(sum_sq / (double)width) +
                         cfg.rms_norm_eps);

        for (size_t i = 0U; i < width; ++i) {
            dot += (double)dy[t * width + i] *
                   (double)weight[i] *
                   (double)x[t * width + i];

            cpu_dw[i] +=
                dy[t * width + i] *
                x[t * width + i] *
                inv;
        }

        const float coeff =
            inv * inv * inv * (float)(dot / (double)width);

        for (size_t i = 0U; i < width; ++i) {
            cpu_dx[t * width + i] =
                dy[t * width + i] * weight[i] * inv -
                x[t * width + i] * coeff;
        }
    }

    if (niyah_cuda_model_state_create(&ms, &model) != 0 ||
        niyah_cuda_train_state_create(
            &ts, &model, tokens, workspace_count) != 0 ||
        niyah_cuda_train_state_copy_workspace_from_host(
            &ts, x_off, x, count) != 0 ||
        niyah_cuda_train_state_copy_workspace_from_host(
            &ts, dy_off, dy, count) != 0 ||
        niyah_cuda_train_state_zero_workspace(
            &ts, dx_off, count) != 0 ||
        niyah_cuda_train_state_zero_gradients(&ts) != 0 ||
        niyah_cuda_train_rmsnorm_backward(
            &ms, &ts, layer.attn_norm,
            x_off, dy_off, dx_off,
            tokens, width, cfg.rms_norm_eps) != 0 ||
        niyah_cuda_train_state_copy_workspace_to_host(
            &ts, dx_off, gpu_dx, count) != 0 ||
        niyah_cuda_train_state_copy_gradient_range_to_host(
            &ts, layer.attn_norm, gpu_dw, width) != 0)
        return 1;

    const float ex = max_abs(cpu_dx, gpu_dx, count);
    const float ew = max_abs(cpu_dw, gpu_dw, width);

    printf("rmsnorm_dx_max_abs=%.9g\n", ex);
    printf("rmsnorm_dweight_max_abs=%.9g\n", ew);

    niyah_cuda_train_state_destroy(&ts);
    niyah_cuda_model_state_destroy(&ms);
    niyah_model_destroy(&model);

    free(x);
    free(dy);
    free(cpu_dx);
    free(gpu_dx);
    free(cpu_dw);
    free(gpu_dw);

    return ex <= 2.0e-4f &&
           ew <= 2.0e-4f ? 0 : 1;
}
