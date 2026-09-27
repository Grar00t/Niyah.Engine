#include "niyah/niyah.h"
#include "niyah_cuda_matvec.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>


static float max_abs_region(
    const float *full,
    size_t offset,
    const float *expected,
    size_t count)
{
    float result = 0.0f;

    for (size_t i = 0U; i < count; ++i) {
        const float d =
            fabsf(full[offset + i] - expected[i]);

        if (d > result)
            result = d;
    }

    return result;
}


static float max_abs_outside(
    const float *values,
    size_t count,
    size_t begin,
    size_t length)
{
    float result = 0.0f;
    const size_t end = begin + length;

    for (size_t i = 0U; i < count; ++i) {
        float d;

        if (i >= begin && i < end)
            continue;

        d = fabsf(values[i]);

        if (d > result)
            result = d;
    }

    return result;
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
            &cfg,
            &model.layout,
            0U,
            &layer) != NIYAH_OK) {
        return 1;
    }

    const size_t matrix_count = rows * cols;
    const size_t x_count = tokens * cols;
    const size_t y_count = tokens * rows;

    const size_t x_off = 0U;
    const size_t dy_off = x_off + x_count;
    const size_t dx_off = dy_off + y_count;
    const size_t workspace_count = dx_off + x_count;

    float *x = calloc(x_count, sizeof(float));
    float *dy = calloc(y_count, sizeof(float));
    float *cpu_dw = calloc(matrix_count, sizeof(float));
    float *gpu_accum =
        calloc(model.weight_count, sizeof(float));

    if (x == NULL ||
        dy == NULL ||
        cpu_dw == NULL ||
        gpu_accum == NULL) {
        return 1;
    }

    for (size_t i = 0U; i < x_count; ++i) {
        x[i] =
            ((int)(i % 17U) - 8) * 0.015625f;
    }

    for (size_t i = 0U; i < y_count; ++i) {
        dy[i] =
            ((int)(i % 13U) - 6) * 0.0078125f;
    }

    for (size_t t = 0U; t < tokens; ++t) {
        for (size_t r = 0U; r < rows; ++r) {
            const float d = dy[t * rows + r];

            for (size_t c = 0U; c < cols; ++c) {
                cpu_dw[r * cols + c] +=
                    d * x[t * cols + c];
            }
        }
    }

    if (niyah_cuda_model_state_create(
            &ms,
            &model) != 0 ||
        niyah_cuda_train_state_create(
            &ts,
            &model,
            tokens,
            workspace_count) != 0 ||
        niyah_cuda_train_state_copy_workspace_from_host(
            &ts,
            x_off,
            x,
            x_count) != 0 ||
        niyah_cuda_train_state_copy_workspace_from_host(
            &ts,
            dy_off,
            dy,
            y_count) != 0 ||
        niyah_cuda_train_state_zero_accumulated_gradients(
            &ts) != 0) {
        return 1;
    }

    /*
     * Two identical sample gradients followed by device scaling:
     *
     * 0.125 * g + 0.375 * g = 0.5 * g
     * 2.0 * (0.5 * g) = g
     *
     * This proves both accumulation and scaling without changing
     * the expected CPU reference.
     */
    if (niyah_cuda_train_state_zero_gradients(&ts) != 0 ||
        niyah_cuda_train_state_zero_workspace(
            &ts,
            dx_off,
            x_count) != 0 ||
        niyah_cuda_train_linear_backward(
            &ms,
            &ts,
            layer.w_gate,
            x_off,
            dy_off,
            dx_off,
            tokens,
            rows,
            cols) != 0 ||
        niyah_cuda_train_state_accumulate_gradients(
            &ts,
            0.125f) != 0) {
        return 1;
    }

    if (niyah_cuda_train_state_zero_gradients(&ts) != 0 ||
        niyah_cuda_train_state_zero_workspace(
            &ts,
            dx_off,
            x_count) != 0 ||
        niyah_cuda_train_linear_backward(
            &ms,
            &ts,
            layer.w_gate,
            x_off,
            dy_off,
            dx_off,
            tokens,
            rows,
            cols) != 0 ||
        niyah_cuda_train_state_accumulate_gradients(
            &ts,
            0.375f) != 0 ||
        niyah_cuda_train_state_scale_accumulated_gradients(
            &ts,
            2.0f) != 0 ||
        niyah_cuda_train_state_copy_accumulated_gradients_to_host(
            &ts,
            gpu_accum,
            model.weight_count) != 0) {
        return 1;
    }

    const float region_error =
        max_abs_region(
            gpu_accum,
            layer.w_gate,
            cpu_dw,
            matrix_count);

    const float outside_error =
        max_abs_outside(
            gpu_accum,
            model.weight_count,
            layer.w_gate,
            matrix_count);

    printf(
        "cuda_accumulated_gradient_max_abs=%.9g\n",
        region_error);

    printf(
        "cuda_accumulated_outside_max_abs=%.9g\n",
        outside_error);

    niyah_cuda_train_state_destroy(&ts);
    niyah_cuda_model_state_destroy(&ms);
    niyah_model_destroy(&model);

    free(gpu_accum);
    free(cpu_dw);
    free(dy);
    free(x);

    return
        region_error <= 2.0e-4f &&
        outside_error <= 2.0e-7f
            ? 0
            : 1;
}
