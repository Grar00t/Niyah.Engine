#include "test_numeric_compare.h"
#include "niyah/niyah.h"
#include "niyah/optimizer.h"
#include "niyah/train.h"
#include "niyah_cuda_matvec.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static float max_abs_region(
    const float *full,
    size_t offset,
    const float *expected,
    size_t count)
{
    return niyah_test_max_abs_diff(full + offset, expected, count);
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

        if (!isfinite(values[i])) return INFINITY;
        d = fabsf(values[i]);

        if (d > result)
            result = d;
    }

    return result;
}



/* Real CUDA full backward + device accumulation + CPU AdamW, compared with
 * independent CPU backwards on the identical model, moments, masks and updates. */
static int test_accumulated_update_parity(void)
{
    const uint32_t tokens[2][4] = {{1U, 3U, 1U, 5U}, {2U, 2U, 4U, 1U}};
    const uint32_t targets[2][4] = {{3U, 1U, 5U, 2U}, {2U, 4U, 1U, 7U}};
    const uint32_t segments[2][4] = {{0U, 1U, 0U, 1U}, {1U, 1U, 0U, 0U}};
    const size_t starts[2] = {2U, 1U};
    NiyahModelConfig cfg = {0};
    NiyahModel cpu = {0}, gpu = {0};
    NiyahModelGradients sample = {0}, sum = {0}, device_sum = {0};
    NiyahAdamWState cpu_opt = {0}, gpu_opt = {0};
    NiyahAdamWConfig opt = {0.005f, 0.9f, 0.999f, 1.0e-8f, 0.01f, 1.0f};
    NiyahCudaModelState ms = {0};
    NiyahCudaTrainState ts = {0};
    float *workspace = NULL;
    size_t cpu_count = 0U, cuda_count = 0U, i, update, k;
    int failed = 1;
    cfg.vocab_size = 16U; cfg.context_length = 4U; cfg.embedding_dim = 8U;
    cfg.n_layers = 2U; cfg.n_heads = 2U; cfg.n_kv_heads = 1U;
    cfg.ffn_hidden_dim = 16U; cfg.rms_norm_eps = 1.0e-5f;
    cfg.tie_word_embeddings = 1; cfg.n_segments = 2U;
#define REQUIRE(expr) do { if (!(expr)) { \
    fprintf(stderr, "CUDA update parity failed at %d: %s\n", __LINE__, #expr); \
    goto cleanup; \
} } while (0)
    REQUIRE(niyah_model_create(&cpu, &cfg) == NIYAH_OK);
    REQUIRE(niyah_model_create(&gpu, &cfg) == NIYAH_OK);
    REQUIRE(niyah_model_reset_parameters(&cpu, 901U) == NIYAH_OK);
    REQUIRE(niyah_model_reset_parameters(&gpu, 901U) == NIYAH_OK);
    REQUIRE(niyah_model_gradients_create(&sample, &cpu) == NIYAH_OK);
    REQUIRE(niyah_model_gradients_create(&sum, &cpu) == NIYAH_OK);
    REQUIRE(niyah_model_gradients_create(&device_sum, &gpu) == NIYAH_OK);
    REQUIRE(niyah_adamw_state_create(&cpu_opt, &cpu) == NIYAH_OK);
    REQUIRE(niyah_adamw_state_create(&gpu_opt, &gpu) == NIYAH_OK);
    for (i = 0U; i < cpu.weight_count; ++i) {
        cpu_opt.m[i] = gpu_opt.m[i] = 0.001f;
        cpu_opt.v[i] = gpu_opt.v[i] = 0.002f;
    }
    cpu_opt.step = gpu_opt.step = 5U;
    REQUIRE(niyah_train_backward_workspace_floats(&cfg, 4U, &cpu_count) == NIYAH_OK);
    REQUIRE(niyah_cuda_train_backward_workspace_floats(&cfg, 4U, &cuda_count) == 0);
    workspace = (float *)calloc(cpu_count, sizeof(float));
    REQUIRE(workspace != NULL);
    REQUIRE(niyah_cuda_model_state_create(&ms, &gpu) == 0);
    REQUIRE(niyah_cuda_train_state_create(&ts, &gpu, 4U, cuda_count) == 0);
    for (update = 0U; update < 2U; ++update) {
        double cpu_loss_sum = 0.0, gpu_loss_sum = 0.0;
        float gradient_error, weight_error, m_error, v_error;
        niyah_model_gradients_zero(&sum);
        REQUIRE(niyah_cuda_model_state_sync(&ms, &gpu) == 0);
        REQUIRE(niyah_cuda_train_state_zero_accumulated_gradients(&ts) == 0);
        for (k = 0U; k < 2U; ++k) {
            float cpu_loss = NAN, gpu_loss = NAN;
            const float supervised = (float)(4U - starts[k]);
            REQUIRE(niyah_train_backward_masked_with_segments(&cpu,
                tokens[k], targets[k], 4U, starts[k], segments[k],
                &cpu_loss, &sample, workspace, cpu_count) == NIYAH_OK);
            REQUIRE(niyah_cuda_train_backward_full(&ms, &ts,
                tokens[k], targets[k], 4U, starts[k], segments[k], &gpu_loss) == 0);
            REQUIRE(isfinite(cpu_loss) && isfinite(gpu_loss));
            for (i = 0U; i < sum.count; ++i)
                sum.values[i] += sample.values[i] * supervised;
            REQUIRE(niyah_cuda_train_state_accumulate_gradients(&ts, supervised) == 0);
            cpu_loss_sum += (double)cpu_loss * supervised;
            gpu_loss_sum += (double)gpu_loss * supervised;
        }
        for (i = 0U; i < sum.count; ++i) sum.values[i] *= 1.0f / 5.0f;
        REQUIRE(niyah_cuda_train_state_scale_accumulated_gradients(&ts, 1.0f / 5.0f) == 0);
        REQUIRE(niyah_cuda_train_state_copy_accumulated_gradients_to_host(
                    &ts, device_sum.values, device_sum.count) == 0);
        gradient_error = niyah_test_max_abs_diff(sum.values, device_sum.values, sum.count);
        REQUIRE(gradient_error <= 2.0e-4f);
        REQUIRE(fabs(cpu_loss_sum - gpu_loss_sum) / 5.0 <= 2.0e-4);
        REQUIRE(niyah_adamw_step(&cpu, &sum, &cpu_opt, &opt) == NIYAH_OK);
        REQUIRE(niyah_adamw_step(&gpu, &device_sum, &gpu_opt, &opt) == NIYAH_OK);
        weight_error = niyah_test_max_abs_diff(cpu.weights, gpu.weights, cpu.weight_count);
        m_error = niyah_test_max_abs_diff(cpu_opt.m, gpu_opt.m, cpu_opt.count);
        v_error = niyah_test_max_abs_diff(cpu_opt.v, gpu_opt.v, cpu_opt.count);
        printf("cuda_update_%zu_gradient=%.9g weights=%.9g m=%.9g v=%.9g\n",
               update, gradient_error, weight_error, m_error, v_error);
        REQUIRE(weight_error <= 1.0e-5f && m_error <= 2.0e-5f && v_error <= 2.0e-5f);
        REQUIRE(cpu_opt.step == gpu_opt.step && cpu_opt.step == 6U + update);
    }
    /* Invalid scalar/count inputs must not launch a mutating kernel. */
    REQUIRE(niyah_cuda_train_state_accumulate_gradients(&ts, NAN) != 0);
    REQUIRE(niyah_cuda_train_state_accumulate_gradients(&ts, INFINITY) != 0);
    REQUIRE(niyah_cuda_train_state_scale_accumulated_gradients(&ts, -INFINITY) != 0);
    {
        const size_t saved = ts.gradient_capacity;
        ts.gradient_capacity = SIZE_MAX;
        REQUIRE(niyah_cuda_train_state_accumulate_gradients(&ts, 1.0f) != 0);
        REQUIRE(niyah_cuda_train_state_scale_accumulated_gradients(&ts, 1.0f) != 0);
        ts.gradient_capacity = saved;
    }
    REQUIRE(niyah_cuda_train_state_copy_accumulated_gradients_to_host(
                &ts, sum.values, sum.count) == 0);
    REQUIRE(memcmp(sum.values, device_sum.values, sum.count * sizeof(float)) == 0);
    failed = 0;
cleanup:
    niyah_cuda_train_state_destroy(&ts); niyah_cuda_model_state_destroy(&ms);
    free(workspace);
    niyah_adamw_state_destroy(&cpu_opt); niyah_adamw_state_destroy(&gpu_opt);
    niyah_model_gradients_destroy(&sample); niyah_model_gradients_destroy(&sum);
    niyah_model_gradients_destroy(&device_sum);
    niyah_model_destroy(&cpu); niyah_model_destroy(&gpu);
#undef REQUIRE
    return failed;
}

int main(void)
{
    if (test_accumulated_update_parity() != 0) return 1;
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
