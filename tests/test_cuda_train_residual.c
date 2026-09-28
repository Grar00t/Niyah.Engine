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
        niyah_model_reset_parameters(&model, 12345U) != NIYAH_OK) {
        return 1;
    }

    const size_t a_off = 0U;
    const size_t b_off = a_off + count;
    const size_t sum_off = b_off + count;
    const size_t accum_off = sum_off + count;
    const size_t workspace_count = accum_off + count;

    float *a = calloc(count, sizeof(float));
    float *b = calloc(count, sizeof(float));
    float *accum = calloc(count, sizeof(float));

    float *cpu_sum = calloc(count, sizeof(float));
    float *cpu_accum = calloc(count, sizeof(float));

    float *gpu_sum = calloc(count, sizeof(float));
    float *gpu_accum = calloc(count, sizeof(float));

    if (!a || !b || !accum ||
        !cpu_sum || !cpu_accum ||
        !gpu_sum || !gpu_accum) {
        return 1;
    }

    for (size_t i = 0U; i < count; ++i) {
        a[i] =
            ((int)(i % 17U) - 8) * 0.03125f;

        b[i] =
            ((int)(i % 13U) - 6) * 0.015625f;

        accum[i] =
            ((int)(i % 11U) - 5) * 0.0234375f;

        cpu_sum[i] = a[i] + b[i];
        cpu_accum[i] = accum[i] + b[i];
    }

    if (niyah_cuda_train_state_create(
            &ts, &model, tokens, workspace_count) != 0 ||
        niyah_cuda_train_state_copy_workspace_from_host(
            &ts, a_off, a, count) != 0 ||
        niyah_cuda_train_state_copy_workspace_from_host(
            &ts, b_off, b, count) != 0 ||
        niyah_cuda_train_state_copy_workspace_from_host(
            &ts, accum_off, accum, count) != 0 ||
        niyah_cuda_train_state_zero_workspace(
            &ts, sum_off, count) != 0) {
        return 1;
    }

    if (niyah_cuda_train_add(
            &ts,
            sum_off,
            a_off,
            b_off,
            count) != 0 ||
        niyah_cuda_train_add_inplace(
            &ts,
            accum_off,
            b_off,
            count) != 0 ||
        niyah_cuda_train_state_copy_workspace_to_host(
            &ts, sum_off, gpu_sum, count) != 0 ||
        niyah_cuda_train_state_copy_workspace_to_host(
            &ts, accum_off, gpu_accum, count) != 0) {
        return 1;
    }

    const float es =
        max_abs(cpu_sum, gpu_sum, count);
    const float ea =
        max_abs(cpu_accum, gpu_accum, count);

    printf("residual_sum_max_abs=%.9g\n", es);
    printf("residual_accum_max_abs=%.9g\n", ea);

    niyah_cuda_train_state_destroy(&ts);
    niyah_model_destroy(&model);

    free(a);
    free(b);
    free(accum);
    free(cpu_sum);
    free(cpu_accum);
    free(gpu_sum);
    free(gpu_accum);

    return es <= 2.0e-4f &&
           ea <= 2.0e-4f ? 0 : 1;
}
