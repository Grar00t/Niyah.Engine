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

static void apply_rope_signed(
    float *vector,
    size_t n_heads,
    size_t head_dim,
    size_t position,
    float sign)
{
    const float pos = (float)position;

    for (size_t head = 0U; head < n_heads; ++head) {
        float *head_vector = vector + head * head_dim;

        for (size_t i = 0U; i + 1U < head_dim; i += 2U) {
            const float exponent =
                -((float)i / (float)head_dim);
            const float inv_freq =
                powf(10000.0f, exponent);
            const float angle =
                sign * pos * inv_freq;
            const float c = cosf(angle);
            const float s = sinf(angle);
            const float x0 = head_vector[i];
            const float x1 = head_vector[i + 1U];

            head_vector[i] = x0 * c - x1 * s;
            head_vector[i + 1U] = x0 * s + x1 * c;
        }
    }
}

int main(void)
{
    const size_t tokens = 4U;

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
        niyah_model_reset_parameters(&model, 12345U) != NIYAH_OK)
        return 1;

    const size_t head_dim =
        (size_t)cfg.embedding_dim / (size_t)cfg.n_heads;
    const size_t q_width =
        (size_t)cfg.n_heads * head_dim;
    const size_t k_width =
        (size_t)cfg.n_kv_heads * head_dim;
    const size_t q_count = tokens * q_width;
    const size_t k_count = tokens * k_width;

    const size_t q_off = 0U;
    const size_t k_off = q_off + q_count;
    const size_t workspace_count = k_off + k_count;

    float *cpu_q = calloc(q_count, sizeof(float));
    float *gpu_q = calloc(q_count, sizeof(float));
    float *cpu_k = calloc(k_count, sizeof(float));
    float *gpu_k = calloc(k_count, sizeof(float));

    if (!cpu_q || !gpu_q || !cpu_k || !gpu_k)
        return 1;

    for (size_t i = 0U; i < q_count; ++i)
        cpu_q[i] =
            ((int)(i % 19U) - 9) * 0.03125f;

    for (size_t i = 0U; i < k_count; ++i)
        cpu_k[i] =
            ((int)(i % 17U) - 8) * 0.0234375f;

    if (niyah_cuda_train_state_create(
            &ts, &model, tokens, workspace_count) != 0 ||
        niyah_cuda_train_state_copy_workspace_from_host(
            &ts, q_off, cpu_q, q_count) != 0 ||
        niyah_cuda_train_state_copy_workspace_from_host(
            &ts, k_off, cpu_k, k_count) != 0)
        return 1;

    for (size_t t = 0U; t < tokens; ++t) {
        apply_rope_signed(
            cpu_q + t * q_width,
            (size_t)cfg.n_heads,
            head_dim,
            t,
            -1.0f);

        apply_rope_signed(
            cpu_k + t * k_width,
            (size_t)cfg.n_kv_heads,
            head_dim,
            t,
            -1.0f);
    }

    if (niyah_cuda_train_rope_backward(
            &ts,
            q_off,
            tokens,
            (size_t)cfg.n_heads,
            head_dim) != 0 ||
        niyah_cuda_train_rope_backward(
            &ts,
            k_off,
            tokens,
            (size_t)cfg.n_kv_heads,
            head_dim) != 0 ||
        niyah_cuda_train_state_copy_workspace_to_host(
            &ts, q_off, gpu_q, q_count) != 0 ||
        niyah_cuda_train_state_copy_workspace_to_host(
            &ts, k_off, gpu_k, k_count) != 0)
        return 1;

    const float eq = max_abs(cpu_q, gpu_q, q_count);
    const float ek = max_abs(cpu_k, gpu_k, k_count);

    printf("rope_q_max_abs=%.9g\n", eq);
    printf("rope_k_max_abs=%.9g\n", ek);

    niyah_cuda_train_state_destroy(&ts);
    niyah_model_destroy(&model);

    free(cpu_q);
    free(gpu_q);
    free(cpu_k);
    free(gpu_k);

    return eq <= 2.0e-4f &&
           ek <= 2.0e-4f ? 0 : 1;
}
