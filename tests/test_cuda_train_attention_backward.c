#include "niyah/niyah.h"
#include "niyah_cuda_matvec.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float max_abs(
    const float *a,
    const float *b,
    size_t n)
{
    float m = 0.0f;

    for (size_t i = 0U; i < n; ++i) {
        const float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }

    return m;
}

static int cpu_attention_backward(
    float *dq,
    float *dk,
    float *dv,
    const float *da,
    const float *q,
    const float *k,
    const float *v,
    size_t tokens,
    size_t n_heads,
    size_t n_kv_heads,
    size_t head_dim)
{
    const size_t dim =
        n_heads * head_dim;
    const size_t kv_dim =
        n_kv_heads * head_dim;
    const size_t group_size =
        n_heads / n_kv_heads;
    const float scale =
        1.0f / sqrtf((float)head_dim);

    float *probs =
        calloc(tokens, sizeof(float));
    float *dp =
        calloc(tokens, sizeof(float));

    if (!probs || !dp)
        return 1;

    memset(
        dq,
        0,
        tokens * dim * sizeof(float));
    memset(
        dk,
        0,
        tokens * kv_dim * sizeof(float));
    memset(
        dv,
        0,
        tokens * kv_dim * sizeof(float));

    for (size_t t = 0U; t < tokens; ++t) {
        for (size_t h = 0U; h < n_heads; ++h) {
            const size_t kh =
                h / group_size;
            const float *qh =
                q + t * dim + h * head_dim;
            const float *dah =
                da + t * dim + h * head_dim;

            float max_score = -FLT_MAX;
            float sum_exp = 0.0f;
            float mean_dp = 0.0f;

            for (size_t src = 0U;
                 src <= t;
                 ++src) {
                const float *khv =
                    k + src * kv_dim +
                    kh * head_dim;
                float score = 0.0f;

                for (size_t d = 0U;
                     d < head_dim;
                     ++d) {
                    score += qh[d] * khv[d];
                }

                score *= scale;
                probs[src] = score;

                if (score > max_score)
                    max_score = score;
            }

            for (size_t src = 0U;
                 src <= t;
                 ++src) {
                probs[src] =
                    expf(probs[src] - max_score);

                sum_exp += probs[src];
            }

            if (!(sum_exp > 0.0f) ||
                !isfinite(sum_exp)) {
                free(probs);
                free(dp);
                return 1;
            }

            for (size_t src = 0U;
                 src <= t;
                 ++src) {
                const float *vh =
                    v + src * kv_dim +
                    kh * head_dim;
                float dprob = 0.0f;

                probs[src] /= sum_exp;

                for (size_t d = 0U;
                     d < head_dim;
                     ++d) {
                    dprob += dah[d] * vh[d];
                }

                dp[src] = dprob;
                mean_dp +=
                    probs[src] * dprob;
            }

            for (size_t src = 0U;
                 src <= t;
                 ++src) {
                const float ds =
                    probs[src] *
                    (dp[src] - mean_dp);

                const float *khv =
                    k + src * kv_dim +
                    kh * head_dim;

                float *dkh =
                    dk + src * kv_dim +
                    kh * head_dim;

                float *dvh =
                    dv + src * kv_dim +
                    kh * head_dim;

                float *dqh =
                    dq + t * dim +
                    h * head_dim;

                for (size_t d = 0U;
                     d < head_dim;
                     ++d) {
                    dqh[d] +=
                        ds * scale * khv[d];

                    dkh[d] +=
                        ds * scale * qh[d];

                    dvh[d] +=
                        probs[src] * dah[d];
                }
            }
        }
    }

    free(probs);
    free(dp);
    return 0;
}

int main(void)
{
    const size_t tokens = 4U;
    const size_t n_heads = 2U;
    const size_t n_kv_heads = 1U;
    const size_t head_dim = 4U;
    const size_t dim =
        n_heads * head_dim;
    const size_t kv_dim =
        n_kv_heads * head_dim;
    const size_t q_count =
        tokens * dim;
    const size_t kv_count =
        tokens * kv_dim;

    NiyahModelConfig cfg = {0};
    NiyahModel model = {0};
    NiyahCudaTrainState ts = {0};

    cfg.vocab_size = 16U;
    cfg.context_length = 8U;
    cfg.embedding_dim = (uint32_t)dim;
    cfg.n_layers = 1U;
    cfg.n_heads = (uint32_t)n_heads;
    cfg.n_kv_heads = (uint32_t)n_kv_heads;
    cfg.ffn_hidden_dim = 16U;
    cfg.rms_norm_eps = 1.0e-5f;

    if (niyah_model_create(
            &model, &cfg) != NIYAH_OK ||
        niyah_model_reset_parameters(
            &model, 12345U) != NIYAH_OK) {
        return 1;
    }

    float *q =
        calloc(q_count, sizeof(float));
    float *k =
        calloc(kv_count, sizeof(float));
    float *v =
        calloc(kv_count, sizeof(float));
    float *da =
        calloc(q_count, sizeof(float));

    float *cpu_dq =
        calloc(q_count, sizeof(float));
    float *cpu_dk =
        calloc(kv_count, sizeof(float));
    float *cpu_dv =
        calloc(kv_count, sizeof(float));

    float *gpu_dq =
        calloc(q_count, sizeof(float));
    float *gpu_dk =
        calloc(kv_count, sizeof(float));
    float *gpu_dv =
        calloc(kv_count, sizeof(float));

    if (!q || !k || !v || !da ||
        !cpu_dq || !cpu_dk || !cpu_dv ||
        !gpu_dq || !gpu_dk || !gpu_dv) {
        return 1;
    }

    for (size_t i = 0U; i < q_count; ++i) {
        q[i] =
            ((int)(i % 19U) - 9) *
            0.03125f;

        da[i] =
            ((int)(i % 13U) - 6) *
            0.015625f;
    }

    for (size_t i = 0U; i < kv_count; ++i) {
        k[i] =
            ((int)(i % 17U) - 8) *
            0.0234375f;

        v[i] =
            ((int)(i % 11U) - 5) *
            0.02734375f;
    }

    if (cpu_attention_backward(
            cpu_dq,
            cpu_dk,
            cpu_dv,
            da,
            q,
            k,
            v,
            tokens,
            n_heads,
            n_kv_heads,
            head_dim) != 0) {
        return 1;
    }

    const size_t q_off = 0U;
    const size_t k_off = q_off + q_count;
    const size_t v_off = k_off + kv_count;
    const size_t da_off = v_off + kv_count;
    const size_t dq_off = da_off + q_count;
    const size_t dk_off = dq_off + q_count;
    const size_t dv_off = dk_off + kv_count;
    const size_t workspace_count =
        dv_off + kv_count;

    if (niyah_cuda_train_state_create(
            &ts,
            &model,
            tokens,
            workspace_count) != 0 ||
        niyah_cuda_train_state_copy_workspace_from_host(
            &ts, q_off, q, q_count) != 0 ||
        niyah_cuda_train_state_copy_workspace_from_host(
            &ts, k_off, k, kv_count) != 0 ||
        niyah_cuda_train_state_copy_workspace_from_host(
            &ts, v_off, v, kv_count) != 0 ||
        niyah_cuda_train_state_copy_workspace_from_host(
            &ts, da_off, da, q_count) != 0) {
        return 1;
    }

    if (niyah_cuda_train_attention_backward(
            &ts,
            q_off,
            k_off,
            v_off,
            da_off,
            dq_off,
            dk_off,
            dv_off,
            tokens,
            n_heads,
            n_kv_heads,
            head_dim) != 0 ||
        niyah_cuda_train_state_copy_workspace_to_host(
            &ts,
            dq_off,
            gpu_dq,
            q_count) != 0 ||
        niyah_cuda_train_state_copy_workspace_to_host(
            &ts,
            dk_off,
            gpu_dk,
            kv_count) != 0 ||
        niyah_cuda_train_state_copy_workspace_to_host(
            &ts,
            dv_off,
            gpu_dv,
            kv_count) != 0) {
        return 1;
    }

    const float e_dq =
        max_abs(cpu_dq, gpu_dq, q_count);
    const float e_dk =
        max_abs(cpu_dk, gpu_dk, kv_count);
    const float e_dv =
        max_abs(cpu_dv, gpu_dv, kv_count);

    printf(
        "attention_dq_max_abs=%.9g\n",
        e_dq);
    printf(
        "attention_dk_max_abs=%.9g\n",
        e_dk);
    printf(
        "attention_dv_max_abs=%.9g\n",
        e_dv);

    niyah_cuda_train_state_destroy(&ts);
    niyah_model_destroy(&model);

    free(q);
    free(k);
    free(v);
    free(da);

    free(cpu_dq);
    free(cpu_dk);
    free(cpu_dv);

    free(gpu_dq);
    free(gpu_dk);
    free(gpu_dv);

    return
        e_dq <= 2.0e-4f &&
        e_dk <= 2.0e-4f &&
        e_dv <= 2.0e-4f
            ? 0
            : 1;
}
