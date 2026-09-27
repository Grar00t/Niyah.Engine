#include "niyah/niyah.h"
#include "niyah_cuda_matvec.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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

static float silu(float x)
{
    return x / (1.0f + expf(-x));
}

static float silu_derivative(float x)
{
    const float s =
        1.0f / (1.0f + expf(-x));

    return s + x * s * (1.0f - s);
}

static void rmsnorm_forward(
    float *out,
    const float *x,
    const float *weight,
    size_t width,
    float eps)
{
    double sum_sq = 0.0;

    for (size_t i = 0U; i < width; ++i) {
        const double xv = (double)x[i];
        sum_sq += xv * xv;
    }

    const float inv =
        1.0f /
        sqrtf(
            (float)(sum_sq / (double)width) +
            eps);

    for (size_t i = 0U; i < width; ++i)
        out[i] = x[i] * inv * weight[i];
}

static void rmsnorm_backward(
    float *dx,
    float *dw,
    const float *dy,
    const float *x,
    const float *weight,
    size_t width,
    float eps)
{
    double sum_sq = 0.0;
    double dot = 0.0;

    for (size_t i = 0U; i < width; ++i) {
        const double xv = (double)x[i];
        sum_sq += xv * xv;
    }

    const float inv =
        1.0f /
        sqrtf(
            (float)(sum_sq / (double)width) +
            eps);

    for (size_t i = 0U; i < width; ++i) {
        dot +=
            (double)dy[i] *
            (double)weight[i] *
            (double)x[i];

        dw[i] +=
            dy[i] * x[i] * inv;
    }

    const float coeff =
        inv * inv * inv *
        (float)(dot / (double)width);

    for (size_t i = 0U; i < width; ++i) {
        dx[i] =
            dy[i] * weight[i] * inv -
            x[i] * coeff;
    }
}

static void linear_forward(
    float *out,
    const float *w,
    const float *x,
    size_t rows,
    size_t cols)
{
    for (size_t r = 0U; r < rows; ++r) {
        float v = 0.0f;

        for (size_t c = 0U; c < cols; ++c)
            v += w[r * cols + c] * x[c];

        out[r] = v;
    }
}

int main(void)
{
    const size_t tokens = 4U;
    const size_t dim = 8U;
    const size_t ffn = 16U;
    const size_t td = tokens * dim;
    const size_t tf = tokens * ffn;
    const size_t down_count = dim * ffn;
    const size_t gu_count = ffn * dim;

    NiyahModelConfig cfg = {0};
    NiyahModel model = {0};
    NiyahLayerLayout layer = {0};
    NiyahCudaModelState ms = {0};
    NiyahCudaTrainState ts = {0};

    cfg.vocab_size = 16U;
    cfg.context_length = 8U;
    cfg.embedding_dim = (uint32_t)dim;
    cfg.n_layers = 1U;
    cfg.n_heads = 2U;
    cfg.n_kv_heads = 1U;
    cfg.ffn_hidden_dim = (uint32_t)ffn;
    cfg.rms_norm_eps = 1.0e-5f;

    if (niyah_model_create(
            &model, &cfg) != NIYAH_OK ||
        niyah_model_reset_parameters(
            &model, 12345U) != NIYAH_OK ||
        niyah_model_layer_layout(
            &cfg,
            &model.layout,
            0U,
            &layer) != NIYAH_OK) {
        return 1;
    }

    const float *w_norm =
        model.weights + layer.ffn_norm;
    const float *w_gate =
        model.weights + layer.w_gate;
    const float *w_up =
        model.weights + layer.w_up;
    const float *w_down =
        model.weights + layer.w_down;

    float *hidden = calloc(td, sizeof(float));
    float *norm2 = calloc(td, sizeof(float));
    float *gate = calloc(tf, sizeof(float));
    float *up = calloc(tf, sizeof(float));
    float *act = calloc(tf, sizeof(float));
    float *dy = calloc(td, sizeof(float));
    float *dh_base = calloc(td, sizeof(float));

    float *cpu_dact = calloc(tf, sizeof(float));
    float *cpu_dg = calloc(tf, sizeof(float));
    float *cpu_du = calloc(tf, sizeof(float));
    float *cpu_dn2 = calloc(td, sizeof(float));
    float *cpu_dtmp = calloc(td, sizeof(float));
    float *cpu_dh = calloc(td, sizeof(float));

    float *cpu_dw_down =
        calloc(down_count, sizeof(float));
    float *cpu_dw_gate =
        calloc(gu_count, sizeof(float));
    float *cpu_dw_up =
        calloc(gu_count, sizeof(float));
    float *cpu_dw_norm =
        calloc(dim, sizeof(float));

    float *gpu_dw_down =
        calloc(down_count, sizeof(float));
    float *gpu_dw_gate =
        calloc(gu_count, sizeof(float));
    float *gpu_dw_up =
        calloc(gu_count, sizeof(float));
    float *gpu_dw_norm =
        calloc(dim, sizeof(float));
    float *gpu_dh =
        calloc(td, sizeof(float));

    if (!hidden || !norm2 || !gate || !up ||
        !act || !dy || !dh_base ||
        !cpu_dact || !cpu_dg || !cpu_du ||
        !cpu_dn2 || !cpu_dtmp || !cpu_dh ||
        !cpu_dw_down || !cpu_dw_gate ||
        !cpu_dw_up || !cpu_dw_norm ||
        !gpu_dw_down || !gpu_dw_gate ||
        !gpu_dw_up || !gpu_dw_norm ||
        !gpu_dh) {
        return 1;
    }

    for (size_t i = 0U; i < td; ++i) {
        hidden[i] =
            ((int)(i % 17U) - 8) *
            0.03125f;

        dy[i] =
            ((int)(i % 13U) - 6) *
            0.015625f;

        dh_base[i] =
            ((int)(i % 11U) - 5) *
            0.0234375f;
    }

    for (size_t t = 0U; t < tokens; ++t) {
        rmsnorm_forward(
            norm2 + t * dim,
            hidden + t * dim,
            w_norm,
            dim,
            cfg.rms_norm_eps);

        linear_forward(
            gate + t * ffn,
            w_gate,
            norm2 + t * dim,
            ffn,
            dim);

        linear_forward(
            up + t * ffn,
            w_up,
            norm2 + t * dim,
            ffn,
            dim);

        for (size_t i = 0U; i < ffn; ++i) {
            act[t * ffn + i] =
                silu(gate[t * ffn + i]) *
                up[t * ffn + i];
        }
    }

    for (size_t t = 0U; t < tokens; ++t) {
        const float *dy_t =
            dy + t * dim;
        const float *act_t =
            act + t * ffn;
        const float *norm_t =
            norm2 + t * dim;

        for (size_t r = 0U; r < dim; ++r) {
            for (size_t c = 0U; c < ffn; ++c) {
                cpu_dw_down[r * ffn + c] +=
                    dy_t[r] * act_t[c];
            }
        }

        for (size_t i = 0U; i < ffn; ++i) {
            float dact = 0.0f;

            for (size_t r = 0U; r < dim; ++r) {
                dact +=
                    w_down[r * ffn + i] *
                    dy_t[r];
            }

            cpu_dact[t * ffn + i] = dact;

            const float g =
                gate[t * ffn + i];
            const float u =
                up[t * ffn + i];

            const float dg =
                dact * u *
                silu_derivative(g);

            const float du =
                dact * silu(g);

            cpu_dg[t * ffn + i] = dg;
            cpu_du[t * ffn + i] = du;

            for (size_t c = 0U; c < dim; ++c) {
                cpu_dw_gate[i * dim + c] +=
                    dg * norm_t[c];

                cpu_dw_up[i * dim + c] +=
                    du * norm_t[c];

                cpu_dn2[t * dim + c] +=
                    w_gate[i * dim + c] * dg +
                    w_up[i * dim + c] * du;
            }
        }
    }

    for (size_t t = 0U; t < tokens; ++t) {
        rmsnorm_backward(
            cpu_dtmp + t * dim,
            cpu_dw_norm,
            cpu_dn2 + t * dim,
            hidden + t * dim,
            w_norm,
            dim,
            cfg.rms_norm_eps);

        for (size_t i = 0U; i < dim; ++i) {
            cpu_dh[t * dim + i] =
                dh_base[t * dim + i] +
                cpu_dtmp[t * dim + i];
        }
    }

    const size_t hidden_off = 0U;
    const size_t norm_off = hidden_off + td;
    const size_t gate_off = norm_off + td;
    const size_t up_off = gate_off + tf;
    const size_t act_off = up_off + tf;
    const size_t dy_off = act_off + tf;
    const size_t dact_off = dy_off + td;
    const size_t dg_off = dact_off + tf;
    const size_t du_off = dg_off + tf;
    const size_t dn2_gate_off = du_off + tf;
    const size_t dn2_up_off = dn2_gate_off + td;
    const size_t dn2_off = dn2_up_off + td;
    const size_t dtmp_off = dn2_off + td;
    const size_t dh_off = dtmp_off + td;
    const size_t workspace_count = dh_off + td;

    if (niyah_cuda_model_state_create(
            &ms, &model) != 0 ||
        niyah_cuda_train_state_create(
            &ts,
            &model,
            tokens,
            workspace_count) != 0 ||
        niyah_cuda_train_state_zero_gradients(
            &ts) != 0 ||
        niyah_cuda_train_state_copy_workspace_from_host(
            &ts, hidden_off, hidden, td) != 0 ||
        niyah_cuda_train_state_copy_workspace_from_host(
            &ts, norm_off, norm2, td) != 0 ||
        niyah_cuda_train_state_copy_workspace_from_host(
            &ts, gate_off, gate, tf) != 0 ||
        niyah_cuda_train_state_copy_workspace_from_host(
            &ts, up_off, up, tf) != 0 ||
        niyah_cuda_train_state_copy_workspace_from_host(
            &ts, act_off, act, tf) != 0 ||
        niyah_cuda_train_state_copy_workspace_from_host(
            &ts, dy_off, dy, td) != 0 ||
        niyah_cuda_train_state_copy_workspace_from_host(
            &ts, dh_off, dh_base, td) != 0) {
        return 1;
    }

    if (niyah_cuda_train_linear_backward(
            &ms,
            &ts,
            layer.w_down,
            act_off,
            dy_off,
            dact_off,
            tokens,
            dim,
            ffn) != 0 ||
        niyah_cuda_train_silu_mul_backward(
            &ts,
            gate_off,
            up_off,
            dact_off,
            dg_off,
            du_off,
            tf) != 0 ||
        niyah_cuda_train_linear_backward(
            &ms,
            &ts,
            layer.w_gate,
            norm_off,
            dg_off,
            dn2_gate_off,
            tokens,
            ffn,
            dim) != 0 ||
        niyah_cuda_train_linear_backward(
            &ms,
            &ts,
            layer.w_up,
            norm_off,
            du_off,
            dn2_up_off,
            tokens,
            ffn,
            dim) != 0 ||
        niyah_cuda_train_add(
            &ts,
            dn2_off,
            dn2_gate_off,
            dn2_up_off,
            td) != 0 ||
        niyah_cuda_train_rmsnorm_backward(
            &ms,
            &ts,
            layer.ffn_norm,
            hidden_off,
            dn2_off,
            dtmp_off,
            tokens,
            dim,
            cfg.rms_norm_eps) != 0 ||
        niyah_cuda_train_add_inplace(
            &ts,
            dh_off,
            dtmp_off,
            td) != 0) {
        return 1;
    }

    if (niyah_cuda_train_state_copy_gradient_range_to_host(
            &ts,
            layer.w_down,
            gpu_dw_down,
            down_count) != 0 ||
        niyah_cuda_train_state_copy_gradient_range_to_host(
            &ts,
            layer.w_gate,
            gpu_dw_gate,
            gu_count) != 0 ||
        niyah_cuda_train_state_copy_gradient_range_to_host(
            &ts,
            layer.w_up,
            gpu_dw_up,
            gu_count) != 0 ||
        niyah_cuda_train_state_copy_gradient_range_to_host(
            &ts,
            layer.ffn_norm,
            gpu_dw_norm,
            dim) != 0 ||
        niyah_cuda_train_state_copy_workspace_to_host(
            &ts,
            dh_off,
            gpu_dh,
            td) != 0) {
        return 1;
    }

    const float e_down =
        max_abs(
            cpu_dw_down,
            gpu_dw_down,
            down_count);

    const float e_gate =
        max_abs(
            cpu_dw_gate,
            gpu_dw_gate,
            gu_count);

    const float e_up =
        max_abs(
            cpu_dw_up,
            gpu_dw_up,
            gu_count);

    const float e_norm =
        max_abs(
            cpu_dw_norm,
            gpu_dw_norm,
            dim);

    const float e_dh =
        max_abs(
            cpu_dh,
            gpu_dh,
            td);

    printf(
        "ffn_w_down_max_abs=%.9g\n",
        e_down);
    printf(
        "ffn_w_gate_max_abs=%.9g\n",
        e_gate);
    printf(
        "ffn_w_up_max_abs=%.9g\n",
        e_up);
    printf(
        "ffn_norm_max_abs=%.9g\n",
        e_norm);
    printf(
        "ffn_dh_attn_max_abs=%.9g\n",
        e_dh);

    niyah_cuda_train_state_destroy(&ts);
    niyah_cuda_model_state_destroy(&ms);
    niyah_model_destroy(&model);

    free(hidden);
    free(norm2);
    free(gate);
    free(up);
    free(act);
    free(dy);
    free(dh_base);

    free(cpu_dact);
    free(cpu_dg);
    free(cpu_du);
    free(cpu_dn2);
    free(cpu_dtmp);
    free(cpu_dh);

    free(cpu_dw_down);
    free(cpu_dw_gate);
    free(cpu_dw_up);
    free(cpu_dw_norm);

    free(gpu_dw_down);
    free(gpu_dw_gate);
    free(gpu_dw_up);
    free(gpu_dw_norm);
    free(gpu_dh);

    return
        e_down <= 2.0e-4f &&
        e_gate <= 2.0e-4f &&
        e_up <= 2.0e-4f &&
        e_norm <= 2.0e-4f &&
        e_dh <= 2.0e-4f
            ? 0
            : 1;
}
