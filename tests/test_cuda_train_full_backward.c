#include "niyah/niyah.h"
#include "niyah/train.h"
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
        const float d =
            fabsf(a[i] - b[i]);

        if (d > m)
            m = d;
    }

    return m;
}

static int check_region(
    const char *case_name,
    const char *region,
    const float *cpu,
    const float *gpu,
    size_t offset,
    size_t count,
    float tolerance,
    float *overall)
{
    const float e =
        max_abs(
            cpu + offset,
            gpu + offset,
            count);

    printf(
        "%s_%s_max_abs=%.9g\n",
        case_name,
        region,
        e);

    if (e > *overall)
        *overall = e;

    return e <= tolerance ? 0 : 1;
}

static int run_case(
    const char *case_name,
    int tied,
    uint32_t n_segments,
    int use_segments,
    size_t loss_start)
{
    const size_t token_count = 4U;
    const uint32_t tokens[4] = {
        1U, 3U, 1U, 5U
    };
    const uint32_t targets[4] = {
        2U, 4U, 6U, 8U
    };
    const uint32_t segments[4] = {
        0U, 1U, 0U, 1U
    };

    const float tolerance =
        2.0e-4f;

    NiyahModelConfig cfg = {0};
    NiyahModel model = {0};
    NiyahModelGradients cpu_grad = {0};
    NiyahCudaModelState ms = {0};
    NiyahCudaTrainState ts = {0};

    float *cpu_workspace = NULL;
    float *gpu_grad = NULL;

    size_t cpu_workspace_count = 0U;
    size_t cuda_workspace_count = 0U;

    float cpu_loss = 0.0f;
    float gpu_loss = 0.0f;
    float overall = 0.0f;

    int failed = 0;

    cfg.vocab_size = 16U;
    cfg.context_length = 8U;
    cfg.embedding_dim = 8U;
    cfg.n_layers = 2U;
    cfg.n_heads = 2U;
    cfg.n_kv_heads = 1U;
    cfg.ffn_hidden_dim = 16U;
    cfg.rms_norm_eps = 1.0e-5f;
    cfg.tie_word_embeddings = tied;
    cfg.n_segments = n_segments;

    if (niyah_model_create(
            &model,
            &cfg) != NIYAH_OK ||
        niyah_model_reset_parameters(
            &model,
            12345U) != NIYAH_OK ||
        niyah_model_gradients_create(
            &cpu_grad,
            &model) != NIYAH_OK ||
        niyah_train_backward_workspace_floats(
            &cfg,
            token_count,
            &cpu_workspace_count) != NIYAH_OK ||
        niyah_cuda_train_backward_workspace_floats(
            &cfg,
            token_count,
            &cuda_workspace_count) != 0) {
        return 1;
    }

    cpu_workspace =
        calloc(
            cpu_workspace_count,
            sizeof(float));

    gpu_grad =
        calloc(
            model.weight_count,
            sizeof(float));

    if (!cpu_workspace ||
        !gpu_grad) {
        return 1;
    }

    if (use_segments) {
        if (niyah_train_backward_masked_with_segments(
                &model,
                tokens,
                targets,
                token_count,
                loss_start,
                segments,
                &cpu_loss,
                &cpu_grad,
                cpu_workspace,
                cpu_workspace_count) !=
            NIYAH_OK) {
            return 1;
        }
    } else {
        if (niyah_train_backward_masked(
                &model,
                tokens,
                targets,
                token_count,
                loss_start,
                &cpu_loss,
                &cpu_grad,
                cpu_workspace,
                cpu_workspace_count) !=
            NIYAH_OK) {
            return 1;
        }
    }

    if (niyah_cuda_model_state_create(
            &ms,
            &model) != 0 ||
        niyah_cuda_train_state_create(
            &ts,
            &model,
            token_count,
            cuda_workspace_count) != 0 ||
        niyah_cuda_train_backward_full(
            &ms,
            &ts,
            tokens,
            targets,
            token_count,
            loss_start,
            use_segments ? segments : NULL,
            &gpu_loss) != 0 ||
        niyah_cuda_train_state_copy_gradients_to_host(
            &ts,
            gpu_grad,
            model.weight_count) != 0) {
        return 1;
    }

    const float loss_error =
        fabsf(cpu_loss - gpu_loss);

    printf(
        "%s_loss_cpu=%.9g\n",
        case_name,
        cpu_loss);

    printf(
        "%s_loss_cuda=%.9g\n",
        case_name,
        gpu_loss);

    printf(
        "%s_loss_max_abs=%.9g\n",
        case_name,
        loss_error);

    if (loss_error > overall)
        overall = loss_error;

    if (loss_error > tolerance)
        failed = 1;

    {
        const size_t embedding_count =
            (size_t)cfg.vocab_size *
            (size_t)cfg.embedding_dim;

        failed |= check_region(
            case_name,
            tied
                ? "embedding_lm_head_shared"
                : "token_embedding",
            cpu_grad.values,
            gpu_grad,
            model.layout.token_embedding,
            embedding_count,
            tolerance,
            &overall);
    }

    if (use_segments) {
        const size_t segment_count =
            (size_t)cfg.n_segments *
            (size_t)cfg.embedding_dim;

        failed |= check_region(
            case_name,
            "segment_embedding",
            cpu_grad.values,
            gpu_grad,
            model.layout.segment_embedding,
            segment_count,
            tolerance,
            &overall);
    }

    for (uint32_t layer_index = 0U;
         layer_index < cfg.n_layers;
         ++layer_index) {
        NiyahLayerLayout layer = {0};
        char name[64];

        if (niyah_model_layer_layout(
                &cfg,
                &model.layout,
                layer_index,
                &layer) != NIYAH_OK) {
            return 1;
        }

        snprintf(
            name,
            sizeof(name),
            "layer%u_attn_norm",
            layer_index);

        failed |= check_region(
            case_name,
            name,
            cpu_grad.values,
            gpu_grad,
            layer.attn_norm,
            cfg.embedding_dim,
            tolerance,
            &overall);

        snprintf(
            name,
            sizeof(name),
            "layer%u_wq",
            layer_index);

        failed |= check_region(
            case_name,
            name,
            cpu_grad.values,
            gpu_grad,
            layer.wq,
            (size_t)cfg.embedding_dim *
                cfg.embedding_dim,
            tolerance,
            &overall);

        snprintf(
            name,
            sizeof(name),
            "layer%u_wk",
            layer_index);

        failed |= check_region(
            case_name,
            name,
            cpu_grad.values,
            gpu_grad,
            layer.wk,
            model.layout.kv_dim *
                cfg.embedding_dim,
            tolerance,
            &overall);

        snprintf(
            name,
            sizeof(name),
            "layer%u_wv",
            layer_index);

        failed |= check_region(
            case_name,
            name,
            cpu_grad.values,
            gpu_grad,
            layer.wv,
            model.layout.kv_dim *
                cfg.embedding_dim,
            tolerance,
            &overall);

        snprintf(
            name,
            sizeof(name),
            "layer%u_wo",
            layer_index);

        failed |= check_region(
            case_name,
            name,
            cpu_grad.values,
            gpu_grad,
            layer.wo,
            (size_t)cfg.embedding_dim *
                cfg.embedding_dim,
            tolerance,
            &overall);

        snprintf(
            name,
            sizeof(name),
            "layer%u_ffn_norm",
            layer_index);

        failed |= check_region(
            case_name,
            name,
            cpu_grad.values,
            gpu_grad,
            layer.ffn_norm,
            cfg.embedding_dim,
            tolerance,
            &overall);

        snprintf(
            name,
            sizeof(name),
            "layer%u_w_gate",
            layer_index);

        failed |= check_region(
            case_name,
            name,
            cpu_grad.values,
            gpu_grad,
            layer.w_gate,
            (size_t)cfg.ffn_hidden_dim *
                cfg.embedding_dim,
            tolerance,
            &overall);

        snprintf(
            name,
            sizeof(name),
            "layer%u_w_up",
            layer_index);

        failed |= check_region(
            case_name,
            name,
            cpu_grad.values,
            gpu_grad,
            layer.w_up,
            (size_t)cfg.ffn_hidden_dim *
                cfg.embedding_dim,
            tolerance,
            &overall);

        snprintf(
            name,
            sizeof(name),
            "layer%u_w_down",
            layer_index);

        failed |= check_region(
            case_name,
            name,
            cpu_grad.values,
            gpu_grad,
            layer.w_down,
            (size_t)cfg.embedding_dim *
                cfg.ffn_hidden_dim,
            tolerance,
            &overall);
    }

    failed |= check_region(
        case_name,
        "final_norm",
        cpu_grad.values,
        gpu_grad,
        model.layout.final_norm,
        cfg.embedding_dim,
        tolerance,
        &overall);

    if (!tied) {
        failed |= check_region(
            case_name,
            "lm_head",
            cpu_grad.values,
            gpu_grad,
            model.layout.lm_head,
            (size_t)cfg.vocab_size *
                cfg.embedding_dim,
            tolerance,
            &overall);
    }

    {
        const float full_error =
            max_abs(
                cpu_grad.values,
                gpu_grad,
                model.weight_count);

        printf(
            "%s_full_gradient_max_abs=%.9g\n",
            case_name,
            full_error);

        if (full_error > overall)
            overall = full_error;

        if (full_error > tolerance)
            failed = 1;
    }

    printf(
        "%s_overall_max_abs=%.9g\n",
        case_name,
        overall);

    niyah_cuda_train_state_destroy(&ts);
    niyah_cuda_model_state_destroy(&ms);
    niyah_model_gradients_destroy(&cpu_grad);
    niyah_model_destroy(&model);

    free(cpu_workspace);
    free(gpu_grad);

    return failed ? 1 : 0;
}

int main(void)
{
    if (run_case(
            "untied",
            0,
            0U,
            0,
            0U) != 0) {
        return 1;
    }

    if (run_case(
            "tied_segment_masked",
            1,
            2U,
            1,
            1U) != 0) {
        return 1;
    }

    return 0;
}
