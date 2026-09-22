#include "niyah/optimizer.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct NiyahAdamWDecayPlan {
    size_t total;
    size_t layers_start;
    size_t layer_stride;
    size_t n_layers;
    size_t dim;
    size_t attn_norm_relative;
    size_t ffn_norm_relative;
    size_t final_norm;
    size_t final_norm_end;
} NiyahAdamWDecayPlan;

typedef struct NiyahAdamWCandidate {
    double m;
    double v;
    double weight;
} NiyahAdamWCandidate;

static int niyah_size_add_ok(size_t a, size_t b, size_t *out)
{
    if (out == NULL || a > SIZE_MAX - b) {
        return 0;
    }
    *out = a + b;
    return 1;
}

static int niyah_size_mul_ok(size_t a, size_t b, size_t *out)
{
    if (out == NULL || (a != 0U && b > SIZE_MAX / a)) {
        return 0;
    }
    *out = a * b;
    return 1;
}

static int niyah_config_equal(const NiyahModelConfig *a,
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
           a->tie_word_embeddings == b->tie_word_embeddings &&
           a->n_segments == b->n_segments;
}

static int niyah_layout_equal(const NiyahModelLayout *a,
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

static NiyahStatus niyah_adamw_validate_model(const NiyahModel *model,
                                               NiyahModelLayout *canonical_layout)
{
    NiyahModelLayout canonical;
    NiyahStatus status;

    if (model == NULL || model->weights == NULL || model->weight_count == 0U) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    status = niyah_model_layout_compute(&model->config, &canonical);
    if (status != NIYAH_OK) {
        return status;
    }
    if (!niyah_layout_equal(&model->layout, &canonical) ||
        model->weight_count != canonical.total_floats) {
        return NIYAH_ERR_INVALID_CONFIG;
    }
    if (canonical_layout != NULL) {
        *canonical_layout = canonical;
    }
    return NIYAH_OK;
}

NiyahStatus niyah_adamw_linear_warmup_learning_rate(
    float base_learning_rate,
    uint64_t optimizer_step,
    uint64_t warmup_steps,
    float *out_learning_rate)
{
    double scaled;
    float effective;

    if (out_learning_rate == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    *out_learning_rate = 0.0f;

    if (!isfinite(base_learning_rate) ||
        base_learning_rate <= 0.0f ||
        optimizer_step == UINT64_C(0)) {
        return NIYAH_ERR_INVALID_CONFIG;
    }

    if (warmup_steps == UINT64_C(0) ||
        optimizer_step >= warmup_steps) {
        *out_learning_rate = base_learning_rate;
        return NIYAH_OK;
    }

    scaled =
        (double)base_learning_rate *
        ((double)optimizer_step / (double)warmup_steps);

    effective = (float)scaled;

    if (!isfinite(effective) || effective <= 0.0f) {
        return NIYAH_ERR_OVERFLOW;
    }

    *out_learning_rate = effective;
    return NIYAH_OK;
}

NiyahStatus niyah_adamw_config_validate(const NiyahAdamWConfig *config)
{
    if (config == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    if (!isfinite(config->learning_rate) || config->learning_rate <= 0.0f ||
        !isfinite(config->beta1) || config->beta1 < 0.0f || config->beta1 >= 1.0f ||
        !isfinite(config->beta2) || config->beta2 < 0.0f || config->beta2 >= 1.0f ||
        !isfinite(config->epsilon) || config->epsilon <= 0.0f ||
        !isfinite(config->weight_decay) || config->weight_decay < 0.0f ||
        !isfinite(config->max_grad_norm) || config->max_grad_norm <= 0.0f) {
        return NIYAH_ERR_INVALID_CONFIG;
    }
    return NIYAH_OK;
}

NiyahStatus niyah_adamw_state_create(NiyahAdamWState *state,
                                     const NiyahModel *model)
{
    NiyahModelLayout canonical;
    size_t bytes = 0U;
    NiyahStatus status;

    if (state == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    memset(state, 0, sizeof(*state));

    status = niyah_adamw_validate_model(model, &canonical);
    if (status != NIYAH_OK) {
        return status;
    }
    if (!niyah_size_mul_ok(model->weight_count, sizeof(float), &bytes)) {
        return NIYAH_ERR_OVERFLOW;
    }

    state->m = (float *)calloc(1U, bytes);
    if (state->m == NULL) {
        return NIYAH_ERR_OUT_OF_MEMORY;
    }
    state->v = (float *)calloc(1U, bytes);
    if (state->v == NULL) {
        free(state->m);
        memset(state, 0, sizeof(*state));
        return NIYAH_ERR_OUT_OF_MEMORY;
    }

    state->count = model->weight_count;
    state->step = 0U;
    state->bound_model = model;
    state->bound_weights = model->weights;
    state->model_config = model->config;
    state->model_layout = canonical;
    return NIYAH_OK;
}

void niyah_adamw_state_destroy(NiyahAdamWState *state)
{
    if (state == NULL) {
        return;
    }
    free(state->m);
    free(state->v);
    memset(state, 0, sizeof(*state));
}

NiyahStatus niyah_model_gradients_global_l2_norm(const NiyahModelGradients *gradients,
                                                 double *out_norm)
{
    double scale = 0.0;
    double sumsq = 1.0;
    size_t i;

    if (gradients == NULL || gradients->values == NULL ||
        gradients->count == 0U || out_norm == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    for (i = 0U; i < gradients->count; ++i) {
        const float value = gradients->values[i];
        double magnitude;
        if (!isfinite(value)) {
            return NIYAH_ERR_INVALID_ARGUMENT;
        }
        magnitude = fabs((double)value);
        if (magnitude != 0.0) {
            if (scale < magnitude) {
                const double ratio = scale / magnitude;
                sumsq = 1.0 + sumsq * ratio * ratio;
                scale = magnitude;
            } else {
                const double ratio = magnitude / scale;
                sumsq += ratio * ratio;
            }
        }
    }

    *out_norm = scale == 0.0 ? 0.0 : scale * sqrt(sumsq);
    if (!isfinite(*out_norm)) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    return NIYAH_OK;
}

static int niyah_bias_correction(float beta, uint64_t step, double *out)
{
    double correction;

    if (out == NULL || step == 0U) {
        return 0;
    }
    if (beta == 0.0f) {
        *out = 1.0;
        return 1;
    }

    correction = -expm1((double)step * log((double)beta));
    if (!isfinite(correction) || correction <= 0.0 || correction > 1.0) {
        return 0;
    }
    *out = correction;
    return 1;
}

static int niyah_double_fits_float(double value)
{
    float converted;
    if (!isfinite(value) || value > (double)FLT_MAX || value < -(double)FLT_MAX) {
        return 0;
    }
    converted = (float)value;
    return isfinite(converted);
}

static int niyah_range_bounds(const void *pointer,
                              size_t bytes,
                              uintptr_t *start,
                              uintptr_t *last)
{
    const uintptr_t address = (uintptr_t)pointer;
    if (pointer == NULL || bytes == 0U || start == NULL || last == NULL ||
        address > UINTPTR_MAX - (uintptr_t)(bytes - 1U)) {
        return 0;
    }
    *start = address;
    *last = address + (uintptr_t)(bytes - 1U);
    return 1;
}

static int niyah_ranges_overlap(const void *a,
                                size_t a_bytes,
                                const void *b,
                                size_t b_bytes)
{
    uintptr_t a_start;
    uintptr_t a_last;
    uintptr_t b_start;
    uintptr_t b_last;

    if (!niyah_range_bounds(a, a_bytes, &a_start, &a_last) ||
        !niyah_range_bounds(b, b_bytes, &b_start, &b_last)) {
        return 1;
    }
    return a_start <= b_last && b_start <= a_last;
}

static NiyahStatus niyah_adamw_decay_plan(const NiyahModel *model,
                                           const NiyahModelLayout *layout,
                                           NiyahAdamWDecayPlan *plan)
{
    NiyahLayerLayout layer0;
    size_t layers_bytes = 0U;
    size_t expected_final = 0U;
    size_t attn_end = 0U;
    size_t ffn_end = 0U;
    size_t final_end = 0U;
    NiyahStatus status;

    if (model == NULL || layout == NULL || plan == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    status = niyah_model_layer_layout(&model->config, layout, 0U, &layer0);
    if (status != NIYAH_OK) {
        return status;
    }
    if (layer0.attn_norm < layout->layers || layer0.ffn_norm < layout->layers) {
        return NIYAH_ERR_INVALID_CONFIG;
    }

    memset(plan, 0, sizeof(*plan));
    plan->total = layout->total_floats;
    plan->layers_start = layout->layers;
    plan->layer_stride = layout->layer_stride;
    plan->n_layers = (size_t)model->config.n_layers;
    plan->dim = (size_t)model->config.embedding_dim;
    plan->attn_norm_relative = layer0.attn_norm - layout->layers;
    plan->ffn_norm_relative = layer0.ffn_norm - layout->layers;
    plan->final_norm = layout->final_norm;

    if (!niyah_size_add_ok(plan->attn_norm_relative, plan->dim, &attn_end) ||
        !niyah_size_add_ok(plan->ffn_norm_relative, plan->dim, &ffn_end) ||
        !niyah_size_mul_ok(plan->n_layers, plan->layer_stride, &layers_bytes) ||
        !niyah_size_add_ok(plan->layers_start, layers_bytes, &expected_final) ||
        !niyah_size_add_ok(plan->final_norm, plan->dim, &final_end)) {
        return NIYAH_ERR_OVERFLOW;
    }
    plan->final_norm_end = final_end;

    if (attn_end > plan->ffn_norm_relative ||
        ffn_end > plan->layer_stride ||
        expected_final != plan->final_norm ||
        plan->final_norm_end > plan->total) {
        return NIYAH_ERR_INVALID_CONFIG;
    }
    if (model->config.tie_word_embeddings != 0) {
        if (layout->lm_head != layout->token_embedding ||
            plan->final_norm_end != plan->total) {
            return NIYAH_ERR_INVALID_CONFIG;
        }
    } else if (layout->lm_head != plan->final_norm_end || layout->lm_head >= plan->total) {
        return NIYAH_ERR_INVALID_CONFIG;
    }
    return NIYAH_OK;
}

static void niyah_adamw_candidate(float weight,
                                  float old_m,
                                  float old_v,
                                  float gradient,
                                  double clip_scale,
                                  double correction1,
                                  double correction2,
                                  const NiyahAdamWConfig *config,
                                  int decay_enabled,
                                  NiyahAdamWCandidate *candidate)
{
    const double beta1 = (double)config->beta1;
    const double beta2 = (double)config->beta2;
    const double lr = (double)config->learning_rate;
    const double epsilon = (double)config->epsilon;
    const double g = (double)gradient * clip_scale;
    const double m = beta1 * (double)old_m + (1.0 - beta1) * g;
    const double v = beta2 * (double)old_v + (1.0 - beta2) * g * g;
    const double m_hat = m / correction1;
    const double v_hat = v / correction2;
    const double decay_factor = decay_enabled != 0
        ? 1.0 - lr * (double)config->weight_decay
        : 1.0;

    candidate->m = m;
    candidate->v = v;
    candidate->weight = (double)weight * decay_factor -
                        lr * m_hat / (sqrt(v_hat) + epsilon);
}

static NiyahStatus niyah_adamw_validate_span(const NiyahModel *model,
                                              const NiyahModelGradients *gradients,
                                              const NiyahAdamWState *state,
                                              const NiyahAdamWConfig *config,
                                              size_t begin,
                                              size_t end,
                                              int decay_enabled,
                                              double clip_scale,
                                              double correction1,
                                              double correction2)
{
    size_t i;

    if (begin > end || end > state->count) {
        return NIYAH_ERR_INVALID_CONFIG;
    }
    for (i = begin; i < end; ++i) {
        NiyahAdamWCandidate candidate;
        if (!isfinite(model->weights[i]) || !isfinite(state->m[i]) ||
            !isfinite(state->v[i]) || state->v[i] < 0.0f) {
            return NIYAH_ERR_INVALID_CONFIG;
        }
        niyah_adamw_candidate(model->weights[i], state->m[i], state->v[i],
                              gradients->values[i], clip_scale,
                              correction1, correction2, config,
                              decay_enabled, &candidate);
        if (candidate.v < 0.0) {
            return NIYAH_ERR_INVALID_CONFIG;
        }
        if (!niyah_double_fits_float(candidate.m) ||
            !niyah_double_fits_float(candidate.v) ||
            !niyah_double_fits_float(candidate.weight)) {
            return NIYAH_ERR_OVERFLOW;
        }
    }
    return NIYAH_OK;
}

static void niyah_adamw_commit_span(NiyahModel *model,
                                    const NiyahModelGradients *gradients,
                                    NiyahAdamWState *state,
                                    const NiyahAdamWConfig *config,
                                    size_t begin,
                                    size_t end,
                                    int decay_enabled,
                                    double clip_scale,
                                    double correction1,
                                    double correction2)
{
    size_t i;
    for (i = begin; i < end; ++i) {
        NiyahAdamWCandidate candidate;
        niyah_adamw_candidate(model->weights[i], state->m[i], state->v[i],
                              gradients->values[i], clip_scale,
                              correction1, correction2, config,
                              decay_enabled, &candidate);
        state->m[i] = (float)candidate.m;
        state->v[i] = (float)candidate.v;
        model->weights[i] = (float)candidate.weight;
    }
}

static NiyahStatus niyah_adamw_validate_plan(const NiyahModel *model,
                                              const NiyahModelGradients *gradients,
                                              const NiyahAdamWState *state,
                                              const NiyahAdamWConfig *config,
                                              const NiyahAdamWDecayPlan *plan,
                                              double clip_scale,
                                              double correction1,
                                              double correction2)
{
    size_t layer_index;
    NiyahStatus status;

    status = niyah_adamw_validate_span(model, gradients, state, config,
                                       0U, plan->layers_start, 1,
                                       clip_scale, correction1, correction2);
    if (status != NIYAH_OK) {
        return status;
    }

    for (layer_index = 0U; layer_index < plan->n_layers; ++layer_index) {
        const size_t base = plan->layers_start + layer_index * plan->layer_stride;
        const size_t layer_end = base + plan->layer_stride;
        const size_t attn_start = base + plan->attn_norm_relative;
        const size_t attn_end = attn_start + plan->dim;
        const size_t ffn_start = base + plan->ffn_norm_relative;
        const size_t ffn_end = ffn_start + plan->dim;

        status = niyah_adamw_validate_span(model, gradients, state, config,
                                           base, attn_start, 1,
                                           clip_scale, correction1, correction2);
        if (status != NIYAH_OK) return status;
        status = niyah_adamw_validate_span(model, gradients, state, config,
                                           attn_start, attn_end, 0,
                                           clip_scale, correction1, correction2);
        if (status != NIYAH_OK) return status;
        status = niyah_adamw_validate_span(model, gradients, state, config,
                                           attn_end, ffn_start, 1,
                                           clip_scale, correction1, correction2);
        if (status != NIYAH_OK) return status;
        status = niyah_adamw_validate_span(model, gradients, state, config,
                                           ffn_start, ffn_end, 0,
                                           clip_scale, correction1, correction2);
        if (status != NIYAH_OK) return status;
        status = niyah_adamw_validate_span(model, gradients, state, config,
                                           ffn_end, layer_end, 1,
                                           clip_scale, correction1, correction2);
        if (status != NIYAH_OK) return status;
    }

    status = niyah_adamw_validate_span(model, gradients, state, config,
                                       plan->final_norm, plan->final_norm_end, 0,
                                       clip_scale, correction1, correction2);
    if (status != NIYAH_OK) {
        return status;
    }
    return niyah_adamw_validate_span(model, gradients, state, config,
                                     plan->final_norm_end, plan->total, 1,
                                     clip_scale, correction1, correction2);
}

static void niyah_adamw_commit_plan(NiyahModel *model,
                                    const NiyahModelGradients *gradients,
                                    NiyahAdamWState *state,
                                    const NiyahAdamWConfig *config,
                                    const NiyahAdamWDecayPlan *plan,
                                    double clip_scale,
                                    double correction1,
                                    double correction2)
{
    size_t layer_index;

    niyah_adamw_commit_span(model, gradients, state, config,
                            0U, plan->layers_start, 1,
                            clip_scale, correction1, correction2);

    for (layer_index = 0U; layer_index < plan->n_layers; ++layer_index) {
        const size_t base = plan->layers_start + layer_index * plan->layer_stride;
        const size_t layer_end = base + plan->layer_stride;
        const size_t attn_start = base + plan->attn_norm_relative;
        const size_t attn_end = attn_start + plan->dim;
        const size_t ffn_start = base + plan->ffn_norm_relative;
        const size_t ffn_end = ffn_start + plan->dim;

        niyah_adamw_commit_span(model, gradients, state, config,
                                base, attn_start, 1,
                                clip_scale, correction1, correction2);
        niyah_adamw_commit_span(model, gradients, state, config,
                                attn_start, attn_end, 0,
                                clip_scale, correction1, correction2);
        niyah_adamw_commit_span(model, gradients, state, config,
                                attn_end, ffn_start, 1,
                                clip_scale, correction1, correction2);
        niyah_adamw_commit_span(model, gradients, state, config,
                                ffn_start, ffn_end, 0,
                                clip_scale, correction1, correction2);
        niyah_adamw_commit_span(model, gradients, state, config,
                                ffn_end, layer_end, 1,
                                clip_scale, correction1, correction2);
    }

    niyah_adamw_commit_span(model, gradients, state, config,
                            plan->final_norm, plan->final_norm_end, 0,
                            clip_scale, correction1, correction2);
    niyah_adamw_commit_span(model, gradients, state, config,
                            plan->final_norm_end, plan->total, 1,
                            clip_scale, correction1, correction2);
}

NiyahStatus niyah_adamw_step(NiyahModel *model,
                             const NiyahModelGradients *gradients,
                             NiyahAdamWState *state,
                             const NiyahAdamWConfig *config)
{
    NiyahModelLayout canonical;
    NiyahAdamWDecayPlan plan;
    size_t bytes = 0U;
    uint64_t next_step;
    double norm = 0.0;
    double clip_scale = 1.0;
    double correction1 = 0.0;
    double correction2 = 0.0;
    NiyahStatus status;

    status = niyah_adamw_validate_model(model, &canonical);
    if (status != NIYAH_OK) {
        return status;
    }
    if (gradients == NULL || gradients->values == NULL || state == NULL ||
        state->m == NULL || state->v == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    if (state->bound_model != model || state->bound_weights != model->weights) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    if (state->count != model->weight_count || gradients->count != model->weight_count ||
        !niyah_config_equal(&state->model_config, &model->config) ||
        !niyah_layout_equal(&state->model_layout, &canonical)) {
        return NIYAH_ERR_INVALID_CONFIG;
    }
    status = niyah_adamw_config_validate(config);
    if (status != NIYAH_OK) {
        return status;
    }
    if (state->step == UINT64_MAX) {
        return NIYAH_ERR_OVERFLOW;
    }
    if (!niyah_size_mul_ok(model->weight_count, sizeof(float), &bytes)) {
        return NIYAH_ERR_OVERFLOW;
    }
    if (niyah_ranges_overlap(model->weights, bytes, gradients->values, bytes) ||
        niyah_ranges_overlap(model->weights, bytes, state->m, bytes) ||
        niyah_ranges_overlap(model->weights, bytes, state->v, bytes) ||
        niyah_ranges_overlap(gradients->values, bytes, state->m, bytes) ||
        niyah_ranges_overlap(gradients->values, bytes, state->v, bytes) ||
        niyah_ranges_overlap(state->m, bytes, state->v, bytes)) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    status = niyah_model_gradients_global_l2_norm(gradients, &norm);
    if (status != NIYAH_OK) {
        return status;
    }
    if (norm > (double)config->max_grad_norm) {
        clip_scale = (double)config->max_grad_norm / norm;
    }

    next_step = state->step + UINT64_C(1);
    if (!niyah_bias_correction(config->beta1, next_step, &correction1) ||
        !niyah_bias_correction(config->beta2, next_step, &correction2)) {
        return NIYAH_ERR_INVALID_CONFIG;
    }
    status = niyah_adamw_decay_plan(model, &canonical, &plan);
    if (status != NIYAH_OK) {
        return status;
    }

    status = niyah_adamw_validate_plan(model, gradients, state, config, &plan,
                                       clip_scale, correction1, correction2);
    if (status != NIYAH_OK) {
        return status;
    }

    niyah_adamw_commit_plan(model, gradients, state, config, &plan,
                            clip_scale, correction1, correction2);
    state->step = next_step;
    return NIYAH_OK;
}
