#include "niyah/train.h"
#include "niyah/transformer.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct NiyahTrainShape {
    size_t token_count;
    size_t dim;
    size_t kv_dim;
    size_t ffn;
    size_t vocab;
    size_t td;
    size_t tkv;
    size_t tffn;
    size_t tv;
    size_t layer_stride;
} NiyahTrainShape;

typedef struct NiyahLayerCacheView {
    float *hidden_in;
    float *norm1;
    float *q;
    float *k;
    float *v;
    float *attn;
    float *hidden_attn;
    float *norm2;
    float *gate;
    float *up;
    float *act;
    float *hidden_out;
} NiyahLayerCacheView;

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

static float niyah_silu(float x)
{
    return x / (1.0f + expf(-x));
}

static float niyah_silu_derivative(float x)
{
    const float s = 1.0f / (1.0f + expf(-x));
    return s + x * s * (1.0f - s);
}

static void niyah_apply_rope_signed(float *vector,
                                    size_t n_heads,
                                    size_t head_dim,
                                    size_t position,
                                    float sign)
{
    size_t head;
    const float pos = (float)position;

    for (head = 0U; head < n_heads; ++head) {
        float *head_vector = vector + head * head_dim;
        size_t i;
        for (i = 0U; i + 1U < head_dim; i += 2U) {
            const float exponent = -((float)i / (float)head_dim);
            const float inv_freq = powf(10000.0f, exponent);
            const float angle = sign * pos * inv_freq;
            const float c = cosf(angle);
            const float s = sinf(angle);
            const float x0 = head_vector[i];
            const float x1 = head_vector[i + 1U];
            head_vector[i] = x0 * c - x1 * s;
            head_vector[i + 1U] = x0 * s + x1 * c;
        }
    }
}

static NiyahStatus niyah_train_shape(const NiyahModelConfig *config,
                                     size_t token_count,
                                     NiyahTrainShape *shape)
{
    NiyahTrainShape s;
    size_t head_dim;
    size_t term;
    NiyahStatus status;

    if (shape == NULL || token_count == 0U) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    status = niyah_model_config_validate(config);
    if (status != NIYAH_OK) {
        return status;
    }
    if (token_count > (size_t)config->context_length) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    memset(&s, 0, sizeof(s));
    s.token_count = token_count;
    s.dim = (size_t)config->embedding_dim;
    head_dim = s.dim / (size_t)config->n_heads;
    if (head_dim < 2U || (head_dim % 2U) != 0U) {
        return NIYAH_ERR_INVALID_CONFIG;
    }
    if (!niyah_size_mul_ok(head_dim, (size_t)config->n_kv_heads, &s.kv_dim)) {
        return NIYAH_ERR_OVERFLOW;
    }
    s.ffn = (size_t)config->ffn_hidden_dim;
    s.vocab = (size_t)config->vocab_size;
    if (!niyah_size_mul_ok(token_count, s.dim, &s.td) ||
        !niyah_size_mul_ok(token_count, s.kv_dim, &s.tkv) ||
        !niyah_size_mul_ok(token_count, s.ffn, &s.tffn) ||
        !niyah_size_mul_ok(token_count, s.vocab, &s.tv)) {
        return NIYAH_ERR_OVERFLOW;
    }

    /* hidden_in,norm1,q,attn,hidden_attn,norm2,hidden_out = 7*td */
    if (!niyah_size_mul_ok(7U, s.td, &s.layer_stride) ||
        !niyah_size_mul_ok(2U, s.tkv, &term) ||
        !niyah_size_add_ok(s.layer_stride, term, &s.layer_stride) ||
        !niyah_size_mul_ok(3U, s.tffn, &term) ||
        !niyah_size_add_ok(s.layer_stride, term, &s.layer_stride)) {
        return NIYAH_ERR_OVERFLOW;
    }
    *shape = s;
    return NIYAH_OK;
}

NiyahStatus niyah_train_backward_workspace_floats(const NiyahModelConfig *config,
                                                  size_t token_count,
                                                  size_t *out_floats)
{
    NiyahTrainShape s;
    size_t total = 0U;
    size_t term;
    NiyahStatus status;

    if (out_floats == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    status = niyah_train_shape(config, token_count, &s);
    if (status != NIYAH_OK) {
        return status;
    }

    if (!niyah_size_mul_ok((size_t)config->n_layers, s.layer_stride, &total) ||
        !niyah_size_add_ok(total, s.td, &total) ||
        !niyah_size_mul_ok(2U, s.tv, &term) ||
        !niyah_size_add_ok(total, term, &total) ||
        !niyah_size_mul_ok(7U, s.td, &term) ||
        !niyah_size_add_ok(total, term, &total) ||
        !niyah_size_mul_ok(2U, s.tkv, &term) ||
        !niyah_size_add_ok(total, term, &total) ||
        !niyah_size_mul_ok(2U, s.token_count, &term) ||
        !niyah_size_add_ok(total, term, &total)) {
        return NIYAH_ERR_OVERFLOW;
    }
    *out_floats = total;
    return NIYAH_OK;
}

static void niyah_layer_cache_view(float *base,
                                   const NiyahTrainShape *s,
                                   NiyahLayerCacheView *view)
{
    float *p = base;
    view->hidden_in = p; p += s->td;
    view->norm1 = p; p += s->td;
    view->q = p; p += s->td;
    view->k = p; p += s->tkv;
    view->v = p; p += s->tkv;
    view->attn = p; p += s->td;
    view->hidden_attn = p; p += s->td;
    view->norm2 = p; p += s->td;
    view->gate = p; p += s->tffn;
    view->up = p; p += s->tffn;
    view->act = p; p += s->tffn;
    view->hidden_out = p;
}

static NiyahStatus niyah_attention_forward(float *out,
                                           const float *q,
                                           const float *k,
                                           const float *v,
                                           float *scores,
                                           const NiyahModelConfig *config,
                                           const NiyahTrainShape *s)
{
    const size_t head_dim = s->dim / (size_t)config->n_heads;
    const size_t group_size = (size_t)config->n_heads / (size_t)config->n_kv_heads;
    const float scale = 1.0f / sqrtf((float)head_dim);
    size_t t;

    memset(out, 0, s->td * sizeof(float));
    for (t = 0U; t < s->token_count; ++t) {
        size_t h;
        for (h = 0U; h < (size_t)config->n_heads; ++h) {
            const size_t kh = h / group_size;
            const float *qh = q + t * s->dim + h * head_dim;
            float max_score = -FLT_MAX;
            float normalizer = 0.0f;
            size_t src;
            size_t d;

            for (src = 0U; src <= t; ++src) {
                const float *khv = k + src * s->kv_dim + kh * head_dim;
                float dot = 0.0f;
                for (d = 0U; d < head_dim; ++d) {
                    dot += qh[d] * khv[d];
                }
                scores[src] = dot * scale;
                if (scores[src] > max_score) {
                    max_score = scores[src];
                }
            }
            for (src = 0U; src <= t; ++src) {
                scores[src] = expf(scores[src] - max_score);
                normalizer += scores[src];
            }
            if (!(normalizer > 0.0f) || !isfinite(normalizer)) {
                return NIYAH_ERR_INVALID_CONFIG;
            }
            for (d = 0U; d < head_dim; ++d) {
                float value = 0.0f;
                for (src = 0U; src <= t; ++src) {
                    const float *vh = v + src * s->kv_dim + kh * head_dim;
                    value += (scores[src] / normalizer) * vh[d];
                }
                out[t * s->dim + h * head_dim + d] = value;
            }
        }
    }
    return NIYAH_OK;
}

static NiyahStatus niyah_cached_forward(const NiyahModel *model,
                                        const uint32_t *tokens,
                                        const uint32_t *segment_ids,
                                        const NiyahTrainShape *s,
                                        float *layer_cache_base,
                                        float *final_norm,
                                        float *logits,
                                        float *scores)
{
    const NiyahModelConfig *config = &model->config;
    size_t t;
    uint32_t layer_index;
    NiyahStatus status;

    for (layer_index = 0U; layer_index < config->n_layers; ++layer_index) {
        NiyahLayerCacheView c;
        NiyahLayerLayout layer;
        float *base = layer_cache_base + (size_t)layer_index * s->layer_stride;
        niyah_layer_cache_view(base, s, &c);
        status = niyah_model_layer_layout(config, &model->layout, layer_index, &layer);
        if (status != NIYAH_OK) {
            return status;
        }

        if (layer_index == 0U) {
            for (t = 0U; t < s->token_count; ++t) {
                if ((size_t)tokens[t] >= s->vocab) {
                    return NIYAH_ERR_INVALID_ARGUMENT;
                }
                memcpy(c.hidden_in + t * s->dim,
                       model->weights + model->layout.token_embedding + (size_t)tokens[t] * s->dim,
                       s->dim * sizeof(float));
                if (segment_ids != NULL) {
                    const uint32_t segment = segment_ids[t];
                    size_t i;
                    if ((size_t)segment >= (size_t)config->n_segments) {
                        return NIYAH_ERR_INVALID_ARGUMENT;
                    }
                    for (i = 0U; i < s->dim; ++i) {
                        c.hidden_in[t * s->dim + i] +=
                            model->weights[model->layout.segment_embedding + (size_t)segment * s->dim + i];
                    }
                }
            }
        } else {
            NiyahLayerCacheView prev;
            niyah_layer_cache_view(layer_cache_base + ((size_t)layer_index - 1U) * s->layer_stride,
                                   s, &prev);
            memcpy(c.hidden_in, prev.hidden_out, s->td * sizeof(float));
        }

        for (t = 0U; t < s->token_count; ++t) {
            status = niyah_rmsnorm(c.norm1 + t * s->dim,
                                   c.hidden_in + t * s->dim,
                                   model->weights + layer.attn_norm,
                                   s->dim,
                                   config->rms_norm_eps);
            if (status != NIYAH_OK) {
                return status;
            }
            niyah_matvec(c.q + t * s->dim, model->weights + layer.wq,
                         c.norm1 + t * s->dim, s->dim, s->dim);
            niyah_matvec(c.k + t * s->kv_dim, model->weights + layer.wk,
                         c.norm1 + t * s->dim, s->kv_dim, s->dim);
            niyah_matvec(c.v + t * s->kv_dim, model->weights + layer.wv,
                         c.norm1 + t * s->dim, s->kv_dim, s->dim);
            niyah_apply_rope_signed(c.q + t * s->dim, (size_t)config->n_heads,
                                    s->dim / (size_t)config->n_heads, t, 1.0f);
            niyah_apply_rope_signed(c.k + t * s->kv_dim, (size_t)config->n_kv_heads,
                                    s->dim / (size_t)config->n_heads, t, 1.0f);
        }

        status = niyah_attention_forward(c.attn, c.q, c.k, c.v, scores, config, s);
        if (status != NIYAH_OK) {
            return status;
        }

        for (t = 0U; t < s->token_count; ++t) {
            size_t i;
            niyah_matvec(c.hidden_attn + t * s->dim,
                         model->weights + layer.wo,
                         c.attn + t * s->dim,
                         s->dim, s->dim);
            for (i = 0U; i < s->dim; ++i) {
                c.hidden_attn[t * s->dim + i] += c.hidden_in[t * s->dim + i];
            }
            status = niyah_rmsnorm(c.norm2 + t * s->dim,
                                   c.hidden_attn + t * s->dim,
                                   model->weights + layer.ffn_norm,
                                   s->dim,
                                   config->rms_norm_eps);
            if (status != NIYAH_OK) {
                return status;
            }
            niyah_matvec(c.gate + t * s->ffn, model->weights + layer.w_gate,
                         c.norm2 + t * s->dim, s->ffn, s->dim);
            niyah_matvec(c.up + t * s->ffn, model->weights + layer.w_up,
                         c.norm2 + t * s->dim, s->ffn, s->dim);
            for (i = 0U; i < s->ffn; ++i) {
                c.act[t * s->ffn + i] = niyah_silu(c.gate[t * s->ffn + i]) * c.up[t * s->ffn + i];
            }
            niyah_matvec(c.hidden_out + t * s->dim, model->weights + layer.w_down,
                         c.act + t * s->ffn, s->dim, s->ffn);
            for (i = 0U; i < s->dim; ++i) {
                c.hidden_out[t * s->dim + i] += c.hidden_attn[t * s->dim + i];
            }
        }
    }

    {
        NiyahLayerCacheView last;
        niyah_layer_cache_view(layer_cache_base + ((size_t)config->n_layers - 1U) * s->layer_stride,
                               s, &last);
        for (t = 0U; t < s->token_count; ++t) {
            status = niyah_rmsnorm(final_norm + t * s->dim,
                                   last.hidden_out + t * s->dim,
                                   model->weights + model->layout.final_norm,
                                   s->dim,
                                   config->rms_norm_eps);
            if (status != NIYAH_OK) {
                return status;
            }
            niyah_matvec(logits + t * s->vocab,
                         model->weights + model->layout.lm_head,
                         final_norm + t * s->dim,
                         s->vocab, s->dim);
        }
    }
    return NIYAH_OK;
}

static void niyah_rmsnorm_backward(float *dx,
                                   float *dweight,
                                   const float *dy,
                                   const float *x,
                                   const float *weight,
                                   size_t n,
                                   float eps)
{
    size_t i;
    double sum_sq = 0.0;
    double dot = 0.0;
    float inv;
    float coeff;

    for (i = 0U; i < n; ++i) {
        const double xv = (double)x[i];
        sum_sq += xv * xv;
    }
    inv = 1.0f / sqrtf((float)(sum_sq / (double)n) + eps);
    for (i = 0U; i < n; ++i) {
        dot += (double)dy[i] * (double)weight[i] * (double)x[i];
        dweight[i] += dy[i] * x[i] * inv;
    }
    coeff = inv * inv * inv * (float)(dot / (double)n);
    for (i = 0U; i < n; ++i) {
        dx[i] = dy[i] * weight[i] * inv - x[i] * coeff;
    }
}

static void niyah_linear_backward_accum(const float *weight,
                                        float *dweight,
                                        const float *x,
                                        const float *dy,
                                        float *dx,
                                        size_t rows,
                                        size_t cols)
{
    size_t r;
    size_t c;

    for (r = 0U; r < rows; ++r) {
        const float dyr = dy[r];
        const float *wr = weight + r * cols;
        float *dwr = dweight + r * cols;
        for (c = 0U; c < cols; ++c) {
            dwr[c] += dyr * x[c];
            dx[c] += wr[c] * dyr;
        }
    }
}

static void niyah_attention_backward(float *dq,
                                     float *dk,
                                     float *dv,
                                     const float *da,
                                     const float *q,
                                     const float *k,
                                     const float *v,
                                     float *probs,
                                     float *dp,
                                     const NiyahModelConfig *config,
                                     const NiyahTrainShape *s)
{
    const size_t head_dim = s->dim / (size_t)config->n_heads;
    const size_t group_size = (size_t)config->n_heads / (size_t)config->n_kv_heads;
    const float scale = 1.0f / sqrtf((float)head_dim);
    size_t t;

    memset(dq, 0, s->td * sizeof(float));
    memset(dk, 0, s->tkv * sizeof(float));
    memset(dv, 0, s->tkv * sizeof(float));

    for (t = 0U; t < s->token_count; ++t) {
        size_t h;
        for (h = 0U; h < (size_t)config->n_heads; ++h) {
            const size_t kh = h / group_size;
            const float *qh = q + t * s->dim + h * head_dim;
            const float *dah = da + t * s->dim + h * head_dim;
            float max_score = -FLT_MAX;
            float sum_exp = 0.0f;
            float mean_dp = 0.0f;
            size_t src;
            size_t d;

            for (src = 0U; src <= t; ++src) {
                const float *khv = k + src * s->kv_dim + kh * head_dim;
                float score = 0.0f;
                for (d = 0U; d < head_dim; ++d) {
                    score += qh[d] * khv[d];
                }
                score *= scale;
                probs[src] = score;
                if (score > max_score) {
                    max_score = score;
                }
            }
            for (src = 0U; src <= t; ++src) {
                probs[src] = expf(probs[src] - max_score);
                sum_exp += probs[src];
            }
            for (src = 0U; src <= t; ++src) {
                const float *vh = v + src * s->kv_dim + kh * head_dim;
                float dprob = 0.0f;
                probs[src] /= sum_exp;
                for (d = 0U; d < head_dim; ++d) {
                    dprob += dah[d] * vh[d];
                }
                dp[src] = dprob;
                mean_dp += probs[src] * dprob;
            }
            for (src = 0U; src <= t; ++src) {
                const float ds = probs[src] * (dp[src] - mean_dp);
                const float *khv = k + src * s->kv_dim + kh * head_dim;
                float *dkh = dk + src * s->kv_dim + kh * head_dim;
                float *dvh = dv + src * s->kv_dim + kh * head_dim;
                float *dqh = dq + t * s->dim + h * head_dim;
                for (d = 0U; d < head_dim; ++d) {
                    dqh[d] += ds * scale * khv[d];
                    dkh[d] += ds * scale * qh[d];
                    dvh[d] += probs[src] * dah[d];
                }
            }
        }
    }
}

static NiyahStatus niyah_train_backward_impl(
    const NiyahModel *model,
    const uint32_t *tokens,
    const uint32_t *targets,
    size_t token_count,
    size_t loss_start,
    const uint32_t *segment_ids,
    float *out_loss,
    NiyahModelGradients *gradients,
    float *workspace,
    size_t workspace_count)
{
    NiyahTrainShape s;
    size_t required = 0U;
    size_t layers_total;
    float *cursor;
    float *layer_cache_base;
    float *final_norm;
    float *logits;
    float *dlogits;
    float *dh;
    float *dh_attn;
    float *dn2;
    float *da;
    float *dq;
    float *dn1;
    float *dtmp;
    float *dk;
    float *dv;
    float *probs;
    float *dp;
    NiyahStatus status;
    size_t t;
    uint32_t layer_index;

    if (model == NULL || model->weights == NULL || tokens == NULL || targets == NULL ||
        out_loss == NULL || gradients == NULL || gradients->values == NULL ||
        workspace == NULL || gradients->count != model->weight_count) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    status = niyah_train_shape(&model->config, token_count, &s);
    if (status != NIYAH_OK) {
        return status;
    }
    status = niyah_train_backward_workspace_floats(&model->config, token_count, &required);
    if (status != NIYAH_OK) {
        return status;
    }
    if (workspace_count < required) {
        return NIYAH_ERR_BUFFER_TOO_SMALL;
    }
    if (!niyah_size_mul_ok((size_t)model->config.n_layers, s.layer_stride, &layers_total)) {
        return NIYAH_ERR_OVERFLOW;
    }

    cursor = workspace;
    layer_cache_base = cursor; cursor += layers_total;
    final_norm = cursor; cursor += s.td;
    logits = cursor; cursor += s.tv;
    dlogits = cursor; cursor += s.tv;
    dh = cursor; cursor += s.td;
    dh_attn = cursor; cursor += s.td;
    dn2 = cursor; cursor += s.td;
    da = cursor; cursor += s.td;
    dq = cursor; cursor += s.td;
    dn1 = cursor; cursor += s.td;
    dtmp = cursor; cursor += s.td;
    dk = cursor; cursor += s.tkv;
    dv = cursor; cursor += s.tkv;
    probs = cursor; cursor += s.token_count;
    dp = cursor;

    status = niyah_cached_forward(model, tokens, segment_ids, &s, layer_cache_base, final_norm, logits, probs);
    if (status != NIYAH_OK) {
        return status;
    }
    status = niyah_cross_entropy_loss_masked(
        logits,
        targets,
        token_count,
        s.vocab,
        loss_start,
        out_loss,
        dlogits,
        s.tv);
    if (status != NIYAH_OK) {
        return status;
    }

    niyah_model_gradients_zero(gradients);
    memset(dh, 0, s.td * sizeof(float));
    for (t = 0U; t < s.token_count; ++t) {
        const float *dy = dlogits + t * s.vocab;
        const float *x = final_norm + t * s.dim;
        float *dx = dh + t * s.dim;
        niyah_linear_backward_accum(model->weights + model->layout.lm_head,
                                    gradients->values + model->layout.lm_head,
                                    x, dy, dx, s.vocab, s.dim);
    }

    {
        NiyahLayerCacheView last;
        niyah_layer_cache_view(layer_cache_base + ((size_t)model->config.n_layers - 1U) * s.layer_stride,
                               &s, &last);
        memset(dtmp, 0, s.td * sizeof(float));
        for (t = 0U; t < s.token_count; ++t) {
            niyah_rmsnorm_backward(dtmp + t * s.dim,
                                   gradients->values + model->layout.final_norm,
                                   dh + t * s.dim,
                                   last.hidden_out + t * s.dim,
                                   model->weights + model->layout.final_norm,
                                   s.dim, model->config.rms_norm_eps);
        }
        memcpy(dh, dtmp, s.td * sizeof(float));
    }

    for (layer_index = model->config.n_layers; layer_index-- > 0U;) {
        NiyahLayerCacheView c;
        NiyahLayerLayout layer;
        const float *w_down;
        const float *w_gate;
        const float *w_up;
        const float *wo;
        const float *wq;
        const float *wk;
        const float *wv;
        float *gd_down;
        float *gd_gate;
        float *gd_up;
        float *gd_wo;
        float *gd_wq;
        float *gd_wk;
        float *gd_wv;
        size_t i;

        niyah_layer_cache_view(layer_cache_base + (size_t)layer_index * s.layer_stride, &s, &c);
        status = niyah_model_layer_layout(&model->config, &model->layout, layer_index, &layer);
        if (status != NIYAH_OK) {
            return status;
        }
        w_down = model->weights + layer.w_down;
        w_gate = model->weights + layer.w_gate;
        w_up = model->weights + layer.w_up;
        wo = model->weights + layer.wo;
        wq = model->weights + layer.wq;
        wk = model->weights + layer.wk;
        wv = model->weights + layer.wv;
        gd_down = gradients->values + layer.w_down;
        gd_gate = gradients->values + layer.w_gate;
        gd_up = gradients->values + layer.w_up;
        gd_wo = gradients->values + layer.wo;
        gd_wq = gradients->values + layer.wq;
        gd_wk = gradients->values + layer.wk;
        gd_wv = gradients->values + layer.wv;

        memcpy(dh_attn, dh, s.td * sizeof(float));
        memset(dn2, 0, s.td * sizeof(float));
        for (t = 0U; t < s.token_count; ++t) {
            const float *dy = dh + t * s.dim;
            const float *act = c.act + t * s.ffn;
            const float *norm2 = c.norm2 + t * s.dim;
            size_t r;
            size_t col;

            for (r = 0U; r < s.dim; ++r) {
                float *dwr = gd_down + r * s.ffn;
                for (col = 0U; col < s.ffn; ++col) {
                    dwr[col] += dy[r] * act[col];
                }
            }
            for (i = 0U; i < s.ffn; ++i) {
                float dact = 0.0f;
                float dg;
                float du;
                const float g = c.gate[t * s.ffn + i];
                const float u = c.up[t * s.ffn + i];
                const float silu_g = niyah_silu(g);

                for (r = 0U; r < s.dim; ++r) {
                    dact += w_down[r * s.ffn + i] * dy[r];
                }
                dg = dact * u * niyah_silu_derivative(g);
                du = dact * silu_g;
                for (col = 0U; col < s.dim; ++col) {
                    gd_gate[i * s.dim + col] += dg * norm2[col];
                    gd_up[i * s.dim + col] += du * norm2[col];
                    dn2[t * s.dim + col] += w_gate[i * s.dim + col] * dg +
                                             w_up[i * s.dim + col] * du;
                }
            }
        }

        memset(dtmp, 0, s.td * sizeof(float));
        for (t = 0U; t < s.token_count; ++t) {
            niyah_rmsnorm_backward(dtmp + t * s.dim,
                                   gradients->values + layer.ffn_norm,
                                   dn2 + t * s.dim,
                                   c.hidden_attn + t * s.dim,
                                   model->weights + layer.ffn_norm,
                                   s.dim, model->config.rms_norm_eps);
        }
        for (i = 0U; i < s.td; ++i) {
            dh_attn[i] += dtmp[i];
        }

        memset(da, 0, s.td * sizeof(float));
        for (t = 0U; t < s.token_count; ++t) {
            niyah_linear_backward_accum(wo, gd_wo,
                                        c.attn + t * s.dim,
                                        dh_attn + t * s.dim,
                                        da + t * s.dim,
                                        s.dim, s.dim);
        }

        niyah_attention_backward(dq, dk, dv, da, c.q, c.k, c.v,
                                 probs, dp, &model->config, &s);
        for (t = 0U; t < s.token_count; ++t) {
            niyah_apply_rope_signed(dq + t * s.dim, (size_t)model->config.n_heads,
                                    s.dim / (size_t)model->config.n_heads, t, -1.0f);
            niyah_apply_rope_signed(dk + t * s.kv_dim, (size_t)model->config.n_kv_heads,
                                    s.dim / (size_t)model->config.n_heads, t, -1.0f);
        }

        memset(dn1, 0, s.td * sizeof(float));
        for (t = 0U; t < s.token_count; ++t) {
            niyah_linear_backward_accum(wq, gd_wq, c.norm1 + t * s.dim,
                                        dq + t * s.dim, dn1 + t * s.dim,
                                        s.dim, s.dim);
            niyah_linear_backward_accum(wk, gd_wk, c.norm1 + t * s.dim,
                                        dk + t * s.kv_dim, dn1 + t * s.dim,
                                        s.kv_dim, s.dim);
            niyah_linear_backward_accum(wv, gd_wv, c.norm1 + t * s.dim,
                                        dv + t * s.kv_dim, dn1 + t * s.dim,
                                        s.kv_dim, s.dim);
        }

        memset(dtmp, 0, s.td * sizeof(float));
        for (t = 0U; t < s.token_count; ++t) {
            niyah_rmsnorm_backward(dtmp + t * s.dim,
                                   gradients->values + layer.attn_norm,
                                   dn1 + t * s.dim,
                                   c.hidden_in + t * s.dim,
                                   model->weights + layer.attn_norm,
                                   s.dim, model->config.rms_norm_eps);
        }
        for (i = 0U; i < s.td; ++i) {
            dh[i] = dh_attn[i] + dtmp[i];
        }
    }

    for (t = 0U; t < s.token_count; ++t) {
        float *dembed = gradients->values + model->layout.token_embedding +
                        (size_t)tokens[t] * s.dim;
        size_t d;
        for (d = 0U; d < s.dim; ++d) {
            dembed[d] += dh[t * s.dim + d];
        }
        if (segment_ids != NULL) {
            float *dsegment = gradients->values + model->layout.segment_embedding +
                              (size_t)segment_ids[t] * s.dim;
            for (d = 0U; d < s.dim; ++d) {
                dsegment[d] += dh[t * s.dim + d];
            }
        }
    }
    return NIYAH_OK;
}

NiyahStatus niyah_train_backward(
    const NiyahModel *model,
    const uint32_t *tokens,
    const uint32_t *targets,
    size_t token_count,
    float *out_loss,
    NiyahModelGradients *gradients,
    float *workspace,
    size_t workspace_count)
{
    return niyah_train_backward_impl(
        model,
        tokens,
        targets,
        token_count,
        0U,
        NULL,
        out_loss,
        gradients,
        workspace,
        workspace_count);
}

NiyahStatus niyah_train_backward_masked(
    const NiyahModel *model,
    const uint32_t *tokens,
    const uint32_t *targets,
    size_t token_count,
    size_t loss_start,
    float *out_loss,
    NiyahModelGradients *gradients,
    float *workspace,
    size_t workspace_count)
{
    return niyah_train_backward_impl(
        model,
        tokens,
        targets,
        token_count,
        loss_start,
        NULL,
        out_loss,
        gradients,
        workspace,
        workspace_count);
}

NiyahStatus niyah_train_backward_masked_with_segments(
    const NiyahModel *model,
    const uint32_t *tokens,
    const uint32_t *targets,
    size_t token_count,
    size_t loss_start,
    const uint32_t *segment_ids,
    float *out_loss,
    NiyahModelGradients *gradients,
    float *workspace,
    size_t workspace_count)
{
    if (segment_ids == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    return niyah_train_backward_impl(
        model,
        tokens,
        targets,
        token_count,
        loss_start,
        segment_ids,
        out_loss,
        gradients,
        workspace,
        workspace_count);
}
