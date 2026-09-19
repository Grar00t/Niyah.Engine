#include "niyah/transformer.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

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

static void niyah_apply_rope(float *vector,
                             size_t n_heads,
                             size_t head_dim,
                             size_t position)
{
    size_t head;
    const float pos = (float)position;

    for (head = 0U; head < n_heads; ++head) {
        float *head_vector = vector + head * head_dim;
        size_t i;
        for (i = 0U; i + 1U < head_dim; i += 2U) {
            const float exponent = -((float)i / (float)head_dim);
            const float inv_freq = powf(10000.0f, exponent);
            const float angle = pos * inv_freq;
            const float c = cosf(angle);
            const float s = sinf(angle);
            const float x0 = head_vector[i];
            const float x1 = head_vector[i + 1U];
            head_vector[i] = x0 * c - x1 * s;
            head_vector[i + 1U] = x0 * s + x1 * c;
        }
    }
}

NiyahStatus niyah_transformer_workspace_floats(const NiyahModelConfig *config,
                                               size_t token_count,
                                               size_t *out_floats)
{
    size_t per_token = 0U;
    size_t total = 0U;
    size_t dim;
    size_t head_dim;
    size_t kv_dim;
    size_t ffn;
    size_t term;
    NiyahStatus status;

    if (out_floats == NULL || token_count == 0U) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    status = niyah_model_config_validate(config);
    if (status != NIYAH_OK) {
        return status;
    }
    if (token_count > (size_t)config->context_length) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    dim = (size_t)config->embedding_dim;
    head_dim = dim / (size_t)config->n_heads;
    if (head_dim < 2U || (head_dim % 2U) != 0U) {
        return NIYAH_ERR_INVALID_CONFIG;
    }
    if (!niyah_size_mul_ok(head_dim, (size_t)config->n_kv_heads, &kv_dim)) {
        return NIYAH_ERR_OVERFLOW;
    }
    ffn = (size_t)config->ffn_hidden_dim;

    /* hidden, norm, q, attention output, projection = 5 * dim */
    if (!niyah_size_mul_ok(5U, dim, &per_token)) {
        return NIYAH_ERR_OVERFLOW;
    }
    /* k + v = 2 * kv_dim */
    if (!niyah_size_mul_ok(2U, kv_dim, &term) ||
        !niyah_size_add_ok(per_token, term, &per_token)) {
        return NIYAH_ERR_OVERFLOW;
    }
    /* gate + up = 2 * ffn */
    if (!niyah_size_mul_ok(2U, ffn, &term) ||
        !niyah_size_add_ok(per_token, term, &per_token)) {
        return NIYAH_ERR_OVERFLOW;
    }
    /* one causal-attention score slot per token */
    if (!niyah_size_add_ok(per_token, 1U, &per_token) ||
        !niyah_size_mul_ok(per_token, token_count, &total)) {
        return NIYAH_ERR_OVERFLOW;
    }

    *out_floats = total;
    return NIYAH_OK;
}

static NiyahStatus niyah_attention(float *out,
                                   const float *q,
                                   const float *k,
                                   const float *v,
                                   float *scores,
                                   size_t token_count,
                                   size_t dim,
                                   size_t n_heads,
                                   size_t n_kv_heads,
                                   size_t head_dim,
                                   size_t kv_dim)
{
    size_t position;
    size_t group_size;
    float scale;

    if (out == NULL || q == NULL || k == NULL || v == NULL || scores == NULL ||
        token_count == 0U || dim == 0U || n_heads == 0U || n_kv_heads == 0U ||
        head_dim == 0U) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    group_size = n_heads / n_kv_heads;
    if (group_size == 0U) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    scale = 1.0f / sqrtf((float)head_dim);

    memset(out, 0, token_count * dim * sizeof(float));

    for (position = 0U; position < token_count; ++position) {
        size_t head;
        for (head = 0U; head < n_heads; ++head) {
            const size_t kv_head = head / group_size;
            const float *q_head = q + position * dim + head * head_dim;
            float max_score = -FLT_MAX;
            float normalizer = 0.0f;
            size_t source;
            size_t d;

            for (source = 0U; source <= position; ++source) {
                const float *k_head = k + source * kv_dim + kv_head * head_dim;
                float dot = 0.0f;
                for (d = 0U; d < head_dim; ++d) {
                    dot += q_head[d] * k_head[d];
                }
                scores[source] = dot * scale;
                if (scores[source] > max_score) {
                    max_score = scores[source];
                }
            }

            for (source = 0U; source <= position; ++source) {
                scores[source] = expf(scores[source] - max_score);
                normalizer += scores[source];
            }
            if (!(normalizer > 0.0f) || !isfinite(normalizer)) {
                return NIYAH_ERR_INVALID_CONFIG;
            }

            for (d = 0U; d < head_dim; ++d) {
                float value = 0.0f;
                for (source = 0U; source <= position; ++source) {
                    const float *v_head = v + source * kv_dim + kv_head * head_dim;
                    value += (scores[source] / normalizer) * v_head[d];
                }
                out[position * dim + head * head_dim + d] = value;
            }
        }
    }
    return NIYAH_OK;
}

static NiyahStatus niyah_transformer_forward_impl(const NiyahModel *model,
                                                  const uint32_t *tokens,
                                                  size_t token_count,
                                                  const uint32_t *segment_ids,
                                                  float *logits,
                                                  size_t logits_count,
                                                  float *workspace,
                                                  size_t workspace_count)
{
    const NiyahModelConfig *config;
    size_t required_workspace = 0U;
    size_t required_logits = 0U;
    size_t dim;
    size_t head_dim;
    size_t kv_dim;
    size_t ffn;
    size_t vocab;
    size_t td;
    size_t tkv;
    size_t tffn;
    float *cursor;
    float *hidden;
    float *norm;
    float *q;
    float *k;
    float *v;
    float *attn;
    float *proj;
    float *gate;
    float *up;
    float *scores;
    size_t position;
    uint32_t layer_index;
    NiyahStatus status;

    if (model == NULL || model->weights == NULL || tokens == NULL || logits == NULL ||
        workspace == NULL || token_count == 0U) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    config = &model->config;
    if (segment_ids != NULL && config->n_segments == 0U) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    status = niyah_transformer_workspace_floats(config, token_count, &required_workspace);
    if (status != NIYAH_OK) {
        return status;
    }
    if (workspace_count < required_workspace) {
        return NIYAH_ERR_BUFFER_TOO_SMALL;
    }
    if (!niyah_size_mul_ok(token_count, (size_t)config->vocab_size, &required_logits)) {
        return NIYAH_ERR_OVERFLOW;
    }
    if (logits_count < required_logits) {
        return NIYAH_ERR_BUFFER_TOO_SMALL;
    }

    dim = (size_t)config->embedding_dim;
    head_dim = dim / (size_t)config->n_heads;
    if (!niyah_size_mul_ok(head_dim, (size_t)config->n_kv_heads, &kv_dim)) {
        return NIYAH_ERR_OVERFLOW;
    }
    ffn = (size_t)config->ffn_hidden_dim;
    vocab = (size_t)config->vocab_size;
    if (!niyah_size_mul_ok(token_count, dim, &td) ||
        !niyah_size_mul_ok(token_count, kv_dim, &tkv) ||
        !niyah_size_mul_ok(token_count, ffn, &tffn)) {
        return NIYAH_ERR_OVERFLOW;
    }

    cursor = workspace;
    hidden = cursor; cursor += td;
    norm = cursor; cursor += td;
    q = cursor; cursor += td;
    k = cursor; cursor += tkv;
    v = cursor; cursor += tkv;
    attn = cursor; cursor += td;
    proj = cursor; cursor += td;
    gate = cursor; cursor += tffn;
    up = cursor; cursor += tffn;
    scores = cursor;

    for (position = 0U; position < token_count; ++position) {
        const uint32_t token = tokens[position];
        if ((size_t)token >= vocab) {
            return NIYAH_ERR_INVALID_ARGUMENT;
        }
        memcpy(hidden + position * dim,
               model->weights + model->layout.token_embedding + (size_t)token * dim,
               dim * sizeof(float));
        if (segment_ids != NULL) {
            const uint32_t segment = segment_ids[position];
            size_t i;
            if ((size_t)segment >= (size_t)config->n_segments) {
                return NIYAH_ERR_INVALID_ARGUMENT;
            }
            for (i = 0U; i < dim; ++i) {
                hidden[position * dim + i] +=
                    model->weights[model->layout.segment_embedding + (size_t)segment * dim + i];
            }
        }
    }

    for (layer_index = 0U; layer_index < config->n_layers; ++layer_index) {
        NiyahLayerLayout layer;
        const float *attn_norm_weight;
        const float *wq;
        const float *wk;
        const float *wv;
        const float *wo;
        const float *ffn_norm_weight;
        const float *w_gate;
        const float *w_up;
        const float *w_down;

        status = niyah_model_layer_layout(config, &model->layout, layer_index, &layer);
        if (status != NIYAH_OK) {
            return status;
        }
        attn_norm_weight = model->weights + layer.attn_norm;
        wq = model->weights + layer.wq;
        wk = model->weights + layer.wk;
        wv = model->weights + layer.wv;
        wo = model->weights + layer.wo;
        ffn_norm_weight = model->weights + layer.ffn_norm;
        w_gate = model->weights + layer.w_gate;
        w_up = model->weights + layer.w_up;
        w_down = model->weights + layer.w_down;

        for (position = 0U; position < token_count; ++position) {
            status = niyah_rmsnorm(norm + position * dim,
                                   hidden + position * dim,
                                   attn_norm_weight,
                                   dim,
                                   config->rms_norm_eps);
            if (status != NIYAH_OK) {
                return status;
            }
            niyah_matvec(q + position * dim, wq, norm + position * dim, dim, dim);
            niyah_matvec(k + position * kv_dim, wk, norm + position * dim, kv_dim, dim);
            niyah_matvec(v + position * kv_dim, wv, norm + position * dim, kv_dim, dim);
            niyah_apply_rope(q + position * dim,
                             (size_t)config->n_heads,
                             head_dim,
                             position);
            niyah_apply_rope(k + position * kv_dim,
                             (size_t)config->n_kv_heads,
                             head_dim,
                             position);
        }

        status = niyah_attention(attn,
                                 q,
                                 k,
                                 v,
                                 scores,
                                 token_count,
                                 dim,
                                 (size_t)config->n_heads,
                                 (size_t)config->n_kv_heads,
                                 head_dim,
                                 kv_dim);
        if (status != NIYAH_OK) {
            return status;
        }

        for (position = 0U; position < token_count; ++position) {
            size_t i;
            niyah_matvec(proj + position * dim,
                         wo,
                         attn + position * dim,
                         dim,
                         dim);
            for (i = 0U; i < dim; ++i) {
                hidden[position * dim + i] += proj[position * dim + i];
            }

            status = niyah_rmsnorm(norm + position * dim,
                                   hidden + position * dim,
                                   ffn_norm_weight,
                                   dim,
                                   config->rms_norm_eps);
            if (status != NIYAH_OK) {
                return status;
            }
            niyah_matvec(gate + position * ffn,
                         w_gate,
                         norm + position * dim,
                         ffn,
                         dim);
            niyah_matvec(up + position * ffn,
                         w_up,
                         norm + position * dim,
                         ffn,
                         dim);
            for (i = 0U; i < ffn; ++i) {
                gate[position * ffn + i] =
                    niyah_silu(gate[position * ffn + i]) * up[position * ffn + i];
            }
            niyah_matvec(proj + position * dim,
                         w_down,
                         gate + position * ffn,
                         dim,
                         ffn);
            for (i = 0U; i < dim; ++i) {
                hidden[position * dim + i] += proj[position * dim + i];
            }
        }
    }

    for (position = 0U; position < token_count; ++position) {
        status = niyah_rmsnorm(norm + position * dim,
                               hidden + position * dim,
                               model->weights + model->layout.final_norm,
                               dim,
                               config->rms_norm_eps);
        if (status != NIYAH_OK) {
            return status;
        }
        niyah_matvec(logits + position * vocab,
                     model->weights + model->layout.lm_head,
                     norm + position * dim,
                     vocab,
                     dim);
    }

    return NIYAH_OK;
}

NiyahStatus niyah_transformer_forward(const NiyahModel *model,
                                      const uint32_t *tokens,
                                      size_t token_count,
                                      float *logits,
                                      size_t logits_count,
                                      float *workspace,
                                      size_t workspace_count)
{
    return niyah_transformer_forward_impl(model, tokens, token_count, NULL,
                                          logits, logits_count, workspace, workspace_count);
}

NiyahStatus niyah_transformer_forward_with_segments(const NiyahModel *model,
                                                    const uint32_t *tokens,
                                                    size_t token_count,
                                                    const uint32_t *segment_ids,
                                                    float *logits,
                                                    size_t logits_count,
                                                    float *workspace,
                                                    size_t workspace_count)
{
    if (segment_ids == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    return niyah_transformer_forward_impl(model, tokens, token_count, segment_ids,
                                          logits, logits_count, workspace, workspace_count);
}
