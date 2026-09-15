#include "niyah/decode.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
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

static NiyahStatus niyah_cache_shape(const NiyahModelConfig *config,
                                     size_t *head_dim,
                                     size_t *kv_dim,
                                     size_t *values_per_tensor)
{
    size_t hd;
    size_t kd;
    size_t per_layer;
    size_t total;
    NiyahStatus status;

    if (head_dim == NULL || kv_dim == NULL || values_per_tensor == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    status = niyah_model_config_validate(config);
    if (status != NIYAH_OK) {
        return status;
    }

    hd = (size_t)config->embedding_dim / (size_t)config->n_heads;
    if (hd < 2U || (hd % 2U) != 0U) {
        return NIYAH_ERR_INVALID_CONFIG;
    }
    if (!niyah_size_mul_ok(hd, (size_t)config->n_kv_heads, &kd) ||
        !niyah_size_mul_ok((size_t)config->context_length, kd, &per_layer) ||
        !niyah_size_mul_ok((size_t)config->n_layers, per_layer, &total)) {
        return NIYAH_ERR_OVERFLOW;
    }

    *head_dim = hd;
    *kv_dim = kd;
    *values_per_tensor = total;
    return NIYAH_OK;
}

NiyahStatus niyah_kv_cache_create(NiyahKVCache *cache,
                                  const NiyahModelConfig *config)
{
    size_t head_dim = 0U;
    size_t kv_dim = 0U;
    size_t values_per_tensor = 0U;
    size_t bytes = 0U;
    NiyahStatus status;

    if (cache == NULL || config == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    memset(cache, 0, sizeof(*cache));

    status = niyah_cache_shape(config, &head_dim, &kv_dim, &values_per_tensor);
    if (status != NIYAH_OK) {
        return status;
    }
    if (!niyah_size_mul_ok(values_per_tensor, sizeof(float), &bytes)) {
        return NIYAH_ERR_OVERFLOW;
    }

    cache->keys = (float *)calloc(1U, bytes);
    if (cache->keys == NULL) {
        return NIYAH_ERR_OUT_OF_MEMORY;
    }
    cache->values = (float *)calloc(1U, bytes);
    if (cache->values == NULL) {
        free(cache->keys);
        memset(cache, 0, sizeof(*cache));
        return NIYAH_ERR_OUT_OF_MEMORY;
    }

    cache->n_layers = config->n_layers;
    cache->n_heads = config->n_heads;
    cache->n_kv_heads = config->n_kv_heads;
    cache->context_length = (size_t)config->context_length;
    cache->head_dim = head_dim;
    cache->kv_dim = kv_dim;
    cache->values_per_tensor = values_per_tensor;
    cache->next_position = 0U;
    return NIYAH_OK;
}

void niyah_kv_cache_destroy(NiyahKVCache *cache)
{
    if (cache == NULL) {
        return;
    }
    free(cache->keys);
    free(cache->values);
    memset(cache, 0, sizeof(*cache));
}

void niyah_kv_cache_reset(NiyahKVCache *cache)
{
    if (cache == NULL || cache->keys == NULL || cache->values == NULL) {
        return;
    }
    memset(cache->keys, 0, cache->values_per_tensor * sizeof(float));
    memset(cache->values, 0, cache->values_per_tensor * sizeof(float));
    cache->next_position = 0U;
}

size_t niyah_kv_cache_position(const NiyahKVCache *cache)
{
    return cache != NULL ? cache->next_position : 0U;
}

NiyahStatus niyah_decode_workspace_floats(const NiyahModelConfig *config,
                                          size_t *out_floats)
{
    size_t head_dim = 0U;
    size_t kv_dim = 0U;
    size_t ignored_cache_values = 0U;
    size_t total = 0U;
    size_t term = 0U;
    size_t dim;
    size_t ffn;
    NiyahStatus status;

    if (out_floats == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    status = niyah_cache_shape(config, &head_dim, &kv_dim, &ignored_cache_values);
    if (status != NIYAH_OK) {
        return status;
    }
    (void)head_dim;
    (void)ignored_cache_values;

    dim = (size_t)config->embedding_dim;
    ffn = (size_t)config->ffn_hidden_dim;

    if (!niyah_size_mul_ok(5U, dim, &total) ||
        !niyah_size_mul_ok(2U, kv_dim, &term) ||
        !niyah_size_add_ok(total, term, &total) ||
        !niyah_size_mul_ok(2U, ffn, &term) ||
        !niyah_size_add_ok(total, term, &total) ||
        !niyah_size_add_ok(total, (size_t)config->context_length, &total)) {
        return NIYAH_ERR_OVERFLOW;
    }

    *out_floats = total;
    return NIYAH_OK;
}

static int niyah_cache_matches(const NiyahKVCache *cache,
                               const NiyahModelConfig *config,
                               size_t head_dim,
                               size_t kv_dim,
                               size_t expected_values)
{
    return cache != NULL && cache->keys != NULL && cache->values != NULL &&
           cache->n_layers == config->n_layers &&
           cache->n_heads == config->n_heads &&
           cache->n_kv_heads == config->n_kv_heads &&
           cache->context_length == (size_t)config->context_length &&
           cache->head_dim == head_dim &&
           cache->kv_dim == kv_dim &&
           cache->values_per_tensor == expected_values;
}

static NiyahStatus niyah_attention_one(float *out,
                                       const float *q,
                                       const NiyahKVCache *cache,
                                       uint32_t layer_index,
                                       size_t position,
                                       float *scores,
                                       size_t dim,
                                       size_t n_heads,
                                       size_t n_kv_heads,
                                       size_t head_dim,
                                       size_t kv_dim)
{
    const size_t group_size = n_heads / n_kv_heads;
    const size_t layer_base = (size_t)layer_index * cache->context_length * kv_dim;
    const float scale = 1.0f / sqrtf((float)head_dim);
    size_t head;

    if (out == NULL || q == NULL || cache == NULL || scores == NULL ||
        group_size == 0U || position >= cache->context_length) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    memset(out, 0, dim * sizeof(float));

    for (head = 0U; head < n_heads; ++head) {
        const size_t kv_head = head / group_size;
        const float *q_head = q + head * head_dim;
        float max_score = -FLT_MAX;
        float normalizer = 0.0f;
        size_t source;
        size_t d;

        for (source = 0U; source <= position; ++source) {
            const float *k_head = cache->keys + layer_base + source * kv_dim + kv_head * head_dim;
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
                const float *v_head = cache->values + layer_base + source * kv_dim + kv_head * head_dim;
                value += (scores[source] / normalizer) * v_head[d];
            }
            out[head * head_dim + d] = value;
        }
    }

    return NIYAH_OK;
}

NiyahStatus niyah_transformer_decode_token(const NiyahModel *model,
                                           NiyahKVCache *cache,
                                           uint32_t token,
                                           float *logits,
                                           size_t logits_count,
                                           float *workspace,
                                           size_t workspace_count)
{
    const NiyahModelConfig *config;
    size_t required_workspace = 0U;
    size_t expected_cache_values = 0U;
    size_t head_dim = 0U;
    size_t kv_dim = 0U;
    size_t dim;
    size_t ffn;
    size_t vocab;
    size_t position;
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
    uint32_t layer_index;
    NiyahStatus status;

    if (model == NULL || model->weights == NULL || cache == NULL ||
        logits == NULL || workspace == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    config = &model->config;
    status = niyah_cache_shape(config, &head_dim, &kv_dim, &expected_cache_values);
    if (status != NIYAH_OK) {
        return status;
    }
    if (!niyah_cache_matches(cache, config, head_dim, kv_dim, expected_cache_values)) {
        return NIYAH_ERR_INVALID_CONFIG;
    }
    if (cache->next_position >= cache->context_length) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    vocab = (size_t)config->vocab_size;
    if ((size_t)token >= vocab) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    if (logits_count < vocab) {
        return NIYAH_ERR_BUFFER_TOO_SMALL;
    }
    status = niyah_decode_workspace_floats(config, &required_workspace);
    if (status != NIYAH_OK) {
        return status;
    }
    if (workspace_count < required_workspace) {
        return NIYAH_ERR_BUFFER_TOO_SMALL;
    }

    dim = (size_t)config->embedding_dim;
    ffn = (size_t)config->ffn_hidden_dim;
    position = cache->next_position;

    cursor = workspace;
    hidden = cursor; cursor += dim;
    norm = cursor; cursor += dim;
    q = cursor; cursor += dim;
    k = cursor; cursor += kv_dim;
    v = cursor; cursor += kv_dim;
    attn = cursor; cursor += dim;
    proj = cursor; cursor += dim;
    gate = cursor; cursor += ffn;
    up = cursor; cursor += ffn;
    scores = cursor;

    memcpy(hidden,
           model->weights + model->layout.token_embedding + (size_t)token * dim,
           dim * sizeof(float));

    for (layer_index = 0U; layer_index < config->n_layers; ++layer_index) {
        NiyahLayerLayout layer;
        const size_t layer_base = (size_t)layer_index * cache->context_length * kv_dim;
        const size_t cache_offset = layer_base + position * kv_dim;
        size_t i;

        status = niyah_model_layer_layout(config, &model->layout, layer_index, &layer);
        if (status != NIYAH_OK) {
            return status;
        }

        status = niyah_rmsnorm(norm,
                               hidden,
                               model->weights + layer.attn_norm,
                               dim,
                               config->rms_norm_eps);
        if (status != NIYAH_OK) {
            return status;
        }

        niyah_matvec(q, model->weights + layer.wq, norm, dim, dim);
        niyah_matvec(k, model->weights + layer.wk, norm, kv_dim, dim);
        niyah_matvec(v, model->weights + layer.wv, norm, kv_dim, dim);
        niyah_apply_rope(q, (size_t)config->n_heads, head_dim, position);
        niyah_apply_rope(k, (size_t)config->n_kv_heads, head_dim, position);

        memcpy(cache->keys + cache_offset, k, kv_dim * sizeof(float));
        memcpy(cache->values + cache_offset, v, kv_dim * sizeof(float));

        status = niyah_attention_one(attn,
                                     q,
                                     cache,
                                     layer_index,
                                     position,
                                     scores,
                                     dim,
                                     (size_t)config->n_heads,
                                     (size_t)config->n_kv_heads,
                                     head_dim,
                                     kv_dim);
        if (status != NIYAH_OK) {
            return status;
        }

        niyah_matvec(proj, model->weights + layer.wo, attn, dim, dim);
        for (i = 0U; i < dim; ++i) {
            hidden[i] += proj[i];
        }

        status = niyah_rmsnorm(norm,
                               hidden,
                               model->weights + layer.ffn_norm,
                               dim,
                               config->rms_norm_eps);
        if (status != NIYAH_OK) {
            return status;
        }
        niyah_matvec(gate, model->weights + layer.w_gate, norm, ffn, dim);
        niyah_matvec(up, model->weights + layer.w_up, norm, ffn, dim);
        for (i = 0U; i < ffn; ++i) {
            gate[i] = niyah_silu(gate[i]) * up[i];
        }
        niyah_matvec(proj, model->weights + layer.w_down, gate, dim, ffn);
        for (i = 0U; i < dim; ++i) {
            hidden[i] += proj[i];
        }
    }

    status = niyah_rmsnorm(norm,
                           hidden,
                           model->weights + model->layout.final_norm,
                           dim,
                           config->rms_norm_eps);
    if (status != NIYAH_OK) {
        return status;
    }
    niyah_matvec(logits,
                 model->weights + model->layout.lm_head,
                 norm,
                 vocab,
                 dim);

    cache->next_position = position + 1U;
    return NIYAH_OK;
}
