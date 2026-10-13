#include "niyah/niyah.h"
#include "niyah_model_internal.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/mman.h>
#endif

#include <math.h>
#include <stdlib.h>
#include <string.h>

static int niyah_checked_add(size_t a, size_t b, size_t *out)
{
    if (out == NULL || a > SIZE_MAX - b) {
        return 0;
    }
    *out = a + b;
    return 1;
}

static int niyah_checked_mul(size_t a, size_t b, size_t *out)
{
    if (out == NULL || (a != 0U && b > SIZE_MAX / a)) {
        return 0;
    }
    *out = a * b;
    return 1;
}

static NiyahStatus niyah_advance(size_t *cursor, size_t count, size_t *offset)
{
    size_t next = 0U;
    if (cursor == NULL || offset == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    *offset = *cursor;
    if (!niyah_checked_add(*cursor, count, &next)) {
        return NIYAH_ERR_OVERFLOW;
    }
    *cursor = next;
    return NIYAH_OK;
}

NiyahStatus niyah_model_config_validate(const NiyahModelConfig *config)
{
    size_t head_dim;

    if (config == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    if (config->vocab_size < 2U ||
        config->context_length == 0U ||
        config->embedding_dim == 0U ||
        config->n_layers == 0U ||
        config->n_heads == 0U ||
        config->n_kv_heads == 0U ||
        config->ffn_hidden_dim == 0U ||
        !isfinite(config->rms_norm_eps) ||
        config->rms_norm_eps <= 0.0f) {
        return NIYAH_ERR_INVALID_CONFIG;
    }
    if ((config->embedding_dim % config->n_heads) != 0U ||
        (config->n_heads % config->n_kv_heads) != 0U ||
        config->n_kv_heads > config->n_heads) {
        return NIYAH_ERR_INVALID_CONFIG;
    }

    head_dim = (size_t)config->embedding_dim / (size_t)config->n_heads;
    if (head_dim < 2U || (head_dim % 2U) != 0U) {
        return NIYAH_ERR_INVALID_CONFIG;
    }
    return NIYAH_OK;
}

NiyahStatus niyah_model_layout_compute(const NiyahModelConfig *config,
                                       NiyahModelLayout *layout)
{
    NiyahModelLayout tmp;
    size_t cursor = 0U;
    size_t token_embedding_count = 0U;
    size_t segment_embedding_count = 0U;
    size_t q_count = 0U;
    size_t kv_count = 0U;
    size_t ffn_up_count = 0U;
    size_t ffn_down_count = 0U;
    size_t layer_stride = 0U;
    size_t all_layers_count = 0U;
    size_t lm_head_count = 0U;
    size_t dim = 0U;
    size_t ffn = 0U;
    size_t vocab = 0U;
    NiyahStatus status;

    if (layout == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    status = niyah_model_config_validate(config);
    if (status != NIYAH_OK) {
        return status;
    }

    memset(&tmp, 0, sizeof(tmp));
    dim = (size_t)config->embedding_dim;
    ffn = (size_t)config->ffn_hidden_dim;
    vocab = (size_t)config->vocab_size;
    tmp.head_dim = dim / (size_t)config->n_heads;
    if (!niyah_checked_mul(tmp.head_dim, (size_t)config->n_kv_heads, &tmp.kv_dim)) {
        return NIYAH_ERR_OVERFLOW;
    }

    if (!niyah_checked_mul(vocab, dim, &token_embedding_count) ||
        !niyah_checked_mul((size_t)config->n_segments, dim, &segment_embedding_count) ||
        !niyah_checked_mul(dim, dim, &q_count) ||
        !niyah_checked_mul(tmp.kv_dim, dim, &kv_count) ||
        !niyah_checked_mul(ffn, dim, &ffn_up_count) ||
        !niyah_checked_mul(dim, ffn, &ffn_down_count)) {
        return NIYAH_ERR_OVERFLOW;
    }

    tmp.token_embedding = cursor;
    if (!niyah_checked_add(cursor, token_embedding_count, &cursor)) {
        return NIYAH_ERR_OVERFLOW;
    }

    tmp.segment_embedding = cursor;
    if (!niyah_checked_add(cursor, segment_embedding_count, &cursor)) {
        return NIYAH_ERR_OVERFLOW;
    }

    tmp.layers = cursor;
    layer_stride = 0U;
    if (!niyah_checked_add(layer_stride, dim, &layer_stride) ||
        !niyah_checked_add(layer_stride, q_count, &layer_stride) ||
        !niyah_checked_add(layer_stride, kv_count, &layer_stride) ||
        !niyah_checked_add(layer_stride, kv_count, &layer_stride) ||
        !niyah_checked_add(layer_stride, q_count, &layer_stride) ||
        !niyah_checked_add(layer_stride, dim, &layer_stride) ||
        !niyah_checked_add(layer_stride, ffn_up_count, &layer_stride) ||
        !niyah_checked_add(layer_stride, ffn_up_count, &layer_stride) ||
        !niyah_checked_add(layer_stride, ffn_down_count, &layer_stride)) {
        return NIYAH_ERR_OVERFLOW;
    }
    tmp.layer_stride = layer_stride;

    if (!niyah_checked_mul(layer_stride, (size_t)config->n_layers, &all_layers_count) ||
        !niyah_checked_add(cursor, all_layers_count, &cursor)) {
        return NIYAH_ERR_OVERFLOW;
    }

    tmp.final_norm = cursor;
    if (!niyah_checked_add(cursor, dim, &cursor)) {
        return NIYAH_ERR_OVERFLOW;
    }

    if (config->tie_word_embeddings != 0) {
        tmp.lm_head = tmp.token_embedding;
    } else {
        tmp.lm_head = cursor;
        if (!niyah_checked_mul(vocab, dim, &lm_head_count) ||
            !niyah_checked_add(cursor, lm_head_count, &cursor)) {
            return NIYAH_ERR_OVERFLOW;
        }
    }

    tmp.total_floats = cursor;
    *layout = tmp;
    return NIYAH_OK;
}

NiyahStatus niyah_model_layer_layout(const NiyahModelConfig *config,
                                     const NiyahModelLayout *layout,
                                     uint32_t layer_index,
                                     NiyahLayerLayout *layer)
{
    size_t cursor = 0U;
    size_t layer_delta = 0U;
    size_t dim = 0U;
    size_t q_count = 0U;
    size_t kv_count = 0U;
    size_t ffn_up_count = 0U;
    size_t ffn_down_count = 0U;
    NiyahStatus status;

    if (config == NULL || layout == NULL || layer == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    status = niyah_model_config_validate(config);
    if (status != NIYAH_OK) {
        return status;
    }
    if (layer_index >= config->n_layers) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    dim = (size_t)config->embedding_dim;
    if (!niyah_checked_mul(dim, dim, &q_count) ||
        !niyah_checked_mul(layout->kv_dim, dim, &kv_count) ||
        !niyah_checked_mul((size_t)config->ffn_hidden_dim, dim, &ffn_up_count) ||
        !niyah_checked_mul(dim, (size_t)config->ffn_hidden_dim, &ffn_down_count) ||
        !niyah_checked_mul((size_t)layer_index, layout->layer_stride, &layer_delta) ||
        !niyah_checked_add(layout->layers, layer_delta, &cursor)) {
        return NIYAH_ERR_OVERFLOW;
    }

    status = niyah_advance(&cursor, dim, &layer->attn_norm);
    if (status != NIYAH_OK) return status;
    status = niyah_advance(&cursor, q_count, &layer->wq);
    if (status != NIYAH_OK) return status;
    status = niyah_advance(&cursor, kv_count, &layer->wk);
    if (status != NIYAH_OK) return status;
    status = niyah_advance(&cursor, kv_count, &layer->wv);
    if (status != NIYAH_OK) return status;
    status = niyah_advance(&cursor, q_count, &layer->wo);
    if (status != NIYAH_OK) return status;
    status = niyah_advance(&cursor, dim, &layer->ffn_norm);
    if (status != NIYAH_OK) return status;
    status = niyah_advance(&cursor, ffn_up_count, &layer->w_gate);
    if (status != NIYAH_OK) return status;
    status = niyah_advance(&cursor, ffn_up_count, &layer->w_up);
    if (status != NIYAH_OK) return status;
    status = niyah_advance(&cursor, ffn_down_count, &layer->w_down);
    if (status != NIYAH_OK) return status;

    if (cursor != layout->layers + ((size_t)layer_index + 1U) * layout->layer_stride) {
        return NIYAH_ERR_INVALID_CONFIG;
    }
    return NIYAH_OK;
}

NiyahStatus niyah_model_create(NiyahModel *model, const NiyahModelConfig *config)
{
    NiyahModelLayout layout;
    size_t bytes = 0U;
    NiyahStatus status;

    if (model == NULL || config == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    memset(model, 0, sizeof(*model));

    status = niyah_model_layout_compute(config, &layout);
    if (status != NIYAH_OK) {
        return status;
    }
    if (!niyah_checked_mul(layout.total_floats, sizeof(float), &bytes)) {
        return NIYAH_ERR_OVERFLOW;
    }

    model->weights = (float *)calloc(1U, bytes);
    if (model->weights == NULL) {
        return NIYAH_ERR_OUT_OF_MEMORY;
    }
    model->config = *config;
    model->layout = layout;
    model->weight_count = layout.total_floats;
    model->storage_kind = NIYAH_MODEL_STORAGE_OWNED;
    return NIYAH_OK;
}

void niyah_model_destroy(NiyahModel *model)
{
    if (model == NULL) {
        return;
    }
    if (model->storage_kind == NIYAH_MODEL_STORAGE_CASPER_MMAP &&
        model->mapping_base != NULL) {
#if defined(_WIN32)
        (void)UnmapViewOfFile(model->mapping_base);
#else
        if (model->mapping_bytes != 0U) {
            (void)munmap(model->mapping_base, model->mapping_bytes);
        }
#endif
    } else {
        free(model->weights);
    }
    memset(model, 0, sizeof(*model));
}

static uint64_t niyah_rng_next(uint64_t *state)
{
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * UINT64_C(2685821657736338717);
}

static float niyah_random_weight(uint64_t *state)
{
    const uint64_t bits = (niyah_rng_next(state) >> 40) & UINT64_C(0xFFFFFF);
    const float unit = (float)bits / 16777215.0f;
    return (unit * 2.0f - 1.0f) * 0.02f;
}

NiyahStatus niyah_model_reset_parameters(NiyahModel *model, uint64_t seed)
{
    uint64_t state;
    uint32_t layer_index;
    size_t i;
    NiyahLayerLayout layer;
    NiyahStatus status;

    if (model == NULL || model->weights == NULL || model->weight_count == 0U ||
        model->storage_kind != NIYAH_MODEL_STORAGE_OWNED) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    state = seed != 0U ? seed : UINT64_C(0x4e495941485f4c4d);
    for (i = 0U; i < model->weight_count; ++i) {
        model->weights[i] = niyah_random_weight(&state);
    }

    for (layer_index = 0U; layer_index < model->config.n_layers; ++layer_index) {
        status = niyah_model_layer_layout(&model->config, &model->layout, layer_index, &layer);
        if (status != NIYAH_OK) {
            return status;
        }
        for (i = 0U; i < (size_t)model->config.embedding_dim; ++i) {
            model->weights[layer.attn_norm + i] = 1.0f;
            model->weights[layer.ffn_norm + i] = 1.0f;
        }
    }
    for (i = 0U; i < (size_t)model->config.embedding_dim; ++i) {
        model->weights[model->layout.final_norm + i] = 1.0f;
    }
    return NIYAH_OK;
}


int niyah_model_is_read_only(const NiyahModel *model)
{
    return model != NULL &&
           model->storage_kind == NIYAH_MODEL_STORAGE_CASPER_MMAP;
}

const float *niyah_model_token_embedding_weights(const NiyahModel *model)
{
    size_t offset;
    if (model == NULL || model->weights == NULL) return NULL;
    if (model->storage_kind == NIYAH_MODEL_STORAGE_CASPER_MMAP) {
        offset = (size_t)model->config.n_layers * model->layout.layer_stride;
        return model->weights + offset;
    }
    return model->weights + model->layout.token_embedding;
}

const float *niyah_model_segment_embedding_weights(const NiyahModel *model)
{
    if (model == NULL || model->weights == NULL ||
        model->storage_kind != NIYAH_MODEL_STORAGE_OWNED) return NULL;
    return model->weights + model->layout.segment_embedding;
}
const float *niyah_model_final_norm_weights(const NiyahModel *model)
{
    size_t offset;
    size_t token_count;
    if (model == NULL || model->weights == NULL) return NULL;
    if (model->storage_kind == NIYAH_MODEL_STORAGE_CASPER_MMAP) {
        token_count = (size_t)model->config.vocab_size *
                      (size_t)model->config.embedding_dim;
        offset = (size_t)model->config.n_layers * model->layout.layer_stride +
                 token_count;
        return model->weights + offset;
    }
    return model->weights + model->layout.final_norm;
}

const float *niyah_model_lm_head_weights(const NiyahModel *model)
{
    const float *final_norm;
    if (model == NULL || model->weights == NULL) return NULL;
    if (model->storage_kind == NIYAH_MODEL_STORAGE_CASPER_MMAP) {
        final_norm = niyah_model_final_norm_weights(model);
        return final_norm == NULL ? NULL :
               final_norm + (size_t)model->config.embedding_dim;
    }
    return model->weights + model->layout.lm_head;
}
NiyahStatus niyah_model_layer_weights_view(const NiyahModel *model,
                                           uint32_t layer_index,
                                           NiyahLayerWeightsView *out)
{
    NiyahLayerLayout layer;
    size_t cursor;
    const size_t dim = model != NULL ? (size_t)model->config.embedding_dim : 0U;
    const size_t kv = model != NULL ? model->layout.kv_dim : 0U;
    const size_t ffn = model != NULL ? (size_t)model->config.ffn_hidden_dim : 0U;

    if (model == NULL || out == NULL || model->weights == NULL ||
        layer_index >= model->config.n_layers) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    if (model->storage_kind == NIYAH_MODEL_STORAGE_OWNED) {
        NiyahStatus status = niyah_model_layer_layout(
            &model->config, &model->layout, layer_index, &layer);
        if (status != NIYAH_OK) return status;
        out->attn_norm = model->weights + layer.attn_norm;
        out->wq = model->weights + layer.wq;
        out->wk = model->weights + layer.wk;
        out->wv = model->weights + layer.wv;
        out->wo = model->weights + layer.wo;
        out->ffn_norm = model->weights + layer.ffn_norm;
        out->w_gate = model->weights + layer.w_gate;
        out->w_up = model->weights + layer.w_up;
        out->w_down = model->weights + layer.w_down;
        return NIYAH_OK;
    }
    if (model->storage_kind != NIYAH_MODEL_STORAGE_CASPER_MMAP) {
        return NIYAH_ERR_INVALID_CONFIG;
    }

    cursor = (size_t)layer_index * model->layout.layer_stride;
    out->wq = model->weights + cursor; cursor += dim * dim;
    out->wk = model->weights + cursor; cursor += kv * dim;
    out->wv = model->weights + cursor; cursor += kv * dim;
    out->wo = model->weights + cursor; cursor += dim * dim;
    out->w_gate = model->weights + cursor; cursor += ffn * dim;
    out->w_up = model->weights + cursor; cursor += ffn * dim;
    out->w_down = model->weights + cursor; cursor += dim * ffn;
    out->attn_norm = model->weights + cursor; cursor += dim;
    out->ffn_norm = model->weights + cursor;
    return NIYAH_OK;
}
