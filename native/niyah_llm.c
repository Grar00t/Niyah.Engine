#include "niyah.h"
#include "niyah_runtime.h"
#include "niyah_sampler_internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static NiyahStatus niyah_llm_validate_config(const NiyahModelConfig* config)
{
    if (!config) {
        return NIYAH_ERR_INVALID_ARG;
    }

    if (config->n_vocab <= 0 || config->n_vocab > NIYAH_MAX_VOCAB ||
        config->n_embd <= 0 || config->n_embd > NIYAH_MAX_DIM ||
        config->n_head <= 0 || config->n_head > NIYAH_MAX_HEADS ||
        config->n_layer <= 0 || config->n_layer > NIYAH_MAX_LAYERS ||
        config->n_ctx <= 0 || config->n_ctx > NIYAH_MAX_SEQ_LEN ||
        config->n_kv_head <= 0 || config->n_kv_head > config->n_head ||
        config->n_ff <= 0) {
        return NIYAH_ERR_SHAPE;
    }
    if (config->n_embd % config->n_head != 0 ||
        config->n_head % config->n_kv_head != 0) {
        return NIYAH_ERR_SHAPE;
    }

    return NIYAH_OK;
}

static NiyahStatus niyah_llm_map_weights(NiyahModelWeights* out,
                                         const NiyahModel* model,
                                         NiyahLayerWeights* layers,
                                         int32_t layer_capacity)
{
    if (!out || !model || !layers) {
        return NIYAH_ERR_INVALID_ARG;
    }
    if (!model->weights) {
        return NIYAH_ERR_NO_WEIGHTS;
    }

    NiyahModelConfig c = model->config;
    niyah_model_config_normalize(&c);

    const NiyahStatus config_status = niyah_llm_validate_config(&c);
    if (config_status != NIYAH_OK || layer_capacity < c.n_layer) {
        return config_status == NIYAH_OK ? NIYAH_ERR_SHAPE : config_status;
    }

    const size_t dim = (size_t)c.n_embd;
    const size_t head_dim = dim / (size_t)c.n_head;
    const size_t kv_dim = (size_t)c.n_kv_head * head_dim;
    const size_t ff = (size_t)c.n_ff;
    const size_t vocab = (size_t)c.n_vocab;

    if (model->weights_size % sizeof(float) != 0u) {
        return NIYAH_ERR_SHAPE;
    }

    const size_t expected_floats = niyah_model_expected_floats(&c);
    if (expected_floats == 0u || (dim != 0u && vocab > SIZE_MAX / dim)) {
        return NIYAH_ERR_SHAPE;
    }

    size_t required_floats = expected_floats;
    if (c.tie_word_embeddings) {
        const size_t lm_head_floats = vocab * dim;
        if (required_floats < lm_head_floats) {
            return NIYAH_ERR_SHAPE;
        }
        required_floats -= lm_head_floats;
    }

    if (model->weights_size / sizeof(float) != required_floats) {
        return NIYAH_ERR_SHAPE;
    }

    memset(out, 0, sizeof(*out));
    memset(layers, 0, (size_t)c.n_layer * sizeof(*layers));
    out->layers = layers;
    out->n_layer = c.n_layer;

    const float* p = (const float*)model->weights;
    out->embedding = p;
    p += vocab * dim;

    for (int32_t l = 0; l < c.n_layer; ++l) {
        NiyahLayerWeights* w = &layers[l];
        w->attn_norm = p; p += dim;
        w->wq        = p; p += dim * dim;
        w->wk        = p; p += kv_dim * dim;
        w->wv        = p; p += kv_dim * dim;
        w->wo        = p; p += dim * dim;
        w->ffn_norm  = p; p += dim;
        w->ffn_gate  = p; p += ff * dim;
        w->ffn_up    = p; p += ff * dim;
        w->ffn_down  = p; p += dim * ff;
    }

    out->final_norm = p;
    p += dim;
    out->lm_head = c.tie_word_embeddings ? out->embedding : p;
    return NIYAH_OK;
}

static NiyahStatus niyah_llm_runtime_acquire(NiyahLLM* llm,
                                             bool* initialized_here,
                                             NiyahRuntimeConfig* saved_config)
{
    if (!llm || !initialized_here || !saved_config) {
        return NIYAH_ERR_INVALID_ARG;
    }

    *initialized_here = false;
    *saved_config = llm->runtime.config;

    if (llm->runtime.context) {
        return niyah_runtime_capacity(&llm->runtime) > 0u
            ? NIYAH_OK : NIYAH_ERR_INVALID_ARG;
    }

    const NiyahStatus status =
        niyah_runtime_init_inplace(&llm->runtime, saved_config);
    if (status == NIYAH_OK) {
        *initialized_here = true;
    }
    return status;
}

static void niyah_llm_runtime_release(NiyahLLM* llm,
                                      bool initialized_here,
                                      const NiyahRuntimeConfig* saved_config)
{
    if (!llm || !initialized_here || !saved_config) {
        return;
    }

    niyah_runtime_deinit_inplace(&llm->runtime);
    llm->runtime.config = *saved_config;
}

static NiyahStatus niyah_llm_forward_mapped(NiyahLLM* llm,
                                            int32_t token,
                                            int32_t position,
                                            NiyahKVCache* cache,
                                            float* logits,
                                            const NiyahModelWeights* weights)
{
    if (!llm || !cache || !logits || !weights || !weights->layers) {
        return NIYAH_ERR_INVALID_ARG;
    }
    if (!llm->model.weights) {
        return NIYAH_ERR_NO_WEIGHTS;
    }

    NiyahModelConfig c = llm->model.config;
    niyah_model_config_normalize(&c);

    NiyahStatus status = niyah_llm_validate_config(&c);
    if (status != NIYAH_OK) {
        return status;
    }
    if (token < 0 || token >= c.n_vocab ||
        position < 0 || position >= c.n_ctx) {
        return NIYAH_ERR_INVALID_ARG;
    }

    const size_t mark = niyah_runtime_used(&llm->runtime);
    const size_t scratch_floats = niyah_transformer_scratch_floats(&c);
    if (scratch_floats == 0u) {
        return NIYAH_ERR_SHAPE;
    }

    float* hidden = niyah_runtime_alloc_floats(&llm->runtime,
                                               (size_t)c.n_embd);
    float* scratch = niyah_runtime_alloc_floats(&llm->runtime,
                                                scratch_floats);
    if (!hidden || !scratch) {
        status = NIYAH_ERR_OUT_OF_MEMORY;
        goto cleanup;
    }

    memcpy(hidden,
           weights->embedding + (size_t)token * (size_t)c.n_embd,
           (size_t)c.n_embd * sizeof(float));

    for (int32_t l = 0; l < c.n_layer; ++l) {
        status = niyah_transformer_layer_forward_weighted(
            hidden, &weights->layers[l], &c, cache, l, position, scratch);
        if (status != NIYAH_OK) {
            goto cleanup;
        }
    }

    niyah_rmsnorm(hidden, weights->final_norm, c.n_embd, c.norm_eps);
    niyah_matvec(logits, weights->lm_head, hidden, c.n_vocab, c.n_embd);

cleanup:
    {
        const NiyahStatus rewind_status =
            niyah_runtime_rewind(&llm->runtime, mark);
        if (status == NIYAH_OK && rewind_status != NIYAH_OK) {
            status = rewind_status;
        }
    }
    return status;
}

NiyahStatus niyah_llm_forward(NiyahLLM* llm,
                              int32_t token,
                              int32_t position,
                              NiyahKVCache* cache,
                              float* logits,
                              float* scratch)
{
    (void)scratch;

    if (!llm) {
        return NIYAH_ERR_INVALID_ARG;
    }
    if (!llm->model.weights) {
        return NIYAH_ERR_NO_WEIGHTS;
    }
    if (!cache || !logits) {
        return NIYAH_ERR_INVALID_ARG;
    }

    NiyahModelConfig c = llm->model.config;
    niyah_model_config_normalize(&c);

    NiyahStatus status = niyah_llm_validate_config(&c);
    if (status != NIYAH_OK) {
        return status;
    }
    if (token < 0 || token >= c.n_vocab ||
        position < 0 || position >= c.n_ctx) {
        return NIYAH_ERR_INVALID_ARG;
    }

    bool initialized_here = false;
    NiyahRuntimeConfig saved_config;
    status = niyah_llm_runtime_acquire(llm, &initialized_here, &saved_config);
    if (status != NIYAH_OK) {
        return status;
    }

    const size_t mark = niyah_runtime_used(&llm->runtime);
    NiyahLayerWeights* layers = (NiyahLayerWeights*)niyah_runtime_alloc(
        &llm->runtime, (size_t)c.n_layer * sizeof(NiyahLayerWeights));
    if (!layers) {
        status = NIYAH_ERR_OUT_OF_MEMORY;
        goto cleanup;
    }

    NiyahModelWeights weights;
    status = niyah_llm_map_weights(&weights, &llm->model,
                                   layers, c.n_layer);
    if (status == NIYAH_OK) {
        status = niyah_llm_forward_mapped(llm, token, position,
                                          cache, logits, &weights);
    }

cleanup:
    {
        const NiyahStatus rewind_status =
            niyah_runtime_rewind(&llm->runtime, mark);
        if (status == NIYAH_OK && rewind_status != NIYAH_OK) {
            status = rewind_status;
        }
    }
    niyah_llm_runtime_release(llm, initialized_here, &saved_config);
    return status;
}

NiyahLLMOutput niyah_llm_generate(NiyahLLM* llm,
                                  const char* prompt,
                                  int32_t max_tokens)
{
    NiyahLLMOutput output;
    memset(&output, 0, sizeof(output));
    output.status = NIYAH_ERR_INVALID_ARG;

    if (!llm || !prompt || max_tokens <= 0) {
        return output;
    }
    if (!llm->model.weights) {
        output.status = NIYAH_ERR_NO_WEIGHTS;
        return output;
    }

    NiyahModelConfig c = llm->model.config;
    niyah_model_config_normalize(&c);

    NiyahStatus status = niyah_llm_validate_config(&c);
    if (status != NIYAH_OK) {
        output.status = status;
        return output;
    }

    const int32_t head_dim = c.n_embd / c.n_head;
    bool initialized_here = false;
    NiyahRuntimeConfig saved_config;
    status = niyah_llm_runtime_acquire(llm, &initialized_here, &saved_config);
    if (status != NIYAH_OK) {
        output.status = status;
        return output;
    }

    const size_t arena_mark = niyah_runtime_used(&llm->runtime);
    int32_t* const saved_tokenizer_tokens = llm->tokenizer.tokens;
    const int32_t saved_tokenizer_n_tokens = llm->tokenizer.n_tokens;
    bool telemetry_started = false;
    bool cache_initialized = false;

    NiyahKVCache cache;
    memset(&cache, 0, sizeof(cache));

    NiyahLayerWeights* layers = (NiyahLayerWeights*)niyah_runtime_alloc(
        &llm->runtime, (size_t)c.n_layer * sizeof(NiyahLayerWeights));
    int32_t* tokens = (int32_t*)niyah_runtime_alloc(
        &llm->runtime, (size_t)c.n_ctx * sizeof(int32_t));
    float* logits = niyah_runtime_alloc_floats(&llm->runtime,
                                               (size_t)c.n_vocab);
    float* sampler_probs = niyah_runtime_alloc_floats(&llm->runtime,
                                                      (size_t)c.n_vocab);
    NiyahSamplerCandidate* sampler_pool =
        (NiyahSamplerCandidate*)niyah_runtime_alloc(
            &llm->runtime,
            (size_t)c.n_vocab * sizeof(NiyahSamplerCandidate));

    if (!layers || !tokens || !logits || !sampler_probs || !sampler_pool) {
        status = NIYAH_ERR_OUT_OF_MEMORY;
        goto cleanup;
    }

    NiyahModelWeights weights;
    status = niyah_llm_map_weights(&weights, &llm->model,
                                   layers, c.n_layer);
    if (status != NIYAH_OK) {
        goto cleanup;
    }

    niyah_telemetry_start(&output.telemetry);
    telemetry_started = true;

    /*
     * KV cache is generation-scoped arena memory. The standalone calloc
     * initializer remains available for tests without a NiyahRuntime.
     */
    status = niyah_kv_cache_init_arena(&cache, &llm->runtime,
                                       c.n_layer, c.n_kv_head,
                                       head_dim, c.n_ctx);
    if (status != NIYAH_OK) {
        goto cleanup;
    }
    cache_initialized = true;

    int32_t n_prompt = niyah_tokenize(&llm->tokenizer, prompt,
                                      tokens, c.n_ctx);
    if (n_prompt <= 0) {
        if (c.bos_token_id > 0 && c.bos_token_id < c.n_vocab) {
            tokens[0] = c.bos_token_id;
            n_prompt = 1;
        } else {
            status = NIYAH_ERR_INVALID_ARG;
            goto cleanup;
        }
    }

    int32_t position = 0;
    for (; position < n_prompt && position < c.n_ctx; ++position) {
        status = niyah_llm_forward_mapped(llm, tokens[position], position,
                                          &cache, logits, &weights);
        if (status != NIYAH_OK) {
            goto cleanup;
        }
    }

    int32_t total = n_prompt;
    int32_t generated = 0;

    while (generated < max_tokens && total < c.n_ctx) {
        niyah_sampler_apply_repetition_penalty(logits, c.n_vocab,
                                               tokens, total, 1.1f);

        const int32_t next = niyah_sample_with_scratch(
            logits, c.n_vocab, &llm->sampler,
            sampler_probs, sampler_pool, c.n_vocab);
        if (next < 0 || next >= c.n_vocab) {
            status = NIYAH_ERR_SHAPE;
            break;
        }

        tokens[total++] = next;
        ++generated;

        if (c.eos_token_id > 0 && next == c.eos_token_id) {
            break;
        }
        if (total >= c.n_ctx) {
            break;
        }

        status = niyah_llm_forward_mapped(llm, next, total - 1,
                                          &cache, logits, &weights);
        if (status != NIYAH_OK) {
            break;
        }
    }

    /*
     * Returned text/logits are caller-owned heap materialization performed
     * only after the token loop; no output allocation occurs inside decode.
     */
    if (status == NIYAH_OK && generated > 0) {
        char* text = niyah_detokenize(&llm->tokenizer,
                                      tokens + n_prompt, generated);
        float* result_logits = (float*)malloc((size_t)c.n_vocab * sizeof(float));
        if (!text || !result_logits) {
            free(text);
            free(result_logits);
            status = NIYAH_ERR_OUT_OF_MEMORY;
        } else {
            memcpy(result_logits, logits,
                   (size_t)c.n_vocab * sizeof(float));
            output.text = text;
            output.logits = result_logits;
            output.n_tokens = generated;
        }
    } else if (status == NIYAH_OK) {
        status = NIYAH_ERR_SHAPE;
    }

cleanup:
    if (telemetry_started) {
        niyah_telemetry_end(&output.telemetry);
    }
    output.telemetry.tokens_processed = output.n_tokens;
    output.telemetry.memory_used = (int64_t)llm->model.weights_size;

    if (cache_initialized) {
        niyah_kv_cache_free(&cache);
    }

    llm->tokenizer.tokens = saved_tokenizer_tokens;
    llm->tokenizer.n_tokens = saved_tokenizer_n_tokens;

    {
        const NiyahStatus rewind_status =
            niyah_runtime_rewind(&llm->runtime, arena_mark);
        if (status == NIYAH_OK && rewind_status != NIYAH_OK) {
            status = rewind_status;
        }
    }

    niyah_llm_runtime_release(llm, initialized_here, &saved_config);

    if (status != NIYAH_OK) {
        free(output.text);
        free(output.logits);
        output.text = NULL;
        output.logits = NULL;
        output.n_tokens = 0;
    }
    output.status = status;
    return output;
}

void niyah_llm_output_free(NiyahLLMOutput* output)
{
    if (!output) {
        return;
    }
    free(output->text);
    free(output->logits);
    output->text = NULL;
    output->logits = NULL;
    output->n_tokens = 0;
}
