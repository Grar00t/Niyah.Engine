#include "niyah/decode.h"
#include "niyah/generate.h"
#include "niyah/tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expr) \
    do { \
        if (!(expr)) { \
            fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
            return 1; \
        } \
    } while (0)

int main(void)
{
    static const uint8_t corpus[] =
        "hello world hello world native model native model\n"
        "\xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7 \xd8\xa8\xd8\xa7\xd9\x84\xd8\xb9\xd8\xa7\xd9\x84\xd9\x85 "
        "\xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7 \xd8\xa8\xd8\xa7\xd9\x84\xd8\xb9\xd8\xa7\xd9\x84\xd9\x85\n";
    static const uint8_t prompt[] =
        "hello \xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7";
    NiyahTokenizerTrainConfig tokenizer_config;
    NiyahTokenizer *tokenizer = NULL;
    NiyahModelConfig model_config;
    NiyahModel model;
    NiyahKVCache cache;
    NiyahGenerationConfig generation_config;
    NiyahGenerationResult generation_result;
    size_t vocab_size;
    uint32_t prompt_tokens[128];
    size_t prompt_count = 0U;
    size_t decode_workspace_count = 0U;
    size_t generation_workspace_count = 0U;
    float *decode_workspace = NULL;
    float *generation_workspace = NULL;
    float *logits = NULL;
    uint32_t generated[4] = {0U, 0U, 0U, 0U};
    size_t decoded_size = 0U;
    uint8_t *decoded = NULL;
    size_t i;

    memset(&tokenizer_config, 0, sizeof(tokenizer_config));
    tokenizer_config.target_vocab_size = 280U;
    tokenizer_config.min_pair_frequency = 2U;

    CHECK(niyah_tokenizer_train(corpus,
                                sizeof(corpus) - 1U,
                                &tokenizer_config,
                                &tokenizer) == NIYAH_OK);
    CHECK(tokenizer != NULL);

    vocab_size = niyah_tokenizer_vocab_size(tokenizer);
    CHECK(vocab_size >= (size_t)NIYAH_TOKENIZER_BASE_VOCAB_SIZE);
    CHECK(vocab_size <= (size_t)tokenizer_config.target_vocab_size);
    CHECK(vocab_size <= (size_t)UINT32_MAX);

    memset(&model_config, 0, sizeof(model_config));
    model_config.vocab_size = (uint32_t)vocab_size;
    model_config.context_length = 64U;
    model_config.embedding_dim = 8U;
    model_config.n_layers = 1U;
    model_config.n_heads = 4U;
    model_config.n_kv_heads = 2U;
    model_config.ffn_hidden_dim = 16U;
    model_config.rms_norm_eps = 1.0e-5f;
    model_config.tie_word_embeddings = 1;

    memset(&model, 0, sizeof(model));
    memset(&cache, 0, sizeof(cache));
    memset(&generation_config, 0, sizeof(generation_config));
    memset(&generation_result, 0, sizeof(generation_result));

    CHECK(niyah_model_create(&model, &model_config) == NIYAH_OK);
    CHECK(model.config.vocab_size == (uint32_t)vocab_size);
    CHECK(niyah_model_reset_parameters(&model, UINT64_C(20260916)) == NIYAH_OK);

    CHECK(niyah_tokenizer_encode(tokenizer,
                                 prompt,
                                 sizeof(prompt) - 1U,
                                 NULL,
                                 0U,
                                 &prompt_count) == NIYAH_OK);
    CHECK(prompt_count > 0U);
    CHECK(prompt_count <= (sizeof(prompt_tokens) / sizeof(prompt_tokens[0])));
    CHECK(niyah_tokenizer_encode(tokenizer,
                                 prompt,
                                 sizeof(prompt) - 1U,
                                 prompt_tokens,
                                 sizeof(prompt_tokens) / sizeof(prompt_tokens[0]),
                                 &prompt_count) == NIYAH_OK);

    for (i = 0U; i < prompt_count; ++i) {
        CHECK((size_t)prompt_tokens[i] < vocab_size);
        CHECK(prompt_tokens[i] < model.config.vocab_size);
    }

    CHECK(niyah_kv_cache_create(&cache, &model.config) == NIYAH_OK);
    CHECK(niyah_decode_workspace_floats(&model.config, &decode_workspace_count) == NIYAH_OK);
    decode_workspace = (float *)calloc(decode_workspace_count, sizeof(float));
    logits = (float *)calloc(vocab_size, sizeof(float));
    CHECK(decode_workspace != NULL);
    CHECK(logits != NULL);

    for (i = 0U; i < prompt_count; ++i) {
        CHECK(niyah_transformer_decode_token(&model,
                                             &cache,
                                             prompt_tokens[i],
                                             logits,
                                             vocab_size,
                                             decode_workspace,
                                             decode_workspace_count) == NIYAH_OK);
    }
    CHECK(niyah_kv_cache_position(&cache) == prompt_count);
    niyah_kv_cache_reset(&cache);
    CHECK(niyah_kv_cache_position(&cache) == 0U);

    CHECK(niyah_generation_workspace_floats(&model.config,
                                            &generation_workspace_count) == NIYAH_OK);
    generation_workspace = (float *)calloc(generation_workspace_count, sizeof(float));
    CHECK(generation_workspace != NULL);

    generation_config.max_new_tokens = 4U;
    generation_config.eos_token = NIYAH_TOKEN_EOS;
    generation_config.stop_on_eos = 1;
    generation_config.sampler.temperature = 0.0f;
    generation_config.sampler.seed = UINT64_C(1);

    CHECK(niyah_generate(&model,
                         &cache,
                         prompt_tokens,
                         prompt_count,
                         &generation_config,
                         generated,
                         sizeof(generated) / sizeof(generated[0]),
                         &generation_result,
                         generation_workspace,
                         generation_workspace_count) == NIYAH_OK);
    CHECK(generation_result.prompt_tokens == prompt_count);
    CHECK(generation_result.generated_tokens >= 1U);
    CHECK(generation_result.generated_tokens <= generation_config.max_new_tokens);

    for (i = 0U; i < generation_result.generated_tokens; ++i) {
        CHECK((size_t)generated[i] < vocab_size);
        CHECK(generated[i] < model.config.vocab_size);
    }

    CHECK(niyah_tokenizer_decode(tokenizer,
                                 generated,
                                 generation_result.generated_tokens,
                                 NULL,
                                 0U,
                                 &decoded_size) == NIYAH_OK);
    decoded = (uint8_t *)calloc(decoded_size == 0U ? 1U : decoded_size, sizeof(uint8_t));
    CHECK(decoded != NULL);
    CHECK(niyah_tokenizer_decode(tokenizer,
                                 generated,
                                 generation_result.generated_tokens,
                                 decoded,
                                 decoded_size,
                                 &decoded_size) == NIYAH_OK);

    free(decoded);
    free(generation_workspace);
    free(logits);
    free(decode_workspace);
    niyah_kv_cache_destroy(&cache);
    niyah_model_destroy(&model);
    niyah_tokenizer_destroy(tokenizer);

    puts("NIYAH_TEXT_PIPELINE_P5_1=PASS");
    return 0;
}
