#include "niyah/decode.h"
#include "niyah/generate.h"
#include "niyah/sampler.h"

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

static NiyahModelConfig tiny_config(void)
{
    NiyahModelConfig config;
    memset(&config, 0, sizeof(config));
    config.vocab_size = 16U;
    config.context_length = 12U;
    config.embedding_dim = 8U;
    config.n_layers = 2U;
    config.n_heads = 4U;
    config.n_kv_heads = 2U;
    config.ffn_hidden_dim = 16U;
    config.rms_norm_eps = 1.0e-5f;
    config.tie_word_embeddings = 0;
    return config;
}

static int test_sampler_contract(void)
{
    const float greedy_logits[4] = {1.0f, 3.0f, 3.0f, 2.0f};
    const float sample_logits[4] = {0.0f, 1.0f, 2.0f, 3.0f};
    NiyahSamplerConfig config;
    NiyahSampler a;
    NiyahSampler b;
    uint32_t token = 99U;
    size_t i;

    memset(&config, 0, sizeof(config));
    config.temperature = 0.0f;
    config.seed = UINT64_C(1234);
    CHECK(niyah_sampler_init(&a, &config) == NIYAH_OK);
    CHECK(niyah_sampler_sample(&a, greedy_logits, 4U, &token) == NIYAH_OK);
    CHECK(token == 1U);

    config.temperature = 0.75f;
    config.seed = UINT64_C(0x12345678);
    CHECK(niyah_sampler_init(&a, &config) == NIYAH_OK);
    CHECK(niyah_sampler_init(&b, &config) == NIYAH_OK);
    for (i = 0U; i < 32U; ++i) {
        uint32_t ta = 0U;
        uint32_t tb = 0U;
        CHECK(niyah_sampler_sample(&a, sample_logits, 4U, &ta) == NIYAH_OK);
        CHECK(niyah_sampler_sample(&b, sample_logits, 4U, &tb) == NIYAH_OK);
        CHECK(ta == tb);
        CHECK(ta < 4U);
    }

    config.temperature = -1.0f;
    CHECK(niyah_sampler_init(&a, &config) == NIYAH_ERR_INVALID_CONFIG);
    return 0;
}

static int test_generation_matches_manual_decode(void)
{
    NiyahModelConfig model_config = tiny_config();
    NiyahModel model;
    NiyahKVCache generated_cache;
    NiyahKVCache manual_cache;
    NiyahGenerationConfig generation_config;
    NiyahGenerationResult result;
    NiyahSampler sampler;
    const uint32_t prompt[3] = {1U, 2U, 3U};
    uint32_t generated[4] = {0U, 0U, 0U, 0U};
    float manual_logits[16];
    float *generation_workspace = NULL;
    float *decode_workspace = NULL;
    size_t generation_floats = 0U;
    size_t decode_floats = 0U;
    size_t i;

    memset(&model, 0, sizeof(model));
    memset(&generated_cache, 0, sizeof(generated_cache));
    memset(&manual_cache, 0, sizeof(manual_cache));
    memset(&generation_config, 0, sizeof(generation_config));
    memset(&result, 0, sizeof(result));

    CHECK(niyah_model_create(&model, &model_config) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(&model, UINT64_C(987654321)) == NIYAH_OK);
    CHECK(niyah_kv_cache_create(&generated_cache, &model.config) == NIYAH_OK);
    CHECK(niyah_kv_cache_create(&manual_cache, &model.config) == NIYAH_OK);
    CHECK(niyah_generation_workspace_floats(&model.config, &generation_floats) == NIYAH_OK);
    CHECK(niyah_decode_workspace_floats(&model.config, &decode_floats) == NIYAH_OK);

    generation_workspace = (float *)calloc(generation_floats, sizeof(float));
    decode_workspace = (float *)calloc(decode_floats, sizeof(float));
    CHECK(generation_workspace != NULL);
    CHECK(decode_workspace != NULL);

    generation_config.max_new_tokens = 4U;
    generation_config.stop_on_eos = 0;
    generation_config.eos_token = 0U;
    generation_config.sampler.temperature = 0.0f;
    generation_config.sampler.seed = UINT64_C(1);

    CHECK(niyah_generate(&model,
                         &generated_cache,
                         prompt,
                         3U,
                         &generation_config,
                         generated,
                         4U,
                         &result,
                         generation_workspace,
                         generation_floats) == NIYAH_OK);
    CHECK(result.prompt_tokens == 3U);
    CHECK(result.generated_tokens == 4U);
    CHECK(result.stopped_on_eos == 0);
    CHECK(niyah_kv_cache_position(&generated_cache) == 7U);

    CHECK(niyah_sampler_init(&sampler, &generation_config.sampler) == NIYAH_OK);
    for (i = 0U; i < 3U; ++i) {
        CHECK(niyah_transformer_decode_token(&model,
                                             &manual_cache,
                                             prompt[i],
                                             manual_logits,
                                             16U,
                                             decode_workspace,
                                             decode_floats) == NIYAH_OK);
    }
    for (i = 0U; i < 4U; ++i) {
        uint32_t token = 0U;
        CHECK(niyah_sampler_sample(&sampler, manual_logits, 16U, &token) == NIYAH_OK);
        CHECK(generated[i] == token);
        CHECK(niyah_transformer_decode_token(&model,
                                             &manual_cache,
                                             token,
                                             manual_logits,
                                             16U,
                                             decode_workspace,
                                             decode_floats) == NIYAH_OK);
    }
    CHECK(niyah_kv_cache_position(&manual_cache) == 7U);

    free(decode_workspace);
    free(generation_workspace);
    niyah_kv_cache_destroy(&manual_cache);
    niyah_kv_cache_destroy(&generated_cache);
    niyah_model_destroy(&model);
    return 0;
}

static int test_eos_and_failure_reset(void)
{
    NiyahModelConfig model_config = tiny_config();
    NiyahModel model;
    NiyahKVCache cache;
    NiyahGenerationConfig config;
    NiyahGenerationResult result;
    uint32_t output[3] = {99U, 99U, 99U};
    const uint32_t prompt[1] = {1U};
    const uint32_t bad_prompt[2] = {1U, 99U};
    float *workspace = NULL;
    size_t workspace_floats = 0U;

    memset(&model, 0, sizeof(model));
    memset(&cache, 0, sizeof(cache));
    memset(&config, 0, sizeof(config));
    memset(&result, 0, sizeof(result));

    CHECK(niyah_model_create(&model, &model_config) == NIYAH_OK);
    CHECK(niyah_kv_cache_create(&cache, &model.config) == NIYAH_OK);
    CHECK(niyah_generation_workspace_floats(&model.config, &workspace_floats) == NIYAH_OK);
    workspace = (float *)calloc(workspace_floats, sizeof(float));
    CHECK(workspace != NULL);

    /* Zero-initialized weights produce tied zero logits; greedy selects token 0. */
    config.max_new_tokens = 3U;
    config.eos_token = 0U;
    config.stop_on_eos = 1;
    config.sampler.temperature = 0.0f;
    config.sampler.seed = UINT64_C(7);

    CHECK(niyah_generate(&model,
                         &cache,
                         prompt,
                         1U,
                         &config,
                         output,
                         3U,
                         &result,
                         workspace,
                         workspace_floats) == NIYAH_OK);
    CHECK(result.prompt_tokens == 1U);
    CHECK(result.generated_tokens == 1U);
    CHECK(result.stopped_on_eos == 1);
    CHECK(output[0] == 0U);
    CHECK(niyah_kv_cache_position(&cache) == 1U);

    niyah_kv_cache_reset(&cache);
    memset(&result, 0, sizeof(result));
    CHECK(niyah_generate(&model,
                         &cache,
                         bad_prompt,
                         2U,
                         &config,
                         output,
                         3U,
                         &result,
                         workspace,
                         workspace_floats) == NIYAH_ERR_INVALID_ARGUMENT);
    CHECK(niyah_kv_cache_position(&cache) == 0U);
    CHECK(result.prompt_tokens == 0U);
    CHECK(result.generated_tokens == 0U);
    CHECK(result.stopped_on_eos == 0);

    free(workspace);
    niyah_kv_cache_destroy(&cache);
    niyah_model_destroy(&model);
    return 0;
}

int main(void)
{
    CHECK(test_sampler_contract() == 0);
    CHECK(test_generation_matches_manual_decode() == 0);
    CHECK(test_eos_and_failure_reset() == 0);

    puts("NIYAH_GENERATION_P4=PASS");
    return 0;
}
