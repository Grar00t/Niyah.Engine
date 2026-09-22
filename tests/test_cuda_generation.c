#include "niyah/decode.h"
#include "niyah/generate.h"
#include "niyah_cuda_matvec.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "CHECK failed line %d: %s\n", __LINE__, #expr); \
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

static int results_equal(
    const NiyahGenerationResult *a,
    const NiyahGenerationResult *b)
{
    return a->prompt_tokens == b->prompt_tokens &&
           a->generated_tokens == b->generated_tokens &&
           a->stopped_on_eos == b->stopped_on_eos;
}

int main(void)
{
    const uint32_t prompt[3] = {1U, 2U, 3U};
    const uint32_t bad_prompt[2] = {1U, 99U};
    const NiyahModelConfig model_config = tiny_config();
    NiyahModel model;
    NiyahKVCache cpu_cache;
    NiyahCudaModelState cuda_model;
    NiyahCudaDecodeState cuda_decode;
    NiyahGenerationConfig config;
    NiyahGenerationResult cpu_result;
    NiyahGenerationResult gpu_result;
    uint32_t cpu_output[4];
    uint32_t gpu_output[4];
    float *cpu_workspace = NULL;
    float gpu_logits[16];
    size_t cpu_workspace_count = 0U;
    size_t i;

    memset(&model, 0, sizeof(model));
    memset(&cpu_cache, 0, sizeof(cpu_cache));
    memset(&cuda_model, 0, sizeof(cuda_model));
    memset(&cuda_decode, 0, sizeof(cuda_decode));
    memset(&config, 0, sizeof(config));
    memset(&cpu_result, 0, sizeof(cpu_result));
    memset(&gpu_result, 0, sizeof(gpu_result));
    memset(cpu_output, 0, sizeof(cpu_output));
    memset(gpu_output, 0, sizeof(gpu_output));
    memset(gpu_logits, 0, sizeof(gpu_logits));

    CHECK(niyah_model_create(
              &model,
              &model_config) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(
              &model,
              UINT64_C(987654321)) == NIYAH_OK);

    CHECK(niyah_kv_cache_create(
              &cpu_cache,
              &model.config) == NIYAH_OK);

    CHECK(niyah_generation_workspace_floats(
              &model.config,
              &cpu_workspace_count) == NIYAH_OK);

    cpu_workspace = (float *)calloc(
        cpu_workspace_count,
        sizeof(float));
    CHECK(cpu_workspace != NULL);

    CHECK(niyah_cuda_model_state_create(
              &cuda_model,
              &model) == 0);
    CHECK(niyah_cuda_decode_state_create(
              &cuda_decode,
              &cuda_model) == 0);

    config.max_new_tokens = 4U;
    config.eos_token = 0U;
    config.stop_on_eos = 0;
    config.sampler.temperature = 0.0f;
    config.sampler.seed = UINT64_C(1);

    CHECK(niyah_generate(
              &model,
              &cpu_cache,
              prompt,
              3U,
              &config,
              cpu_output,
              4U,
              &cpu_result,
              cpu_workspace,
              cpu_workspace_count) == NIYAH_OK);

    CHECK(niyah_cuda_generate(
              &cuda_model,
              &cuda_decode,
              prompt,
              3U,
              &config,
              gpu_output,
              4U,
              &gpu_result,
              gpu_logits,
              16U) == NIYAH_OK);

    CHECK(results_equal(
              &cpu_result,
              &gpu_result));

    for (i = 0U; i < 4U; ++i) {
        CHECK(cpu_output[i] == gpu_output[i]);
    }

    CHECK(niyah_kv_cache_position(&cpu_cache) ==
          cuda_decode.next_position);
    CHECK(cuda_decode.next_position == 7U);

    niyah_kv_cache_reset(&cpu_cache);
    CHECK(niyah_cuda_decode_state_reset(
              &cuda_decode) == 0);

    memset(&cpu_result, 0, sizeof(cpu_result));
    memset(&gpu_result, 0, sizeof(gpu_result));
    memset(cpu_output, 0, sizeof(cpu_output));
    memset(gpu_output, 0, sizeof(gpu_output));

    config.max_new_tokens = 4U;
    config.eos_token = 0U;
    config.stop_on_eos = 0;
    config.sampler.temperature = 0.75f;
    config.sampler.seed = UINT64_C(12345);

    CHECK(niyah_generate(
              &model,
              &cpu_cache,
              prompt,
              3U,
              &config,
              cpu_output,
              4U,
              &cpu_result,
              cpu_workspace,
              cpu_workspace_count) == NIYAH_OK);

    CHECK(niyah_cuda_generate(
              &cuda_model,
              &cuda_decode,
              prompt,
              3U,
              &config,
              gpu_output,
              4U,
              &gpu_result,
              gpu_logits,
              16U) == NIYAH_OK);

    CHECK(results_equal(
              &cpu_result,
              &gpu_result));

    for (i = 0U; i < 4U; ++i) {
        CHECK(cpu_output[i] == gpu_output[i]);
    }

    CHECK(niyah_kv_cache_position(&cpu_cache) ==
          cuda_decode.next_position);
    CHECK(cuda_decode.next_position == 7U);

    /*
     * Zero weights force tied zero logits. Greedy chooses token 0,
     * exercising identical EOS behavior without relying on approximate
     * floating-point ordering.
     */
    memset(
        model.weights,
        0,
        model.weight_count * sizeof(float));

    CHECK(niyah_cuda_model_state_sync(
              &cuda_model,
              &model) == 0);

    niyah_kv_cache_reset(&cpu_cache);
    CHECK(niyah_cuda_decode_state_reset(
              &cuda_decode) == 0);

    memset(&cpu_result, 0, sizeof(cpu_result));
    memset(&gpu_result, 0, sizeof(gpu_result));
    memset(cpu_output, 0, sizeof(cpu_output));
    memset(gpu_output, 0, sizeof(gpu_output));

    config.max_new_tokens = 3U;
    config.eos_token = 0U;
    config.stop_on_eos = 1;
    config.sampler.temperature = 0.0f;
    config.sampler.seed = UINT64_C(7);

    CHECK(niyah_generate(
              &model,
              &cpu_cache,
              prompt,
              1U,
              &config,
              cpu_output,
              4U,
              &cpu_result,
              cpu_workspace,
              cpu_workspace_count) == NIYAH_OK);

    CHECK(niyah_cuda_generate(
              &cuda_model,
              &cuda_decode,
              prompt,
              1U,
              &config,
              gpu_output,
              4U,
              &gpu_result,
              gpu_logits,
              16U) == NIYAH_OK);

    CHECK(results_equal(
              &cpu_result,
              &gpu_result));
    CHECK(cpu_result.generated_tokens == 1U);
    CHECK(cpu_result.stopped_on_eos == 1);
    CHECK(cpu_output[0] == 0U);
    CHECK(gpu_output[0] == 0U);
    CHECK(niyah_kv_cache_position(&cpu_cache) == 1U);
    CHECK(cuda_decode.next_position == 1U);

    /*
     * Failure is fail-closed: both generation states return to position 0
     * and the result is zeroed.
     */
    niyah_kv_cache_reset(&cpu_cache);
    CHECK(niyah_cuda_decode_state_reset(
              &cuda_decode) == 0);

    memset(&cpu_result, 0xA5, sizeof(cpu_result));
    memset(&gpu_result, 0xA5, sizeof(gpu_result));

    CHECK(niyah_generate(
              &model,
              &cpu_cache,
              bad_prompt,
              2U,
              &config,
              cpu_output,
              4U,
              &cpu_result,
              cpu_workspace,
              cpu_workspace_count) ==
          NIYAH_ERR_INVALID_ARGUMENT);

    CHECK(niyah_cuda_generate(
              &cuda_model,
              &cuda_decode,
              bad_prompt,
              2U,
              &config,
              gpu_output,
              4U,
              &gpu_result,
              gpu_logits,
              16U) ==
          NIYAH_ERR_INVALID_ARGUMENT);

    CHECK(niyah_kv_cache_position(&cpu_cache) == 0U);
    CHECK(cuda_decode.next_position == 0U);

    CHECK(cpu_result.prompt_tokens == 0U);
    CHECK(cpu_result.generated_tokens == 0U);
    CHECK(cpu_result.stopped_on_eos == 0);

    CHECK(gpu_result.prompt_tokens == 0U);
    CHECK(gpu_result.generated_tokens == 0U);
    CHECK(gpu_result.stopped_on_eos == 0);

    niyah_cuda_decode_state_destroy(&cuda_decode);
    niyah_cuda_model_state_destroy(&cuda_model);
    niyah_kv_cache_destroy(&cpu_cache);
    niyah_model_destroy(&model);
    free(cpu_workspace);

    puts("P7F_CUDA_GENERATION_BACKEND=PASS");
    return 0;
}
