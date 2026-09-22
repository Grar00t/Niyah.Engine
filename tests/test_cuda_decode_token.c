#include "niyah/decode.h"
#include "niyah_cuda_matvec.h"

#include <cuda_runtime_api.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "CHECK failed line %d: %s\n", __LINE__, #expr); \
        return 1; \
    } \
} while (0)

static NiyahModelConfig test_config(void)
{
    NiyahModelConfig config;

    memset(&config, 0, sizeof(config));
    config.vocab_size = 64U;
    config.context_length = 40U;
    config.embedding_dim = 64U;
    config.n_layers = 1U;
    config.n_heads = 2U;
    config.n_kv_heads = 1U;
    config.ffn_hidden_dim = 96U;
    config.rms_norm_eps = 1.0e-5f;
    config.tie_word_embeddings = 0;
    return config;
}

static int vectors_close(
    const float *a,
    const float *b,
    size_t n,
    float tolerance)
{
    size_t i;

    for (i = 0U; i < n; ++i) {
        const float diff = fabsf(a[i] - b[i]);
        const float limit =
            tolerance * (1.0f + fabsf(a[i]));

        if (!isfinite(a[i]) ||
            !isfinite(b[i]) ||
            diff > limit) {
            fprintf(stderr,
                    "mismatch i=%zu cpu=%.9g gpu=%.9g diff=%.9g limit=%.9g\n",
                    i,
                    (double)a[i],
                    (double)b[i],
                    (double)diff,
                    (double)limit);
            return 0;
        }
    }

    return 1;
}

int main(void)
{
    uint32_t tokens[33];
    const size_t token_count =
        sizeof(tokens) / sizeof(tokens[0]);
    const NiyahModelConfig config = test_config();
    NiyahModel model;
    NiyahKVCache cpu_cache;
    NiyahCudaModelState cuda_model;
    NiyahCudaDecodeState cuda_decode;
    size_t cpu_workspace_count = 0U;
    float *cpu_workspace = NULL;
    float *cpu_logits = NULL;
    float *gpu_logits = NULL;
    float *gpu_keys = NULL;
    float *gpu_values = NULL;
    size_t position;
    size_t bytes;

    for (position = 0U; position < token_count; ++position) {
        tokens[position] =
            (uint32_t)(3U + (position * 7U) %
                              ((size_t)config.vocab_size - 3U));
    }

    memset(&model, 0, sizeof(model));
    memset(&cpu_cache, 0, sizeof(cpu_cache));
    memset(&cuda_model, 0, sizeof(cuda_model));
    memset(&cuda_decode, 0, sizeof(cuda_decode));

    CHECK(niyah_model_create(&model, &config) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(
              &model,
              UINT64_C(0x12345678)) == NIYAH_OK);

    CHECK(niyah_kv_cache_create(
              &cpu_cache,
              &config) == NIYAH_OK);

    CHECK(niyah_decode_workspace_floats(
              &config,
              &cpu_workspace_count) == NIYAH_OK);

    cpu_workspace = (float *)calloc(
        cpu_workspace_count,
        sizeof(float));
    cpu_logits = (float *)calloc(
        (size_t)config.vocab_size,
        sizeof(float));
    gpu_logits = (float *)calloc(
        (size_t)config.vocab_size,
        sizeof(float));

    CHECK(cpu_workspace != NULL);
    CHECK(cpu_logits != NULL);
    CHECK(gpu_logits != NULL);

    CHECK(niyah_cuda_model_state_create(
              &cuda_model,
              &model) == 0);
    CHECK(niyah_cuda_decode_state_create(
              &cuda_decode,
              &cuda_model) == 0);

    for (position = 0U;
         position < token_count;
         ++position) {
        memset(
            cpu_logits,
            0,
            (size_t)config.vocab_size * sizeof(float));
        memset(
            gpu_logits,
            0,
            (size_t)config.vocab_size * sizeof(float));

        CHECK(niyah_transformer_decode_token(
                  &model,
                  &cpu_cache,
                  tokens[position],
                  cpu_logits,
                  (size_t)config.vocab_size,
                  cpu_workspace,
                  cpu_workspace_count) == NIYAH_OK);

        CHECK(niyah_cuda_decode_token(
                  &cuda_model,
                  &cuda_decode,
                  tokens[position],
                  gpu_logits,
                  (size_t)config.vocab_size) == 0);

        CHECK(cpu_cache.next_position ==
              position + 1U);
        CHECK(cuda_decode.next_position ==
              position + 1U);

        CHECK(vectors_close(
                  cpu_logits,
                  gpu_logits,
                  (size_t)config.vocab_size,
                  1.0e-3f));
    }

    bytes =
        cuda_decode.values_per_tensor *
        sizeof(float);

    gpu_keys = (float *)malloc(bytes);
    gpu_values = (float *)malloc(bytes);
    CHECK(gpu_keys != NULL);
    CHECK(gpu_values != NULL);

    CHECK(cudaMemcpy(
              gpu_keys,
              cuda_decode.device_keys,
              bytes,
              cudaMemcpyDeviceToHost) == cudaSuccess);

    CHECK(cudaMemcpy(
              gpu_values,
              cuda_decode.device_values,
              bytes,
              cudaMemcpyDeviceToHost) == cudaSuccess);

    CHECK(vectors_close(
              cpu_cache.keys,
              gpu_keys,
              cuda_decode.values_per_tensor,
              1.0e-3f));

    CHECK(vectors_close(
              cpu_cache.values,
              gpu_values,
              cuda_decode.values_per_tensor,
              1.0e-3f));

    {
        const size_t before =
            cuda_decode.next_position;

        CHECK(niyah_cuda_decode_token(
                  &cuda_model,
                  &cuda_decode,
                  config.vocab_size,
                  gpu_logits,
                  (size_t)config.vocab_size) != 0);

        CHECK(cuda_decode.next_position == before);

        CHECK(niyah_cuda_decode_token(
                  &cuda_model,
                  &cuda_decode,
                  1U,
                  gpu_logits,
                  (size_t)config.vocab_size - 1U) != 0);

        CHECK(cuda_decode.next_position == before);
    }

    niyah_kv_cache_reset(&cpu_cache);
    CHECK(niyah_cuda_decode_state_reset(
              &cuda_decode) == 0);

    CHECK(cpu_cache.next_position == 0U);
    CHECK(cuda_decode.next_position == 0U);

    CHECK(niyah_transformer_decode_token(
              &model,
              &cpu_cache,
              tokens[0],
              cpu_logits,
              (size_t)config.vocab_size,
              cpu_workspace,
              cpu_workspace_count) == NIYAH_OK);

    CHECK(niyah_cuda_decode_token(
              &cuda_model,
              &cuda_decode,
              tokens[0],
              gpu_logits,
              (size_t)config.vocab_size) == 0);

    CHECK(vectors_close(
              cpu_logits,
              gpu_logits,
              (size_t)config.vocab_size,
              1.0e-3f));

    free(gpu_values);
    free(gpu_keys);
    free(gpu_logits);
    free(cpu_logits);
    free(cpu_workspace);

    niyah_cuda_decode_state_destroy(&cuda_decode);
    niyah_cuda_model_state_destroy(&cuda_model);
    niyah_kv_cache_destroy(&cpu_cache);
    niyah_model_destroy(&model);

    puts("P7E_CUDA_DECODE_TOKEN_PARITY=PASS");
    return 0;
}
