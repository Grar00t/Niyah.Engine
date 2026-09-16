#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif

#include "niyah/decode.h"
#include "niyah/generate.h"
#include "niyah_cuda_matvec.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
static double now_seconds(void)
{
    LARGE_INTEGER counter;
    LARGE_INTEGER frequency;

    if (!QueryPerformanceFrequency(&frequency) ||
        !QueryPerformanceCounter(&counter) ||
        frequency.QuadPart == 0) {
        return -1.0;
    }

    return (double)counter.QuadPart /
           (double)frequency.QuadPart;
}
#else
#include <time.h>
static double now_seconds(void)
{
    struct timespec ts;

    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return -1.0;
    }

    return (double)ts.tv_sec +
           (double)ts.tv_nsec / 1000000000.0;
}
#endif

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "CHECK failed line %d: %s\n", \
                __LINE__, #expr); \
        goto fail; \
    } \
} while (0)

static NiyahModelConfig benchmark_config(void)
{
    NiyahModelConfig config;

    memset(&config, 0, sizeof(config));
    config.vocab_size = 4096U;
    config.context_length = 128U;
    config.embedding_dim = 256U;
    config.n_layers = 6U;
    config.n_heads = 8U;
    config.n_kv_heads = 4U;
    config.ffn_hidden_dim = 768U;
    config.rms_norm_eps = 1.0e-5f;
    config.tie_word_embeddings = 0;
    return config;
}

int main(void)
{
    enum {
        PROMPT_COUNT = 8,
        GENERATED_COUNT = 32,
        DECODE_COUNT = 64,
        REPETITIONS = 5
    };

    const NiyahModelConfig model_config =
        benchmark_config();
    const uint32_t prompt[PROMPT_COUNT] = {
        1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U
    };

    NiyahModel model;
    NiyahCudaModelState cuda_model;
    NiyahCudaDecodeState cuda_decode;
    NiyahGenerationConfig generation_config;
    NiyahGenerationResult result;
    uint32_t *output = NULL;
    float *logits = NULL;

    double decode_seconds = 0.0;
    double generation_seconds = 0.0;
    size_t rep;
    size_t i;
    int exit_code = 1;

    memset(&model, 0, sizeof(model));
    memset(&cuda_model, 0, sizeof(cuda_model));
    memset(&cuda_decode, 0, sizeof(cuda_decode));
    memset(&generation_config, 0, sizeof(generation_config));
    memset(&result, 0, sizeof(result));

    CHECK(niyah_model_create(
              &model,
              &model_config) == NIYAH_OK);

    CHECK(niyah_model_reset_parameters(
              &model,
              UINT64_C(987654321)) == NIYAH_OK);

    CHECK(niyah_cuda_model_state_create(
              &cuda_model,
              &model) == 0);

    CHECK(niyah_cuda_decode_state_create(
              &cuda_decode,
              &cuda_model) == 0);

    output = (uint32_t *)calloc(
        GENERATED_COUNT,
        sizeof(*output));
    logits = (float *)calloc(
        (size_t)model_config.vocab_size,
        sizeof(*logits));

    CHECK(output != NULL);
    CHECK(logits != NULL);

    generation_config.max_new_tokens =
        GENERATED_COUNT;
    generation_config.eos_token = 0U;
    generation_config.stop_on_eos = 0;
    generation_config.sampler.temperature = 0.0f;
    generation_config.sampler.seed = UINT64_C(1);

    /*
     * Warm up the complete generation path once.
     */
    CHECK(niyah_cuda_decode_state_reset(
              &cuda_decode) == 0);

    CHECK(niyah_cuda_generate(
              &cuda_model,
              &cuda_decode,
              prompt,
              PROMPT_COUNT,
              &generation_config,
              output,
              GENERATED_COUNT,
              &result,
              logits,
              (size_t)model_config.vocab_size) ==
          NIYAH_OK);

    /*
     * Decode benchmark:
     * includes the existing per-token logits D2H completion boundary.
     */
    for (rep = 0U; rep < REPETITIONS; ++rep) {
        double begin;
        double end;

        CHECK(niyah_cuda_decode_state_reset(
                  &cuda_decode) == 0);

        begin = now_seconds();
        CHECK(begin >= 0.0);

        for (i = 0U; i < DECODE_COUNT; ++i) {
            const uint32_t token =
                (uint32_t)((i % 255U) + 1U);

            CHECK(niyah_cuda_decode_token(
                      &cuda_model,
                      &cuda_decode,
                      token,
                      logits,
                      (size_t)model_config.vocab_size) == 0);
        }

        end = now_seconds();
        CHECK(end >= begin);

        decode_seconds += end - begin;
    }

    /*
     * Generation benchmark:
     * measures niyah_cuda_generate() end-to-end, including prompt
     * prefill, CPU sampling, and per-token logits D2H.
     */
    for (rep = 0U; rep < REPETITIONS; ++rep) {
        double begin;
        double end;

        CHECK(niyah_cuda_decode_state_reset(
                  &cuda_decode) == 0);

        memset(&result, 0, sizeof(result));

        begin = now_seconds();
        CHECK(begin >= 0.0);

        CHECK(niyah_cuda_generate(
                  &cuda_model,
                  &cuda_decode,
                  prompt,
                  PROMPT_COUNT,
                  &generation_config,
                  output,
                  GENERATED_COUNT,
                  &result,
                  logits,
                  (size_t)model_config.vocab_size) ==
              NIYAH_OK);

        end = now_seconds();
        CHECK(end >= begin);

        CHECK(result.prompt_tokens == PROMPT_COUNT);
        CHECK(result.generated_tokens == GENERATED_COUNT);
        CHECK(result.stopped_on_eos == 0);

        generation_seconds += end - begin;
    }

    {
        const double decode_tokens =
            (double)DECODE_COUNT *
            (double)REPETITIONS;
        const double generation_runs =
            (double)REPETITIONS;
        const double generated_tokens =
            (double)GENERATED_COUNT *
            generation_runs;
        const double total_generation_tokens =
            (double)(PROMPT_COUNT + GENERATED_COUNT) *
            generation_runs;

        CHECK(decode_seconds > 0.0);
        CHECK(generation_seconds > 0.0);

        printf(
            "P7L_CUDA_GENERATION_BENCHMARK=PASS\n"
            "config vocab=%u context=%u dim=%u layers=%u "
            "heads=%u kv_heads=%u ffn=%u\n"
            "decode repetitions=%d tokens_per_run=%d "
            "total_seconds=%.6f ms_per_token=%.3f "
            "tokens_per_second=%.2f\n"
            "generation repetitions=%d prompt_tokens=%d "
            "generated_tokens=%d total_seconds=%.6f "
            "end_to_end_ms_per_generated_token=%.3f "
            "end_to_end_generated_tokens_per_second=%.2f "
            "decoded_tokens_per_second=%.2f\n",
            model_config.vocab_size,
            model_config.context_length,
            model_config.embedding_dim,
            model_config.n_layers,
            model_config.n_heads,
            model_config.n_kv_heads,
            model_config.ffn_hidden_dim,
            REPETITIONS,
            DECODE_COUNT,
            decode_seconds,
            (decode_seconds * 1000.0) / decode_tokens,
            decode_tokens / decode_seconds,
            REPETITIONS,
            PROMPT_COUNT,
            GENERATED_COUNT,
            generation_seconds,
            (generation_seconds * 1000.0) /
                generated_tokens,
            generated_tokens / generation_seconds,
            total_generation_tokens /
                generation_seconds);
    }

    exit_code = 0;

fail:
    free(logits);
    free(output);
    niyah_cuda_decode_state_destroy(&cuda_decode);
    niyah_cuda_model_state_destroy(&cuda_model);
    niyah_model_destroy(&model);
    return exit_code;
}
