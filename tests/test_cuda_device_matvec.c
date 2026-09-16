#include "niyah/niyah.h"
#include "niyah_cuda_matvec.h"

#include <cuda_runtime_api.h>

#include <math.h>
#include <stdio.h>
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
    config.vocab_size = 32U;
    config.context_length = 8U;
    config.embedding_dim = 8U;
    config.n_layers = 2U;
    config.n_heads = 4U;
    config.n_kv_heads = 2U;
    config.ffn_hidden_dim = 16U;
    config.rms_norm_eps = 1.0e-5f;
    config.tie_word_embeddings = 0;
    return config;
}

static int vectors_close(const float *a, const float *b, size_t n)
{
    size_t i;

    for (i = 0U; i < n; ++i) {
        const float diff = fabsf(a[i] - b[i]);
        const float tolerance = 1.0e-4f * (1.0f + fabsf(a[i]));

        if (!isfinite(a[i]) || !isfinite(b[i]) || diff > tolerance) {
            fprintf(stderr,
                    "mismatch i=%zu cpu=%.9g gpu=%.9g diff=%.9g\n",
                    i,
                    (double)a[i],
                    (double)b[i],
                    (double)diff);
            return 0;
        }
    }

    return 1;
}

int main(void)
{
    const NiyahModelConfig config = tiny_config();
    const float input[8] = {
        1.0f, -0.5f, 0.25f, 2.0f,
        -1.0f, 0.75f, 0.125f, -0.25f
    };
    NiyahModel model;
    NiyahLayerLayout layer;
    NiyahCudaModelState cuda_model;
    NiyahCudaDecodeState decode;
    float expected[8];
    float actual[8];

    memset(&model, 0, sizeof(model));
    memset(&layer, 0, sizeof(layer));
    memset(&cuda_model, 0, sizeof(cuda_model));
    memset(&decode, 0, sizeof(decode));
    memset(expected, 0, sizeof(expected));
    memset(actual, 0, sizeof(actual));

    CHECK(niyah_model_create(&model, &config) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(
              &model, UINT64_C(0xabcdef)) == NIYAH_OK);
    CHECK(niyah_model_layer_layout(
              &model.config,
              &model.layout,
              0U,
              &layer) == NIYAH_OK);

    CHECK(niyah_cuda_model_state_create(&cuda_model, &model) == 0);
    CHECK(niyah_cuda_decode_state_create(&decode, &cuda_model) == 0);

    niyah_matvec(expected,
                 model.weights + layer.wq,
                 input,
                 8U,
                 8U);

    CHECK(cudaMemcpy(decode.device_workspace,
                     input,
                     sizeof(input),
                     cudaMemcpyHostToDevice) == cudaSuccess);

    CHECK(niyah_cuda_model_state_matvec_device(
              &cuda_model,
              layer.wq,
              decode.device_workspace,
              decode.device_logits,
              8U,
              8U) == 0);

    CHECK(cudaMemcpy(actual,
                     decode.device_logits,
                     sizeof(actual),
                     cudaMemcpyDeviceToHost) == cudaSuccess);

    CHECK(vectors_close(expected, actual, 8U));

    CHECK(niyah_cuda_model_state_matvec_device(
              &cuda_model,
              model.weight_count,
              decode.device_workspace,
              decode.device_logits,
              8U,
              8U) != 0);

    CHECK(niyah_cuda_model_state_matvec_device(
              &cuda_model,
              layer.wq,
              decode.device_workspace,
              decode.device_workspace,
              8U,
              8U) != 0);

    niyah_cuda_decode_state_destroy(&decode);
    niyah_cuda_model_state_destroy(&cuda_model);
    niyah_model_destroy(&model);

    puts("P7D_CUDA_DEVICE_MATVEC=PASS");
    return 0;
}
