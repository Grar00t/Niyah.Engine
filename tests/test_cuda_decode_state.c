#include "niyah/niyah.h"
#include "niyah_cuda_matvec.h"

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

int main(void)
{
    const NiyahModelConfig config = tiny_config();
    NiyahModel model;
    NiyahCudaModelState cuda_model;
    NiyahCudaModelState bad_model;
    NiyahCudaDecodeState decode;
    NiyahCudaDecodeState bad_decode;

    memset(&model, 0, sizeof(model));
    memset(&cuda_model, 0, sizeof(cuda_model));
    memset(&bad_model, 0, sizeof(bad_model));
    memset(&decode, 0, sizeof(decode));
    memset(&bad_decode, 0, sizeof(bad_decode));

    CHECK(niyah_model_create(&model, &config) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(
              &model, UINT64_C(0x12345678)) == NIYAH_OK);
    CHECK(niyah_cuda_model_state_create(&cuda_model, &model) == 0);

    CHECK(niyah_cuda_decode_state_create(&decode, &cuda_model) == 0);

    CHECK(decode.device_keys != NULL);
    CHECK(decode.device_values != NULL);
    CHECK(decode.device_workspace != NULL);
    CHECK(decode.device_logits != NULL);

    CHECK(decode.context_length == 8U);
    CHECK(decode.head_dim == 2U);
    CHECK(decode.kv_dim == 4U);
    CHECK(decode.values_per_tensor == 64U);
    CHECK(decode.workspace_floats == 88U);
    CHECK(decode.logits_capacity == 32U);
    CHECK(decode.next_position == 0U);

    decode.next_position = 5U;
    CHECK(niyah_cuda_decode_state_reset(&decode) == 0);
    CHECK(decode.next_position == 0U);

    /*
     * Geometry mismatch must fail before device allocations become a valid
     * decode state.
     */
    bad_model = cuda_model;
    bad_model.layout.kv_dim += 1U;
    CHECK(niyah_cuda_decode_state_create(
              &bad_decode, &bad_model) != 0);
    CHECK(bad_decode.device_keys == NULL);
    CHECK(bad_decode.device_values == NULL);
    CHECK(bad_decode.device_workspace == NULL);
    CHECK(bad_decode.device_logits == NULL);

    niyah_cuda_decode_state_destroy(&bad_decode);
    niyah_cuda_decode_state_destroy(&decode);
    niyah_cuda_model_state_destroy(&cuda_model);
    niyah_model_destroy(&model);

    puts("P7C_CUDA_DECODE_STATE=PASS");
    return 0;
}
