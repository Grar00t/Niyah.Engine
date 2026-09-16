#include "niyah/niyah.h"
#include "niyah_cuda_matvec.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "CHECK failed line %d: %s\n", __LINE__, #expr); \
        return 1; \
    } \
} while (0)

static int close_enough(float a, float b)
{
    const float diff = fabsf(a - b);
    const float scale = 1.0f + fabsf(a);
    return diff <= 1.0e-4f * scale;
}

static int vectors_close(const float *a, const float *b, size_t n)
{
    size_t i;
    for (i = 0U; i < n; ++i) {
        if (!close_enough(a[i], b[i])) {
            return 0;
        }
    }
    return 1;
}

static NiyahModelConfig test_config(void)
{
    NiyahModelConfig c;
    memset(&c, 0, sizeof(c));
    c.vocab_size = 8U;
    c.context_length = 4U;
    c.embedding_dim = 4U;
    c.n_layers = 1U;
    c.n_heads = 2U;
    c.n_kv_heads = 1U;
    c.ffn_hidden_dim = 8U;
    c.rms_norm_eps = 1.0e-5f;
    c.tie_word_embeddings = 1;
    return c;
}

int main(void)
{
    const NiyahModelConfig config = test_config();
    const float x[4] = {1.0f, -0.5f, 0.25f, 2.0f};
    NiyahModel model;
    NiyahLayerLayout layer;
    NiyahCudaModelState cuda_model;
    float cpu_before[4];
    float cpu_after[4];
    float gpu[4];

    memset(&model, 0, sizeof(model));
    memset(&layer, 0, sizeof(layer));
    memset(&cuda_model, 0, sizeof(cuda_model));

    CHECK(niyah_model_create(&model, &config) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(&model, UINT64_C(42)) == NIYAH_OK);
    CHECK(niyah_model_layer_layout(
              &model.config, &model.layout, 0U, &layer) == NIYAH_OK);

    niyah_matvec(cpu_before,
                 model.weights + layer.wq,
                 x,
                 4U,
                 4U);

    CHECK(niyah_cuda_model_state_create(&cuda_model, &model) == 0);
    CHECK(niyah_cuda_model_state_matvec(
              &cuda_model, layer.wq, x, gpu, 4U, 4U) == 0);
    CHECK(vectors_close(cpu_before, gpu, 4U));

    /* Host mutation must not silently alter the persistent CUDA snapshot. */
    model.weights[layer.wq] += 1.0f;

    niyah_matvec(cpu_after,
                 model.weights + layer.wq,
                 x,
                 4U,
                 4U);

    CHECK(!vectors_close(cpu_before, cpu_after, 4U));

    CHECK(niyah_cuda_model_state_matvec(
              &cuda_model, layer.wq, x, gpu, 4U, 4U) == 0);
    CHECK(vectors_close(cpu_before, gpu, 4U));
    CHECK(!vectors_close(cpu_after, gpu, 4U));

    /* Explicit synchronization publishes the new canonical host weights. */
    CHECK(niyah_cuda_model_state_sync(&cuda_model, &model) == 0);
    CHECK(niyah_cuda_model_state_matvec(
              &cuda_model, layer.wq, x, gpu, 4U, 4U) == 0);
    CHECK(vectors_close(cpu_after, gpu, 4U));

    /* Weight-range validation must fail closed. */
    CHECK(niyah_cuda_model_state_matvec(
              &cuda_model, model.weight_count, x, gpu, 4U, 4U) != 0);

    niyah_cuda_model_state_destroy(&cuda_model);
    niyah_model_destroy(&model);

    puts("P7B_CUDA_MODEL_STATE=PASS");
    return 0;
}
