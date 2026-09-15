#include "niyah/niyah.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(expr) \
    do { \
        if (!(expr)) { \
            fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
            return 1; \
        } \
    } while (0)

static NiyahModelConfig tiny_config(int tied)
{
    NiyahModelConfig config;
    memset(&config, 0, sizeof(config));
    config.vocab_size = 16U;
    config.context_length = 8U;
    config.embedding_dim = 8U;
    config.n_layers = 2U;
    config.n_heads = 4U;
    config.n_kv_heads = 2U;
    config.ffn_hidden_dim = 16U;
    config.rms_norm_eps = 1.0e-5f;
    config.tie_word_embeddings = tied;
    return config;
}

static int test_config_and_layout(void)
{
    NiyahModelConfig config = tiny_config(1);
    NiyahModelLayout layout;
    NiyahLayerLayout layer0;
    NiyahLayerLayout layer1;

    CHECK(niyah_model_config_validate(&config) == NIYAH_OK);
    CHECK(niyah_model_layout_compute(&config, &layout) == NIYAH_OK);
    CHECK(layout.head_dim == 2U);
    CHECK(layout.kv_dim == 4U);
    CHECK(layout.token_embedding == 0U);
    CHECK(layout.layers == 128U);
    CHECK(layout.layer_stride == 592U);
    CHECK(layout.final_norm == 1312U);
    CHECK(layout.lm_head == layout.token_embedding);
    CHECK(layout.total_floats == 1320U);

    CHECK(niyah_model_layer_layout(&config, &layout, 0U, &layer0) == NIYAH_OK);
    CHECK(layer0.attn_norm == 128U);
    CHECK(layer0.wq == 136U);
    CHECK(layer0.wk == 200U);
    CHECK(layer0.wv == 232U);
    CHECK(layer0.wo == 264U);
    CHECK(layer0.ffn_norm == 328U);
    CHECK(layer0.w_gate == 336U);
    CHECK(layer0.w_up == 464U);
    CHECK(layer0.w_down == 592U);

    CHECK(niyah_model_layer_layout(&config, &layout, 1U, &layer1) == NIYAH_OK);
    CHECK(layer1.attn_norm == 720U);
    CHECK(layer1.w_down == 1184U);
    CHECK(niyah_model_layer_layout(&config, &layout, 2U, &layer1) == NIYAH_ERR_INVALID_ARGUMENT);

    config.tie_word_embeddings = 0;
    CHECK(niyah_model_layout_compute(&config, &layout) == NIYAH_OK);
    CHECK(layout.final_norm == 1312U);
    CHECK(layout.lm_head == 1320U);
    CHECK(layout.total_floats == 1448U);
    return 0;
}

static int test_invalid_configs(void)
{
    NiyahModelConfig config = tiny_config(1);

    config.embedding_dim = 10U;
    CHECK(niyah_model_config_validate(&config) == NIYAH_ERR_INVALID_CONFIG);

    config = tiny_config(1);
    config.n_heads = 3U;
    CHECK(niyah_model_config_validate(&config) == NIYAH_ERR_INVALID_CONFIG);

    config = tiny_config(1);
    config.n_kv_heads = 3U;
    CHECK(niyah_model_config_validate(&config) == NIYAH_ERR_INVALID_CONFIG);

    config = tiny_config(1);
    config.rms_norm_eps = 0.0f;
    CHECK(niyah_model_config_validate(&config) == NIYAH_ERR_INVALID_CONFIG);
    return 0;
}

static int test_model_and_initialization(void)
{
    NiyahModelConfig config = tiny_config(0);
    NiyahModel a;
    NiyahModel b;
    NiyahLayerLayout layer;
    size_t bytes;
    size_t i;

    CHECK(niyah_model_create(&a, &config) == NIYAH_OK);
    CHECK(niyah_model_create(&b, &config) == NIYAH_OK);
    CHECK(a.weight_count == 1448U);
    CHECK(b.weight_count == a.weight_count);

    CHECK(niyah_model_reset_parameters(&a, UINT64_C(1234567)) == NIYAH_OK);
    CHECK(niyah_model_reset_parameters(&b, UINT64_C(1234567)) == NIYAH_OK);

    bytes = a.weight_count * sizeof(float);
    CHECK(memcmp(a.weights, b.weights, bytes) == 0);

    CHECK(niyah_model_layer_layout(&a.config, &a.layout, 0U, &layer) == NIYAH_OK);
    for (i = 0U; i < (size_t)a.config.embedding_dim; ++i) {
        CHECK(a.weights[layer.attn_norm + i] == 1.0f);
        CHECK(a.weights[layer.ffn_norm + i] == 1.0f);
        CHECK(a.weights[a.layout.final_norm + i] == 1.0f);
    }

    CHECK(a.layout.lm_head != a.layout.token_embedding);
    niyah_model_destroy(&a);
    niyah_model_destroy(&b);
    CHECK(a.weights == NULL);
    CHECK(a.weight_count == 0U);
    return 0;
}

static int test_math(void)
{
    const float matrix[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float x[2] = {1.0f, 1.0f};
    float out[2] = {0.0f, 0.0f};
    const float norm_x[2] = {3.0f, 4.0f};
    const float norm_w[2] = {1.0f, 1.0f};
    float norm_out[2] = {0.0f, 0.0f};
    const float eps = 1.0e-5f;
    const float inv = 1.0f / sqrtf(12.5f + eps);

    niyah_matvec(out, matrix, x, 2U, 2U);
    CHECK(fabsf(out[0] - 3.0f) < 1.0e-6f);
    CHECK(fabsf(out[1] - 7.0f) < 1.0e-6f);

    CHECK(niyah_rmsnorm(norm_out, norm_x, norm_w, 2U, eps) == NIYAH_OK);
    CHECK(fabsf(norm_out[0] - 3.0f * inv) < 1.0e-6f);
    CHECK(fabsf(norm_out[1] - 4.0f * inv) < 1.0e-6f);
    return 0;
}

int main(void)
{
    CHECK(test_config_and_layout() == 0);
    CHECK(test_invalid_configs() == 0);
    CHECK(test_model_and_initialization() == 0);
    CHECK(test_math() == 0);

    puts("NIYAH_MODEL_CORE_P0=PASS");
    return 0;
}
