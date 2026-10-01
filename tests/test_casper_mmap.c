#include "niyah/casper.h"
#include "niyah/transformer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expr) do { if (!(expr)) { \
    fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
    return 1; } } while (0)

static void store_u32(unsigned char *p, uint32_t v)
{
    p[0] = (unsigned char)(v & 0xffU);
    p[1] = (unsigned char)((v >> 8U) & 0xffU);
    p[2] = (unsigned char)((v >> 16U) & 0xffU);
    p[3] = (unsigned char)((v >> 24U) & 0xffU);
}

static uint32_t float_bits(float v)
{
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    return bits;
}
static int write_fixture(const char *path, size_t *out_weight_count)
{
    unsigned char header[64] = {0};
    NiyahModelConfig config;
    NiyahModelLayout layout;
    FILE *f;
    float zeros[256] = {0.0f};
    size_t left;

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
    CHECK(niyah_model_layout_compute(&config, &layout) == NIYAH_OK);

    store_u32(header + 0U, UINT32_C(0x4E595148));
    store_u32(header + 4U, UINT32_C(0x0005));
    store_u32(header + 8U, config.embedding_dim);
    store_u32(header + 12U, config.n_heads);
    store_u32(header + 16U, config.n_kv_heads);
    store_u32(header + 20U, config.n_layers);
    store_u32(header + 24U, 2U);
    store_u32(header + 28U, config.vocab_size);
    store_u32(header + 32U, config.context_length);
    store_u32(header + 36U, float_bits(10000.0f));
    store_u32(header + 40U, float_bits(config.rms_norm_eps));
    store_u32(header + 44U, 0U);

    f = fopen(path, "wb");
    CHECK(f != NULL);
    CHECK(fwrite(header, 1U, sizeof(header), f) == sizeof(header));
    left = layout.total_floats;
    while (left != 0U) {
        const size_t n = left > 256U ? 256U : left;
        CHECK(fwrite(zeros, sizeof(float), n, f) == n);
        left -= n;
    }
    CHECK(fclose(f) == 0);
    *out_weight_count = layout.total_floats;
    return 0;
}
int main(void)
{
    const char *path = "niyah_casper_mmap_test.bin";
    NiyahModel model;
    const uint32_t tokens[2] = {1U, 2U};
    float *workspace = NULL;
    float logits[32];
    size_t workspace_count = 0U;
    size_t expected_weights = 0U;

    memset(&model, 0, sizeof(model));
    CHECK(write_fixture(path, &expected_weights) == 0);
    CHECK(niyah_casper_mmap_load(path, &model) == NIYAH_OK);
    CHECK(model.storage_kind == NIYAH_MODEL_STORAGE_CASPER_MMAP);
    CHECK(model.mapping_base != NULL);
    CHECK(model.mapping_bytes == 64U + expected_weights * sizeof(float));
    CHECK(model.weights == (float *)((unsigned char *)model.mapping_base + 64U));
    CHECK(model.weight_count == expected_weights);
    CHECK(niyah_model_reset_parameters(&model, UINT64_C(7)) ==
          NIYAH_ERR_INVALID_ARGUMENT);
    CHECK(niyah_transformer_workspace_floats(&model.config, 2U,
                                             &workspace_count) == NIYAH_OK);
    workspace = (float *)calloc(workspace_count, sizeof(float));
    CHECK(workspace != NULL);
    CHECK(niyah_transformer_forward(&model, tokens, 2U, logits, 32U,
                                    workspace, workspace_count) == NIYAH_OK);
    free(workspace);
    niyah_model_destroy(&model);
    CHECK(model.weights == NULL);
    CHECK(model.mapping_base == NULL);
    CHECK(model.mapping_bytes == 0U);
    CHECK(remove(path) == 0);

    puts("NIYAH_CASPER_MMAP_ZERO_COPY=PASS");
    return 0;
}
