#include "niyah/casper.h"
#include "niyah/transformer.h"
#include "../src/niyah_sha256.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REQUIRE(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "REQUIRE failed at %s:%d: %s\n", \
                __FILE__, __LINE__, #expr); \
        goto cleanup; \
    } \
} while (0)

static int all_finite(const float *values, size_t count)
{
    size_t i;

    for (i = 0U; i < count; ++i) {
        if (!isfinite(values[i])) {
            return 0;
        }
    }

    return 1;
}

static int hex_digit(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int identity_matches(const NiyahModel *model, const char *expected)
{
    NiyahSha256 state;
    unsigned char digest[32];
    size_t i;
    if (strlen(expected) != 2U * sizeof(digest)) return 0;
    niyah_sha256_init(&state);
    if (!niyah_sha256_update(&state, model->mapping_base,
                            (size_t)model->mapping_bytes)) return 0;
    niyah_sha256_final(&state, digest);
    for (i = 0U; i < sizeof(digest); ++i) {
        int high = hex_digit(expected[2U * i]);
        int low = hex_digit(expected[2U * i + 1U]);
        if (high < 0 || low < 0 || (unsigned int)(16 * high + low) != digest[i])
            return 0;
    }
    return 1;
}

int main(void)
{
    const char *path = getenv("NIYAH_CASPER_MODEL_PATH");
    const char *expected_sha256 = getenv("NIYAH_CASPER_MODEL_SHA256");
    NiyahModel model;
    const uint32_t tokens[2] = {1U, 2U};
    float *workspace = NULL;
    float *logits = NULL;
    size_t workspace_count = 0U;
    size_t logits_count = 0U;
    const size_t token_count = sizeof(tokens) / sizeof(tokens[0]);
    int rc = 1;

    memset(&model, 0, sizeof(model));

    if (path == NULL || path[0] == '\0' ||
        expected_sha256 == NULL || expected_sha256[0] == '\0') {
        puts("SKIP: NIYAH_CASPER_MODEL_PATH and "
             "NIYAH_CASPER_MODEL_SHA256 must be set");
        return 77;
    }

    REQUIRE(niyah_casper_mmap_load(path, &model) == NIYAH_OK);
    REQUIRE(model.storage_kind == NIYAH_MODEL_STORAGE_CASPER_MMAP);
    REQUIRE(model.mapping_base != NULL);
    REQUIRE(model.mapping_bytes == UINT64_C(5997120));
    REQUIRE(identity_matches(&model, expected_sha256));
    REQUIRE(model.weights ==
            (float *)((unsigned char *)model.mapping_base + 64U));
    REQUIRE(model.weight_count == UINT64_C(1499264));

    REQUIRE(model.config.embedding_dim == 128U);
    REQUIRE(model.config.n_heads == 8U);
    REQUIRE(model.config.n_kv_heads == 8U);
    REQUIRE(model.config.n_layers == 4U);
    REQUIRE(model.config.ffn_hidden_dim == 512U);
    REQUIRE(model.config.vocab_size == 1756U);
    REQUIRE(model.config.context_length == 64U);
    REQUIRE(fabsf(model.config.rms_norm_eps - 1.0e-5f) < 1.0e-10f);

    REQUIRE(niyah_transformer_workspace_floats(
                &model.config, token_count, &workspace_count) == NIYAH_OK);

    workspace = (float *)calloc(workspace_count, sizeof(*workspace));
    REQUIRE(workspace != NULL);

    REQUIRE(token_count != 0U);
    REQUIRE((size_t)model.config.vocab_size <= SIZE_MAX / token_count);
    logits_count = token_count * (size_t)model.config.vocab_size;
    logits = (float *)calloc(logits_count, sizeof(*logits));
    REQUIRE(logits != NULL);

    REQUIRE(niyah_transformer_forward(
                &model, tokens, token_count, logits, logits_count,
                workspace, workspace_count) == NIYAH_OK);

    REQUIRE(all_finite(logits, logits_count));

    puts("NIYAH_CASPER_EXTERNAL_MMAP_FORWARD=PASS");
    rc = 0;

cleanup:
    free(logits);
    free(workspace);
    niyah_model_destroy(&model);
    return rc;
}
