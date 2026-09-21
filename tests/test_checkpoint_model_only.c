#include "niyah/checkpoint.h"
#include "niyah/optimizer.h"
#include "niyah/tokenizer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        failures += 1; \
    } \
} while (0)

static FILE *test_fopen(const char *path, const char *mode)
{
#if defined(_MSC_VER)
    FILE *file = NULL;
    if (fopen_s(&file, path, mode) != 0) {
        return NULL;
    }
    return file;
#else
    return fopen(path, mode);
#endif
}

static uint64_t load_u64_le(const unsigned char *p)
{
    uint64_t value = UINT64_C(0);
    size_t i;
    for (i = 0U; i < 8U; ++i) {
        value |= ((uint64_t)p[i]) << (i * 8U);
    }
    return value;
}

static void store_u32_le(unsigned char *p, uint32_t value)
{
    p[0] = (unsigned char)(value & UINT32_C(0xff));
    p[1] = (unsigned char)((value >> 8) & UINT32_C(0xff));
    p[2] = (unsigned char)((value >> 16) & UINT32_C(0xff));
    p[3] = (unsigned char)((value >> 24) & UINT32_C(0xff));
}

static uint32_t crc32_reference(const unsigned char *data, size_t size)
{
    uint32_t crc = UINT32_C(0xffffffff);
    size_t i;
    for (i = 0U; i < size; ++i) {
        unsigned bit;
        crc ^= (uint32_t)data[i];
        for (bit = 0U; bit < 8U; ++bit) {
            crc = (crc & UINT32_C(1)) != 0U
                ? (crc >> 1) ^ UINT32_C(0xedb88320)
                : crc >> 1;
        }
    }
    return crc ^ UINT32_C(0xffffffff);
}

static unsigned char *read_bytes(const char *path, size_t *out_size)
{
    FILE *file = test_fopen(path, "rb");
    long end;
    unsigned char *bytes;

    if (file == NULL || out_size == NULL) {
        if (file != NULL) (void)fclose(file);
        return NULL;
    }
    if (fseek(file, 0L, SEEK_END) != 0) {
        (void)fclose(file);
        return NULL;
    }
    end = ftell(file);
    if (end <= 0L || fseek(file, 0L, SEEK_SET) != 0) {
        (void)fclose(file);
        return NULL;
    }
    bytes = (unsigned char *)malloc((size_t)end);
    if (bytes == NULL) {
        (void)fclose(file);
        return NULL;
    }
    if (fread(bytes, 1U, (size_t)end, file) != (size_t)end ||
        fclose(file) != 0) {
        free(bytes);
        return NULL;
    }
    *out_size = (size_t)end;
    return bytes;
}

static int write_bytes(const char *path, const unsigned char *bytes, size_t size)
{
    FILE *file = test_fopen(path, "wb");
    int ok;
    if (file == NULL) return 0;
    ok = fwrite(bytes, 1U, size, file) == size;
    if (fclose(file) != 0) ok = 0;
    return ok;
}

static int model_config_equal(const NiyahModelConfig *a, const NiyahModelConfig *b)
{
    return a->vocab_size == b->vocab_size &&
           a->context_length == b->context_length &&
           a->embedding_dim == b->embedding_dim &&
           a->n_layers == b->n_layers &&
           a->n_heads == b->n_heads &&
           a->n_kv_heads == b->n_kv_heads &&
           a->ffn_hidden_dim == b->ffn_hidden_dim &&
           a->rms_norm_eps == b->rms_norm_eps &&
           a->tie_word_embeddings == b->tie_word_embeddings &&
           a->n_segments == b->n_segments;
}

static NiyahAdamWConfig optimizer_config(void)
{
    NiyahAdamWConfig config;
    config.learning_rate = 5.0e-4f;
    config.beta1 = 0.9f;
    config.beta2 = 0.999f;
    config.epsilon = 1.0e-8f;
    config.weight_decay = 0.01f;
    config.max_grad_norm = 1.0f;
    return config;
}

int main(void)
{
    static const uint8_t corpus_a[] = "alpha alpha alpha beta beta gamma";
    static const uint8_t corpus_b[] = "zeta zeta zeta eta eta theta";
    const char *valid_path = "niyah_model_only_valid.bin";
    const char *corrupt_path = "niyah_model_only_corrupt.bin";
    NiyahTokenizerTrainConfig train_config;
    NiyahTokenizer *tokenizer_a = NULL;
    NiyahTokenizer *tokenizer_b = NULL;
    NiyahModelConfig model_config;
    NiyahModel model;
    NiyahModel loaded;
    NiyahAdamWState state;
    NiyahAdamWConfig opt_config;
    unsigned char *bytes = NULL;
    size_t byte_count = 0U;
    size_t tensor_bytes;
    size_t m_payload_offset;
    size_t footer_offset;
    NiyahStatus status;

    memset(&train_config, 0, sizeof(train_config));
    memset(&model_config, 0, sizeof(model_config));
    memset(&model, 0, sizeof(model));
    memset(&loaded, 0, sizeof(loaded));
    memset(&state, 0, sizeof(state));

    train_config.target_vocab_size = 260U;
    train_config.min_pair_frequency = 1U;

    CHECK(niyah_tokenizer_train(
              corpus_a, sizeof(corpus_a) - 1U,
              &train_config, &tokenizer_a) == NIYAH_OK);
    CHECK(niyah_tokenizer_train(
              corpus_b, sizeof(corpus_b) - 1U,
              &train_config, &tokenizer_b) == NIYAH_OK);
    if (tokenizer_a == NULL || tokenizer_b == NULL) goto cleanup;

    model_config.vocab_size = (uint32_t)niyah_tokenizer_vocab_size(tokenizer_a);
    model_config.context_length = 8U;
    model_config.embedding_dim = 8U;
    model_config.n_layers = 1U;
    model_config.n_heads = 2U;
    model_config.n_kv_heads = 1U;
    model_config.ffn_hidden_dim = 16U;
    model_config.rms_norm_eps = 1.0e-5f;
    model_config.tie_word_embeddings = 1;
    model_config.n_segments = 0U;

    CHECK(niyah_model_create(&model, &model_config) == NIYAH_OK);
    if (model.weights == NULL) goto cleanup;
    CHECK(niyah_model_reset_parameters(&model, UINT64_C(20260921)) == NIYAH_OK);
    CHECK(niyah_adamw_state_create(&state, &model) == NIYAH_OK);
    if (state.m == NULL || state.v == NULL) goto cleanup;

    opt_config = optimizer_config();
    CHECK(niyah_checkpoint_save_with_tokenizer(
              valid_path, &model, &state, &opt_config,
              tokenizer_a) == NIYAH_OK);

    status = niyah_checkpoint_load_model_with_tokenizer(
        valid_path, tokenizer_a, &loaded);
    CHECK(status == NIYAH_OK);
    if (status == NIYAH_OK) {
        CHECK(model_config_equal(&model.config, &loaded.config));
        CHECK(model.weight_count == loaded.weight_count);
        CHECK(memcmp(model.weights, loaded.weights,
                     model.weight_count * sizeof(float)) == 0);
    }
    niyah_model_destroy(&loaded);
    memset(&loaded, 0, sizeof(loaded));

    status = niyah_checkpoint_load_model_with_tokenizer(
        valid_path, tokenizer_b, &loaded);
    CHECK(status == NIYAH_ERR_INVALID_CONFIG);
    CHECK(loaded.weights == NULL && loaded.weight_count == 0U);

    bytes = read_bytes(valid_path, &byte_count);
    CHECK(bytes != NULL);
    if (bytes != NULL) {
        CHECK(byte_count > 80U);
        CHECK(load_u64_le(bytes + 64U) == (uint64_t)model.weight_count);
        tensor_bytes = model.weight_count * sizeof(float);
        m_payload_offset = 72U + 16U + tensor_bytes + 16U;
        footer_offset = byte_count - 8U;
        CHECK(m_payload_offset + sizeof(uint32_t) < footer_offset);
        if (m_payload_offset + sizeof(uint32_t) < footer_offset) {
            store_u32_le(bytes + m_payload_offset, UINT32_C(0x7fc00000));
            store_u32_le(bytes + footer_offset + 4U,
                         crc32_reference(bytes, footer_offset));
            CHECK(write_bytes(corrupt_path, bytes, byte_count));
            status = niyah_checkpoint_load_model_with_tokenizer(
                corrupt_path, tokenizer_a, &loaded);
            CHECK(status == NIYAH_ERR_CORRUPT_DATA);
            CHECK(loaded.weights == NULL && loaded.weight_count == 0U);
        }
    }

    CHECK(niyah_checkpoint_load_model_with_tokenizer(
              NULL, tokenizer_a, &loaded) == NIYAH_ERR_INVALID_ARGUMENT);
    CHECK(niyah_checkpoint_load_model_with_tokenizer(
              valid_path, NULL, &loaded) == NIYAH_ERR_INVALID_ARGUMENT);

cleanup:
    free(bytes);
    niyah_model_destroy(&loaded);
    niyah_adamw_state_destroy(&state);
    niyah_model_destroy(&model);
    niyah_tokenizer_destroy(tokenizer_b);
    niyah_tokenizer_destroy(tokenizer_a);
    (void)remove(corrupt_path);
    (void)remove(valid_path);

    if (failures != 0) {
        fprintf(stderr, "niyah_checkpoint_model_only_test: %d failure(s)\n", failures);
        return 1;
    }
    puts("NIYAH_CHECKPOINT_MODEL_ONLY=PASS");
    return 0;
}
