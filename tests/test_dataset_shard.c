#include "niyah/dataset.h"
#include "niyah/tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", \
                __FILE__, __LINE__, #expr); \
        failures += 1; \
    } \
} while (0)

static FILE *tfopen(const char *path, const char *mode)
{
#if defined(_MSC_VER)
    FILE *file = NULL;
    if (fopen_s(&file, path, mode) != 0) return NULL;
    return file;
#else
    return fopen(path, mode);
#endif
}

static unsigned char *read_all(const char *path, size_t *out_size)
{
    FILE *file = tfopen(path, "rb");
    long end;
    unsigned char *data;

    *out_size = 0U;
    if (file == NULL) return NULL;
    if (fseek(file, 0L, SEEK_END) != 0) {
        fclose(file);
        return NULL;
    }
    end = ftell(file);
    if (end < 0L || fseek(file, 0L, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }

    data = (unsigned char *)malloc(
        end == 0L ? 1U : (size_t)end);
    if (data == NULL) {
        fclose(file);
        return NULL;
    }
    if (end != 0L &&
        fread(data, 1U, (size_t)end, file) != (size_t)end) {
        free(data);
        fclose(file);
        return NULL;
    }
    if (fclose(file) != 0) {
        free(data);
        return NULL;
    }

    *out_size = (size_t)end;
    return data;
}

static int write_all(const char *path,
                     const unsigned char *data,
                     size_t size)
{
    FILE *file = tfopen(path, "wb");
    if (file == NULL) return 0;
    if (size != 0U && fwrite(data, 1U, size, file) != size) {
        fclose(file);
        return 0;
    }
    return fclose(file) == 0;
}

static NiyahTokenizer *make_tokenizer(const uint8_t *corpus,
                                      size_t corpus_size)
{
    NiyahTokenizerTrainConfig config;
    NiyahTokenizer *tokenizer = NULL;

    config.target_vocab_size = 264U;
    config.min_pair_frequency = 2U;
    CHECK(niyah_tokenizer_train(
              corpus, corpus_size, &config, &tokenizer) == NIYAH_OK);
    return tokenizer;
}

static void test_build_and_samples(void)
{
    static const uint8_t corpus[] =
        "aaaaaaaaaaaaaaaa bbbbbbbbbbbbbbbb aaaaaaaa bbbbbbbb";
    static const uint8_t text[] =
        "aaaaaaaa bbbbbbbb aaaaaaaa bbbbbbbb";
    NiyahTokenizer *tokenizer =
        make_tokenizer(corpus, sizeof(corpus) - 1U);
    NiyahDatasetShard shard;
    size_t covered = 0U;
    size_t i;

    memset(&shard, 0, sizeof(shard));
    CHECK(tokenizer != NULL);
    if (tokenizer == NULL) return;

    CHECK(niyah_dataset_shard_build_text(
              tokenizer, text, sizeof(text) - 1U,
              3U, &shard) == NIYAH_OK);

    if (shard.tokens != NULL) {
        CHECK(shard.token_count >= 2U);
        CHECK(shard.tokens[0] == NIYAH_TOKEN_BOS);
        CHECK(shard.tokens[shard.token_count - 1U] == NIYAH_TOKEN_EOS);
        CHECK(shard.sequence_length == 3U);
        CHECK(shard.sample_count ==
              1U + (shard.token_count - 2U) / 3U);

        for (i = 0U; i < shard.sample_count; ++i) {
            const uint32_t *tokens = NULL;
            const uint32_t *targets = NULL;
            size_t count = 0U;

            CHECK(niyah_dataset_shard_sample(
                      &shard, i, &tokens, &targets, &count) == NIYAH_OK);
            CHECK(tokens != NULL);
            CHECK(targets == tokens + 1U);
            CHECK(count > 0U && count <= 3U);
            covered += count;
        }
        CHECK(covered == shard.token_count - 1U);
    }

    niyah_dataset_shard_destroy(&shard);
    niyah_tokenizer_destroy(tokenizer);
}

static void test_persistence_and_identity(void)
{
    static const char path_a[] = "niyah_dataset_shard_v1_a.bin";
    static const char path_b[] = "niyah_dataset_shard_v1_b.bin";
    static const uint8_t corpus[] =
        "aaaaaaaaaaaaaaaa bbbbbbbbbbbbbbbb aaaaaaaa bbbbbbbb";
    static const uint8_t other_corpus[] =
        "zzzzzzzzzzzzzzzz yyyyyyyyyyyyyyyy zzzzzzzz yyyyyyyy";
    static const uint8_t text[] =
        "aaaaaaaa bbbbbbbb aaaaaaaa bbbbbbbb";

    NiyahTokenizer *tokenizer =
        make_tokenizer(corpus, sizeof(corpus) - 1U);
    NiyahTokenizer *other =
        make_tokenizer(other_corpus, sizeof(other_corpus) - 1U);
    NiyahDatasetShard shard;
    NiyahDatasetShard loaded;
    NiyahDatasetShard mismatch;
    unsigned char *a = NULL;
    unsigned char *b = NULL;
    uint8_t id_a[NIYAH_TOKENIZER_IDENTITY_SHA256_SIZE];
    uint8_t id_b[NIYAH_TOKENIZER_IDENTITY_SHA256_SIZE];
    size_t as = 0U;
    size_t bs = 0U;

    memset(&shard, 0, sizeof(shard));
    memset(&loaded, 0, sizeof(loaded));
    memset(&mismatch, 0, sizeof(mismatch));
    mismatch.sequence_length = 123U;
    mismatch.sample_count = 456U;

    (void)remove(path_a);
    (void)remove(path_b);

    CHECK(tokenizer != NULL);
    CHECK(other != NULL);
    if (tokenizer == NULL || other == NULL) goto done;

    CHECK(niyah_tokenizer_identity_sha256(tokenizer, id_a) == NIYAH_OK);
    CHECK(niyah_tokenizer_identity_sha256(other, id_b) == NIYAH_OK);
    CHECK(memcmp(id_a, id_b, sizeof(id_a)) != 0);

    CHECK(niyah_dataset_shard_build_text(
              tokenizer, text, sizeof(text) - 1U,
              4U, &shard) == NIYAH_OK);

    CHECK(niyah_dataset_shard_save(
              &shard, tokenizer, path_a) == NIYAH_OK);
    CHECK(niyah_dataset_shard_save(
              &shard, tokenizer, path_b) == NIYAH_OK);

    a = read_all(path_a, &as);
    b = read_all(path_b, &bs);
    CHECK(a != NULL);
    CHECK(b != NULL);
    if (a != NULL && b != NULL) {
        CHECK(as == bs);
        CHECK(memcmp(a, b, as) == 0);
    }

    CHECK(niyah_dataset_shard_load(
              path_a, tokenizer, &loaded) == NIYAH_OK);
    if (loaded.tokens != NULL) {
        CHECK(loaded.token_count == shard.token_count);
        CHECK(loaded.sequence_length == shard.sequence_length);
        CHECK(loaded.sample_count == shard.sample_count);
        CHECK(memcmp(loaded.tokenizer_identity,
                     shard.tokenizer_identity,
                     sizeof(shard.tokenizer_identity)) == 0);
        CHECK(memcmp(loaded.tokens, shard.tokens,
                     shard.token_count * sizeof(uint32_t)) == 0);
    }

    CHECK(niyah_dataset_shard_load(
              path_a, other, &mismatch) == NIYAH_ERR_INVALID_CONFIG);
    CHECK(mismatch.tokens == NULL);
    CHECK(mismatch.sequence_length == 123U);
    CHECK(mismatch.sample_count == 456U);

done:
    free(b);
    free(a);
    niyah_dataset_shard_destroy(&loaded);
    niyah_dataset_shard_destroy(&shard);
    niyah_tokenizer_destroy(other);
    niyah_tokenizer_destroy(tokenizer);
    (void)remove(path_a);
    (void)remove(path_b);
}

static void test_corruption_rejection(void)
{
    static const char valid_path[] = "niyah_dataset_shard_valid.bin";
    static const char bad_path[] = "niyah_dataset_shard_bad.bin";
    static const uint8_t corpus[] =
        "aaaaaaaaaaaaaaaa bbbbbbbbbbbbbbbb aaaaaaaa bbbbbbbb";
    static const uint8_t text[] =
        "aaaaaaaa bbbbbbbb aaaaaaaa bbbbbbbb";
    NiyahTokenizer *tokenizer =
        make_tokenizer(corpus, sizeof(corpus) - 1U);
    NiyahDatasetShard shard;
    NiyahDatasetShard rejected;
    unsigned char *data = NULL;
    unsigned char *extra = NULL;
    size_t size = 0U;

    memset(&shard, 0, sizeof(shard));
    memset(&rejected, 0, sizeof(rejected));
    (void)remove(valid_path);
    (void)remove(bad_path);

    CHECK(tokenizer != NULL);
    if (tokenizer == NULL) return;

    CHECK(niyah_dataset_shard_build_text(
              tokenizer, text, sizeof(text) - 1U,
              4U, &shard) == NIYAH_OK);
    CHECK(niyah_dataset_shard_save(
              &shard, tokenizer, valid_path) == NIYAH_OK);

    data = read_all(valid_path, &size);
    CHECK(data != NULL);
    CHECK(size > 80U);
    if (data == NULL || size <= 80U) goto done;

    data[size - 1U] ^= 1U;
    CHECK(write_all(bad_path, data, size));
    CHECK(niyah_dataset_shard_load(
              bad_path, tokenizer, &rejected) ==
          NIYAH_ERR_CORRUPT_DATA);
    CHECK(rejected.tokens == NULL);
    data[size - 1U] ^= 1U;

    data[8U] = 2U;
    CHECK(write_all(bad_path, data, size));
    CHECK(niyah_dataset_shard_load(
              bad_path, tokenizer, &rejected) ==
          NIYAH_ERR_UNSUPPORTED_VERSION);
    CHECK(rejected.tokens == NULL);
    data[8U] = 1U;

    CHECK(write_all(bad_path, data, size - 1U));
    CHECK(niyah_dataset_shard_load(
              bad_path, tokenizer, &rejected) ==
          NIYAH_ERR_CORRUPT_DATA);
    CHECK(rejected.tokens == NULL);

    extra = (unsigned char *)malloc(size + 1U);
    CHECK(extra != NULL);
    if (extra != NULL) {
        memcpy(extra, data, size);
        extra[size] = 0U;
        CHECK(write_all(bad_path, extra, size + 1U));
        CHECK(niyah_dataset_shard_load(
                  bad_path, tokenizer, &rejected) ==
              NIYAH_ERR_CORRUPT_DATA);
        CHECK(rejected.tokens == NULL);
    }

done:
    free(extra);
    free(data);
    niyah_dataset_shard_destroy(&rejected);
    niyah_dataset_shard_destroy(&shard);
    niyah_tokenizer_destroy(tokenizer);
    (void)remove(valid_path);
    (void)remove(bad_path);
}

static void test_invalid_inputs(void)
{
    static const uint8_t corpus[] = "aaaaaaaaaaaaaaaa";
    static const uint8_t text[] = "aaaa";
    NiyahTokenizer *tokenizer =
        make_tokenizer(corpus, sizeof(corpus) - 1U);
    NiyahDatasetShard shard;

    memset(&shard, 0, sizeof(shard));
    CHECK(tokenizer != NULL);
    if (tokenizer == NULL) return;

    CHECK(niyah_dataset_shard_build_text(
              tokenizer, text, sizeof(text) - 1U,
              0U, &shard) == NIYAH_ERR_INVALID_CONFIG);
    CHECK(shard.tokens == NULL);

    CHECK(niyah_dataset_shard_build_text(
              tokenizer, NULL, 0U,
              4U, &shard) == NIYAH_ERR_INVALID_ARGUMENT);
    CHECK(shard.tokens == NULL);

    niyah_tokenizer_destroy(tokenizer);
}

int main(void)
{
    test_build_and_samples();
    test_persistence_and_identity();
    test_corruption_rejection();
    test_invalid_inputs();

    if (failures != 0) {
        fprintf(stderr, "niyah_dataset_shard_test: %d failure(s)\n", failures);
        return 1;
    }

    puts("P6H_DATASET_PREPROCESS=PASS");
    puts("P6H_BINARY_SHARD_V1=PASS");
    puts("P6H_TOKENIZER_BINDING=PASS");
    puts("P6H_CORRUPTION_REJECTION=PASS");
    return 0;
}
