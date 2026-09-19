#include "niyah/dataset.h"
#include "niyah/tokenizer.h"
#include "niyah/training_loop.h"

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
            if (tokens != NULL) {
                CHECK(targets == tokens + 1U);
            }
            CHECK(count > 0U && count <= 3U);
            covered += count;
        }
        CHECK(covered == shard.token_count - 1U);
    }

    niyah_dataset_shard_destroy(&shard);
    niyah_tokenizer_destroy(tokenizer);
}


static void test_shard_content_identity(void)
{
    static const uint8_t corpus[] =
        "aaaaaaaaaaaaaaaa bbbbbbbbbbbbbbbb aaaaaaaa bbbbbbbb";
    static const uint8_t text_a[] =
        "aaaaaaaa bbbbbbbb aaaaaaaa bbbbbbbb";
    static const uint8_t text_b[] =
        "aaaaaaaa bbbbbbbb aaaaaaaa bbbbbbb";
    NiyahTokenizer *tokenizer =
        make_tokenizer(corpus, sizeof(corpus) - 1U);
    NiyahDatasetShard a;
    NiyahDatasetShard a_same;
    NiyahDatasetShard content_changed;
    NiyahDatasetShard geometry_changed;
    uint8_t id_a[NIYAH_DATASET_SHARD_IDENTITY_SHA256_SIZE];
    uint8_t id_same[NIYAH_DATASET_SHARD_IDENTITY_SHA256_SIZE];
    uint8_t id_content[NIYAH_DATASET_SHARD_IDENTITY_SHA256_SIZE];
    uint8_t id_geometry[NIYAH_DATASET_SHARD_IDENTITY_SHA256_SIZE];

    memset(&a, 0, sizeof(a));
    memset(&a_same, 0, sizeof(a_same));
    memset(&content_changed, 0, sizeof(content_changed));
    memset(&geometry_changed, 0, sizeof(geometry_changed));

    CHECK(tokenizer != NULL);
    if (tokenizer == NULL) return;

    CHECK(niyah_dataset_shard_build_text(
              tokenizer, text_a, sizeof(text_a) - 1U,
              4U, &a) == NIYAH_OK);
    CHECK(niyah_dataset_shard_build_text(
              tokenizer, text_a, sizeof(text_a) - 1U,
              4U, &a_same) == NIYAH_OK);
    CHECK(niyah_dataset_shard_build_text(
              tokenizer, text_b, sizeof(text_b) - 1U,
              4U, &content_changed) == NIYAH_OK);
    CHECK(niyah_dataset_shard_build_text(
              tokenizer, text_a, sizeof(text_a) - 1U,
              3U, &geometry_changed) == NIYAH_OK);

    CHECK(niyah_dataset_shard_identity_sha256(&a, id_a) == NIYAH_OK);
    CHECK(niyah_dataset_shard_identity_sha256(
              &a_same, id_same) == NIYAH_OK);
    CHECK(niyah_dataset_shard_identity_sha256(
              &content_changed, id_content) == NIYAH_OK);
    CHECK(niyah_dataset_shard_identity_sha256(
              &geometry_changed, id_geometry) == NIYAH_OK);

    CHECK(memcmp(id_a, id_same, sizeof(id_a)) == 0);
    CHECK(memcmp(id_a, id_content, sizeof(id_a)) != 0);
    CHECK(memcmp(id_a, id_geometry, sizeof(id_a)) != 0);

    niyah_dataset_shard_destroy(&geometry_changed);
    niyah_dataset_shard_destroy(&content_changed);
    niyah_dataset_shard_destroy(&a_same);
    niyah_dataset_shard_destroy(&a);
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

    data[8U] = 4U;
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

static void test_boundary_aware_v2(void)
{
    static const char path[] =
        "niyah_dataset_shard_v2.bin";
    static const uint8_t corpus[] =
        "aaaa\n\nbbbb";
    static const size_t offsets[] = {0U, 6U};
    static const size_t lengths[] = {4U, 4U};

    NiyahTokenizerTrainConfig config;
    NiyahTokenizer *tokenizer = NULL;
    NiyahDatasetShard shard;
    NiyahDatasetShard loaded;
    const uint32_t *tokens = NULL;
    const uint32_t *targets = NULL;
    size_t count = 0U;
    unsigned char *bytes = NULL;
    size_t byte_count = 0U;
    uint8_t id_a[NIYAH_DATASET_SHARD_IDENTITY_SHA256_SIZE];
    uint8_t id_b[NIYAH_DATASET_SHARD_IDENTITY_SHA256_SIZE];

    memset(&shard, 0, sizeof(shard));
    memset(&loaded, 0, sizeof(loaded));
    (void)remove(path);

    config.target_vocab_size =
        NIYAH_TOKENIZER_BASE_VOCAB_SIZE;
    config.min_pair_frequency = 1U;

    CHECK(niyah_tokenizer_train(
              corpus,
              sizeof(corpus) - 1U,
              &config,
              &tokenizer) == NIYAH_OK);

    CHECK(tokenizer != NULL);
    if (tokenizer == NULL) goto done;

    CHECK(niyah_dataset_shard_build_records(
              tokenizer,
              corpus,
              sizeof(corpus) - 1U,
              offsets,
              lengths,
              2U,
              16U,
              &shard) == NIYAH_OK);

    CHECK(shard.has_explicit_samples != 0);
    CHECK(shard.sample_count == 2U);
    CHECK(shard.sample_offsets != NULL);
    CHECK(shard.sample_lengths != NULL);

    CHECK(niyah_dataset_shard_sample(
              &shard,
              0U,
              &tokens,
              &targets,
              &count) == NIYAH_OK);
    CHECK(count == 5U);
    CHECK(tokens[0U] == NIYAH_TOKEN_BOS);
    CHECK(targets[count - 1U] == NIYAH_TOKEN_EOS);

    CHECK(niyah_dataset_shard_sample(
              &shard,
              1U,
              &tokens,
              &targets,
              &count) == NIYAH_OK);
    CHECK(count == 5U);
    CHECK(tokens[0U] == NIYAH_TOKEN_BOS);
    CHECK(targets[count - 1U] == NIYAH_TOKEN_EOS);

    CHECK(niyah_dataset_shard_identity_sha256(
              &shard,
              id_a) == NIYAH_OK);

    CHECK(niyah_dataset_shard_save(
              &shard,
              tokenizer,
              path) == NIYAH_OK);

    bytes = read_all(path, &byte_count);
    CHECK(bytes != NULL);
    CHECK(byte_count > 12U);

    if (bytes != NULL && byte_count > 12U) {
        CHECK(bytes[8U] == 2U);
        CHECK(bytes[9U] == 0U);
        CHECK(bytes[10U] == 0U);
        CHECK(bytes[11U] == 0U);
    }

    CHECK(niyah_dataset_shard_load(
              path,
              tokenizer,
              &loaded) == NIYAH_OK);

    CHECK(loaded.has_explicit_samples != 0);
    CHECK(loaded.sample_count == shard.sample_count);
    CHECK(loaded.token_count == shard.token_count);
    CHECK(memcmp(
              loaded.sample_offsets,
              shard.sample_offsets,
              shard.sample_count * sizeof(size_t)) == 0);
    CHECK(memcmp(
              loaded.sample_lengths,
              shard.sample_lengths,
              shard.sample_count * sizeof(size_t)) == 0);

    CHECK(niyah_dataset_shard_identity_sha256(
              &loaded,
              id_b) == NIYAH_OK);
    CHECK(memcmp(id_a, id_b, sizeof(id_a)) == 0);

done:
    free(bytes);
    niyah_dataset_shard_destroy(&loaded);
    niyah_dataset_shard_destroy(&shard);
    niyah_tokenizer_destroy(tokenizer);
    (void)remove(path);
}

static void test_supervised_v3(void)
{
    static const char path[] =
        "niyah_dataset_shard_v3.bin";
    static const uint8_t text[] =
        "Question AAnswer AQuestion BAnswer B";

    static const NiyahDatasetSupervisedRecord records[] = {
        {0U, 10U, 10U, 8U},
        {18U, 10U, 28U, 8U}
    };

    NiyahTokenizerTrainConfig config;
    NiyahTokenizer *tokenizer = NULL;
    NiyahDatasetShard shard;
    NiyahDatasetShard loaded;
    NiyahTrainingSample samples[2];
    const uint32_t *tokens = NULL;
    const uint32_t *targets = NULL;
    size_t token_count = 0U;
    size_t loss_start = 0U;
    size_t sample_count = 0U;
    unsigned char *bytes = NULL;
    size_t byte_count = 0U;
    uint8_t id_a[NIYAH_DATASET_SHARD_IDENTITY_SHA256_SIZE];
    uint8_t id_b[NIYAH_DATASET_SHARD_IDENTITY_SHA256_SIZE];

    memset(&shard, 0, sizeof(shard));
    memset(&loaded, 0, sizeof(loaded));
    memset(samples, 0, sizeof(samples));
    (void)remove(path);

    config.target_vocab_size =
        NIYAH_TOKENIZER_BASE_VOCAB_SIZE;
    config.min_pair_frequency = 1U;

    CHECK(niyah_tokenizer_train(
              text,
              sizeof(text) - 1U,
              &config,
              &tokenizer) == NIYAH_OK);

    CHECK(tokenizer != NULL);
    if (tokenizer == NULL)
        goto done;

    CHECK(niyah_dataset_shard_build_supervised_records(
              tokenizer,
              text,
              sizeof(text) - 1U,
              records,
              2U,
              64U,
              &shard) == NIYAH_OK);

    CHECK(shard.has_explicit_samples != 0);
    CHECK(shard.has_loss_starts != 0);
    CHECK(shard.sample_count == 2U);
    CHECK(shard.sample_loss_starts != NULL);

    CHECK(niyah_dataset_shard_sample_with_loss(
              &shard,
              0U,
              &tokens,
              &targets,
              &token_count,
              &loss_start) == NIYAH_OK);

    CHECK(tokens != NULL);
    if (tokens != NULL) {
        CHECK(targets == tokens + 1U);
    }

    CHECK(token_count == 19U);
    CHECK(loss_start == 10U);

    if (tokens != NULL && token_count > 0U) {
        CHECK(tokens[0U] == NIYAH_TOKEN_BOS);
    }
    if (targets != NULL && token_count > 0U) {
        CHECK(targets[token_count - 1U] == NIYAH_TOKEN_EOS);
    }

    CHECK(niyah_training_samples_from_shard(
              &shard,
              samples,
              2U,
              &sample_count) == NIYAH_OK);

    CHECK(sample_count == 2U);
    CHECK(samples[0U].loss_start == 10U);
    CHECK(samples[1U].loss_start == 10U);
    CHECK(samples[0U].token_count == 19U);
    CHECK(samples[1U].token_count == 19U);

    CHECK(niyah_dataset_shard_identity_sha256(
              &shard,
              id_a) == NIYAH_OK);

    CHECK(niyah_dataset_shard_save(
              &shard,
              tokenizer,
              path) == NIYAH_OK);

    bytes = read_all(
        path,
        &byte_count);

    CHECK(bytes != NULL);
    CHECK(byte_count > 12U);

    if (bytes != NULL &&
        byte_count > 12U) {
        CHECK(bytes[8U] == 3U);
        CHECK(bytes[9U] == 0U);
        CHECK(bytes[10U] == 0U);
        CHECK(bytes[11U] == 0U);
    }

    CHECK(niyah_dataset_shard_load(
              path,
              tokenizer,
              &loaded) == NIYAH_OK);

    CHECK(loaded.has_explicit_samples != 0);
    CHECK(loaded.has_loss_starts != 0);
    CHECK(loaded.sample_count == shard.sample_count);

    CHECK(memcmp(
              loaded.sample_offsets,
              shard.sample_offsets,
              shard.sample_count * sizeof(size_t)) == 0);

    CHECK(memcmp(
              loaded.sample_lengths,
              shard.sample_lengths,
              shard.sample_count * sizeof(size_t)) == 0);

    CHECK(memcmp(
              loaded.sample_loss_starts,
              shard.sample_loss_starts,
              shard.sample_count * sizeof(size_t)) == 0);

    CHECK(niyah_dataset_shard_identity_sha256(
              &loaded,
              id_b) == NIYAH_OK);

    CHECK(memcmp(
              id_a,
              id_b,
              sizeof(id_a)) == 0);

done:
    free(bytes);
    niyah_dataset_shard_destroy(&loaded);
    niyah_dataset_shard_destroy(&shard);
    niyah_tokenizer_destroy(tokenizer);
    (void)remove(path);
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
    test_shard_content_identity();
    test_persistence_and_identity();
    test_corruption_rejection();
    test_boundary_aware_v2();
    test_supervised_v3();
    test_invalid_inputs();

    if (failures != 0) {
        fprintf(stderr, "niyah_dataset_shard_test: %d failure(s)\n", failures);
        return 1;
    }

    puts("P6H_DATASET_PREPROCESS=PASS");
    puts("P6H_BINARY_SHARD_V1=PASS");
    puts("P6H_TOKENIZER_BINDING=PASS");
    puts("P6H_CORRUPTION_REJECTION=PASS");
    puts("P8D_BOUNDARY_AWARE_SHARD_V2=PASS");
    puts("P8F_SUPERVISED_SHARD_V3=PASS");
    return 0;
}
