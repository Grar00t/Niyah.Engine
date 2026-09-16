#include "niyah/tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        return 1; \
    } \
} while (0)

static int check_deterministic_merges(const NiyahTokenizer *a, const NiyahTokenizer *b)
{
    size_t i;
    CHECK(niyah_tokenizer_vocab_size(a) == niyah_tokenizer_vocab_size(b));
    CHECK(niyah_tokenizer_merge_count(a) == niyah_tokenizer_merge_count(b));

    for (i = 0U; i < niyah_tokenizer_merge_count(a); ++i) {
        uint32_t al;
        uint32_t ar;
        uint32_t ao;
        uint32_t bl;
        uint32_t br;
        uint32_t bo;
        CHECK(niyah_tokenizer_merge_at(a, i, &al, &ar, &ao) == NIYAH_OK);
        CHECK(niyah_tokenizer_merge_at(b, i, &bl, &br, &bo) == NIYAH_OK);
        CHECK(al == bl);
        CHECK(ar == br);
        CHECK(ao == bo);
    }
    return 0;
}

static int check_roundtrip(const NiyahTokenizer *tokenizer,
                           const uint8_t *input,
                           size_t input_size)
{
    uint32_t tokens[128];
    uint8_t decoded[256];
    size_t token_count = 0U;
    size_t decoded_size = 0U;

    CHECK(input_size <= 128U);
    CHECK(niyah_tokenizer_encode(tokenizer, input, input_size,
                                 NULL, 0U, &token_count) == NIYAH_OK);
    CHECK(token_count <= input_size);
    CHECK(token_count <= 128U);
    CHECK(niyah_tokenizer_encode(tokenizer, input, input_size,
                                 tokens, 128U, &token_count) == NIYAH_OK);

    CHECK(niyah_tokenizer_decode(tokenizer, tokens, token_count,
                                 NULL, 0U, &decoded_size) == NIYAH_OK);
    CHECK(decoded_size == input_size);
    CHECK(decoded_size <= sizeof(decoded));
    CHECK(niyah_tokenizer_decode(tokenizer, tokens, token_count,
                                 decoded, sizeof(decoded), &decoded_size) == NIYAH_OK);
    CHECK(memcmp(decoded, input, input_size) == 0);
    return 0;
}


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

static unsigned char *read_file(const char *path, size_t *out_size)
{
    FILE *file;
    long end;
    unsigned char *data;

    *out_size = 0U;

    file = test_fopen(path, "rb");
    if (file == NULL) return NULL;

    if (fseek(file, 0L, SEEK_END) != 0) {
        fclose(file);
        return NULL;
    }

    end = ftell(file);
    if (end < 0L ||
        fseek(file, 0L, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }

    data = (unsigned char *)malloc(
        (size_t)end == 0U ? 1U : (size_t)end);

    if (data == NULL) {
        fclose(file);
        return NULL;
    }

    if ((size_t)end != 0U &&
        fread(data, 1U, (size_t)end, file) !=
            (size_t)end) {
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

static int write_file(const char *path,
                      const unsigned char *data,
                      size_t size)
{
    FILE *file = test_fopen(path, "wb");

    if (file == NULL) return 0;

    if (size != 0U &&
        fwrite(data, 1U, size, file) != size) {
        fclose(file);
        return 0;
    }

    return fclose(file) == 0;
}

static int check_persistence(const NiyahTokenizer *tokenizer,
                             const uint8_t *sample,
                             size_t sample_size)
{
    static const char path[] = "niyah_tokenizer_v1.bin";
    static const char bad[] = "niyah_tokenizer_v1_bad.bin";

    NiyahTokenizer *loaded = NULL;
    NiyahTokenizer *rejected = NULL;
    unsigned char *data = NULL;
    uint32_t a[128];
    uint32_t b[128];
    size_t ac = 0U;
    size_t bc = 0U;
    size_t size = 0U;

    (void)remove(path);
    (void)remove(bad);

    CHECK(niyah_tokenizer_save(tokenizer, path) == NIYAH_OK);
    CHECK(niyah_tokenizer_load(path, &loaded) == NIYAH_OK);
    CHECK(loaded != NULL);

    CHECK(check_deterministic_merges(
              tokenizer, loaded) == 0);
    {
        uint8_t before[NIYAH_TOKENIZER_IDENTITY_SHA256_SIZE];
        uint8_t after[NIYAH_TOKENIZER_IDENTITY_SHA256_SIZE];
        CHECK(niyah_tokenizer_identity_sha256(tokenizer, before) == NIYAH_OK);
        CHECK(niyah_tokenizer_identity_sha256(loaded, after) == NIYAH_OK);
        CHECK(memcmp(before, after, sizeof(before)) == 0);
    }

    CHECK(niyah_tokenizer_encode(
              tokenizer,
              sample,
              sample_size,
              a,
              sizeof(a) / sizeof(a[0]),
              &ac) == NIYAH_OK);

    CHECK(niyah_tokenizer_encode(
              loaded,
              sample,
              sample_size,
              b,
              sizeof(b) / sizeof(b[0]),
              &bc) == NIYAH_OK);

    CHECK(ac == bc);
    CHECK(memcmp(a, b, ac * sizeof(uint32_t)) == 0);

    data = read_file(path, &size);
    CHECK(data != NULL);
    CHECK(size > 40U);

    data[size - 1U] ^= 1U;
    CHECK(write_file(bad, data, size));
    CHECK(niyah_tokenizer_load(
              bad, &rejected) == NIYAH_ERR_CORRUPT_DATA);
    CHECK(rejected == NULL);
    data[size - 1U] ^= 1U;

    data[8U] = 2U;
    CHECK(write_file(bad, data, size));
    CHECK(niyah_tokenizer_load(
              bad, &rejected) ==
          NIYAH_ERR_UNSUPPORTED_VERSION);
    CHECK(rejected == NULL);
    data[8U] = 1U;

    CHECK(write_file(bad, data, size - 1U));
    CHECK(niyah_tokenizer_load(
              bad, &rejected) == NIYAH_ERR_CORRUPT_DATA);
    CHECK(rejected == NULL);

    free(data);
    niyah_tokenizer_destroy(loaded);
    (void)remove(path);
    (void)remove(bad);

    return 0;
}

int main(void)
{
    static const uint8_t corpus[] =
        "banana banana banana bandana banana banana\n"
        "native language model native language model\n";
    static const uint8_t sample[] = "banana banana banana";
    static const uint8_t arabic_utf8[] = {
        0xd9U, 0x85U, 0xd8U, 0xb1U, 0xd8U, 0xadU, 0xd8U, 0xa8U, 0xd8U, 0xa7U,
        0x20U,
        0xd8U, 0xa8U, 0xd8U, 0xa7U, 0xd9U, 0x84U, 0xd8U, 0xb9U, 0xd8U, 0xa7U,
        0xd9U, 0x84U, 0xd9U, 0x85U
    };
    NiyahTokenizerTrainConfig config;
    NiyahTokenizer *first = NULL;
    NiyahTokenizer *second = NULL;
    uint8_t first_identity[NIYAH_TOKENIZER_IDENTITY_SHA256_SIZE];
    uint8_t second_identity[NIYAH_TOKENIZER_IDENTITY_SHA256_SIZE];
    size_t sample_tokens = 0U;
    uint32_t special_tokens[3];
    uint8_t special_decoded[4];
    size_t special_size = 0U;

    config.target_vocab_size = 274U;
    config.min_pair_frequency = 2U;

    CHECK(niyah_tokenizer_train(corpus, sizeof(corpus) - 1U,
                                &config, &first) == NIYAH_OK);
    CHECK(niyah_tokenizer_train(corpus, sizeof(corpus) - 1U,
                                &config, &second) == NIYAH_OK);

    CHECK(first != NULL);
    CHECK(second != NULL);
    CHECK(niyah_tokenizer_vocab_size(first) >= NIYAH_TOKENIZER_BASE_VOCAB_SIZE);
    CHECK(niyah_tokenizer_vocab_size(first) <= config.target_vocab_size);
    CHECK(niyah_tokenizer_merge_count(first) > 0U);
    CHECK(check_deterministic_merges(first, second) == 0);
    CHECK(niyah_tokenizer_identity_sha256(first, first_identity) == NIYAH_OK);
    CHECK(niyah_tokenizer_identity_sha256(second, second_identity) == NIYAH_OK);
    CHECK(memcmp(first_identity, second_identity, sizeof(first_identity)) == 0);
    niyah_tokenizer_destroy(second);
    second = NULL;

    CHECK(niyah_tokenizer_encode(first, sample, sizeof(sample) - 1U,
                                 NULL, 0U, &sample_tokens) == NIYAH_OK);
    CHECK(sample_tokens < sizeof(sample) - 1U);

    CHECK(check_roundtrip(first, sample, sizeof(sample) - 1U) == 0);
    CHECK(check_roundtrip(first, arabic_utf8, sizeof(arabic_utf8)) == 0);
    CHECK(check_persistence(first, sample, sizeof(sample) - 1U) == 0);

    special_tokens[0] = NIYAH_TOKEN_BOS;
    special_tokens[1] = (uint32_t)'A';
    special_tokens[2] = NIYAH_TOKEN_EOS;
    CHECK(niyah_tokenizer_decode(first, special_tokens, 3U,
                                 special_decoded, sizeof(special_decoded),
                                 &special_size) == NIYAH_OK);
    CHECK(special_size == 1U);
    CHECK(special_decoded[0] == (uint8_t)'A');

    config.target_vocab_size = NIYAH_TOKENIZER_BASE_VOCAB_SIZE - 1U;
    CHECK(niyah_tokenizer_train(corpus, sizeof(corpus) - 1U,
                                &config, &second) == NIYAH_ERR_INVALID_CONFIG);
    CHECK(second == NULL);

    niyah_tokenizer_destroy(first);

    puts("NIYAH_TOKENIZER_TEST=PASS");
    return 0;
}
