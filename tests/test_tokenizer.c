#include "niyah/tokenizer.h"

#include <stdio.h>
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
    niyah_tokenizer_destroy(second);
    second = NULL;

    CHECK(niyah_tokenizer_encode(first, sample, sizeof(sample) - 1U,
                                 NULL, 0U, &sample_tokens) == NIYAH_OK);
    CHECK(sample_tokens < sizeof(sample) - 1U);

    CHECK(check_roundtrip(first, sample, sizeof(sample) - 1U) == 0);
    CHECK(check_roundtrip(first, arabic_utf8, sizeof(arabic_utf8)) == 0);

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
