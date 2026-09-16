#include "niyah/tokenizer.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct NiyahTokenBytes {
    uint8_t *data;
    size_t length;
} NiyahTokenBytes;

typedef struct NiyahMerge {
    uint32_t left;
    uint32_t right;
    uint32_t output;
} NiyahMerge;

struct NiyahTokenizer {
    NiyahTokenBytes *vocab;
    size_t vocab_size;
    size_t vocab_capacity;
    NiyahMerge *merges;
    size_t merge_count;
    size_t merge_capacity;
};

static int niyah_size_mul_ok(size_t a, size_t b, size_t *out)
{
    if (out == NULL || (a != 0U && b > SIZE_MAX / a)) {
        return 0;
    }
    *out = a * b;
    return 1;
}

static int niyah_size_add_ok(size_t a, size_t b, size_t *out)
{
    if (out == NULL || a > SIZE_MAX - b) {
        return 0;
    }
    *out = a + b;
    return 1;
}

static int niyah_u64_compare(const void *a, const void *b)
{
    const uint64_t lhs = *(const uint64_t *)a;
    const uint64_t rhs = *(const uint64_t *)b;
    if (lhs < rhs) {
        return -1;
    }
    if (lhs > rhs) {
        return 1;
    }
    return 0;
}

static uint64_t niyah_pair_key(uint32_t left, uint32_t right)
{
    return ((uint64_t)left << 32U) | (uint64_t)right;
}

static NiyahStatus niyah_tokenizer_alloc_base(uint32_t target_vocab_size,
                                               NiyahTokenizer **out_tokenizer)
{
    NiyahTokenizer *tokenizer;
    size_t vocab_bytes = 0U;
    size_t merge_capacity;
    size_t merge_bytes = 0U;
    uint32_t i;

    if (out_tokenizer == NULL || target_vocab_size < NIYAH_TOKENIZER_BASE_VOCAB_SIZE) {
        return NIYAH_ERR_INVALID_CONFIG;
    }
    if (!niyah_size_mul_ok((size_t)target_vocab_size, sizeof(NiyahTokenBytes), &vocab_bytes)) {
        return NIYAH_ERR_OVERFLOW;
    }

    tokenizer = (NiyahTokenizer *)calloc(1U, sizeof(*tokenizer));
    if (tokenizer == NULL) {
        return NIYAH_ERR_OUT_OF_MEMORY;
    }

    tokenizer->vocab = (NiyahTokenBytes *)calloc(1U, vocab_bytes);
    if (tokenizer->vocab == NULL) {
        free(tokenizer);
        return NIYAH_ERR_OUT_OF_MEMORY;
    }

    merge_capacity = (size_t)target_vocab_size - (size_t)NIYAH_TOKENIZER_BASE_VOCAB_SIZE;
    if (merge_capacity != 0U) {
        if (!niyah_size_mul_ok(merge_capacity, sizeof(NiyahMerge), &merge_bytes)) {
            free(tokenizer->vocab);
            free(tokenizer);
            return NIYAH_ERR_OVERFLOW;
        }
        tokenizer->merges = (NiyahMerge *)calloc(1U, merge_bytes);
        if (tokenizer->merges == NULL) {
            free(tokenizer->vocab);
            free(tokenizer);
            return NIYAH_ERR_OUT_OF_MEMORY;
        }
    }

    tokenizer->vocab_capacity = (size_t)target_vocab_size;
    tokenizer->merge_capacity = merge_capacity;
    tokenizer->vocab_size = (size_t)NIYAH_TOKENIZER_BASE_VOCAB_SIZE;

    for (i = 0U; i < NIYAH_TOKENIZER_BYTE_VOCAB_SIZE; ++i) {
        tokenizer->vocab[i].data = (uint8_t *)malloc(1U);
        if (tokenizer->vocab[i].data == NULL) {
            niyah_tokenizer_destroy(tokenizer);
            return NIYAH_ERR_OUT_OF_MEMORY;
        }
        tokenizer->vocab[i].data[0] = (uint8_t)i;
        tokenizer->vocab[i].length = 1U;
    }

    *out_tokenizer = tokenizer;
    return NIYAH_OK;
}

static NiyahStatus niyah_tokenizer_add_merge(NiyahTokenizer *tokenizer,
                                             uint32_t left,
                                             uint32_t right,
                                             uint32_t *out_token)
{
    NiyahTokenBytes *left_bytes;
    NiyahTokenBytes *right_bytes;
    NiyahTokenBytes *new_bytes;
    size_t total_length = 0U;
    uint32_t token_id;

    if (tokenizer == NULL || out_token == NULL ||
        (size_t)left >= tokenizer->vocab_size ||
        (size_t)right >= tokenizer->vocab_size ||
        tokenizer->vocab_size >= tokenizer->vocab_capacity ||
        tokenizer->merge_count >= tokenizer->merge_capacity ||
        tokenizer->vocab_size > (size_t)UINT32_MAX) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    left_bytes = &tokenizer->vocab[left];
    right_bytes = &tokenizer->vocab[right];
    if (!niyah_size_add_ok(left_bytes->length, right_bytes->length, &total_length)) {
        return NIYAH_ERR_OVERFLOW;
    }
    if (total_length == 0U) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    new_bytes = &tokenizer->vocab[tokenizer->vocab_size];
    new_bytes->data = (uint8_t *)malloc(total_length);
    if (new_bytes->data == NULL) {
        return NIYAH_ERR_OUT_OF_MEMORY;
    }
    memcpy(new_bytes->data, left_bytes->data, left_bytes->length);
    memcpy(new_bytes->data + left_bytes->length, right_bytes->data, right_bytes->length);
    new_bytes->length = total_length;

    token_id = (uint32_t)tokenizer->vocab_size;
    tokenizer->merges[tokenizer->merge_count].left = left;
    tokenizer->merges[tokenizer->merge_count].right = right;
    tokenizer->merges[tokenizer->merge_count].output = token_id;
    tokenizer->merge_count += 1U;
    tokenizer->vocab_size += 1U;
    *out_token = token_id;
    return NIYAH_OK;
}

static void niyah_apply_merge(uint32_t *tokens,
                              size_t *token_count,
                              uint32_t left,
                              uint32_t right,
                              uint32_t output)
{
    size_t read_index = 0U;
    size_t write_index = 0U;
    size_t count;

    if (tokens == NULL || token_count == NULL) {
        return;
    }

    count = *token_count;
    while (read_index < count) {
        if (read_index + 1U < count &&
            tokens[read_index] == left &&
            tokens[read_index + 1U] == right) {
            tokens[write_index++] = output;
            read_index += 2U;
        } else {
            tokens[write_index++] = tokens[read_index++];
        }
    }
    *token_count = write_index;
}

NiyahStatus niyah_tokenizer_train(const uint8_t *corpus,
                                  size_t corpus_size,
                                  const NiyahTokenizerTrainConfig *config,
                                  NiyahTokenizer **out_tokenizer)
{
    NiyahTokenizer *tokenizer = NULL;
    uint32_t *tokens = NULL;
    size_t token_bytes = 0U;
    size_t token_count;
    size_t i;
    NiyahStatus status;

    if (corpus == NULL || corpus_size == 0U || config == NULL || out_tokenizer == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    *out_tokenizer = NULL;
    if (config->target_vocab_size < NIYAH_TOKENIZER_BASE_VOCAB_SIZE ||
        config->min_pair_frequency == 0U) {
        return NIYAH_ERR_INVALID_CONFIG;
    }
    if (!niyah_size_mul_ok(corpus_size, sizeof(uint32_t), &token_bytes)) {
        return NIYAH_ERR_OVERFLOW;
    }

    status = niyah_tokenizer_alloc_base(config->target_vocab_size, &tokenizer);
    if (status != NIYAH_OK) {
        return status;
    }

    tokens = (uint32_t *)malloc(token_bytes);
    if (tokens == NULL) {
        niyah_tokenizer_destroy(tokenizer);
        return NIYAH_ERR_OUT_OF_MEMORY;
    }
    for (i = 0U; i < corpus_size; ++i) {
        tokens[i] = (uint32_t)corpus[i];
    }
    token_count = corpus_size;

    while (tokenizer->vocab_size < tokenizer->vocab_capacity && token_count > 1U) {
        uint64_t *pairs;
        size_t pair_count = token_count - 1U;
        size_t pair_bytes = 0U;
        uint64_t best_key = 0U;
        size_t best_frequency = 0U;
        size_t run_start;
        uint32_t left;
        uint32_t right;
        uint32_t output;

        if (!niyah_size_mul_ok(pair_count, sizeof(uint64_t), &pair_bytes)) {
            status = NIYAH_ERR_OVERFLOW;
            goto fail;
        }
        pairs = (uint64_t *)malloc(pair_bytes);
        if (pairs == NULL) {
            status = NIYAH_ERR_OUT_OF_MEMORY;
            goto fail;
        }
        for (i = 0U; i < pair_count; ++i) {
            pairs[i] = niyah_pair_key(tokens[i], tokens[i + 1U]);
        }
        qsort(pairs, pair_count, sizeof(uint64_t), niyah_u64_compare);

        run_start = 0U;
        while (run_start < pair_count) {
            size_t run_end = run_start + 1U;
            size_t frequency;
            while (run_end < pair_count && pairs[run_end] == pairs[run_start]) {
                run_end += 1U;
            }
            frequency = run_end - run_start;
            if (frequency > best_frequency) {
                best_frequency = frequency;
                best_key = pairs[run_start];
            }
            run_start = run_end;
        }
        free(pairs);

        if (best_frequency < (size_t)config->min_pair_frequency) {
            break;
        }

        left = (uint32_t)(best_key >> 32U);
        right = (uint32_t)(best_key & UINT64_C(0xffffffff));
        status = niyah_tokenizer_add_merge(tokenizer, left, right, &output);
        if (status != NIYAH_OK) {
            goto fail;
        }
        niyah_apply_merge(tokens, &token_count, left, right, output);
    }

    free(tokens);
    *out_tokenizer = tokenizer;
    return NIYAH_OK;

fail:
    free(tokens);
    niyah_tokenizer_destroy(tokenizer);
    return status;
}

void niyah_tokenizer_destroy(NiyahTokenizer *tokenizer)
{
    size_t i;
    if (tokenizer == NULL) {
        return;
    }
    if (tokenizer->vocab != NULL) {
        for (i = 0U; i < tokenizer->vocab_size; ++i) {
            free(tokenizer->vocab[i].data);
        }
    }
    free(tokenizer->merges);
    free(tokenizer->vocab);
    free(tokenizer);
}

size_t niyah_tokenizer_vocab_size(const NiyahTokenizer *tokenizer)
{
    return tokenizer == NULL ? 0U : tokenizer->vocab_size;
}

size_t niyah_tokenizer_merge_count(const NiyahTokenizer *tokenizer)
{
    return tokenizer == NULL ? 0U : tokenizer->merge_count;
}

NiyahStatus niyah_tokenizer_merge_at(const NiyahTokenizer *tokenizer,
                                     size_t merge_index,
                                     uint32_t *left_token,
                                     uint32_t *right_token,
                                     uint32_t *output_token)
{
    const NiyahMerge *merge;
    if (tokenizer == NULL || left_token == NULL || right_token == NULL ||
        output_token == NULL || merge_index >= tokenizer->merge_count) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    merge = &tokenizer->merges[merge_index];
    *left_token = merge->left;
    *right_token = merge->right;
    *output_token = merge->output;
    return NIYAH_OK;
}


_Static_assert(CHAR_BIT == 8, "tokenizer v1 requires 8-bit bytes");

#define P6C_TOKENIZER_IDENTITY_SHA256 1

typedef struct NiyahTokenizerSha256 {
    uint32_t h[8];
    uint64_t bit_count;
    unsigned char block[64];
    size_t block_used;
} NiyahTokenizerSha256;

static uint32_t niyah_sha256_rotr(uint32_t x, unsigned n)
{
    return (x >> n) | (x << (32U - n));
}

static void niyah_sha256_transform(NiyahTokenizerSha256 *s,
                                   const unsigned char block[64])
{
    static const uint32_t k[64] = {
        UINT32_C(0x428a2f98), UINT32_C(0x71374491), UINT32_C(0xb5c0fbcf), UINT32_C(0xe9b5dba5),
        UINT32_C(0x3956c25b), UINT32_C(0x59f111f1), UINT32_C(0x923f82a4), UINT32_C(0xab1c5ed5),
        UINT32_C(0xd807aa98), UINT32_C(0x12835b01), UINT32_C(0x243185be), UINT32_C(0x550c7dc3),
        UINT32_C(0x72be5d74), UINT32_C(0x80deb1fe), UINT32_C(0x9bdc06a7), UINT32_C(0xc19bf174),
        UINT32_C(0xe49b69c1), UINT32_C(0xefbe4786), UINT32_C(0x0fc19dc6), UINT32_C(0x240ca1cc),
        UINT32_C(0x2de92c6f), UINT32_C(0x4a7484aa), UINT32_C(0x5cb0a9dc), UINT32_C(0x76f988da),
        UINT32_C(0x983e5152), UINT32_C(0xa831c66d), UINT32_C(0xb00327c8), UINT32_C(0xbf597fc7),
        UINT32_C(0xc6e00bf3), UINT32_C(0xd5a79147), UINT32_C(0x06ca6351), UINT32_C(0x14292967),
        UINT32_C(0x27b70a85), UINT32_C(0x2e1b2138), UINT32_C(0x4d2c6dfc), UINT32_C(0x53380d13),
        UINT32_C(0x650a7354), UINT32_C(0x766a0abb), UINT32_C(0x81c2c92e), UINT32_C(0x92722c85),
        UINT32_C(0xa2bfe8a1), UINT32_C(0xa81a664b), UINT32_C(0xc24b8b70), UINT32_C(0xc76c51a3),
        UINT32_C(0xd192e819), UINT32_C(0xd6990624), UINT32_C(0xf40e3585), UINT32_C(0x106aa070),
        UINT32_C(0x19a4c116), UINT32_C(0x1e376c08), UINT32_C(0x2748774c), UINT32_C(0x34b0bcb5),
        UINT32_C(0x391c0cb3), UINT32_C(0x4ed8aa4a), UINT32_C(0x5b9cca4f), UINT32_C(0x682e6ff3),
        UINT32_C(0x748f82ee), UINT32_C(0x78a5636f), UINT32_C(0x84c87814), UINT32_C(0x8cc70208),
        UINT32_C(0x90befffa), UINT32_C(0xa4506ceb), UINT32_C(0xbef9a3f7), UINT32_C(0xc67178f2)
    };
    uint32_t w[64];
    uint32_t a, b, c, d, e, f, g, h;
    size_t i;

    for (i = 0U; i < 16U; ++i) {
        const size_t j = i * 4U;
        w[i] = ((uint32_t)block[j] << 24) |
               ((uint32_t)block[j + 1U] << 16) |
               ((uint32_t)block[j + 2U] << 8) |
               (uint32_t)block[j + 3U];
    }
    for (i = 16U; i < 64U; ++i) {
        const uint32_t x = w[i - 15U];
        const uint32_t y = w[i - 2U];
        const uint32_t s0 = niyah_sha256_rotr(x, 7U) ^
                            niyah_sha256_rotr(x, 18U) ^ (x >> 3U);
        const uint32_t s1 = niyah_sha256_rotr(y, 17U) ^
                            niyah_sha256_rotr(y, 19U) ^ (y >> 10U);
        w[i] = w[i - 16U] + s0 + w[i - 7U] + s1;
    }

    a = s->h[0]; b = s->h[1]; c = s->h[2]; d = s->h[3];
    e = s->h[4]; f = s->h[5]; g = s->h[6]; h = s->h[7];

    for (i = 0U; i < 64U; ++i) {
        const uint32_t s1 = niyah_sha256_rotr(e, 6U) ^
                            niyah_sha256_rotr(e, 11U) ^
                            niyah_sha256_rotr(e, 25U);
        const uint32_t ch = (e & f) ^ ((~e) & g);
        const uint32_t t1 = h + s1 + ch + k[i] + w[i];
        const uint32_t s0 = niyah_sha256_rotr(a, 2U) ^
                            niyah_sha256_rotr(a, 13U) ^
                            niyah_sha256_rotr(a, 22U);
        const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t t2 = s0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    s->h[0] += a; s->h[1] += b; s->h[2] += c; s->h[3] += d;
    s->h[4] += e; s->h[5] += f; s->h[6] += g; s->h[7] += h;
}

static void niyah_sha256_init(NiyahTokenizerSha256 *s)
{
    static const uint32_t initial[8] = {
        UINT32_C(0x6a09e667), UINT32_C(0xbb67ae85),
        UINT32_C(0x3c6ef372), UINT32_C(0xa54ff53a),
        UINT32_C(0x510e527f), UINT32_C(0x9b05688c),
        UINT32_C(0x1f83d9ab), UINT32_C(0x5be0cd19)
    };
    memcpy(s->h, initial, sizeof(initial));
    s->bit_count = UINT64_C(0);
    s->block_used = 0U;
}

static int niyah_sha256_update(NiyahTokenizerSha256 *s,
                               const unsigned char *data,
                               size_t size)
{
    size_t offset = 0U;

    if (size > (size_t)(UINT64_MAX / UINT64_C(8)) ||
        s->bit_count > UINT64_MAX - (uint64_t)size * UINT64_C(8)) {
        return 0;
    }
    s->bit_count += (uint64_t)size * UINT64_C(8);

    while (offset < size) {
        const size_t available = 64U - s->block_used;
        const size_t remaining = size - offset;
        const size_t take = remaining < available ? remaining : available;

        memcpy(s->block + s->block_used, data + offset, take);
        s->block_used += take;
        offset += take;

        if (s->block_used == 64U) {
            niyah_sha256_transform(s, s->block);
            s->block_used = 0U;
        }
    }
    return 1;
}

static void niyah_sha256_final(NiyahTokenizerSha256 *s,
                               unsigned char out[32])
{
    size_t i;
    uint64_t bits = s->bit_count;

    s->block[s->block_used++] = 0x80U;
    if (s->block_used > 56U) {
        memset(s->block + s->block_used, 0, 64U - s->block_used);
        niyah_sha256_transform(s, s->block);
        s->block_used = 0U;
    }
    memset(s->block + s->block_used, 0, 56U - s->block_used);

    for (i = 0U; i < 8U; ++i) {
        s->block[63U - i] = (unsigned char)(bits & UINT64_C(0xff));
        bits >>= 8U;
    }
    niyah_sha256_transform(s, s->block);

    for (i = 0U; i < 8U; ++i) {
        out[i * 4U + 0U] = (unsigned char)(s->h[i] >> 24);
        out[i * 4U + 1U] = (unsigned char)(s->h[i] >> 16);
        out[i * 4U + 2U] = (unsigned char)(s->h[i] >> 8);
        out[i * 4U + 3U] = (unsigned char)s->h[i];
    }
}


#define NIYAH_TOKENIZER_FILE_VERSION UINT32_C(1)
#define NIYAH_TOKENIZER_FILE_FLAGS UINT32_C(0)
#define NIYAH_TOKENIZER_FILE_RESERVED UINT32_C(0)
#define NIYAH_TOKENIZER_CHECKSUM_CRC32 UINT32_C(1)

static const unsigned char NIYAH_TOKENIZER_MAGIC[8] = {
    'N', 'I', 'Y', 'A', 'H', 'T', 'O', 'K'
};

typedef struct NiyahTokenizerCrc32 {
    uint32_t value;
    uint32_t table[256];
} NiyahTokenizerCrc32;

static FILE *niyah_tokenizer_fopen(const char *path, const char *mode)
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

static void niyah_tok_store_u32_le(unsigned char out[4], uint32_t value)
{
    out[0] = (unsigned char)(value & UINT32_C(0xff));
    out[1] = (unsigned char)((value >> 8) & UINT32_C(0xff));
    out[2] = (unsigned char)((value >> 16) & UINT32_C(0xff));
    out[3] = (unsigned char)((value >> 24) & UINT32_C(0xff));
}

static uint32_t niyah_tok_load_u32_le(const unsigned char in[4])
{
    return ((uint32_t)in[0]) |
           ((uint32_t)in[1] << 8) |
           ((uint32_t)in[2] << 16) |
           ((uint32_t)in[3] << 24);
}

static void niyah_tok_crc_init(NiyahTokenizerCrc32 *crc)
{
    uint32_t i;

    for (i = 0U; i < UINT32_C(256); ++i) {
        uint32_t c = i;
        unsigned bit;

        for (bit = 0U; bit < 8U; ++bit) {
            c = (c & UINT32_C(1)) != 0U
                ? UINT32_C(0xedb88320) ^ (c >> 1)
                : c >> 1;
        }
        crc->table[i] = c;
    }
    crc->value = UINT32_C(0xffffffff);
}

static void niyah_tok_crc_update(NiyahTokenizerCrc32 *crc,
                                 const unsigned char *data,
                                 size_t size)
{
    size_t i;

    for (i = 0U; i < size; ++i) {
        const uint32_t index =
            (crc->value ^ (uint32_t)data[i]) & UINT32_C(0xff);
        crc->value = crc->table[index] ^ (crc->value >> 8);
    }
}

static uint32_t niyah_tok_crc_final(const NiyahTokenizerCrc32 *crc)
{
    return crc->value ^ UINT32_C(0xffffffff);
}

static NiyahStatus niyah_tok_write(FILE *file,
                                   const void *data,
                                   size_t size,
                                   NiyahTokenizerCrc32 *crc)
{
    if (size != 0U && fwrite(data, 1U, size, file) != size) {
        return NIYAH_ERR_IO;
    }
    if (crc != NULL && size != 0U) {
        niyah_tok_crc_update(crc,
                             (const unsigned char *)data,
                             size);
    }
    return NIYAH_OK;
}

static NiyahStatus niyah_tok_write_u32(FILE *file,
                                       NiyahTokenizerCrc32 *crc,
                                       uint32_t value)
{
    unsigned char b[4];
    niyah_tok_store_u32_le(b, value);
    return niyah_tok_write(file, b, sizeof(b), crc);
}

static NiyahStatus niyah_tok_read(FILE *file,
                                  void *data,
                                  size_t size,
                                  NiyahTokenizerCrc32 *crc)
{
    const size_t got = fread(data, 1U, size, file);

    if (got != 0U && crc != NULL) {
        niyah_tok_crc_update(crc,
                             (const unsigned char *)data,
                             got);
    }

    if (got != size) {
        return ferror(file) != 0
            ? NIYAH_ERR_IO
            : NIYAH_ERR_CORRUPT_DATA;
    }
    return NIYAH_OK;
}

static NiyahStatus niyah_tok_validate(const NiyahTokenizer *tokenizer)
{
    size_t i;

    if (tokenizer == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    if (tokenizer->vocab == NULL ||
        tokenizer->vocab_size <
            (size_t)NIYAH_TOKENIZER_BASE_VOCAB_SIZE ||
        tokenizer->vocab_size > (size_t)UINT32_MAX ||
        tokenizer->merge_count !=
            tokenizer->vocab_size -
            (size_t)NIYAH_TOKENIZER_BASE_VOCAB_SIZE ||
        tokenizer->vocab_capacity < tokenizer->vocab_size ||
        tokenizer->merge_capacity < tokenizer->merge_count ||
        (tokenizer->merge_count != 0U &&
         tokenizer->merges == NULL)) {
        return NIYAH_ERR_INVALID_CONFIG;
    }

    for (i = 0U; i < tokenizer->merge_count; ++i) {
        const NiyahMerge *m = &tokenizer->merges[i];
        const uint32_t expected =
            NIYAH_TOKENIZER_BASE_VOCAB_SIZE + (uint32_t)i;

        if (m->output != expected ||
            m->left >= expected ||
            m->right >= expected ||
            m->left == NIYAH_TOKEN_BOS ||
            m->left == NIYAH_TOKEN_EOS ||
            m->right == NIYAH_TOKEN_BOS ||
            m->right == NIYAH_TOKEN_EOS) {
            return NIYAH_ERR_INVALID_CONFIG;
        }
    }

    return NIYAH_OK;
}


NiyahStatus niyah_tokenizer_identity_sha256(
    const NiyahTokenizer *tokenizer,
    uint8_t out_identity[NIYAH_TOKENIZER_IDENTITY_SHA256_SIZE])
{
    static const unsigned char domain[] = {
        'N','I','Y','A','H','-','T','O','K','E','N','I','Z','E','R','-','V','1'
    };
    NiyahTokenizerSha256 sha;
    unsigned char u32[4];
    size_t i;
    NiyahStatus status;

    if (tokenizer == NULL || out_identity == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    status = niyah_tok_validate(tokenizer);
    if (status != NIYAH_OK) {
        return status;
    }

    niyah_sha256_init(&sha);
    if (!niyah_sha256_update(&sha, domain, sizeof(domain))) {
        return NIYAH_ERR_OVERFLOW;
    }

    niyah_tok_store_u32_le(u32, NIYAH_TOKENIZER_BASE_VOCAB_SIZE);
    if (!niyah_sha256_update(&sha, u32, sizeof(u32))) return NIYAH_ERR_OVERFLOW;

    niyah_tok_store_u32_le(u32, (uint32_t)tokenizer->vocab_size);
    if (!niyah_sha256_update(&sha, u32, sizeof(u32))) return NIYAH_ERR_OVERFLOW;

    niyah_tok_store_u32_le(u32, (uint32_t)tokenizer->merge_count);
    if (!niyah_sha256_update(&sha, u32, sizeof(u32))) return NIYAH_ERR_OVERFLOW;

    for (i = 0U; i < tokenizer->merge_count; ++i) {
        const NiyahMerge *merge = &tokenizer->merges[i];

        niyah_tok_store_u32_le(u32, merge->left);
        if (!niyah_sha256_update(&sha, u32, sizeof(u32))) return NIYAH_ERR_OVERFLOW;

        niyah_tok_store_u32_le(u32, merge->right);
        if (!niyah_sha256_update(&sha, u32, sizeof(u32))) return NIYAH_ERR_OVERFLOW;

        niyah_tok_store_u32_le(u32, merge->output);
        if (!niyah_sha256_update(&sha, u32, sizeof(u32))) return NIYAH_ERR_OVERFLOW;
    }

    niyah_sha256_final(&sha, out_identity);
    return NIYAH_OK;
}

NiyahStatus niyah_tokenizer_save(const NiyahTokenizer *tokenizer,
                                 const char *path)
{
    FILE *file;
    NiyahTokenizerCrc32 crc;
    unsigned char footer[8];
    NiyahStatus status;
    size_t i;
    int close_result;

    if (path == NULL || path[0] == '\0') {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    status = niyah_tok_validate(tokenizer);
    if (status != NIYAH_OK) {
        return status;
    }

    file = niyah_tokenizer_fopen(path, "wb");
    if (file == NULL) {
        return NIYAH_ERR_IO;
    }

    niyah_tok_crc_init(&crc);

    status = niyah_tok_write(
        file,
        NIYAH_TOKENIZER_MAGIC,
        sizeof(NIYAH_TOKENIZER_MAGIC),
        &crc);

    if (status == NIYAH_OK)
        status = niyah_tok_write_u32(
            file, &crc, NIYAH_TOKENIZER_FILE_VERSION);

    if (status == NIYAH_OK)
        status = niyah_tok_write_u32(
            file, &crc, NIYAH_TOKENIZER_FILE_FLAGS);

    if (status == NIYAH_OK)
        status = niyah_tok_write_u32(
            file, &crc, NIYAH_TOKENIZER_BASE_VOCAB_SIZE);

    if (status == NIYAH_OK)
        status = niyah_tok_write_u32(
            file, &crc, (uint32_t)tokenizer->vocab_size);

    if (status == NIYAH_OK)
        status = niyah_tok_write_u32(
            file, &crc, (uint32_t)tokenizer->merge_count);

    if (status == NIYAH_OK)
        status = niyah_tok_write_u32(
            file, &crc, NIYAH_TOKENIZER_FILE_RESERVED);

    for (i = 0U;
         status == NIYAH_OK && i < tokenizer->merge_count;
         ++i) {
        const NiyahMerge *m = &tokenizer->merges[i];

        status = niyah_tok_write_u32(file, &crc, m->left);
        if (status == NIYAH_OK)
            status = niyah_tok_write_u32(file, &crc, m->right);
        if (status == NIYAH_OK)
            status = niyah_tok_write_u32(file, &crc, m->output);
    }

    if (status == NIYAH_OK) {
        niyah_tok_store_u32_le(
            footer + 0U,
            NIYAH_TOKENIZER_CHECKSUM_CRC32);
        niyah_tok_store_u32_le(
            footer + 4U,
            niyah_tok_crc_final(&crc));

        status = niyah_tok_write(
            file, footer, sizeof(footer), NULL);
    }

    if (status == NIYAH_OK && fflush(file) != 0) {
        status = NIYAH_ERR_IO;
    }

    close_result = fclose(file);
    if (status == NIYAH_OK && close_result != 0) {
        status = NIYAH_ERR_IO;
    }

    if (status != NIYAH_OK) {
        (void)remove(path);
    }

    return status;
}

NiyahStatus niyah_tokenizer_load(const char *path,
                                 NiyahTokenizer **out_tokenizer)
{
    FILE *file = NULL;
    NiyahTokenizer *tokenizer = NULL;
    NiyahTokenizerCrc32 crc;
    unsigned char header[32];
    unsigned char triple[12];
    unsigned char footer[8];
    unsigned char extra;
    uint32_t version;
    uint32_t flags;
    uint32_t base_vocab;
    uint32_t vocab_size;
    uint32_t merge_count;
    uint32_t reserved;
    NiyahStatus status;
    size_t i;
    int close_result;

    if (path == NULL ||
        path[0] == '\0' ||
        out_tokenizer == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    *out_tokenizer = NULL;

    file = niyah_tokenizer_fopen(path, "rb");
    if (file == NULL) {
        return NIYAH_ERR_IO;
    }

    niyah_tok_crc_init(&crc);

    status = niyah_tok_read(
        file, header, sizeof(header), &crc);

    if (status != NIYAH_OK) {
        goto done;
    }

    if (memcmp(header,
               NIYAH_TOKENIZER_MAGIC,
               sizeof(NIYAH_TOKENIZER_MAGIC)) != 0) {
        status = NIYAH_ERR_CORRUPT_DATA;
        goto done;
    }

    version = niyah_tok_load_u32_le(header + 8U);
    flags = niyah_tok_load_u32_le(header + 12U);
    base_vocab = niyah_tok_load_u32_le(header + 16U);
    vocab_size = niyah_tok_load_u32_le(header + 20U);
    merge_count = niyah_tok_load_u32_le(header + 24U);
    reserved = niyah_tok_load_u32_le(header + 28U);

    if (version != NIYAH_TOKENIZER_FILE_VERSION) {
        status = NIYAH_ERR_UNSUPPORTED_VERSION;
        goto done;
    }

    if (flags != NIYAH_TOKENIZER_FILE_FLAGS ||
        reserved != NIYAH_TOKENIZER_FILE_RESERVED ||
        base_vocab != NIYAH_TOKENIZER_BASE_VOCAB_SIZE ||
        vocab_size < NIYAH_TOKENIZER_BASE_VOCAB_SIZE ||
        merge_count !=
            vocab_size - NIYAH_TOKENIZER_BASE_VOCAB_SIZE) {
        status = NIYAH_ERR_CORRUPT_DATA;
        goto done;
    }

    status = niyah_tokenizer_alloc_base(
        vocab_size, &tokenizer);
    if (status != NIYAH_OK) {
        goto done;
    }

    for (i = 0U; i < (size_t)merge_count; ++i) {
        uint32_t left;
        uint32_t right;
        uint32_t output;
        uint32_t created;
        const uint32_t expected =
            NIYAH_TOKENIZER_BASE_VOCAB_SIZE + (uint32_t)i;

        status = niyah_tok_read(
            file, triple, sizeof(triple), &crc);
        if (status != NIYAH_OK) {
            goto done;
        }

        left = niyah_tok_load_u32_le(triple + 0U);
        right = niyah_tok_load_u32_le(triple + 4U);
        output = niyah_tok_load_u32_le(triple + 8U);

        if (output != expected ||
            left >= expected ||
            right >= expected ||
            left == NIYAH_TOKEN_BOS ||
            left == NIYAH_TOKEN_EOS ||
            right == NIYAH_TOKEN_BOS ||
            right == NIYAH_TOKEN_EOS) {
            status = NIYAH_ERR_CORRUPT_DATA;
            goto done;
        }

        status = niyah_tokenizer_add_merge(
            tokenizer, left, right, &created);

        if (status != NIYAH_OK) {
            goto done;
        }

        if (created != output) {
            status = NIYAH_ERR_CORRUPT_DATA;
            goto done;
        }
    }

    status = niyah_tok_read(
        file, footer, sizeof(footer), NULL);

    if (status != NIYAH_OK) {
        goto done;
    }

    if (niyah_tok_load_u32_le(footer + 0U) !=
            NIYAH_TOKENIZER_CHECKSUM_CRC32 ||
        niyah_tok_load_u32_le(footer + 4U) !=
            niyah_tok_crc_final(&crc)) {
        status = NIYAH_ERR_CORRUPT_DATA;
        goto done;
    }

    if (fread(&extra, 1U, 1U, file) != 0U) {
        status = NIYAH_ERR_CORRUPT_DATA;
        goto done;
    }

    if (ferror(file) != 0) {
        status = NIYAH_ERR_IO;
        goto done;
    }

done:
    close_result = fclose(file);

    if (status == NIYAH_OK && close_result != 0) {
        status = NIYAH_ERR_IO;
    }

    if (status != NIYAH_OK) {
        niyah_tokenizer_destroy(tokenizer);
        return status;
    }

    *out_tokenizer = tokenizer;
    return NIYAH_OK;
}

NiyahStatus niyah_tokenizer_encode(const NiyahTokenizer *tokenizer,
                                   const uint8_t *input,
                                   size_t input_size,
                                   uint32_t *output_tokens,
                                   size_t output_capacity,
                                   size_t *output_count)
{
    uint32_t *work = NULL;
    size_t work_bytes = 0U;
    size_t count;
    size_t i;

    if (tokenizer == NULL || output_count == NULL || (input == NULL && input_size != 0U)) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    if (output_tokens == NULL && output_capacity != 0U) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    if (input_size == 0U) {
        *output_count = 0U;
        return NIYAH_OK;
    }
    if (!niyah_size_mul_ok(input_size, sizeof(uint32_t), &work_bytes)) {
        return NIYAH_ERR_OVERFLOW;
    }
    work = (uint32_t *)malloc(work_bytes);
    if (work == NULL) {
        return NIYAH_ERR_OUT_OF_MEMORY;
    }
    for (i = 0U; i < input_size; ++i) {
        work[i] = (uint32_t)input[i];
    }
    count = input_size;

    for (i = 0U; i < tokenizer->merge_count; ++i) {
        const NiyahMerge *merge = &tokenizer->merges[i];
        niyah_apply_merge(work, &count, merge->left, merge->right, merge->output);
    }

    *output_count = count;
    if (output_tokens == NULL) {
        free(work);
        return output_capacity == 0U ? NIYAH_OK : NIYAH_ERR_INVALID_ARGUMENT;
    }
    if (output_capacity < count) {
        free(work);
        return NIYAH_ERR_BUFFER_TOO_SMALL;
    }
    memcpy(output_tokens, work, count * sizeof(uint32_t));
    free(work);
    return NIYAH_OK;
}

NiyahStatus niyah_tokenizer_decode(const NiyahTokenizer *tokenizer,
                                   const uint32_t *tokens,
                                   size_t token_count,
                                   uint8_t *output_bytes,
                                   size_t output_capacity,
                                   size_t *output_size)
{
    size_t required = 0U;
    size_t cursor = 0U;
    size_t i;

    if (tokenizer == NULL || output_size == NULL || (tokens == NULL && token_count != 0U)) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    if (output_bytes == NULL && output_capacity != 0U) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    for (i = 0U; i < token_count; ++i) {
        const uint32_t token = tokens[i];
        if (token == NIYAH_TOKEN_BOS || token == NIYAH_TOKEN_EOS) {
            continue;
        }
        if ((size_t)token >= tokenizer->vocab_size) {
            return NIYAH_ERR_INVALID_ARGUMENT;
        }
        if (!niyah_size_add_ok(required, tokenizer->vocab[token].length, &required)) {
            return NIYAH_ERR_OVERFLOW;
        }
    }

    *output_size = required;
    if (output_bytes == NULL) {
        return output_capacity == 0U ? NIYAH_OK : NIYAH_ERR_INVALID_ARGUMENT;
    }
    if (output_capacity < required) {
        return NIYAH_ERR_BUFFER_TOO_SMALL;
    }

    for (i = 0U; i < token_count; ++i) {
        const uint32_t token = tokens[i];
        const NiyahTokenBytes *token_bytes;
        if (token == NIYAH_TOKEN_BOS || token == NIYAH_TOKEN_EOS) {
            continue;
        }
        token_bytes = &tokenizer->vocab[token];
        memcpy(output_bytes + cursor, token_bytes->data, token_bytes->length);
        cursor += token_bytes->length;
    }
    return NIYAH_OK;
}
