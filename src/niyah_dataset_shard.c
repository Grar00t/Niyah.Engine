#include "niyah/dataset.h"
#include "niyah_sha256.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(CHAR_BIT == 8, "dataset shard v1 requires 8-bit bytes");

#define NIYAH_DATASET_SHARD_VERSION UINT32_C(1)
#define NIYAH_DATASET_SHARD_FLAGS UINT32_C(0)
#define NIYAH_DATASET_SHARD_CHECKSUM_CRC32 UINT32_C(1)
#define NIYAH_DATASET_SHARD_HEADER_SIZE 72U

static const unsigned char NIYAH_DATASET_SHARD_MAGIC[8] = {
    'N','I','Y','A','H','S','R','D'
};

typedef struct NiyahDatasetShardCrc32 {
    uint32_t value;
    uint32_t table[256];
} NiyahDatasetShardCrc32;

static FILE *shard_fopen(const char *path, const char *mode)
{
#if defined(_MSC_VER)
    FILE *file = NULL;
    if (fopen_s(&file, path, mode) != 0) return NULL;
    return file;
#else
    return fopen(path, mode);
#endif
}

static int size_mul_ok(size_t a, size_t b, size_t *out)
{
    if (out == NULL || (a != 0U && b > SIZE_MAX / a)) return 0;
    *out = a * b;
    return 1;
}

static int size_add_ok(size_t a, size_t b, size_t *out)
{
    if (out == NULL || a > SIZE_MAX - b) return 0;
    *out = a + b;
    return 1;
}

static size_t sample_count_for(size_t token_count, size_t sequence_length)
{
    if (token_count < 2U || sequence_length == 0U) return 0U;
    return 1U + (token_count - 2U) / sequence_length;
}

static void store_u32_le(unsigned char out[4], uint32_t value)
{
    out[0] = (unsigned char)(value & UINT32_C(0xff));
    out[1] = (unsigned char)((value >> 8U) & UINT32_C(0xff));
    out[2] = (unsigned char)((value >> 16U) & UINT32_C(0xff));
    out[3] = (unsigned char)((value >> 24U) & UINT32_C(0xff));
}

static void store_u64_le(unsigned char out[8], uint64_t value)
{
    size_t i;
    for (i = 0U; i < 8U; ++i) {
        out[i] = (unsigned char)(value & UINT64_C(0xff));
        value >>= 8U;
    }
}

static uint32_t load_u32_le(const unsigned char in[4])
{
    return (uint32_t)in[0] |
           ((uint32_t)in[1] << 8U) |
           ((uint32_t)in[2] << 16U) |
           ((uint32_t)in[3] << 24U);
}

static uint64_t load_u64_le(const unsigned char in[8])
{
    uint64_t value = UINT64_C(0);
    size_t i;
    for (i = 0U; i < 8U; ++i)
        value |= (uint64_t)in[i] << (8U * i);
    return value;
}

static void crc_init(NiyahDatasetShardCrc32 *crc)
{
    uint32_t i;
    for (i = 0U; i < UINT32_C(256); ++i) {
        uint32_t c = i;
        unsigned bit;
        for (bit = 0U; bit < 8U; ++bit)
            c = (c & UINT32_C(1))
                ? UINT32_C(0xedb88320) ^ (c >> 1U)
                : c >> 1U;
        crc->table[i] = c;
    }
    crc->value = UINT32_C(0xffffffff);
}

static void crc_update(NiyahDatasetShardCrc32 *crc,
                       const unsigned char *data,
                       size_t size)
{
    size_t i;
    for (i = 0U; i < size; ++i) {
        const uint32_t index =
            (crc->value ^ (uint32_t)data[i]) & UINT32_C(0xff);
        crc->value = crc->table[index] ^ (crc->value >> 8U);
    }
}

static uint32_t crc_final(const NiyahDatasetShardCrc32 *crc)
{
    return crc->value ^ UINT32_C(0xffffffff);
}

static NiyahStatus write_crc(FILE *file,
                             const void *data,
                             size_t size,
                             NiyahDatasetShardCrc32 *crc)
{
    if (size != 0U && fwrite(data, 1U, size, file) != size)
        return NIYAH_ERR_IO;
    if (size != 0U)
        crc_update(crc, (const unsigned char *)data, size);
    return NIYAH_OK;
}

static NiyahStatus read_crc(FILE *file,
                            void *data,
                            size_t size,
                            NiyahDatasetShardCrc32 *crc)
{
    const size_t got = fread(data, 1U, size, file);
    if (got != 0U)
        crc_update(crc, (const unsigned char *)data, got);
    if (got != size)
        return ferror(file) ? NIYAH_ERR_IO : NIYAH_ERR_CORRUPT_DATA;
    return NIYAH_OK;
}

static NiyahStatus tokenizer_identity(
    const NiyahTokenizer *tokenizer,
    uint8_t out[NIYAH_DATASET_TOKENIZER_IDENTITY_SIZE])
{
    if (NIYAH_DATASET_TOKENIZER_IDENTITY_SIZE !=
        NIYAH_TOKENIZER_IDENTITY_SHA256_SIZE)
        return NIYAH_ERR_INVALID_CONFIG;
    return niyah_tokenizer_identity_sha256(tokenizer, out);
}

static NiyahStatus validate_shard(const NiyahDatasetShard *shard,
                                  const NiyahTokenizer *tokenizer)
{
    uint8_t identity[NIYAH_DATASET_TOKENIZER_IDENTITY_SIZE];
    size_t expected_samples;
    size_t vocab_size;
    size_t i;
    NiyahStatus status;

    if (shard == NULL || tokenizer == NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;
    if (shard->tokens == NULL ||
        shard->token_count < 2U ||
        shard->sequence_length == 0U)
        return NIYAH_ERR_INVALID_CONFIG;

    expected_samples =
        sample_count_for(shard->token_count, shard->sequence_length);
    if (expected_samples == 0U ||
        shard->sample_count != expected_samples ||
        shard->tokens[0] != NIYAH_TOKEN_BOS ||
        shard->tokens[shard->token_count - 1U] != NIYAH_TOKEN_EOS)
        return NIYAH_ERR_INVALID_CONFIG;

    status = tokenizer_identity(tokenizer, identity);
    if (status != NIYAH_OK) return status;
    if (memcmp(identity, shard->tokenizer_identity, sizeof(identity)) != 0)
        return NIYAH_ERR_INVALID_CONFIG;

    vocab_size = niyah_tokenizer_vocab_size(tokenizer);
    if (vocab_size == 0U) return NIYAH_ERR_INVALID_CONFIG;
    for (i = 0U; i < shard->token_count; ++i) {
        if ((size_t)shard->tokens[i] >= vocab_size)
            return NIYAH_ERR_INVALID_CONFIG;
    }
    return NIYAH_OK;
}

NiyahStatus niyah_dataset_shard_build_text(
    const NiyahTokenizer *tokenizer,
    const uint8_t *text,
    size_t text_size,
    size_t sequence_length,
    NiyahDatasetShard *out_shard)
{
    NiyahDatasetShard temp;
    size_t encoded_count = 0U;
    size_t token_count = 0U;
    size_t token_bytes = 0U;
    NiyahStatus status;

    if (tokenizer == NULL || text == NULL || text_size == 0U ||
        out_shard == NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;
    if (out_shard->tokens != NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;
    if (sequence_length == 0U)
        return NIYAH_ERR_INVALID_CONFIG;

    memset(&temp, 0, sizeof(temp));

    status = niyah_tokenizer_encode(
        tokenizer, text, text_size, NULL, 0U, &encoded_count);
    if (status != NIYAH_OK) return status;

    if (!size_add_ok(encoded_count, 2U, &token_count) ||
        !size_mul_ok(token_count, sizeof(uint32_t), &token_bytes))
        return NIYAH_ERR_OVERFLOW;

    temp.tokens = (uint32_t *)malloc(token_bytes);
    if (temp.tokens == NULL) return NIYAH_ERR_OUT_OF_MEMORY;

    temp.tokens[0] = NIYAH_TOKEN_BOS;
    status = niyah_tokenizer_encode(
        tokenizer, text, text_size,
        temp.tokens + 1U, encoded_count, &encoded_count);
    if (status != NIYAH_OK) {
        niyah_dataset_shard_destroy(&temp);
        return status;
    }
    temp.tokens[token_count - 1U] = NIYAH_TOKEN_EOS;

    temp.token_count = token_count;
    temp.sequence_length = sequence_length;
    temp.sample_count = sample_count_for(token_count, sequence_length);

    status = tokenizer_identity(tokenizer, temp.tokenizer_identity);
    if (status != NIYAH_OK) {
        niyah_dataset_shard_destroy(&temp);
        return status;
    }

    status = validate_shard(&temp, tokenizer);
    if (status != NIYAH_OK) {
        niyah_dataset_shard_destroy(&temp);
        return status;
    }

    *out_shard = temp;
    return NIYAH_OK;
}

void niyah_dataset_shard_destroy(NiyahDatasetShard *shard)
{
    if (shard == NULL) return;
    free(shard->tokens);
    memset(shard, 0, sizeof(*shard));
}

NiyahStatus niyah_dataset_shard_sample(
    const NiyahDatasetShard *shard,
    size_t sample_index,
    const uint32_t **out_tokens,
    const uint32_t **out_targets,
    size_t *out_token_count)
{
    size_t start;
    size_t remaining;
    size_t count;

    if (shard == NULL || out_tokens == NULL ||
        out_targets == NULL || out_token_count == NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;
    if (shard->tokens == NULL || shard->token_count < 2U ||
        shard->sequence_length == 0U ||
        shard->sample_count !=
            sample_count_for(shard->token_count, shard->sequence_length))
        return NIYAH_ERR_INVALID_CONFIG;
    if (sample_index >= shard->sample_count)
        return NIYAH_ERR_INVALID_ARGUMENT;
    if (sample_index > SIZE_MAX / shard->sequence_length)
        return NIYAH_ERR_OVERFLOW;

    start = sample_index * shard->sequence_length;
    if (start >= shard->token_count - 1U)
        return NIYAH_ERR_INVALID_CONFIG;

    remaining = shard->token_count - 1U - start;
    count = remaining < shard->sequence_length
        ? remaining
        : shard->sequence_length;

    *out_tokens = shard->tokens + start;
    *out_targets = shard->tokens + start + 1U;
    *out_token_count = count;
    return NIYAH_OK;
}


NiyahStatus niyah_dataset_shard_identity_sha256(
    const NiyahDatasetShard *shard,
    uint8_t out_identity[NIYAH_DATASET_SHARD_IDENTITY_SHA256_SIZE])
{
    static const unsigned char domain[] = {
        'N','I','Y','A','H','-','D','A','T','A','S','E','T','-',
        'S','H','A','R','D','-','V','1'
    };
    NiyahSha256 sha;
    unsigned char u64[8];
    unsigned char u32[4];
    size_t expected_samples;
    size_t i;

    if (shard == NULL || out_identity == NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;
    if (shard->tokens == NULL ||
        shard->token_count < 2U ||
        shard->sequence_length == 0U)
        return NIYAH_ERR_INVALID_CONFIG;

    expected_samples =
        sample_count_for(shard->token_count, shard->sequence_length);
    if (expected_samples == 0U ||
        shard->sample_count != expected_samples ||
        shard->tokens[0] != NIYAH_TOKEN_BOS ||
        shard->tokens[shard->token_count - 1U] != NIYAH_TOKEN_EOS)
        return NIYAH_ERR_INVALID_CONFIG;
    if (shard->sequence_length > (size_t)UINT64_MAX ||
        shard->token_count > (size_t)UINT64_MAX ||
        shard->sample_count > (size_t)UINT64_MAX)
        return NIYAH_ERR_OVERFLOW;

    niyah_sha256_init(&sha);
    if (!niyah_sha256_update(&sha, domain, sizeof(domain)) ||
        !niyah_sha256_update(&sha, shard->tokenizer_identity,
                             sizeof(shard->tokenizer_identity)))
        return NIYAH_ERR_OVERFLOW;

    store_u64_le(u64, (uint64_t)shard->sequence_length);
    if (!niyah_sha256_update(&sha, u64, sizeof(u64)))
        return NIYAH_ERR_OVERFLOW;
    store_u64_le(u64, (uint64_t)shard->token_count);
    if (!niyah_sha256_update(&sha, u64, sizeof(u64)))
        return NIYAH_ERR_OVERFLOW;
    store_u64_le(u64, (uint64_t)shard->sample_count);
    if (!niyah_sha256_update(&sha, u64, sizeof(u64)))
        return NIYAH_ERR_OVERFLOW;

    for (i = 0U; i < shard->token_count; ++i) {
        store_u32_le(u32, shard->tokens[i]);
        if (!niyah_sha256_update(&sha, u32, sizeof(u32)))
            return NIYAH_ERR_OVERFLOW;
    }

    niyah_sha256_final(&sha, out_identity);
    return NIYAH_OK;
}

NiyahStatus niyah_dataset_collection_identity_sha256(
    const NiyahDatasetShard *shards,
    size_t shard_count,
    uint8_t out_identity[NIYAH_DATASET_COLLECTION_IDENTITY_SHA256_SIZE])
{
    static const unsigned char domain[] = "NIYAH-DATASET-COLLECTION-V1";
    NiyahSha256 sha;
    unsigned char count_bytes[8];
    uint8_t shard_identity[NIYAH_DATASET_SHARD_IDENTITY_SHA256_SIZE];
    size_t i;
    NiyahStatus status;

    if (shards == NULL || out_identity == NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;
    if (shard_count == 0U)
        return NIYAH_ERR_INVALID_CONFIG;
    if (shard_count > (size_t)UINT64_MAX)
        return NIYAH_ERR_OVERFLOW;
    if (NIYAH_DATASET_COLLECTION_IDENTITY_SHA256_SIZE !=
        NIYAH_DATASET_SHARD_IDENTITY_SHA256_SIZE)
        return NIYAH_ERR_INVALID_CONFIG;

    niyah_sha256_init(&sha);
    niyah_sha256_update(&sha, domain, sizeof(domain) - 1U);
    store_u64_le(count_bytes, (uint64_t)shard_count);
    niyah_sha256_update(&sha, count_bytes, sizeof(count_bytes));

    for (i = 0U; i < shard_count; ++i) {
        if (i != 0U &&
            memcmp(shards[0U].tokenizer_identity,
                   shards[i].tokenizer_identity,
                   NIYAH_DATASET_TOKENIZER_IDENTITY_SIZE) != 0)
            return NIYAH_ERR_INVALID_CONFIG;

        status = niyah_dataset_shard_identity_sha256(
            &shards[i], shard_identity);
        if (status != NIYAH_OK)
            return status;
        niyah_sha256_update(&sha, shard_identity, sizeof(shard_identity));
    }

    niyah_sha256_final(&sha, out_identity);
    return NIYAH_OK;
}

NiyahStatus niyah_dataset_shard_save(
    const NiyahDatasetShard *shard,
    const NiyahTokenizer *tokenizer,
    const char *path)
{
    unsigned char header[NIYAH_DATASET_SHARD_HEADER_SIZE];
    unsigned char token_bytes[4];
    unsigned char footer[8];
    NiyahDatasetShardCrc32 crc;
    FILE *file;
    size_t i;
    NiyahStatus status;
    int close_result;

    if (path == NULL || path[0] == '\0')
        return NIYAH_ERR_INVALID_ARGUMENT;

    status = validate_shard(shard, tokenizer);
    if (status != NIYAH_OK) return status;
    if (shard->token_count > (size_t)UINT64_MAX ||
        shard->sequence_length > (size_t)UINT64_MAX ||
        shard->sample_count > (size_t)UINT64_MAX)
        return NIYAH_ERR_OVERFLOW;

    memset(header, 0, sizeof(header));
    memcpy(header, NIYAH_DATASET_SHARD_MAGIC, 8U);
    store_u32_le(header + 8U, NIYAH_DATASET_SHARD_VERSION);
    store_u32_le(header + 12U, NIYAH_DATASET_SHARD_FLAGS);
    store_u64_le(header + 16U, (uint64_t)shard->sequence_length);
    store_u64_le(header + 24U, (uint64_t)shard->token_count);
    store_u64_le(header + 32U, (uint64_t)shard->sample_count);
    memcpy(header + 40U, shard->tokenizer_identity,
           NIYAH_DATASET_TOKENIZER_IDENTITY_SIZE);

    file = shard_fopen(path, "wb");
    if (file == NULL) return NIYAH_ERR_IO;

    crc_init(&crc);
    status = write_crc(file, header, sizeof(header), &crc);

    for (i = 0U; status == NIYAH_OK && i < shard->token_count; ++i) {
        store_u32_le(token_bytes, shard->tokens[i]);
        status = write_crc(file, token_bytes, sizeof(token_bytes), &crc);
    }

    if (status == NIYAH_OK) {
        store_u32_le(footer + 0U, NIYAH_DATASET_SHARD_CHECKSUM_CRC32);
        store_u32_le(footer + 4U, crc_final(&crc));
        if (fwrite(footer, 1U, sizeof(footer), file) != sizeof(footer))
            status = NIYAH_ERR_IO;
    }
    if (status == NIYAH_OK && fflush(file) != 0)
        status = NIYAH_ERR_IO;

    close_result = fclose(file);
    if (status == NIYAH_OK && close_result != 0)
        status = NIYAH_ERR_IO;
    if (status != NIYAH_OK)
        (void)remove(path);
    return status;
}

NiyahStatus niyah_dataset_shard_load(
    const char *path,
    const NiyahTokenizer *tokenizer,
    NiyahDatasetShard *out_shard)
{
    unsigned char header[NIYAH_DATASET_SHARD_HEADER_SIZE];
    unsigned char token_bytes[4];
    unsigned char footer[8];
    unsigned char extra;
    uint8_t identity[NIYAH_DATASET_TOKENIZER_IDENTITY_SIZE];
    NiyahDatasetShardCrc32 crc;
    NiyahDatasetShard temp;
    FILE *file;
    uint32_t version;
    uint32_t flags;
    uint64_t sequence64;
    uint64_t token_count64;
    uint64_t sample_count64;
    size_t token_bytes_size;
    size_t expected_samples;
    size_t vocab_size;
    size_t i;
    NiyahStatus status;
    int close_result;

    if (path == NULL || path[0] == '\0' ||
        tokenizer == NULL || out_shard == NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;
    if (out_shard->tokens != NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;

    memset(&temp, 0, sizeof(temp));
    file = shard_fopen(path, "rb");
    if (file == NULL) return NIYAH_ERR_IO;

    crc_init(&crc);
    status = read_crc(file, header, sizeof(header), &crc);
    if (status != NIYAH_OK) goto done;

    if (memcmp(header, NIYAH_DATASET_SHARD_MAGIC, 8U) != 0) {
        status = NIYAH_ERR_CORRUPT_DATA;
        goto done;
    }

    version = load_u32_le(header + 8U);
    flags = load_u32_le(header + 12U);
    sequence64 = load_u64_le(header + 16U);
    token_count64 = load_u64_le(header + 24U);
    sample_count64 = load_u64_le(header + 32U);

    if (version != NIYAH_DATASET_SHARD_VERSION) {
        status = NIYAH_ERR_UNSUPPORTED_VERSION;
        goto done;
    }
    if (flags != NIYAH_DATASET_SHARD_FLAGS ||
        sequence64 == UINT64_C(0) ||
        token_count64 < UINT64_C(2) ||
        sample_count64 == UINT64_C(0) ||
        sequence64 > (uint64_t)SIZE_MAX ||
        token_count64 > (uint64_t)SIZE_MAX ||
        sample_count64 > (uint64_t)SIZE_MAX) {
        status = NIYAH_ERR_CORRUPT_DATA;
        goto done;
    }

    temp.sequence_length = (size_t)sequence64;
    temp.token_count = (size_t)token_count64;
    temp.sample_count = (size_t)sample_count64;
    expected_samples =
        sample_count_for(temp.token_count, temp.sequence_length);
    if (temp.sample_count != expected_samples) {
        status = NIYAH_ERR_CORRUPT_DATA;
        goto done;
    }

    status = tokenizer_identity(tokenizer, identity);
    if (status != NIYAH_OK) goto done;
    if (memcmp(identity, header + 40U, sizeof(identity)) != 0) {
        status = NIYAH_ERR_INVALID_CONFIG;
        goto done;
    }
    memcpy(temp.tokenizer_identity, header + 40U, sizeof(identity));

    if (!size_mul_ok(temp.token_count, sizeof(uint32_t), &token_bytes_size)) {
        status = NIYAH_ERR_OVERFLOW;
        goto done;
    }
    temp.tokens = (uint32_t *)malloc(token_bytes_size);
    if (temp.tokens == NULL) {
        status = NIYAH_ERR_OUT_OF_MEMORY;
        goto done;
    }

    vocab_size = niyah_tokenizer_vocab_size(tokenizer);
    if (vocab_size == 0U) {
        status = NIYAH_ERR_INVALID_CONFIG;
        goto done;
    }

    for (i = 0U; i < temp.token_count; ++i) {
        status = read_crc(file, token_bytes, sizeof(token_bytes), &crc);
        if (status != NIYAH_OK) goto done;
        temp.tokens[i] = load_u32_le(token_bytes);
        if ((size_t)temp.tokens[i] >= vocab_size) {
            status = NIYAH_ERR_CORRUPT_DATA;
            goto done;
        }
    }

    if (fread(footer, 1U, sizeof(footer), file) != sizeof(footer)) {
        status = ferror(file) ? NIYAH_ERR_IO : NIYAH_ERR_CORRUPT_DATA;
        goto done;
    }
    if (load_u32_le(footer + 0U) !=
            NIYAH_DATASET_SHARD_CHECKSUM_CRC32 ||
        load_u32_le(footer + 4U) != crc_final(&crc)) {
        status = NIYAH_ERR_CORRUPT_DATA;
        goto done;
    }

    if (fread(&extra, 1U, 1U, file) != 0U) {
        status = NIYAH_ERR_CORRUPT_DATA;
        goto done;
    }
    if (ferror(file)) {
        status = NIYAH_ERR_IO;
        goto done;
    }

    status = validate_shard(&temp, tokenizer);

done:
    close_result = fclose(file);
    if (status == NIYAH_OK && close_result != 0)
        status = NIYAH_ERR_IO;

    if (status != NIYAH_OK) {
        niyah_dataset_shard_destroy(&temp);
        return status;
    }

    *out_shard = temp;
    return NIYAH_OK;
}
