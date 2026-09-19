#include "niyah/dataset.h"
#include "niyah_sha256.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(CHAR_BIT == 8, "dataset shard requires 8-bit bytes");
_Static_assert(NIYAH_DATASET_TOKENIZER_IDENTITY_SIZE ==
                   NIYAH_TOKENIZER_IDENTITY_SHA256_SIZE,
               "dataset tokenizer identity size");
_Static_assert(NIYAH_DATASET_COLLECTION_IDENTITY_SHA256_SIZE ==
                   NIYAH_DATASET_SHARD_IDENTITY_SHA256_SIZE,
               "collection/shard identity size");

#define NIYAH_DATASET_SHARD_VERSION_V1 UINT32_C(1)
#define NIYAH_DATASET_SHARD_VERSION_V2 UINT32_C(2)
#define NIYAH_DATASET_SHARD_VERSION_V3 UINT32_C(3)
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


static int explicit_geometry_valid(const NiyahDatasetShard *shard)
{
    size_t previous_end = 0U;
    size_t i;

    if (shard == NULL ||
        shard->tokens == NULL ||
        shard->token_count < 2U ||
        shard->sequence_length == 0U ||
        shard->sample_count == 0U ||
        shard->sample_offsets == NULL ||
        shard->sample_lengths == NULL ||
        shard->has_explicit_samples == 0)
        return 0;

    if (shard->has_loss_starts != 0) {
        if (shard->sample_loss_starts == NULL)
            return 0;
    } else if (shard->sample_loss_starts != NULL) {
        return 0;
    }

    for (i = 0U; i < shard->sample_count; ++i) {
        const size_t start = shard->sample_offsets[i];
        const size_t count = shard->sample_lengths[i];
        size_t end;

        if (count == 0U ||
            count > shard->sequence_length ||
            start >= shard->token_count - 1U ||
            count > shard->token_count - 1U - start)
            return 0;

        if (shard->has_loss_starts != 0 &&
            shard->sample_loss_starts[i] >= count)
            return 0;

        end = start + count;

        if (i == 0U) {
            if (start != 0U)
                return 0;
        } else if (start < previous_end) {
            return 0;
        }

        previous_end = end;
    }

    return previous_end == shard->token_count - 1U;
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
    return niyah_tokenizer_identity_sha256(tokenizer, out);
}

static NiyahStatus validate_shard(
    const NiyahDatasetShard *shard,
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
        shard->sequence_length == 0U ||
        shard->sample_count == 0U ||
        shard->tokens[0] != NIYAH_TOKEN_BOS ||
        shard->tokens[shard->token_count - 1U] != NIYAH_TOKEN_EOS)
        return NIYAH_ERR_INVALID_CONFIG;

    if (shard->has_explicit_samples != 0) {
        if (!explicit_geometry_valid(shard))
            return NIYAH_ERR_INVALID_CONFIG;
    } else {
        if (shard->sample_offsets != NULL ||
            shard->sample_lengths != NULL ||
            shard->sample_loss_starts != NULL ||
            shard->has_loss_starts != 0)
            return NIYAH_ERR_INVALID_CONFIG;

        expected_samples =
            sample_count_for(
                shard->token_count,
                shard->sequence_length);

        if (expected_samples == 0U ||
            shard->sample_count != expected_samples)
            return NIYAH_ERR_INVALID_CONFIG;
    }

    status = tokenizer_identity(tokenizer, identity);
    if (status != NIYAH_OK)
        return status;

    if (memcmp(
            identity,
            shard->tokenizer_identity,
            sizeof(identity)) != 0)
        return NIYAH_ERR_INVALID_CONFIG;

    vocab_size = niyah_tokenizer_vocab_size(tokenizer);
    if (vocab_size == 0U)
        return NIYAH_ERR_INVALID_CONFIG;

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

NiyahStatus niyah_dataset_shard_build_records(
    const NiyahTokenizer *tokenizer,
    const uint8_t *text,
    size_t text_size,
    const size_t *record_offsets,
    const size_t *record_lengths,
    size_t record_count,
    size_t sequence_length,
    NiyahDatasetShard *out_shard)
{
    NiyahDatasetShard temp;
    size_t total_tokens = 0U;
    size_t total_samples = 0U;
    size_t token_bytes = 0U;
    size_t geometry_bytes = 0U;
    size_t record_index;
    size_t write_index = 0U;
    size_t sample_index = 0U;
    size_t previous_record_end = 0U;
    NiyahStatus status;

    if (tokenizer == NULL || text == NULL || text_size == 0U ||
        record_offsets == NULL || record_lengths == NULL ||
        record_count == 0U || out_shard == NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;
    if (out_shard->tokens != NULL ||
        out_shard->sample_offsets != NULL ||
        out_shard->sample_lengths != NULL ||
        out_shard->sample_loss_starts != NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;
    if (sequence_length == 0U)
        return NIYAH_ERR_INVALID_CONFIG;

    memset(&temp, 0, sizeof(temp));

    for (record_index = 0U;
         record_index < record_count;
         ++record_index) {
        const size_t offset = record_offsets[record_index];
        const size_t length = record_lengths[record_index];
        size_t encoded_count = 0U;
        size_t record_tokens;
        size_t transitions;
        size_t record_samples;
        size_t record_end;

        if (length == 0U ||
            offset > text_size ||
            length > text_size - offset ||
            !size_add_ok(offset, length, &record_end))
            return NIYAH_ERR_INVALID_ARGUMENT;

        if (record_index != 0U && offset < previous_record_end)
            return NIYAH_ERR_INVALID_ARGUMENT;
        previous_record_end = record_end;

        status = niyah_tokenizer_encode(
            tokenizer,
            text + offset,
            length,
            NULL,
            0U,
            &encoded_count);
        if (status != NIYAH_OK)
            return status;

        if (!size_add_ok(encoded_count, 2U, &record_tokens) ||
            !size_add_ok(encoded_count, 1U, &transitions))
            return NIYAH_ERR_OVERFLOW;

        record_samples =
            1U + (transitions - 1U) / sequence_length;

        if (!size_add_ok(total_tokens, record_tokens, &total_tokens) ||
            !size_add_ok(total_samples, record_samples, &total_samples))
            return NIYAH_ERR_OVERFLOW;
    }

    if (!size_mul_ok(total_tokens, sizeof(uint32_t), &token_bytes) ||
        !size_mul_ok(total_samples, sizeof(size_t), &geometry_bytes))
        return NIYAH_ERR_OVERFLOW;

    temp.tokens = (uint32_t *)malloc(token_bytes);
    temp.sample_offsets = (size_t *)malloc(geometry_bytes);
    temp.sample_lengths = (size_t *)malloc(geometry_bytes);

    if (temp.tokens == NULL ||
        temp.sample_offsets == NULL ||
        temp.sample_lengths == NULL) {
        niyah_dataset_shard_destroy(&temp);
        return NIYAH_ERR_OUT_OF_MEMORY;
    }

    temp.token_count = total_tokens;
    temp.sequence_length = sequence_length;
    temp.sample_count = total_samples;
    temp.has_explicit_samples = 1;

    for (record_index = 0U;
         record_index < record_count;
         ++record_index) {
        const size_t offset = record_offsets[record_index];
        const size_t length = record_lengths[record_index];
        const size_t record_base = write_index;
        size_t encoded_count = 0U;
        size_t remaining;
        size_t start;

        status = niyah_tokenizer_encode(
            tokenizer,
            text + offset,
            length,
            NULL,
            0U,
            &encoded_count);
        if (status != NIYAH_OK) {
            niyah_dataset_shard_destroy(&temp);
            return status;
        }

        temp.tokens[write_index++] = NIYAH_TOKEN_BOS;

        if (encoded_count != 0U) {
            size_t written = 0U;
            status = niyah_tokenizer_encode(
                tokenizer,
                text + offset,
                length,
                temp.tokens + write_index,
                encoded_count,
                &written);
            if (status != NIYAH_OK ||
                written != encoded_count) {
                niyah_dataset_shard_destroy(&temp);
                return status == NIYAH_OK
                    ? NIYAH_ERR_INVALID_CONFIG
                    : status;
            }
            write_index += encoded_count;
        }

        temp.tokens[write_index++] = NIYAH_TOKEN_EOS;

        remaining = encoded_count + 1U;
        start = record_base;

        while (remaining != 0U) {
            const size_t count =
                remaining < sequence_length
                    ? remaining
                    : sequence_length;

            if (sample_index >= total_samples) {
                niyah_dataset_shard_destroy(&temp);
                return NIYAH_ERR_INVALID_CONFIG;
            }

            temp.sample_offsets[sample_index] = start;
            temp.sample_lengths[sample_index] = count;
            sample_index += 1U;

            start += count;
            remaining -= count;
        }
    }

    if (write_index != total_tokens ||
        sample_index != total_samples) {
        niyah_dataset_shard_destroy(&temp);
        return NIYAH_ERR_INVALID_CONFIG;
    }

    status = tokenizer_identity(
        tokenizer,
        temp.tokenizer_identity);
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


NiyahStatus niyah_dataset_shard_build_supervised_records(
    const NiyahTokenizer *tokenizer,
    const uint8_t *text,
    size_t text_size,
    const NiyahDatasetSupervisedRecord *records,
    size_t record_count,
    size_t sequence_length,
    NiyahDatasetShard *out_shard)
{
    NiyahDatasetShard temp;
    size_t total_tokens = 0U;
    size_t token_bytes = 0U;
    size_t geometry_bytes = 0U;
    size_t record_index;
    size_t write_index = 0U;
    NiyahStatus status;

    if (tokenizer == NULL ||
        text == NULL ||
        text_size == 0U ||
        records == NULL ||
        record_count == 0U ||
        out_shard == NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;

    if (out_shard->tokens != NULL ||
        out_shard->sample_offsets != NULL ||
        out_shard->sample_lengths != NULL ||
        out_shard->sample_loss_starts != NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;

    if (sequence_length == 0U)
        return NIYAH_ERR_INVALID_CONFIG;

    memset(&temp, 0, sizeof(temp));

    for (record_index = 0U;
         record_index < record_count;
         ++record_index) {
        const NiyahDatasetSupervisedRecord *record =
            &records[record_index];
        size_t prompt_tokens = 0U;
        size_t response_tokens = 0U;
        size_t transitions = 0U;
        size_t record_tokens = 0U;

        if (record->prompt_length == 0U ||
            record->response_length == 0U ||
            record->prompt_offset > text_size ||
            record->prompt_length >
                text_size - record->prompt_offset ||
            record->response_offset > text_size ||
            record->response_length >
                text_size - record->response_offset)
            return NIYAH_ERR_INVALID_ARGUMENT;

        status = niyah_tokenizer_encode(
            tokenizer,
            text + record->prompt_offset,
            record->prompt_length,
            NULL,
            0U,
            &prompt_tokens);
        if (status != NIYAH_OK)
            return status;

        status = niyah_tokenizer_encode(
            tokenizer,
            text + record->response_offset,
            record->response_length,
            NULL,
            0U,
            &response_tokens);
        if (status != NIYAH_OK)
            return status;

        if (prompt_tokens == 0U ||
            response_tokens == 0U)
            return NIYAH_ERR_INVALID_CONFIG;

        if (!size_add_ok(
                prompt_tokens,
                response_tokens,
                &transitions) ||
            !size_add_ok(
                transitions,
                1U,
                &transitions))
            return NIYAH_ERR_OVERFLOW;

        if (transitions > sequence_length)
            return NIYAH_ERR_INVALID_CONFIG;

        if (!size_add_ok(
                transitions,
                1U,
                &record_tokens) ||
            !size_add_ok(
                total_tokens,
                record_tokens,
                &total_tokens))
            return NIYAH_ERR_OVERFLOW;
    }

    if (!size_mul_ok(
            total_tokens,
            sizeof(uint32_t),
            &token_bytes) ||
        !size_mul_ok(
            record_count,
            sizeof(size_t),
            &geometry_bytes))
        return NIYAH_ERR_OVERFLOW;

    temp.tokens =
        (uint32_t *)malloc(token_bytes);
    temp.sample_offsets =
        (size_t *)malloc(geometry_bytes);
    temp.sample_lengths =
        (size_t *)malloc(geometry_bytes);
    temp.sample_loss_starts =
        (size_t *)malloc(geometry_bytes);

    if (temp.tokens == NULL ||
        temp.sample_offsets == NULL ||
        temp.sample_lengths == NULL ||
        temp.sample_loss_starts == NULL) {
        niyah_dataset_shard_destroy(&temp);
        return NIYAH_ERR_OUT_OF_MEMORY;
    }

    temp.token_count = total_tokens;
    temp.sequence_length = sequence_length;
    temp.sample_count = record_count;
    temp.has_explicit_samples = 1;
    temp.has_loss_starts = 1;

    for (record_index = 0U;
         record_index < record_count;
         ++record_index) {
        const NiyahDatasetSupervisedRecord *record =
            &records[record_index];
        const size_t record_base = write_index;
        size_t prompt_tokens = 0U;
        size_t response_tokens = 0U;
        size_t written = 0U;

        status = niyah_tokenizer_encode(
            tokenizer,
            text + record->prompt_offset,
            record->prompt_length,
            NULL,
            0U,
            &prompt_tokens);
        if (status != NIYAH_OK) {
            niyah_dataset_shard_destroy(&temp);
            return status;
        }

        status = niyah_tokenizer_encode(
            tokenizer,
            text + record->response_offset,
            record->response_length,
            NULL,
            0U,
            &response_tokens);
        if (status != NIYAH_OK) {
            niyah_dataset_shard_destroy(&temp);
            return status;
        }

        temp.tokens[write_index++] =
            NIYAH_TOKEN_BOS;

        status = niyah_tokenizer_encode(
            tokenizer,
            text + record->prompt_offset,
            record->prompt_length,
            temp.tokens + write_index,
            prompt_tokens,
            &written);
        if (status != NIYAH_OK ||
            written != prompt_tokens) {
            niyah_dataset_shard_destroy(&temp);
            return status == NIYAH_OK
                ? NIYAH_ERR_INVALID_CONFIG
                : status;
        }
        write_index += prompt_tokens;

        status = niyah_tokenizer_encode(
            tokenizer,
            text + record->response_offset,
            record->response_length,
            temp.tokens + write_index,
            response_tokens,
            &written);
        if (status != NIYAH_OK ||
            written != response_tokens) {
            niyah_dataset_shard_destroy(&temp);
            return status == NIYAH_OK
                ? NIYAH_ERR_INVALID_CONFIG
                : status;
        }
        write_index += response_tokens;

        temp.tokens[write_index++] =
            NIYAH_TOKEN_EOS;

        temp.sample_offsets[record_index] =
            record_base;
        temp.sample_lengths[record_index] =
            prompt_tokens + response_tokens + 1U;
        temp.sample_loss_starts[record_index] =
            prompt_tokens;
    }

    if (write_index != total_tokens) {
        niyah_dataset_shard_destroy(&temp);
        return NIYAH_ERR_INVALID_CONFIG;
    }

    status = tokenizer_identity(
        tokenizer,
        temp.tokenizer_identity);
    if (status != NIYAH_OK) {
        niyah_dataset_shard_destroy(&temp);
        return status;
    }

    status = validate_shard(
        &temp,
        tokenizer);
    if (status != NIYAH_OK) {
        niyah_dataset_shard_destroy(&temp);
        return status;
    }

    *out_shard = temp;
    return NIYAH_OK;
}

void niyah_dataset_shard_destroy(NiyahDatasetShard *shard)
{
    if (shard == NULL)
        return;

    free(shard->sample_loss_starts);
    free(shard->sample_lengths);
    free(shard->sample_offsets);
    free(shard->tokens);
    memset(shard, 0, sizeof(*shard));
}

NiyahStatus niyah_dataset_shard_sample_with_loss(
    const NiyahDatasetShard *shard,
    size_t sample_index,
    const uint32_t **out_tokens,
    const uint32_t **out_targets,
    size_t *out_token_count,
    size_t *out_loss_start)
{
    size_t start;
    size_t remaining;
    size_t count;
    size_t loss_start = 0U;

    if (shard == NULL ||
        out_tokens == NULL ||
        out_targets == NULL ||
        out_token_count == NULL ||
        out_loss_start == NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;

    if (shard->tokens == NULL ||
        shard->token_count < 2U ||
        shard->sequence_length == 0U ||
        shard->sample_count == 0U)
        return NIYAH_ERR_INVALID_CONFIG;

    if (sample_index >= shard->sample_count)
        return NIYAH_ERR_INVALID_ARGUMENT;

    if (shard->has_explicit_samples != 0) {
        if (!explicit_geometry_valid(shard))
            return NIYAH_ERR_INVALID_CONFIG;

        start =
            shard->sample_offsets[sample_index];
        count =
            shard->sample_lengths[sample_index];

        if (shard->has_loss_starts != 0)
            loss_start =
                shard->sample_loss_starts[sample_index];
    } else {
        if (shard->sample_offsets != NULL ||
            shard->sample_lengths != NULL ||
            shard->sample_loss_starts != NULL ||
            shard->has_loss_starts != 0 ||
            shard->sample_count !=
                sample_count_for(
                    shard->token_count,
                    shard->sequence_length))
            return NIYAH_ERR_INVALID_CONFIG;

        if (sample_index >
            SIZE_MAX / shard->sequence_length)
            return NIYAH_ERR_OVERFLOW;

        start =
            sample_index *
            shard->sequence_length;

        if (start >= shard->token_count - 1U)
            return NIYAH_ERR_INVALID_CONFIG;

        remaining =
            shard->token_count - 1U - start;

        count =
            remaining < shard->sequence_length
                ? remaining
                : shard->sequence_length;
    }

    if (loss_start >= count)
        return NIYAH_ERR_INVALID_CONFIG;

    *out_tokens =
        shard->tokens + start;
    *out_targets =
        shard->tokens + start + 1U;
    *out_token_count = count;
    *out_loss_start = loss_start;

    return NIYAH_OK;
}

NiyahStatus niyah_dataset_shard_sample(
    const NiyahDatasetShard *shard,
    size_t sample_index,
    const uint32_t **out_tokens,
    const uint32_t **out_targets,
    size_t *out_token_count)
{
    size_t loss_start = 0U;

    return niyah_dataset_shard_sample_with_loss(
        shard,
        sample_index,
        out_tokens,
        out_targets,
        out_token_count,
        &loss_start);
}


NiyahStatus niyah_dataset_shard_identity_sha256(
    const NiyahDatasetShard *shard,
    uint8_t out_identity[NIYAH_DATASET_SHARD_IDENTITY_SHA256_SIZE])
{
    static const unsigned char domain_v1[] =
        "NIYAH-DATASET-SHARD-V1";
    static const unsigned char domain_v2[] =
        "NIYAH-DATASET-SHARD-V2";
    static const unsigned char domain_v3[] =
        "NIYAH-DATASET-SHARD-V3";

    const unsigned char *domain;
    size_t domain_size;
    NiyahSha256 sha;
    unsigned char u64[8];
    unsigned char u32[4];
    size_t expected_samples;
    size_t i;

    if (shard == NULL ||
        out_identity == NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;

    if (shard->tokens == NULL ||
        shard->token_count < 2U ||
        shard->sequence_length == 0U ||
        shard->sample_count == 0U ||
        shard->tokens[0] != NIYAH_TOKEN_BOS ||
        shard->tokens[shard->token_count - 1U] !=
            NIYAH_TOKEN_EOS)
        return NIYAH_ERR_INVALID_CONFIG;

    if (shard->has_explicit_samples != 0) {
        if (!explicit_geometry_valid(shard))
            return NIYAH_ERR_INVALID_CONFIG;

        if (shard->has_loss_starts != 0) {
            domain = domain_v3;
            domain_size = sizeof(domain_v3) - 1U;
        } else {
            domain = domain_v2;
            domain_size = sizeof(domain_v2) - 1U;
        }
    } else {
        if (shard->sample_offsets != NULL ||
            shard->sample_lengths != NULL ||
            shard->sample_loss_starts != NULL ||
            shard->has_loss_starts != 0)
            return NIYAH_ERR_INVALID_CONFIG;

        expected_samples =
            sample_count_for(
                shard->token_count,
                shard->sequence_length);

        if (expected_samples == 0U ||
            shard->sample_count != expected_samples)
            return NIYAH_ERR_INVALID_CONFIG;

        domain = domain_v1;
        domain_size = sizeof(domain_v1) - 1U;
    }

    if (shard->sequence_length >
            (size_t)UINT64_MAX ||
        shard->token_count >
            (size_t)UINT64_MAX ||
        shard->sample_count >
            (size_t)UINT64_MAX)
        return NIYAH_ERR_OVERFLOW;

    niyah_sha256_init(&sha);

    if (!niyah_sha256_update(
            &sha,
            domain,
            domain_size) ||
        !niyah_sha256_update(
            &sha,
            shard->tokenizer_identity,
            sizeof(shard->tokenizer_identity)))
        return NIYAH_ERR_OVERFLOW;

    store_u64_le(
        u64,
        (uint64_t)shard->sequence_length);
    if (!niyah_sha256_update(
            &sha,
            u64,
            sizeof(u64)))
        return NIYAH_ERR_OVERFLOW;

    store_u64_le(
        u64,
        (uint64_t)shard->token_count);
    if (!niyah_sha256_update(
            &sha,
            u64,
            sizeof(u64)))
        return NIYAH_ERR_OVERFLOW;

    store_u64_le(
        u64,
        (uint64_t)shard->sample_count);
    if (!niyah_sha256_update(
            &sha,
            u64,
            sizeof(u64)))
        return NIYAH_ERR_OVERFLOW;

    for (i = 0U;
         i < shard->token_count;
         ++i) {
        store_u32_le(
            u32,
            shard->tokens[i]);

        if (!niyah_sha256_update(
                &sha,
                u32,
                sizeof(u32)))
            return NIYAH_ERR_OVERFLOW;
    }

    if (shard->has_explicit_samples != 0) {
        for (i = 0U;
             i < shard->sample_count;
             ++i) {
            if (shard->sample_offsets[i] >
                    (size_t)UINT64_MAX ||
                shard->sample_lengths[i] >
                    (size_t)UINT64_MAX)
                return NIYAH_ERR_OVERFLOW;

            store_u64_le(
                u64,
                (uint64_t)
                    shard->sample_offsets[i]);

            if (!niyah_sha256_update(
                    &sha,
                    u64,
                    sizeof(u64)))
                return NIYAH_ERR_OVERFLOW;

            store_u64_le(
                u64,
                (uint64_t)
                    shard->sample_lengths[i]);

            if (!niyah_sha256_update(
                    &sha,
                    u64,
                    sizeof(u64)))
                return NIYAH_ERR_OVERFLOW;

            if (shard->has_loss_starts != 0) {
                if (shard->sample_loss_starts[i] >
                    (size_t)UINT64_MAX)
                    return NIYAH_ERR_OVERFLOW;

                store_u64_le(
                    u64,
                    (uint64_t)
                        shard->sample_loss_starts[i]);

                if (!niyah_sha256_update(
                        &sha,
                        u64,
                        sizeof(u64)))
                    return NIYAH_ERR_OVERFLOW;
            }
        }
    }

    niyah_sha256_final(
        &sha,
        out_identity);

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
    unsigned char u64_bytes[8];
    unsigned char footer[8];
    NiyahDatasetShardCrc32 crc;
    FILE *file;
    uint32_t version;
    size_t i;
    NiyahStatus status;
    int close_result;

    if (path == NULL || path[0] == '\0')
        return NIYAH_ERR_INVALID_ARGUMENT;

    status = validate_shard(
        shard,
        tokenizer);
    if (status != NIYAH_OK)
        return status;

    if (shard->token_count >
            (size_t)UINT64_MAX ||
        shard->sequence_length >
            (size_t)UINT64_MAX ||
        shard->sample_count >
            (size_t)UINT64_MAX)
        return NIYAH_ERR_OVERFLOW;

    if (shard->has_loss_starts != 0)
        version =
            NIYAH_DATASET_SHARD_VERSION_V3;
    else if (shard->has_explicit_samples != 0)
        version =
            NIYAH_DATASET_SHARD_VERSION_V2;
    else
        version =
            NIYAH_DATASET_SHARD_VERSION_V1;

    memset(header, 0, sizeof(header));
    memcpy(
        header,
        NIYAH_DATASET_SHARD_MAGIC,
        8U);

    store_u32_le(
        header + 8U,
        version);
    store_u32_le(
        header + 12U,
        NIYAH_DATASET_SHARD_FLAGS);
    store_u64_le(
        header + 16U,
        (uint64_t)shard->sequence_length);
    store_u64_le(
        header + 24U,
        (uint64_t)shard->token_count);
    store_u64_le(
        header + 32U,
        (uint64_t)shard->sample_count);

    memcpy(
        header + 40U,
        shard->tokenizer_identity,
        NIYAH_DATASET_TOKENIZER_IDENTITY_SIZE);

    file = shard_fopen(path, "wb");
    if (file == NULL)
        return NIYAH_ERR_IO;

    crc_init(&crc);

    status = write_crc(
        file,
        header,
        sizeof(header),
        &crc);

    for (i = 0U;
         status == NIYAH_OK &&
         i < shard->token_count;
         ++i) {
        store_u32_le(
            token_bytes,
            shard->tokens[i]);

        status = write_crc(
            file,
            token_bytes,
            sizeof(token_bytes),
            &crc);
    }

    if (status == NIYAH_OK &&
        (version ==
             NIYAH_DATASET_SHARD_VERSION_V2 ||
         version ==
             NIYAH_DATASET_SHARD_VERSION_V3)) {
        for (i = 0U;
             status == NIYAH_OK &&
             i < shard->sample_count;
             ++i) {
            if (shard->sample_offsets[i] >
                    (size_t)UINT64_MAX ||
                shard->sample_lengths[i] >
                    (size_t)UINT64_MAX) {
                status = NIYAH_ERR_OVERFLOW;
                break;
            }

            store_u64_le(
                u64_bytes,
                (uint64_t)
                    shard->sample_offsets[i]);

            status = write_crc(
                file,
                u64_bytes,
                sizeof(u64_bytes),
                &crc);

            if (status == NIYAH_OK) {
                store_u64_le(
                    u64_bytes,
                    (uint64_t)
                        shard->sample_lengths[i]);

                status = write_crc(
                    file,
                    u64_bytes,
                    sizeof(u64_bytes),
                    &crc);
            }

            if (status == NIYAH_OK &&
                version ==
                    NIYAH_DATASET_SHARD_VERSION_V3) {
                if (shard->sample_loss_starts[i] >
                    (size_t)UINT64_MAX) {
                    status = NIYAH_ERR_OVERFLOW;
                    break;
                }

                store_u64_le(
                    u64_bytes,
                    (uint64_t)
                        shard->sample_loss_starts[i]);

                status = write_crc(
                    file,
                    u64_bytes,
                    sizeof(u64_bytes),
                    &crc);
            }
        }
    }

    if (status == NIYAH_OK) {
        store_u32_le(
            footer + 0U,
            NIYAH_DATASET_SHARD_CHECKSUM_CRC32);
        store_u32_le(
            footer + 4U,
            crc_final(&crc));

        if (fwrite(
                footer,
                1U,
                sizeof(footer),
                file) != sizeof(footer))
            status = NIYAH_ERR_IO;
    }

    if (status == NIYAH_OK &&
        fflush(file) != 0)
        status = NIYAH_ERR_IO;

    close_result = fclose(file);

    if (status == NIYAH_OK &&
        close_result != 0)
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
    unsigned char u64_bytes[8];
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
    size_t geometry_bytes_size;
    size_t expected_samples;
    size_t vocab_size;
    size_t i;
    NiyahStatus status;
    int close_result;

    if (path == NULL ||
        path[0] == '\0' ||
        tokenizer == NULL ||
        out_shard == NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;

    if (out_shard->tokens != NULL ||
        out_shard->sample_offsets != NULL ||
        out_shard->sample_lengths != NULL ||
        out_shard->sample_loss_starts != NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;

    memset(&temp, 0, sizeof(temp));

    file = shard_fopen(path, "rb");
    if (file == NULL)
        return NIYAH_ERR_IO;

    crc_init(&crc);

    status = read_crc(
        file,
        header,
        sizeof(header),
        &crc);
    if (status != NIYAH_OK)
        goto done;

    if (memcmp(
            header,
            NIYAH_DATASET_SHARD_MAGIC,
            8U) != 0) {
        status = NIYAH_ERR_CORRUPT_DATA;
        goto done;
    }

    version =
        load_u32_le(header + 8U);
    flags =
        load_u32_le(header + 12U);
    sequence64 =
        load_u64_le(header + 16U);
    token_count64 =
        load_u64_le(header + 24U);
    sample_count64 =
        load_u64_le(header + 32U);

    if (version !=
            NIYAH_DATASET_SHARD_VERSION_V1 &&
        version !=
            NIYAH_DATASET_SHARD_VERSION_V2 &&
        version !=
            NIYAH_DATASET_SHARD_VERSION_V3) {
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

    temp.sequence_length =
        (size_t)sequence64;
    temp.token_count =
        (size_t)token_count64;
    temp.sample_count =
        (size_t)sample_count64;

    if (version ==
        NIYAH_DATASET_SHARD_VERSION_V1) {
        expected_samples =
            sample_count_for(
                temp.token_count,
                temp.sequence_length);

        if (temp.sample_count !=
            expected_samples) {
            status = NIYAH_ERR_CORRUPT_DATA;
            goto done;
        }
    } else {
        temp.has_explicit_samples = 1;

        if (version ==
            NIYAH_DATASET_SHARD_VERSION_V3)
            temp.has_loss_starts = 1;
    }

    status =
        tokenizer_identity(
            tokenizer,
            identity);
    if (status != NIYAH_OK)
        goto done;

    if (memcmp(
            identity,
            header + 40U,
            sizeof(identity)) != 0) {
        status = NIYAH_ERR_INVALID_CONFIG;
        goto done;
    }

    memcpy(
        temp.tokenizer_identity,
        header + 40U,
        sizeof(identity));

    if (!size_mul_ok(
            temp.token_count,
            sizeof(uint32_t),
            &token_bytes_size)) {
        status = NIYAH_ERR_OVERFLOW;
        goto done;
    }

    temp.tokens =
        (uint32_t *)malloc(token_bytes_size);
    if (temp.tokens == NULL) {
        status = NIYAH_ERR_OUT_OF_MEMORY;
        goto done;
    }

    if (version ==
            NIYAH_DATASET_SHARD_VERSION_V2 ||
        version ==
            NIYAH_DATASET_SHARD_VERSION_V3) {
        if (!size_mul_ok(
                temp.sample_count,
                sizeof(size_t),
                &geometry_bytes_size)) {
            status = NIYAH_ERR_OVERFLOW;
            goto done;
        }

        temp.sample_offsets =
            (size_t *)malloc(
                geometry_bytes_size);
        temp.sample_lengths =
            (size_t *)malloc(
                geometry_bytes_size);

        if (version ==
            NIYAH_DATASET_SHARD_VERSION_V3) {
            temp.sample_loss_starts =
                (size_t *)malloc(
                    geometry_bytes_size);
        }

        if (temp.sample_offsets == NULL ||
            temp.sample_lengths == NULL ||
            (version ==
                 NIYAH_DATASET_SHARD_VERSION_V3 &&
             temp.sample_loss_starts == NULL)) {
            status = NIYAH_ERR_OUT_OF_MEMORY;
            goto done;
        }
    }

    vocab_size =
        niyah_tokenizer_vocab_size(
            tokenizer);
    if (vocab_size == 0U) {
        status = NIYAH_ERR_INVALID_CONFIG;
        goto done;
    }

    for (i = 0U;
         i < temp.token_count;
         ++i) {
        status = read_crc(
            file,
            token_bytes,
            sizeof(token_bytes),
            &crc);
        if (status != NIYAH_OK)
            goto done;

        temp.tokens[i] =
            load_u32_le(token_bytes);

        if ((size_t)temp.tokens[i] >=
            vocab_size) {
            status = NIYAH_ERR_CORRUPT_DATA;
            goto done;
        }
    }

    if (version ==
            NIYAH_DATASET_SHARD_VERSION_V2 ||
        version ==
            NIYAH_DATASET_SHARD_VERSION_V3) {
        for (i = 0U;
             i < temp.sample_count;
             ++i) {
            uint64_t value;

            status = read_crc(
                file,
                u64_bytes,
                sizeof(u64_bytes),
                &crc);
            if (status != NIYAH_OK)
                goto done;

            value =
                load_u64_le(u64_bytes);
            if (value >
                (uint64_t)SIZE_MAX) {
                status =
                    NIYAH_ERR_CORRUPT_DATA;
                goto done;
            }
            temp.sample_offsets[i] =
                (size_t)value;

            status = read_crc(
                file,
                u64_bytes,
                sizeof(u64_bytes),
                &crc);
            if (status != NIYAH_OK)
                goto done;

            value =
                load_u64_le(u64_bytes);
            if (value >
                (uint64_t)SIZE_MAX) {
                status =
                    NIYAH_ERR_CORRUPT_DATA;
                goto done;
            }
            temp.sample_lengths[i] =
                (size_t)value;

            if (version ==
                NIYAH_DATASET_SHARD_VERSION_V3) {
                status = read_crc(
                    file,
                    u64_bytes,
                    sizeof(u64_bytes),
                    &crc);
                if (status != NIYAH_OK)
                    goto done;

                value =
                    load_u64_le(u64_bytes);
                if (value >
                    (uint64_t)SIZE_MAX) {
                    status =
                        NIYAH_ERR_CORRUPT_DATA;
                    goto done;
                }

                temp.sample_loss_starts[i] =
                    (size_t)value;
            }
        }
    }

    if (fread(
            footer,
            1U,
            sizeof(footer),
            file) != sizeof(footer)) {
        status = ferror(file)
            ? NIYAH_ERR_IO
            : NIYAH_ERR_CORRUPT_DATA;
        goto done;
    }

    if (load_u32_le(footer + 0U) !=
            NIYAH_DATASET_SHARD_CHECKSUM_CRC32 ||
        load_u32_le(footer + 4U) !=
            crc_final(&crc)) {
        status = NIYAH_ERR_CORRUPT_DATA;
        goto done;
    }

    if (fread(
            &extra,
            1U,
            1U,
            file) != 0U) {
        status = NIYAH_ERR_CORRUPT_DATA;
        goto done;
    }

    if (ferror(file)) {
        status = NIYAH_ERR_IO;
        goto done;
    }

    status =
        validate_shard(
            &temp,
            tokenizer);

    if (status ==
        NIYAH_ERR_INVALID_CONFIG)
        status =
            NIYAH_ERR_CORRUPT_DATA;

done:
    close_result = fclose(file);

    if (status == NIYAH_OK &&
        close_result != 0)
        status = NIYAH_ERR_IO;

    if (status != NIYAH_OK) {
        niyah_dataset_shard_destroy(&temp);
        return status;
    }

    *out_shard = temp;
    return NIYAH_OK;
}
