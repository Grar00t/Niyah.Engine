#include "niyah/checkpoint.h"
#include "niyah_sha256.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(CHAR_BIT == 8, "checkpoint v1 requires 8-bit bytes");
_Static_assert(sizeof(float) == 4, "checkpoint v1 requires 32-bit float");
_Static_assert(FLT_RADIX == 2, "checkpoint v1 requires binary float");
_Static_assert(FLT_MANT_DIG == 24, "checkpoint v1 requires IEEE-754 binary32 precision");
_Static_assert(FLT_MAX_EXP == 128, "checkpoint v1 requires IEEE-754 binary32 exponent range");

#define NIYAH_CHECKPOINT_VERSION_V1 UINT32_C(1)
#define NIYAH_CHECKPOINT_VERSION_V2 UINT32_C(2)
#define NIYAH_CHECKPOINT_VERSION NIYAH_CHECKPOINT_VERSION_V1
#define NIYAH_CHECKPOINT_HEADER_FLAGS UINT32_C(0)
#define NIYAH_CHECKPOINT_RESERVED UINT32_C(0)
#define NIYAH_CHECKPOINT_SECTION_COUNT_V1 UINT32_C(4)
#define NIYAH_CHECKPOINT_SECTION_COUNT_V2 UINT32_C(5)
#define NIYAH_CHECKPOINT_SECTION_COUNT NIYAH_CHECKPOINT_SECTION_COUNT_V1
#define NIYAH_CHECKPOINT_MAX_SECTIONS UINT32_C(64)
#define NIYAH_CHECKPOINT_SECTION_REQUIRED UINT32_C(1)
#define NIYAH_CHECKPOINT_KNOWN_SECTION_FLAGS NIYAH_CHECKPOINT_SECTION_REQUIRED
#define NIYAH_CHECKPOINT_CHECKSUM_CRC32 UINT32_C(1)
#define NIYAH_CHECKPOINT_IO_BUFFER_SIZE 65536U
#define NIYAH_CHECKPOINT_ADAMW_META_BYTES UINT64_C(32)
#define NIYAH_CHECKPOINT_TOKENIZER_IDENTITY_BYTES UINT64_C(32)

#define NIYAH_CHECKPOINT_SECTION_MODEL_WEIGHTS UINT32_C(1)
#define NIYAH_CHECKPOINT_SECTION_ADAMW_M UINT32_C(2)
#define NIYAH_CHECKPOINT_SECTION_ADAMW_V UINT32_C(3)
#define NIYAH_CHECKPOINT_SECTION_ADAMW_META UINT32_C(4)
#define NIYAH_CHECKPOINT_SECTION_TOKENIZER_IDENTITY UINT32_C(5)

static const unsigned char NIYAH_CHECKPOINT_MAGIC[8] = {
    'N', 'I', 'Y', 'A', 'H', 'C', 'K', 'P'
};

typedef struct NiyahCheckpointCrc32 {
    uint32_t value;
    uint32_t table[256];
} NiyahCheckpointCrc32;

typedef struct NiyahCheckpointHeader {
    NiyahModelConfig config;
    NiyahModelLayout layout;
    size_t weight_count;
    uint64_t tensor_bytes;
    uint32_t section_count;
} NiyahCheckpointHeader;

typedef struct NiyahCheckpointMeta {
    NiyahAdamWConfig optimizer_config;
    uint64_t step;
} NiyahCheckpointMeta;

typedef struct NiyahCheckpointScan {
    NiyahCheckpointHeader header;
    NiyahCheckpointMeta meta;
    uint8_t tokenizer_identity[NIYAH_TOKENIZER_IDENTITY_SHA256_SIZE];
    int has_tokenizer_identity;
    uint32_t crc32;
} NiyahCheckpointScan;

typedef enum NiyahCheckpointTensorKind {
    NIYAH_CHECKPOINT_TENSOR_MODEL = 1,
    NIYAH_CHECKPOINT_TENSOR_M = 2,
    NIYAH_CHECKPOINT_TENSOR_V = 3
} NiyahCheckpointTensorKind;

typedef struct NiyahCheckpointLoadTargets {
    float *weights;
    float *m;
    float *v;
    size_t count;
} NiyahCheckpointLoadTargets;

static FILE *niyah_checkpoint_fopen(const char *path, const char *mode)
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

static void niyah_store_u32_le(unsigned char out[4], uint32_t value)
{
    out[0] = (unsigned char)(value & UINT32_C(0xff));
    out[1] = (unsigned char)((value >> 8) & UINT32_C(0xff));
    out[2] = (unsigned char)((value >> 16) & UINT32_C(0xff));
    out[3] = (unsigned char)((value >> 24) & UINT32_C(0xff));
}

static uint32_t niyah_load_u32_le(const unsigned char in[4])
{
    return ((uint32_t)in[0]) |
           ((uint32_t)in[1] << 8) |
           ((uint32_t)in[2] << 16) |
           ((uint32_t)in[3] << 24);
}

static void niyah_store_u64_le(unsigned char out[8], uint64_t value)
{
    size_t i;
    for (i = 0U; i < 8U; ++i) {
        out[i] = (unsigned char)((value >> (i * 8U)) & UINT64_C(0xff));
    }
}

static uint64_t niyah_load_u64_le(const unsigned char in[8])
{
    uint64_t value = UINT64_C(0);
    size_t i;
    for (i = 0U; i < 8U; ++i) {
        value |= ((uint64_t)in[i]) << (i * 8U);
    }
    return value;
}

static uint32_t niyah_float_bits(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float niyah_bits_float(uint32_t bits)
{
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void niyah_crc32_init(NiyahCheckpointCrc32 *crc)
{
    uint32_t i;
    if (crc == NULL) {
        return;
    }
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

static void niyah_crc32_update(NiyahCheckpointCrc32 *crc,
                               const unsigned char *data,
                               size_t size)
{
    size_t i;
    if (crc == NULL || data == NULL) {
        return;
    }
    for (i = 0U; i < size; ++i) {
        const uint32_t index = (crc->value ^ (uint32_t)data[i]) & UINT32_C(0xff);
        crc->value = crc->table[index] ^ (crc->value >> 8);
    }
}

static uint32_t niyah_crc32_final(const NiyahCheckpointCrc32 *crc)
{
    return crc->value ^ UINT32_C(0xffffffff);
}

static int niyah_size_mul_ok(size_t a, size_t b, size_t *out)
{
    if (out == NULL || (a != 0U && b > SIZE_MAX / a)) {
        return 0;
    }
    *out = a * b;
    return 1;
}

static int niyah_config_equal(const NiyahModelConfig *a,
                              const NiyahModelConfig *b)
{
    return a->vocab_size == b->vocab_size &&
           a->context_length == b->context_length &&
           a->embedding_dim == b->embedding_dim &&
           a->n_layers == b->n_layers &&
           a->n_heads == b->n_heads &&
           a->n_kv_heads == b->n_kv_heads &&
           a->ffn_hidden_dim == b->ffn_hidden_dim &&
           niyah_float_bits(a->rms_norm_eps) == niyah_float_bits(b->rms_norm_eps) &&
           (a->tie_word_embeddings != 0) == (b->tie_word_embeddings != 0);
}

static int niyah_layout_equal(const NiyahModelLayout *a,
                              const NiyahModelLayout *b)
{
    return a->token_embedding == b->token_embedding &&
           a->layers == b->layers &&
           a->layer_stride == b->layer_stride &&
           a->final_norm == b->final_norm &&
           a->lm_head == b->lm_head &&
           a->total_floats == b->total_floats &&
           a->head_dim == b->head_dim &&
           a->kv_dim == b->kv_dim;
}

static NiyahStatus niyah_write_all(FILE *file, const void *data, size_t size)
{
    if (size == 0U) {
        return NIYAH_OK;
    }
    if (file == NULL || data == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    if (fwrite(data, 1U, size, file) != size) {
        return NIYAH_ERR_IO;
    }
    return NIYAH_OK;
}

static NiyahStatus niyah_write_crc(FILE *file,
                                   NiyahCheckpointCrc32 *crc,
                                   const void *data,
                                   size_t size)
{
    NiyahStatus status = niyah_write_all(file, data, size);
    if (status != NIYAH_OK) {
        return status;
    }
    if (size != 0U) {
        niyah_crc32_update(crc, (const unsigned char *)data, size);
    }
    return NIYAH_OK;
}

static NiyahStatus niyah_read_exact(FILE *file,
                                    void *data,
                                    size_t size,
                                    NiyahCheckpointCrc32 *crc,
                                    int include_in_crc)
{
    size_t received;
    if (size == 0U) {
        return NIYAH_OK;
    }
    if (file == NULL || data == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    received = fread(data, 1U, size, file);
    if (received != 0U && include_in_crc != 0) {
        niyah_crc32_update(crc, (const unsigned char *)data, received);
    }
    if (received != size) {
        return ferror(file) != 0 ? NIYAH_ERR_IO : NIYAH_ERR_CORRUPT_DATA;
    }
    return NIYAH_OK;
}

static NiyahStatus niyah_skip_crc(FILE *file,
                                  NiyahCheckpointCrc32 *crc,
                                  uint64_t bytes,
                                  unsigned char *buffer,
                                  size_t buffer_size)
{
    while (bytes != UINT64_C(0)) {
        const size_t chunk = bytes > (uint64_t)buffer_size
            ? buffer_size
            : (size_t)bytes;
        NiyahStatus status = niyah_read_exact(file, buffer, chunk, crc, 1);
        if (status != NIYAH_OK) {
            return status;
        }
        bytes -= (uint64_t)chunk;
    }
    return NIYAH_OK;
}

static NiyahStatus niyah_write_u32_crc(FILE *file,
                                       NiyahCheckpointCrc32 *crc,
                                       uint32_t value)
{
    unsigned char bytes[4];
    niyah_store_u32_le(bytes, value);
    return niyah_write_crc(file, crc, bytes, sizeof(bytes));
}

static NiyahStatus niyah_write_u64_crc(FILE *file,
                                       NiyahCheckpointCrc32 *crc,
                                       uint64_t value)
{
    unsigned char bytes[8];
    niyah_store_u64_le(bytes, value);
    return niyah_write_crc(file, crc, bytes, sizeof(bytes));
}

static NiyahStatus niyah_write_section_header(FILE *file,
                                              NiyahCheckpointCrc32 *crc,
                                              uint32_t section_id,
                                              uint64_t payload_bytes)
{
    NiyahStatus status;
    status = niyah_write_u32_crc(file, crc, section_id);
    if (status != NIYAH_OK) return status;
    status = niyah_write_u32_crc(file, crc, NIYAH_CHECKPOINT_SECTION_REQUIRED);
    if (status != NIYAH_OK) return status;
    return niyah_write_u64_crc(file, crc, payload_bytes);
}

static NiyahStatus niyah_write_float_array(FILE *file,
                                           NiyahCheckpointCrc32 *crc,
                                           const float *values,
                                           size_t count,
                                           unsigned char *buffer,
                                           size_t buffer_size)
{
    const size_t floats_per_chunk = buffer_size / sizeof(uint32_t);
    size_t offset = 0U;
    if (floats_per_chunk == 0U) {
        return NIYAH_ERR_INVALID_CONFIG;
    }
    while (offset < count) {
        const size_t remaining = count - offset;
        const size_t chunk_count = remaining < floats_per_chunk ? remaining : floats_per_chunk;
        const size_t chunk_bytes = chunk_count * sizeof(uint32_t);
        size_t i;
        for (i = 0U; i < chunk_count; ++i) {
            niyah_store_u32_le(buffer + i * sizeof(uint32_t),
                               niyah_float_bits(values[offset + i]));
        }
        {
            NiyahStatus status = niyah_write_crc(file, crc, buffer, chunk_bytes);
            if (status != NIYAH_OK) {
                return status;
            }
        }
        offset += chunk_count;
    }
    return NIYAH_OK;
}

static NiyahStatus niyah_read_float_array(FILE *file,
                                          NiyahCheckpointCrc32 *crc,
                                          float *values,
                                          size_t count,
                                          NiyahCheckpointTensorKind kind,
                                          unsigned char *buffer,
                                          size_t buffer_size)
{
    const size_t floats_per_chunk = buffer_size / sizeof(uint32_t);
    size_t offset = 0U;
    if (values == NULL || floats_per_chunk == 0U) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    while (offset < count) {
        const size_t remaining = count - offset;
        const size_t chunk_count = remaining < floats_per_chunk ? remaining : floats_per_chunk;
        const size_t chunk_bytes = chunk_count * sizeof(uint32_t);
        size_t i;
        NiyahStatus status = niyah_read_exact(file, buffer, chunk_bytes, crc, 1);
        if (status != NIYAH_OK) {
            return status;
        }
        for (i = 0U; i < chunk_count; ++i) {
            const uint32_t bits = niyah_load_u32_le(buffer + i * sizeof(uint32_t));
            const float value = niyah_bits_float(bits);
            if (!isfinite(value) ||
                (kind == NIYAH_CHECKPOINT_TENSOR_V && value < 0.0f)) {
                return NIYAH_ERR_CORRUPT_DATA;
            }
            values[offset + i] = value;
        }
        offset += chunk_count;
    }
    return NIYAH_OK;
}

static int niyah_range_bounds(const void *pointer,
                              size_t bytes,
                              uintptr_t *start,
                              uintptr_t *last)
{
    const uintptr_t address = (uintptr_t)pointer;
    if (pointer == NULL || bytes == 0U || start == NULL || last == NULL ||
        address > UINTPTR_MAX - (uintptr_t)(bytes - 1U)) {
        return 0;
    }
    *start = address;
    *last = address + (uintptr_t)(bytes - 1U);
    return 1;
}

static int niyah_ranges_overlap(const void *a,
                                size_t a_bytes,
                                const void *b,
                                size_t b_bytes)
{
    uintptr_t a_start;
    uintptr_t a_last;
    uintptr_t b_start;
    uintptr_t b_last;
    if (!niyah_range_bounds(a, a_bytes, &a_start, &a_last) ||
        !niyah_range_bounds(b, b_bytes, &b_start, &b_last)) {
        return 1;
    }
    return a_start <= b_last && b_start <= a_last;
}

static NiyahStatus niyah_validate_save_state(const NiyahModel *model,
                                              const NiyahAdamWState *optimizer_state,
                                              const NiyahAdamWConfig *optimizer_config,
                                              uint64_t *out_tensor_bytes)
{
    NiyahModelLayout canonical;
    size_t bytes;
    size_t i;
    NiyahStatus status;

    if (model == NULL || optimizer_state == NULL || optimizer_config == NULL ||
        out_tensor_bytes == NULL || model->weights == NULL ||
        optimizer_state->m == NULL || optimizer_state->v == NULL ||
        model->weight_count == 0U) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    status = niyah_model_layout_compute(&model->config, &canonical);
    if (status != NIYAH_OK) {
        return status;
    }
    if (!niyah_layout_equal(&model->layout, &canonical) ||
        model->weight_count != canonical.total_floats ||
        optimizer_state->count != model->weight_count ||
        optimizer_state->bound_model != model ||
        optimizer_state->bound_weights != model->weights ||
        !niyah_config_equal(&optimizer_state->model_config, &model->config) ||
        !niyah_layout_equal(&optimizer_state->model_layout, &canonical)) {
        return NIYAH_ERR_INVALID_CONFIG;
    }
    status = niyah_adamw_config_validate(optimizer_config);
    if (status != NIYAH_OK) {
        return status;
    }
    if (!niyah_size_mul_ok(model->weight_count, sizeof(float), &bytes)) {
        return NIYAH_ERR_OVERFLOW;
    }
    if ((sizeof(size_t) > sizeof(uint64_t) &&
         model->weight_count > (size_t)UINT64_MAX) ||
        (uint64_t)model->weight_count > UINT64_MAX / UINT64_C(4)) {
        return NIYAH_ERR_OVERFLOW;
    }
    if (niyah_ranges_overlap(model->weights, bytes, optimizer_state->m, bytes) ||
        niyah_ranges_overlap(model->weights, bytes, optimizer_state->v, bytes) ||
        niyah_ranges_overlap(optimizer_state->m, bytes, optimizer_state->v, bytes)) {
        return NIYAH_ERR_INVALID_CONFIG;
    }
    *out_tensor_bytes = (uint64_t)model->weight_count * UINT64_C(4);
    for (i = 0U; i < model->weight_count; ++i) {
        if (!isfinite(model->weights[i]) ||
            !isfinite(optimizer_state->m[i]) ||
            !isfinite(optimizer_state->v[i]) ||
            optimizer_state->v[i] < 0.0f) {
            return NIYAH_ERR_INVALID_CONFIG;
        }
    }
    (void)bytes;
    return NIYAH_OK;
}

static NiyahStatus niyah_write_header(FILE *file,
                                      NiyahCheckpointCrc32 *crc,
                                      const NiyahModel *model,
                                      uint32_t version,
                                      uint32_t section_count)
{
    const uint32_t tie = model->config.tie_word_embeddings != 0 ? UINT32_C(1) : UINT32_C(0);
    NiyahStatus status = niyah_write_crc(file, crc, NIYAH_CHECKPOINT_MAGIC,
                                         sizeof(NIYAH_CHECKPOINT_MAGIC));
    if (status != NIYAH_OK) return status;
    status = niyah_write_u32_crc(file, crc, version);
    if (status != NIYAH_OK) return status;
    status = niyah_write_u32_crc(file, crc, NIYAH_CHECKPOINT_HEADER_FLAGS);
    if (status != NIYAH_OK) return status;
    status = niyah_write_u32_crc(file, crc, section_count);
    if (status != NIYAH_OK) return status;
    status = niyah_write_u32_crc(file, crc, NIYAH_CHECKPOINT_RESERVED);
    if (status != NIYAH_OK) return status;
    status = niyah_write_u32_crc(file, crc, model->config.vocab_size);
    if (status != NIYAH_OK) return status;
    status = niyah_write_u32_crc(file, crc, model->config.context_length);
    if (status != NIYAH_OK) return status;
    status = niyah_write_u32_crc(file, crc, model->config.embedding_dim);
    if (status != NIYAH_OK) return status;
    status = niyah_write_u32_crc(file, crc, model->config.n_layers);
    if (status != NIYAH_OK) return status;
    status = niyah_write_u32_crc(file, crc, model->config.n_heads);
    if (status != NIYAH_OK) return status;
    status = niyah_write_u32_crc(file, crc, model->config.n_kv_heads);
    if (status != NIYAH_OK) return status;
    status = niyah_write_u32_crc(file, crc, model->config.ffn_hidden_dim);
    if (status != NIYAH_OK) return status;
    status = niyah_write_u32_crc(file, crc, niyah_float_bits(model->config.rms_norm_eps));
    if (status != NIYAH_OK) return status;
    status = niyah_write_u32_crc(file, crc, tie);
    if (status != NIYAH_OK) return status;
    return niyah_write_u64_crc(file, crc, (uint64_t)model->weight_count);
}

static NiyahStatus niyah_write_meta(FILE *file,
                                    NiyahCheckpointCrc32 *crc,
                                    const NiyahAdamWState *state,
                                    const NiyahAdamWConfig *config)
{
    unsigned char payload[32];
    niyah_store_u32_le(payload + 0U, niyah_float_bits(config->learning_rate));
    niyah_store_u32_le(payload + 4U, niyah_float_bits(config->beta1));
    niyah_store_u32_le(payload + 8U, niyah_float_bits(config->beta2));
    niyah_store_u32_le(payload + 12U, niyah_float_bits(config->epsilon));
    niyah_store_u32_le(payload + 16U, niyah_float_bits(config->weight_decay));
    niyah_store_u32_le(payload + 20U, niyah_float_bits(config->max_grad_norm));
    niyah_store_u64_le(payload + 24U, state->step);
    return niyah_write_crc(file, crc, payload, sizeof(payload));
}

static NiyahStatus niyah_checkpoint_save_impl(
    const char *path,
    const NiyahModel *model,
    const NiyahAdamWState *optimizer_state,
    const NiyahAdamWConfig *optimizer_config,
    const NiyahTokenizer *tokenizer)
{
    FILE *file = NULL;
    NiyahCheckpointCrc32 crc;
    unsigned char buffer[NIYAH_CHECKPOINT_IO_BUFFER_SIZE];
    unsigned char footer[8];
    uint8_t tokenizer_identity[NIYAH_TOKENIZER_IDENTITY_SHA256_SIZE];
    uint64_t tensor_bytes = UINT64_C(0);
    uint32_t version = NIYAH_CHECKPOINT_VERSION_V1;
    uint32_t section_count = NIYAH_CHECKPOINT_SECTION_COUNT_V1;
    NiyahStatus status;
    int close_result;

    if (path == NULL || path[0] == '\0') {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    status = niyah_validate_save_state(model, optimizer_state, optimizer_config,
                                       &tensor_bytes);
    if (status != NIYAH_OK) {
        return status;
    }

    if (tokenizer != NULL) {
        if (niyah_tokenizer_vocab_size(tokenizer) !=
            (size_t)model->config.vocab_size) {
            return NIYAH_ERR_INVALID_CONFIG;
        }
        status = niyah_tokenizer_identity_sha256(
            tokenizer, tokenizer_identity);
        if (status != NIYAH_OK) {
            return status;
        }
        version = NIYAH_CHECKPOINT_VERSION_V2;
        section_count = NIYAH_CHECKPOINT_SECTION_COUNT_V2;
    }

    file = niyah_checkpoint_fopen(path, "wb");
    if (file == NULL) {
        return NIYAH_ERR_IO;
    }
    niyah_crc32_init(&crc);

    status = niyah_write_header(file, &crc, model, version, section_count);
    if (status == NIYAH_OK) {
        status = niyah_write_section_header(file, &crc,
                                            NIYAH_CHECKPOINT_SECTION_MODEL_WEIGHTS,
                                            tensor_bytes);
    }
    if (status == NIYAH_OK) {
        status = niyah_write_float_array(file, &crc, model->weights,
                                         model->weight_count, buffer, sizeof(buffer));
    }
    if (status == NIYAH_OK) {
        status = niyah_write_section_header(file, &crc,
                                            NIYAH_CHECKPOINT_SECTION_ADAMW_M,
                                            tensor_bytes);
    }
    if (status == NIYAH_OK) {
        status = niyah_write_float_array(file, &crc, optimizer_state->m,
                                         optimizer_state->count, buffer, sizeof(buffer));
    }
    if (status == NIYAH_OK) {
        status = niyah_write_section_header(file, &crc,
                                            NIYAH_CHECKPOINT_SECTION_ADAMW_V,
                                            tensor_bytes);
    }
    if (status == NIYAH_OK) {
        status = niyah_write_float_array(file, &crc, optimizer_state->v,
                                         optimizer_state->count, buffer, sizeof(buffer));
    }
    if (status == NIYAH_OK) {
        status = niyah_write_section_header(file, &crc,
                                            NIYAH_CHECKPOINT_SECTION_ADAMW_META,
                                            NIYAH_CHECKPOINT_ADAMW_META_BYTES);
    }
    if (status == NIYAH_OK) {
        status = niyah_write_meta(file, &crc, optimizer_state, optimizer_config);
    }
    if (status == NIYAH_OK && tokenizer != NULL) {
        status = niyah_write_section_header(
            file, &crc,
            NIYAH_CHECKPOINT_SECTION_TOKENIZER_IDENTITY,
            NIYAH_CHECKPOINT_TOKENIZER_IDENTITY_BYTES);
    }
    if (status == NIYAH_OK && tokenizer != NULL) {
        status = niyah_write_crc(
            file, &crc, tokenizer_identity,
            NIYAH_TOKENIZER_IDENTITY_SHA256_SIZE);
    }
    if (status == NIYAH_OK) {
        niyah_store_u32_le(footer + 0U, NIYAH_CHECKPOINT_CHECKSUM_CRC32);
        niyah_store_u32_le(footer + 4U, niyah_crc32_final(&crc));
        status = niyah_write_all(file, footer, sizeof(footer));
    }
    if (status == NIYAH_OK && fflush(file) != 0) {
        status = NIYAH_ERR_IO;
    }
    close_result = fclose(file);
    file = NULL;
    if (status == NIYAH_OK && close_result != 0) {
        status = NIYAH_ERR_IO;
    }
    if (status != NIYAH_OK) {
        (void)remove(path);
    }
    return status;
}

NiyahStatus niyah_checkpoint_save(const char *path,
                                  const NiyahModel *model,
                                  const NiyahAdamWState *optimizer_state,
                                  const NiyahAdamWConfig *optimizer_config)
{
    return niyah_checkpoint_save_impl(
        path, model, optimizer_state, optimizer_config, NULL);
}

NiyahStatus niyah_checkpoint_save_with_tokenizer(
    const char *path,
    const NiyahModel *model,
    const NiyahAdamWState *optimizer_state,
    const NiyahAdamWConfig *optimizer_config,
    const NiyahTokenizer *tokenizer)
{
    if (tokenizer == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    return niyah_checkpoint_save_impl(
        path, model, optimizer_state, optimizer_config, tokenizer);
}

static NiyahStatus niyah_read_header(FILE *file,
                                     NiyahCheckpointCrc32 *crc,
                                     NiyahCheckpointHeader *out,
                                     uint32_t required_version)
{
    unsigned char bytes[68];
    uint32_t version;
    uint32_t flags;
    uint32_t reserved;
    uint32_t tie;
    uint64_t weight_count_u64;
    NiyahStatus status;

    memset(out, 0, sizeof(*out));
    status = niyah_read_exact(file, bytes, sizeof(bytes), crc, 1);
    if (status != NIYAH_OK) {
        return status;
    }
    if (memcmp(bytes, NIYAH_CHECKPOINT_MAGIC, sizeof(NIYAH_CHECKPOINT_MAGIC)) != 0) {
        return NIYAH_ERR_CORRUPT_DATA;
    }
    version = niyah_load_u32_le(bytes + 8U);
    if (version != required_version) {
        return NIYAH_ERR_UNSUPPORTED_VERSION;
    }
    flags = niyah_load_u32_le(bytes + 12U);
    out->section_count = niyah_load_u32_le(bytes + 16U);
    reserved = niyah_load_u32_le(bytes + 20U);
    if (flags != NIYAH_CHECKPOINT_HEADER_FLAGS ||
        reserved != NIYAH_CHECKPOINT_RESERVED ||
        out->section_count > NIYAH_CHECKPOINT_MAX_SECTIONS) {
        return NIYAH_ERR_CORRUPT_DATA;
    }

    out->config.vocab_size = niyah_load_u32_le(bytes + 24U);
    out->config.context_length = niyah_load_u32_le(bytes + 28U);
    out->config.embedding_dim = niyah_load_u32_le(bytes + 32U);
    out->config.n_layers = niyah_load_u32_le(bytes + 36U);
    out->config.n_heads = niyah_load_u32_le(bytes + 40U);
    out->config.n_kv_heads = niyah_load_u32_le(bytes + 44U);
    out->config.ffn_hidden_dim = niyah_load_u32_le(bytes + 48U);
    out->config.rms_norm_eps = niyah_bits_float(niyah_load_u32_le(bytes + 52U));
    tie = niyah_load_u32_le(bytes + 56U);
    if (tie > UINT32_C(1)) {
        return NIYAH_ERR_CORRUPT_DATA;
    }
    out->config.tie_word_embeddings = tie != 0U ? 1 : 0;
    weight_count_u64 = niyah_load_u64_le(bytes + 60U);
    if ((sizeof(size_t) < sizeof(uint64_t) &&
         weight_count_u64 > (uint64_t)SIZE_MAX) ||
        weight_count_u64 > UINT64_MAX / UINT64_C(4)) {
        return NIYAH_ERR_CORRUPT_DATA;
    }
    out->weight_count = (size_t)weight_count_u64;
    out->tensor_bytes = weight_count_u64 * UINT64_C(4);

    status = niyah_model_config_validate(&out->config);
    if (status != NIYAH_OK) {
        return NIYAH_ERR_CORRUPT_DATA;
    }
    status = niyah_model_layout_compute(&out->config, &out->layout);
    if (status != NIYAH_OK || out->layout.total_floats != out->weight_count) {
        return NIYAH_ERR_CORRUPT_DATA;
    }
    return NIYAH_OK;
}

static NiyahStatus niyah_read_meta(FILE *file,
                                   NiyahCheckpointCrc32 *crc,
                                   NiyahCheckpointMeta *meta)
{
    unsigned char payload[32];
    NiyahStatus status = niyah_read_exact(file, payload, sizeof(payload), crc, 1);
    if (status != NIYAH_OK) {
        return status;
    }
    meta->optimizer_config.learning_rate = niyah_bits_float(niyah_load_u32_le(payload + 0U));
    meta->optimizer_config.beta1 = niyah_bits_float(niyah_load_u32_le(payload + 4U));
    meta->optimizer_config.beta2 = niyah_bits_float(niyah_load_u32_le(payload + 8U));
    meta->optimizer_config.epsilon = niyah_bits_float(niyah_load_u32_le(payload + 12U));
    meta->optimizer_config.weight_decay = niyah_bits_float(niyah_load_u32_le(payload + 16U));
    meta->optimizer_config.max_grad_norm = niyah_bits_float(niyah_load_u32_le(payload + 20U));
    meta->step = niyah_load_u64_le(payload + 24U);
    if (niyah_adamw_config_validate(&meta->optimizer_config) != NIYAH_OK) {
        return NIYAH_ERR_CORRUPT_DATA;
    }
    return NIYAH_OK;
}

static NiyahStatus niyah_read_section_header(FILE *file,
                                             NiyahCheckpointCrc32 *crc,
                                             uint32_t *section_id,
                                             uint32_t *section_flags,
                                             uint64_t *payload_bytes)
{
    unsigned char bytes[16];
    NiyahStatus status = niyah_read_exact(file, bytes, sizeof(bytes), crc, 1);
    if (status != NIYAH_OK) {
        return status;
    }
    *section_id = niyah_load_u32_le(bytes + 0U);
    *section_flags = niyah_load_u32_le(bytes + 4U);
    *payload_bytes = niyah_load_u64_le(bytes + 8U);
    if ((*section_flags & ~NIYAH_CHECKPOINT_KNOWN_SECTION_FLAGS) != 0U) {
        return NIYAH_ERR_CORRUPT_DATA;
    }
    return NIYAH_OK;
}

static int niyah_header_equal(const NiyahCheckpointHeader *a,
                              const NiyahCheckpointHeader *b)
{
    return a->section_count == b->section_count &&
           a->weight_count == b->weight_count &&
           a->tensor_bytes == b->tensor_bytes &&
           niyah_config_equal(&a->config, &b->config) &&
           niyah_layout_equal(&a->layout, &b->layout);
}

static int niyah_optimizer_config_equal(const NiyahAdamWConfig *a,
                                        const NiyahAdamWConfig *b)
{
    return niyah_float_bits(a->learning_rate) == niyah_float_bits(b->learning_rate) &&
           niyah_float_bits(a->beta1) == niyah_float_bits(b->beta1) &&
           niyah_float_bits(a->beta2) == niyah_float_bits(b->beta2) &&
           niyah_float_bits(a->epsilon) == niyah_float_bits(b->epsilon) &&
           niyah_float_bits(a->weight_decay) == niyah_float_bits(b->weight_decay) &&
           niyah_float_bits(a->max_grad_norm) == niyah_float_bits(b->max_grad_norm);
}

static NiyahStatus niyah_scan_checkpoint(
                                         FILE *file,
                                         uint32_t required_version,
                                         const NiyahCheckpointScan *expected,
                                         NiyahCheckpointLoadTargets *targets,
                                         NiyahCheckpointScan *out)
{
    NiyahCheckpointCrc32 crc;
    unsigned char buffer[NIYAH_CHECKPOINT_IO_BUFFER_SIZE];
    unsigned char footer[8];
    uint32_t seen = UINT32_C(0);
    uint32_t section_index;
    NiyahStatus status;

    memset(out, 0, sizeof(*out));
    niyah_crc32_init(&crc);
    status = niyah_read_header(file, &crc, &out->header, required_version);
    if (status != NIYAH_OK) {
        return status;
    }
    if (expected != NULL && !niyah_header_equal(&expected->header, &out->header)) {
        return NIYAH_ERR_CORRUPT_DATA;
    }
    if (targets != NULL && targets->count != out->header.weight_count) {
        return NIYAH_ERR_INVALID_CONFIG;
    }

    for (section_index = 0U; section_index < out->header.section_count; ++section_index) {
        uint32_t section_id;
        uint32_t section_flags;
        uint64_t payload_bytes;
        uint32_t bit = UINT32_C(0);
        status = niyah_read_section_header(file, &crc, &section_id, &section_flags,
                                           &payload_bytes);
        if (status != NIYAH_OK) {
            return status;
        }

        switch (section_id) {
            case NIYAH_CHECKPOINT_SECTION_MODEL_WEIGHTS:
                bit = UINT32_C(1) << 0;
                if ((seen & bit) != 0U ||
                    (section_flags & NIYAH_CHECKPOINT_SECTION_REQUIRED) == 0U ||
                    payload_bytes != out->header.tensor_bytes) {
                    return NIYAH_ERR_CORRUPT_DATA;
                }
                seen |= bit;
                status = targets == NULL
                    ? niyah_skip_crc(file, &crc, payload_bytes, buffer, sizeof(buffer))
                    : niyah_read_float_array(file, &crc, targets->weights,
                                             targets->count,
                                             NIYAH_CHECKPOINT_TENSOR_MODEL,
                                             buffer, sizeof(buffer));
                break;
            case NIYAH_CHECKPOINT_SECTION_ADAMW_M:
                bit = UINT32_C(1) << 1;
                if ((seen & bit) != 0U ||
                    (section_flags & NIYAH_CHECKPOINT_SECTION_REQUIRED) == 0U ||
                    payload_bytes != out->header.tensor_bytes) {
                    return NIYAH_ERR_CORRUPT_DATA;
                }
                seen |= bit;
                status = targets == NULL
                    ? niyah_skip_crc(file, &crc, payload_bytes, buffer, sizeof(buffer))
                    : niyah_read_float_array(file, &crc, targets->m,
                                             targets->count,
                                             NIYAH_CHECKPOINT_TENSOR_M,
                                             buffer, sizeof(buffer));
                break;
            case NIYAH_CHECKPOINT_SECTION_ADAMW_V:
                bit = UINT32_C(1) << 2;
                if ((seen & bit) != 0U ||
                    (section_flags & NIYAH_CHECKPOINT_SECTION_REQUIRED) == 0U ||
                    payload_bytes != out->header.tensor_bytes) {
                    return NIYAH_ERR_CORRUPT_DATA;
                }
                seen |= bit;
                status = targets == NULL
                    ? niyah_skip_crc(file, &crc, payload_bytes, buffer, sizeof(buffer))
                    : niyah_read_float_array(file, &crc, targets->v,
                                             targets->count,
                                             NIYAH_CHECKPOINT_TENSOR_V,
                                             buffer, sizeof(buffer));
                break;
            case NIYAH_CHECKPOINT_SECTION_ADAMW_META:
                bit = UINT32_C(1) << 3;
                if ((seen & bit) != 0U ||
                    (section_flags & NIYAH_CHECKPOINT_SECTION_REQUIRED) == 0U ||
                    payload_bytes != NIYAH_CHECKPOINT_ADAMW_META_BYTES) {
                    return NIYAH_ERR_CORRUPT_DATA;
                }
                seen |= bit;
                status = niyah_read_meta(file, &crc, &out->meta);
                break;
            case NIYAH_CHECKPOINT_SECTION_TOKENIZER_IDENTITY:
                bit = UINT32_C(1) << 4;
                if (required_version != NIYAH_CHECKPOINT_VERSION_V2 ||
                    (seen & bit) != 0U ||
                    (section_flags & NIYAH_CHECKPOINT_SECTION_REQUIRED) == 0U ||
                    payload_bytes != NIYAH_CHECKPOINT_TOKENIZER_IDENTITY_BYTES) {
                    return NIYAH_ERR_CORRUPT_DATA;
                }
                seen |= bit;
                status = niyah_read_exact(
                    file, out->tokenizer_identity,
                    NIYAH_TOKENIZER_IDENTITY_SHA256_SIZE,
                    &crc, 1);
                if (status == NIYAH_OK) {
                    out->has_tokenizer_identity = 1;
                }
                break;
            default:
                if ((section_flags & NIYAH_CHECKPOINT_SECTION_REQUIRED) != 0U) {
                    return NIYAH_ERR_UNSUPPORTED_VERSION;
                }
                status = niyah_skip_crc(file, &crc, payload_bytes, buffer, sizeof(buffer));
                break;
        }
        if (status != NIYAH_OK) {
            return status;
        }
    }

    if ((required_version == NIYAH_CHECKPOINT_VERSION_V1 &&
         seen != UINT32_C(0x0f)) ||
        (required_version == NIYAH_CHECKPOINT_VERSION_V2 &&
         seen != UINT32_C(0x1f))) {
        return NIYAH_ERR_CORRUPT_DATA;
    }
    status = niyah_read_exact(file, footer, sizeof(footer), &crc, 0);
    if (status != NIYAH_OK) {
        return status;
    }
    if (niyah_load_u32_le(footer + 0U) != NIYAH_CHECKPOINT_CHECKSUM_CRC32) {
        return NIYAH_ERR_UNSUPPORTED_VERSION;
    }
    out->crc32 = niyah_crc32_final(&crc);
    if (niyah_load_u32_le(footer + 4U) != out->crc32) {
        return NIYAH_ERR_CORRUPT_DATA;
    }
    {
        const int trailing = fgetc(file);
        if (trailing != EOF) {
            return NIYAH_ERR_CORRUPT_DATA;
        }
        if (ferror(file) != 0) {
            return NIYAH_ERR_IO;
        }
    }
    if (expected != NULL &&
        (out->crc32 != expected->crc32 ||
         out->meta.step != expected->meta.step ||
         !niyah_optimizer_config_equal(&out->meta.optimizer_config,
                                       &expected->meta.optimizer_config))) {
        return NIYAH_ERR_CORRUPT_DATA;
    }
    return NIYAH_OK;
}

static int niyah_model_output_is_empty(const NiyahModel *model)
{
    return model->weights == NULL && model->weight_count == 0U;
}

static int niyah_optimizer_output_is_empty(const NiyahAdamWState *state)
{
    return state->m == NULL && state->v == NULL && state->count == 0U &&
           state->bound_model == NULL && state->bound_weights == NULL;
}

static NiyahStatus niyah_checkpoint_load_impl(
    const char *path,
    const NiyahTokenizer *tokenizer,
    NiyahModel *out_model,
    NiyahAdamWState *out_optimizer_state,
    NiyahAdamWConfig *out_optimizer_config)
{
    FILE *file;
    NiyahCheckpointScan first;
    NiyahCheckpointScan second;
    NiyahModel temp_model;
    NiyahAdamWState temp_state;
    NiyahCheckpointLoadTargets targets;
    uint8_t tokenizer_identity[NIYAH_TOKENIZER_IDENTITY_SHA256_SIZE];
    uint32_t required_version;
    NiyahStatus status;

    if (path == NULL || path[0] == '\0' || out_model == NULL ||
        out_optimizer_state == NULL || out_optimizer_config == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    if (!niyah_model_output_is_empty(out_model) ||
        !niyah_optimizer_output_is_empty(out_optimizer_state)) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    required_version = tokenizer == NULL
        ? NIYAH_CHECKPOINT_VERSION_V1
        : NIYAH_CHECKPOINT_VERSION_V2;

    if (tokenizer != NULL) {
        status = niyah_tokenizer_identity_sha256(
            tokenizer, tokenizer_identity);
        if (status != NIYAH_OK) {
            return status;
        }
    }

    file = niyah_checkpoint_fopen(path, "rb");
    if (file == NULL) {
        return NIYAH_ERR_IO;
    }
    status = niyah_scan_checkpoint(file, required_version, NULL, NULL, &first);
    if (status != NIYAH_OK) {
        (void)fclose(file);
        return status;
    }
    if (tokenizer != NULL &&
        (niyah_tokenizer_vocab_size(tokenizer) !=
             (size_t)first.header.config.vocab_size ||
         first.has_tokenizer_identity == 0 ||
         memcmp(first.tokenizer_identity, tokenizer_identity,
                NIYAH_TOKENIZER_IDENTITY_SHA256_SIZE) != 0)) {
        (void)fclose(file);
        return NIYAH_ERR_INVALID_CONFIG;
    }
    if (fseek(file, 0L, SEEK_SET) != 0) {
        (void)fclose(file);
        return NIYAH_ERR_IO;
    }
    clearerr(file);

    memset(&temp_model, 0, sizeof(temp_model));
    memset(&temp_state, 0, sizeof(temp_state));
    status = niyah_model_create(&temp_model, &first.header.config);
    if (status != NIYAH_OK) {
        (void)fclose(file);
        return status;
    }
    status = niyah_adamw_state_create(&temp_state, &temp_model);
    if (status != NIYAH_OK) {
        niyah_model_destroy(&temp_model);
        (void)fclose(file);
        return status;
    }

    targets.weights = temp_model.weights;
    targets.m = temp_state.m;
    targets.v = temp_state.v;
    targets.count = temp_model.weight_count;
    status = niyah_scan_checkpoint(file, required_version, &first, &targets, &second);
    if (status == NIYAH_OK) {
        temp_state.step = second.meta.step;
        temp_state.model_config = temp_model.config;
        temp_state.model_layout = temp_model.layout;
    }

    {
        const int close_result = fclose(file);
        if (status == NIYAH_OK && close_result != 0) {
            status = NIYAH_ERR_IO;
        }
    }
    if (status != NIYAH_OK) {
        niyah_adamw_state_destroy(&temp_state);
        niyah_model_destroy(&temp_model);
        return status;
    }

    *out_model = temp_model;
    *out_optimizer_state = temp_state;
    *out_optimizer_config = second.meta.optimizer_config;
    out_optimizer_state->bound_model = out_model;
    out_optimizer_state->bound_weights = out_model->weights;
    out_optimizer_state->model_config = out_model->config;
    out_optimizer_state->model_layout = out_model->layout;
    memset(&temp_model, 0, sizeof(temp_model));
    memset(&temp_state, 0, sizeof(temp_state));
    return NIYAH_OK;
}

NiyahStatus niyah_checkpoint_identity_sha256(
    const char *path,
    uint8_t out_identity[NIYAH_CHECKPOINT_IDENTITY_SHA256_SIZE])
{
    FILE *file;
    NiyahSha256 sha;
    unsigned char buffer[NIYAH_CHECKPOINT_IO_BUFFER_SIZE];
    size_t received;
    int close_result;

    if (path == NULL || path[0] == '\0' || out_identity == NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;
    file = niyah_checkpoint_fopen(path, "rb");
    if (file == NULL)
        return NIYAH_ERR_IO;

    niyah_sha256_init(&sha);
    while ((received = fread(buffer, 1U, sizeof(buffer), file)) != 0U)
        niyah_sha256_update(&sha, buffer, received);

    if (ferror(file) != 0) {
        (void)fclose(file);
        return NIYAH_ERR_IO;
    }

    close_result = fclose(file);
    if (close_result != 0)
        return NIYAH_ERR_IO;

    niyah_sha256_final(&sha, out_identity);
    return NIYAH_OK;
}

NiyahStatus niyah_checkpoint_load(const char *path,
                                  NiyahModel *out_model,
                                  NiyahAdamWState *out_optimizer_state,
                                  NiyahAdamWConfig *out_optimizer_config)
{
    return niyah_checkpoint_load_impl(
        path, NULL, out_model, out_optimizer_state, out_optimizer_config);
}

NiyahStatus niyah_checkpoint_load_with_tokenizer(
    const char *path,
    const NiyahTokenizer *tokenizer,
    NiyahModel *out_model,
    NiyahAdamWState *out_optimizer_state,
    NiyahAdamWConfig *out_optimizer_config)
{
    if (tokenizer == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    return niyah_checkpoint_load_impl(
        path, tokenizer, out_model, out_optimizer_state,
        out_optimizer_config);
}
