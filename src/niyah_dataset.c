#include "niyah/dataset.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(CHAR_BIT == 8, "dataset cursor persistence requires 8-bit bytes");

#define NIYAH_DATASET_CURSOR_VERSION_V1 UINT32_C(1)
#define NIYAH_DATASET_CURSOR_VERSION_V2 UINT32_C(2)
#define NIYAH_DATASET_CURSOR_VERSION_V3 UINT32_C(3)
#define NIYAH_DATASET_CURSOR_FLAGS_V1 UINT32_C(0)
#define NIYAH_DATASET_CURSOR_FLAG_DATASET_IDENTITY UINT32_C(1)
#define NIYAH_DATASET_CURSOR_FLAG_CHECKPOINT_IDENTITY UINT32_C(2)
#define NIYAH_DATASET_CURSOR_CHECKSUM_CRC32 UINT32_C(1)

static const unsigned char NIYAH_DATASET_CURSOR_MAGIC[8] = {
    'N','I','Y','A','H','D','S','T'
};

typedef struct NiyahDatasetCrc32 {
    uint32_t value;
    uint32_t table[256];
} NiyahDatasetCrc32;

static FILE *niyah_dataset_fopen(const char *path, const char *mode)
{
#if defined(_MSC_VER)
    FILE *file = NULL;
    if (fopen_s(&file, path, mode) != 0) return NULL;
    return file;
#else
    return fopen(path, mode);
#endif
}

static uint64_t mix64(uint64_t x)
{
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

static uint64_t prng_next(uint64_t *state)
{
    *state = mix64(*state);
    return *state;
}

static uint64_t bounded(uint64_t *state, uint64_t bound)
{
    uint64_t x, limit;
    if (bound <= UINT64_C(1)) return UINT64_C(0);
    limit = UINT64_MAX - (UINT64_MAX % bound);
    do { x = prng_next(state); } while (x >= limit);
    return x % bound;
}

static NiyahStatus build_order(NiyahDatasetCursor *cursor)
{
    uint64_t state;
    size_t i;

    if (cursor == NULL || cursor->order == NULL || cursor->sample_count == 0U)
        return NIYAH_ERR_INVALID_CONFIG;

    for (i = 0U; i < cursor->sample_count; ++i)
        cursor->order[i] = i;

    state = cursor->seed ^ UINT64_C(0x4e495941485f4453) ^ mix64(cursor->epoch);

    for (i = cursor->sample_count; i > 1U; --i) {
        size_t a = i - 1U;
        size_t j = (size_t)bounded(&state, (uint64_t)i);
        size_t tmp = cursor->order[a];
        cursor->order[a] = cursor->order[j];
        cursor->order[j] = tmp;
    }
    return NIYAH_OK;
}

static NiyahStatus validate_cursor(const NiyahDatasetCursor *cursor)
{
    if (cursor == NULL) return NIYAH_ERR_INVALID_ARGUMENT;
    if (cursor->sample_count == 0U || cursor->order == NULL ||
        cursor->position > cursor->sample_count ||
        (cursor->has_dataset_identity != 0 &&
         cursor->has_dataset_identity != 1) ||
        (cursor->has_checkpoint_identity != 0 &&
         cursor->has_checkpoint_identity != 1) ||
        (cursor->has_checkpoint_identity != 0 &&
         cursor->has_dataset_identity == 0))
        return NIYAH_ERR_INVALID_CONFIG;
    return NIYAH_OK;
}

NiyahStatus niyah_dataset_cursor_init(NiyahDatasetCursor *cursor,
                                      size_t sample_count,
                                      uint64_t seed)
{
    NiyahStatus status;

    if (cursor == NULL) return NIYAH_ERR_INVALID_ARGUMENT;
    if (cursor->order != NULL) return NIYAH_ERR_INVALID_ARGUMENT;
    if (sample_count == 0U) return NIYAH_ERR_INVALID_CONFIG;
    if (sample_count > SIZE_MAX / sizeof(size_t)) return NIYAH_ERR_OVERFLOW;

    cursor->order = (size_t *)malloc(sample_count * sizeof(size_t));
    if (cursor->order == NULL) return NIYAH_ERR_OUT_OF_MEMORY;

    cursor->sample_count = sample_count;
    cursor->position = 0U;
    cursor->seed = seed;
    cursor->epoch = UINT64_C(0);
    memset(cursor->dataset_identity, 0, sizeof(cursor->dataset_identity));
    cursor->has_dataset_identity = 0;
    memset(cursor->checkpoint_identity, 0, sizeof(cursor->checkpoint_identity));
    cursor->has_checkpoint_identity = 0;

    status = build_order(cursor);
    if (status != NIYAH_OK) {
        niyah_dataset_cursor_destroy(cursor);
        return status;
    }
    return NIYAH_OK;
}

void niyah_dataset_cursor_destroy(NiyahDatasetCursor *cursor)
{
    if (cursor == NULL) return;
    free(cursor->order);
    memset(cursor, 0, sizeof(*cursor));
}

NiyahStatus niyah_dataset_cursor_next(NiyahDatasetCursor *cursor,
                                      size_t *out_sample_index)
{
    NiyahStatus status;
    if (out_sample_index == NULL) return NIYAH_ERR_INVALID_ARGUMENT;

    status = validate_cursor(cursor);
    if (status != NIYAH_OK) return status;

    if (cursor->position == cursor->sample_count) {
        if (cursor->epoch == UINT64_MAX) return NIYAH_ERR_OVERFLOW;
        cursor->epoch += UINT64_C(1);
        cursor->position = 0U;
        status = build_order(cursor);
        if (status != NIYAH_OK) return status;
    }

    *out_sample_index = cursor->order[cursor->position++];
    return NIYAH_OK;
}

static void store_u32_le(unsigned char out[4], uint32_t v)
{
    out[0]=(unsigned char)(v);
    out[1]=(unsigned char)(v>>8);
    out[2]=(unsigned char)(v>>16);
    out[3]=(unsigned char)(v>>24);
}

static void store_u64_le(unsigned char out[8], uint64_t v)
{
    size_t i;
    for (i=0U;i<8U;++i) { out[i]=(unsigned char)(v & UINT64_C(0xff)); v >>= 8U; }
}

static uint32_t load_u32_le(const unsigned char in[4])
{
    return (uint32_t)in[0] | ((uint32_t)in[1]<<8) |
           ((uint32_t)in[2]<<16) | ((uint32_t)in[3]<<24);
}

static uint64_t load_u64_le(const unsigned char in[8])
{
    uint64_t v=UINT64_C(0); size_t i;
    for (i=0U;i<8U;++i) v |= (uint64_t)in[i] << (8U*i);
    return v;
}

static void crc_init(NiyahDatasetCrc32 *crc)
{
    uint32_t i;
    for (i=0U;i<UINT32_C(256);++i) {
        uint32_t c=i; unsigned bit;
        for (bit=0U;bit<8U;++bit)
            c = (c & 1U) ? UINT32_C(0xedb88320) ^ (c>>1) : c>>1;
        crc->table[i]=c;
    }
    crc->value=UINT32_C(0xffffffff);
}

static void crc_update(NiyahDatasetCrc32 *crc, const unsigned char *data, size_t size)
{
    size_t i;
    for (i=0U;i<size;++i) {
        uint32_t idx=(crc->value ^ (uint32_t)data[i]) & UINT32_C(0xff);
        crc->value=crc->table[idx] ^ (crc->value>>8);
    }
}

static uint32_t crc_final(const NiyahDatasetCrc32 *crc)
{
    return crc->value ^ UINT32_C(0xffffffff);
}

NiyahStatus niyah_dataset_cursor_seek(NiyahDatasetCursor *cursor,
                                      uint64_t epoch,
                                      size_t position)
{
    uint64_t old_epoch;
    size_t old_position;
    NiyahStatus status;

    status = validate_cursor(cursor);
    if (status != NIYAH_OK) return status;
    if (position > cursor->sample_count) return NIYAH_ERR_INVALID_ARGUMENT;

    old_epoch = cursor->epoch;
    old_position = cursor->position;

    cursor->epoch = epoch;
    cursor->position = position;
    status = build_order(cursor);
    if (status != NIYAH_OK) {
        cursor->epoch = old_epoch;
        cursor->position = old_position;
        (void)build_order(cursor);
        return status;
    }

    return NIYAH_OK;
}

NiyahStatus niyah_dataset_cursor_bind_identity(
    NiyahDatasetCursor *cursor,
    const uint8_t identity[NIYAH_DATASET_IDENTITY_SHA256_SIZE])
{
    NiyahStatus status;

    if (identity == NULL) return NIYAH_ERR_INVALID_ARGUMENT;
    status = validate_cursor(cursor);
    if (status != NIYAH_OK) return status;

    memcpy(cursor->dataset_identity, identity, sizeof(cursor->dataset_identity));
    cursor->has_dataset_identity = 1;
    return NIYAH_OK;
}

NiyahStatus niyah_dataset_cursor_bind_checkpoint_identity(
    NiyahDatasetCursor *cursor,
    const uint8_t identity[NIYAH_DATASET_CHECKPOINT_IDENTITY_SHA256_SIZE])
{
    NiyahStatus status;

    if (identity == NULL) return NIYAH_ERR_INVALID_ARGUMENT;
    status = validate_cursor(cursor);
    if (status != NIYAH_OK) return status;
    if (cursor->has_dataset_identity == 0)
        return NIYAH_ERR_INVALID_CONFIG;

    memcpy(cursor->checkpoint_identity, identity,
           sizeof(cursor->checkpoint_identity));
    cursor->has_checkpoint_identity = 1;
    return NIYAH_OK;
}

NiyahStatus niyah_dataset_cursor_save(const NiyahDatasetCursor *cursor,
                                      const char *path)
{
    FILE *file;
    unsigned char header[112], footer[8];
    size_t header_size;
    uint32_t version;
    uint32_t flags;
    NiyahDatasetCrc32 crc;
    NiyahStatus status;
    int close_result;

    if (path == NULL || path[0] == '\0') return NIYAH_ERR_INVALID_ARGUMENT;
    status = validate_cursor(cursor);
    if (status != NIYAH_OK) return status;
    if (cursor->sample_count > (size_t)UINT64_MAX ||
        cursor->position > (size_t)UINT64_MAX)
        return NIYAH_ERR_OVERFLOW;

    if (cursor->has_checkpoint_identity != 0) {
        version = NIYAH_DATASET_CURSOR_VERSION_V3;
        flags = NIYAH_DATASET_CURSOR_FLAG_DATASET_IDENTITY |
                NIYAH_DATASET_CURSOR_FLAG_CHECKPOINT_IDENTITY;
        header_size = sizeof(header);
    } else if (cursor->has_dataset_identity != 0) {
        version = NIYAH_DATASET_CURSOR_VERSION_V2;
        flags = NIYAH_DATASET_CURSOR_FLAG_DATASET_IDENTITY;
        header_size = 80U;
    } else {
        version = NIYAH_DATASET_CURSOR_VERSION_V1;
        flags = NIYAH_DATASET_CURSOR_FLAGS_V1;
        header_size = 48U;
    }

    memcpy(header, NIYAH_DATASET_CURSOR_MAGIC, 8U);
    store_u32_le(header+8U, version);
    store_u32_le(header+12U, flags);
    store_u64_le(header+16U, (uint64_t)cursor->sample_count);
    store_u64_le(header+24U, cursor->seed);
    store_u64_le(header+32U, cursor->epoch);
    store_u64_le(header+40U, (uint64_t)cursor->position);
    if (version == NIYAH_DATASET_CURSOR_VERSION_V2 ||
        version == NIYAH_DATASET_CURSOR_VERSION_V3)
        memcpy(header+48U, cursor->dataset_identity,
               NIYAH_DATASET_IDENTITY_SHA256_SIZE);
    if (version == NIYAH_DATASET_CURSOR_VERSION_V3)
        memcpy(header+80U, cursor->checkpoint_identity,
               NIYAH_DATASET_CHECKPOINT_IDENTITY_SHA256_SIZE);

    crc_init(&crc);
    crc_update(&crc, header, header_size);
    store_u32_le(footer, NIYAH_DATASET_CURSOR_CHECKSUM_CRC32);
    store_u32_le(footer+4U, crc_final(&crc));

    file = niyah_dataset_fopen(path, "wb");
    if (file == NULL) return NIYAH_ERR_IO;

    status = fwrite(header,1U,header_size,file)==header_size ? NIYAH_OK : NIYAH_ERR_IO;
    if (status == NIYAH_OK)
        status = fwrite(footer,1U,sizeof(footer),file)==sizeof(footer) ? NIYAH_OK : NIYAH_ERR_IO;
    if (status == NIYAH_OK && fflush(file) != 0) status = NIYAH_ERR_IO;

    close_result=fclose(file);
    if (status == NIYAH_OK && close_result != 0) status=NIYAH_ERR_IO;
    if (status != NIYAH_OK) (void)remove(path);
    return status;
}

NiyahStatus niyah_dataset_cursor_load(const char *path,
                                      NiyahDatasetCursor *out_cursor)
{
    FILE *file;
    unsigned char header[112], footer[8], extra;
    size_t header_size;
    NiyahDatasetCrc32 crc;
    NiyahDatasetCursor temp;
    uint64_t count64, pos64;
    uint32_t version, flags;
    NiyahStatus status;
    int close_result;

    if (path == NULL || path[0] == '\0' || out_cursor == NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;
    if (out_cursor->order != NULL) return NIYAH_ERR_INVALID_ARGUMENT;

    memset(&temp,0,sizeof(temp));
    file=niyah_dataset_fopen(path,"rb");
    if (file==NULL) return NIYAH_ERR_IO;

    if (fread(header,1U,48U,file)!=48U) {
        status=ferror(file)?NIYAH_ERR_IO:NIYAH_ERR_CORRUPT_DATA; goto done;
    }

    if (memcmp(header,NIYAH_DATASET_CURSOR_MAGIC,8U)!=0) {
        status=NIYAH_ERR_CORRUPT_DATA; goto done;
    }

    version=load_u32_le(header+8U);
    flags=load_u32_le(header+12U);
    count64=load_u64_le(header+16U);
    pos64=load_u64_le(header+40U);

    if (version == NIYAH_DATASET_CURSOR_VERSION_V1) {
        if (flags != NIYAH_DATASET_CURSOR_FLAGS_V1) {
            status=NIYAH_ERR_CORRUPT_DATA; goto done;
        }
        header_size=48U;
    } else if (version == NIYAH_DATASET_CURSOR_VERSION_V2) {
        if (flags != NIYAH_DATASET_CURSOR_FLAG_DATASET_IDENTITY) {
            status=NIYAH_ERR_CORRUPT_DATA; goto done;
        }
        if (fread(header+48U,1U,NIYAH_DATASET_IDENTITY_SHA256_SIZE,file) !=
            NIYAH_DATASET_IDENTITY_SHA256_SIZE) {
            status=ferror(file)?NIYAH_ERR_IO:NIYAH_ERR_CORRUPT_DATA; goto done;
        }
        header_size=80U;
    } else if (version == NIYAH_DATASET_CURSOR_VERSION_V3) {
        if (flags != (NIYAH_DATASET_CURSOR_FLAG_DATASET_IDENTITY |
                      NIYAH_DATASET_CURSOR_FLAG_CHECKPOINT_IDENTITY)) {
            status=NIYAH_ERR_CORRUPT_DATA; goto done;
        }
        if (fread(header+48U,1U,
                  NIYAH_DATASET_IDENTITY_SHA256_SIZE +
                  NIYAH_DATASET_CHECKPOINT_IDENTITY_SHA256_SIZE,file) !=
            NIYAH_DATASET_IDENTITY_SHA256_SIZE +
            NIYAH_DATASET_CHECKPOINT_IDENTITY_SHA256_SIZE) {
            status=ferror(file)?NIYAH_ERR_IO:NIYAH_ERR_CORRUPT_DATA; goto done;
        }
        header_size=112U;
    } else {
        status=NIYAH_ERR_UNSUPPORTED_VERSION; goto done;
    }

    if (count64==UINT64_C(0) ||
        count64>(uint64_t)SIZE_MAX ||
        pos64>count64 ||
        pos64>(uint64_t)SIZE_MAX) {
        status=NIYAH_ERR_CORRUPT_DATA; goto done;
    }

    if (fread(footer,1U,sizeof(footer),file)!=sizeof(footer)) {
        status=ferror(file)?NIYAH_ERR_IO:NIYAH_ERR_CORRUPT_DATA; goto done;
    }

    crc_init(&crc);
    crc_update(&crc,header,header_size);
    if (load_u32_le(footer)!=NIYAH_DATASET_CURSOR_CHECKSUM_CRC32 ||
        load_u32_le(footer+4U)!=crc_final(&crc)) {
        status=NIYAH_ERR_CORRUPT_DATA; goto done;
    }

    if (fread(&extra,1U,1U,file)!=0U) {
        status=NIYAH_ERR_CORRUPT_DATA; goto done;
    }
    if (ferror(file)) { status=NIYAH_ERR_IO; goto done; }

    status=niyah_dataset_cursor_init(&temp,(size_t)count64,load_u64_le(header+24U));
    if (status!=NIYAH_OK) goto done;

    temp.epoch=load_u64_le(header+32U);
    temp.position=(size_t)pos64;
    if (version == NIYAH_DATASET_CURSOR_VERSION_V2 ||
        version == NIYAH_DATASET_CURSOR_VERSION_V3) {
        memcpy(temp.dataset_identity,header+48U,
               NIYAH_DATASET_IDENTITY_SHA256_SIZE);
        temp.has_dataset_identity=1;
    }
    if (version == NIYAH_DATASET_CURSOR_VERSION_V3) {
        memcpy(temp.checkpoint_identity,header+80U,
               NIYAH_DATASET_CHECKPOINT_IDENTITY_SHA256_SIZE);
        temp.has_checkpoint_identity=1;
    }
    status=build_order(&temp);

done:
    close_result=fclose(file);
    if (status==NIYAH_OK && close_result!=0) status=NIYAH_ERR_IO;
    if (status!=NIYAH_OK) {
        niyah_dataset_cursor_destroy(&temp);
        return status;
    }
    *out_cursor=temp;
    return NIYAH_OK;
}
