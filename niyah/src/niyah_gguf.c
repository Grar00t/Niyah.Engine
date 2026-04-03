/*
 * niyah_gguf.c – GGUF v1/v2/v3 zero-copy loader
 *
 * Compile flags: -Wall -Wextra -Werror -O3 -march=native -std=c11
 */

#define _GNU_SOURCE
#include "niyah_gguf.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

/* ── Reader cursor ─────────────────────────────────────────────────── */
typedef struct {
    const uint8_t *base;
    size_t         size;
    size_t         pos;
} reader_t;

static inline bool reader_check(reader_t *r, size_t need) {
    return (r->pos + need) <= r->size;
}

#define READ_SCALAR(r, type, out) do {            \
    if (!reader_check((r), sizeof(type))) {       \
        fprintf(stderr, "gguf: read overrun\n");  \
        return false;                             \
    }                                             \
    memcpy(&(out), (r)->base + (r)->pos, sizeof(type)); \
    (r)->pos += sizeof(type);                     \
} while (0)

static bool read_str(reader_t *r, gguf_str_t *s) {
    uint64_t len;
    READ_SCALAR(r, uint64_t, len);
    if (!reader_check(r, (size_t)len)) return false;
    s->len  = len;
    s->data = (const char *)(r->base + r->pos);
    r->pos += len;
    return true;
}

static bool read_value(reader_t *r, gguf_type_t type, gguf_value_t *v);

static bool read_array(reader_t *r, gguf_value_t *v) {
    uint32_t elem_type_raw;
    READ_SCALAR(r, uint32_t, elem_type_raw);
    uint64_t count;
    READ_SCALAR(r, uint64_t, count);

    v->arr.elem_type = (gguf_type_t)elem_type_raw;
    v->arr.count     = count;
    v->arr.data      = r->base + r->pos;

    /* Skip over the elements without parsing each one */
    /* For simple types we can calculate the skip; for strings we must walk */
    switch (v->arr.elem_type) {
    case GGUF_TYPE_UINT8:  case GGUF_TYPE_INT8:  case GGUF_TYPE_BOOL:
        r->pos += count * 1; break;
    case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16:
        r->pos += count * 2; break;
    case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32:
        r->pos += count * 4; break;
    case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64:
        r->pos += count * 8; break;
    case GGUF_TYPE_STRING:
        for (uint64_t i = 0; i < count; i++) {
            gguf_str_t tmp;
            if (!read_str(r, &tmp)) return false;
        }
        break;
    default:
        fprintf(stderr, "gguf: unsupported array elem type %u\n",
                v->arr.elem_type);
        return false;
    }
    return true;
}

static bool read_value(reader_t *r, gguf_type_t type, gguf_value_t *v) {
    v->type = type;
    switch (type) {
    case GGUF_TYPE_UINT8:   READ_SCALAR(r, uint8_t,  v->u8);  break;
    case GGUF_TYPE_INT8:    READ_SCALAR(r, int8_t,   v->i8);  break;
    case GGUF_TYPE_UINT16:  READ_SCALAR(r, uint16_t, v->u16); break;
    case GGUF_TYPE_INT16:   READ_SCALAR(r, int16_t,  v->i16); break;
    case GGUF_TYPE_UINT32:  READ_SCALAR(r, uint32_t, v->u32); break;
    case GGUF_TYPE_INT32:   READ_SCALAR(r, int32_t,  v->i32); break;
    case GGUF_TYPE_FLOAT32: READ_SCALAR(r, float,    v->f32); break;
    case GGUF_TYPE_BOOL:    READ_SCALAR(r, uint8_t,  v->b);   break;
    case GGUF_TYPE_STRING:  if (!read_str(r, &v->str)) return false; break;
    case GGUF_TYPE_ARRAY:   if (!read_array(r, v))   return false; break;
    case GGUF_TYPE_UINT64:  READ_SCALAR(r, uint64_t, v->u64); break;
    case GGUF_TYPE_INT64:   READ_SCALAR(r, int64_t,  v->i64); break;
    case GGUF_TYPE_FLOAT64: READ_SCALAR(r, double,   v->f64); break;
    default:
        fprintf(stderr, "gguf: unknown value type %u\n", type);
        return false;
    }
    return true;
}

/* ── Public API ────────────────────────────────────────────────────── */

gguf_ctx_t *gguf_load(const char *path) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        perror("gguf_load: open");
        return NULL;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        perror("gguf_load: fstat");
        close(fd);
        return NULL;
    }

    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ,
                     MAP_PRIVATE | MAP_POPULATE, fd, 0);
    if (map == MAP_FAILED) {
        perror("gguf_load: mmap");
        close(fd);
        return NULL;
    }

    /* Advise sequential + willneed for metadata region */
    madvise(map, (size_t)st.st_size, MADV_SEQUENTIAL);

    reader_t r = {
        .base = (const uint8_t *)map,
        .size = (size_t)st.st_size,
        .pos  = 0
    };

    /* Header */
    uint32_t magic;
    READ_SCALAR(&r, uint32_t, magic);
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "gguf_load: bad magic 0x%08x\n", magic);
        goto err;
    }

    uint32_t version;
    READ_SCALAR(&r, uint32_t, version);
    if (version < GGUF_VERSION_MIN || version > GGUF_VERSION_MAX) {
        fprintf(stderr, "gguf_load: unsupported version %u\n", version);
        goto err;
    }

    uint64_t n_tensors, n_kv;
    READ_SCALAR(&r, uint64_t, n_tensors);
    READ_SCALAR(&r, uint64_t, n_kv);

    gguf_ctx_t *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) goto err;
    ctx->fd        = fd;
    ctx->mmap_base = map;
    ctx->mmap_size = (size_t)st.st_size;
    ctx->version   = version;
    ctx->n_tensors = n_tensors;
    ctx->n_kv      = n_kv;
    ctx->alignment = 32;

    /* Parse metadata KV pairs */
    ctx->kv = calloc(n_kv, sizeof(gguf_kv_t));
    if (!ctx->kv) goto err_ctx;

    for (uint64_t i = 0; i < n_kv; i++) {
        gguf_kv_t *kv = &ctx->kv[i];
        if (!read_str(&r, &kv->key)) goto err_ctx;
        uint32_t vtype;
        READ_SCALAR(&r, uint32_t, vtype);
        if (!read_value(&r, (gguf_type_t)vtype, &kv->value)) goto err_ctx;

        /* Check for alignment override */
        if (gguf_str_eq(&kv->key, "general.alignment") &&
            kv->value.type == GGUF_TYPE_UINT32) {
            ctx->alignment = kv->value.u32;
        }
    }

    /* Parse tensor info */
    ctx->tensors = calloc(n_tensors, sizeof(gguf_tensor_info_t));
    if (!ctx->tensors) goto err_ctx;

    for (uint64_t i = 0; i < n_tensors; i++) {
        gguf_tensor_info_t *ti = &ctx->tensors[i];
        if (!read_str(&r, &ti->name)) goto err_ctx;
        READ_SCALAR(&r, uint32_t, ti->n_dims);
        if (ti->n_dims > GGUF_MAX_DIMS) goto err_ctx;
        for (uint32_t d = 0; d < ti->n_dims; d++)
            READ_SCALAR(&r, uint64_t, ti->dims[d]);
        uint32_t ttype;
        READ_SCALAR(&r, uint32_t, ttype);
        ti->type = (ggml_type_t)ttype;
        READ_SCALAR(&r, uint64_t, ti->offset);
    }

    /* Align data section start */
    size_t align = ctx->alignment;
    size_t data_start = (r.pos + align - 1) & ~(align - 1);
    ctx->data_base = (const uint8_t *)map + data_start;

    /* Compute tensor sizes and pointers */
    for (uint64_t i = 0; i < n_tensors; i++) {
        gguf_tensor_info_t *ti = &ctx->tensors[i];
        uint64_t nelems = 1;
        for (uint32_t d = 0; d < ti->n_dims; d++)
            nelems *= ti->dims[d];
        size_t blk = ggml_blksize(ti->type);
        size_t tsz = ggml_type_size(ti->type);
        ti->nbytes = (nelems / blk) * tsz;
        ti->data   = (const uint8_t *)ctx->data_base + ti->offset;
    }

    /* Switch madvise to random for random-access tensor reads */
    madvise(map, (size_t)st.st_size, MADV_RANDOM);

    return ctx;

err_ctx:
    free(ctx->kv);
    free(ctx->tensors);
    free(ctx);
err:
    munmap(map, (size_t)st.st_size);
    close(fd);
    return NULL;
}

void gguf_free(gguf_ctx_t *ctx) {
    if (!ctx) return;
    munmap(ctx->mmap_base, ctx->mmap_size);
    close(ctx->fd);
    free(ctx->kv);
    free(ctx->tensors);
    free(ctx);
}

const gguf_value_t *gguf_get_kv(const gguf_ctx_t *ctx, const char *key) {
    size_t klen = strlen(key);
    for (uint64_t i = 0; i < ctx->n_kv; i++) {
        const gguf_kv_t *kv = &ctx->kv[i];
        if (kv->key.len == klen &&
            memcmp(kv->key.data, key, klen) == 0)
            return &kv->value;
    }
    return NULL;
}

const gguf_tensor_info_t *gguf_get_tensor(const gguf_ctx_t *ctx,
                                           const char *name) {
    size_t nlen = strlen(name);
    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        const gguf_tensor_info_t *ti = &ctx->tensors[i];
        if (ti->name.len == nlen &&
            memcmp(ti->name.data, name, nlen) == 0)
            return ti;
    }
    return NULL;
}

uint32_t gguf_kv_u32(const gguf_ctx_t *ctx, const char *key, uint32_t def) {
    const gguf_value_t *v = gguf_get_kv(ctx, key);
    if (!v || v->type != GGUF_TYPE_UINT32) return def;
    return v->u32;
}

float gguf_kv_f32(const gguf_ctx_t *ctx, const char *key, float def) {
    const gguf_value_t *v = gguf_get_kv(ctx, key);
    if (!v || v->type != GGUF_TYPE_FLOAT32) return def;
    return v->f32;
}

gguf_str_t gguf_kv_str(const gguf_ctx_t *ctx, const char *key) {
    gguf_str_t empty = {0, NULL};
    const gguf_value_t *v = gguf_get_kv(ctx, key);
    if (!v || v->type != GGUF_TYPE_STRING) return empty;
    return v->str;
}
