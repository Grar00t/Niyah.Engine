/*
 * niyah_gguf.h — GGUF v1/v2/v3 zero-copy loader
 *
 * Loads a GGUF model file via mmap. All tensor data pointers point
 * directly into the mapped region — no extra copies.
 *
 * Zero external dependencies. C11 clean. C++17 compatible.
 */
#ifndef NIYAH_GGUF_H
#define NIYAH_GGUF_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGUF_MAGIC       UINT32_C(0x46554747)  /* "GGUF" LE */
#define GGUF_VERSION_MIN UINT32_C(1)
#define GGUF_VERSION_MAX UINT32_C(3)
#define GGUF_MAX_DIMS    UINT32_C(4)

/* ── GGUF value types ────────────────────────────────────────────── */
typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} gguf_type_t;

/* ── GGML tensor types (subset used by niyah) ────────────────────── */
typedef enum {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q8_0 = 8,
} ggml_type_t;

/* ── String view (points into mmap region) ───────────────────────── */
typedef struct {
    uint64_t    len;
    const char *data;
} gguf_str_t;

static inline bool gguf_str_eq(const gguf_str_t *s, const char *cstr) {
    size_t n = __builtin_strlen(cstr);
    return s->len == (uint64_t)n &&
           __builtin_memcmp(s->data, cstr, n) == 0;
}

/* ── Value union ─────────────────────────────────────────────────── */
typedef struct gguf_value gguf_value_t;
struct gguf_value {
    gguf_type_t type;
    union {
        uint8_t     u8;  int8_t   i8;
        uint16_t    u16; int16_t  i16;
        uint32_t    u32; int32_t  i32;
        uint64_t    u64; int64_t  i64;
        float       f32; double   f64;
        uint8_t     b;
        gguf_str_t  str;
        struct { gguf_type_t elem_type; uint64_t count; const uint8_t *data; } arr;
    };
};

/* ── Key-value metadata entry ────────────────────────────────────── */
typedef struct {
    gguf_str_t  key;
    gguf_value_t value;
} gguf_kv_t;

/* ── Tensor descriptor ───────────────────────────────────────────── */
typedef struct {
    gguf_str_t      name;
    uint32_t        n_dims;
    uint64_t        dims[GGUF_MAX_DIMS];
    ggml_type_t     type;
    uint64_t        offset;   /* byte offset from data section start */
    size_t          nbytes;   /* total bytes for this tensor         */
    const uint8_t  *data;     /* pointer into mmap region            */
} gguf_tensor_info_t;

/* ── Context (owns the mmap) ─────────────────────────────────────── */
typedef struct {
    int                  fd;
    void                *mmap_base;
    size_t               mmap_size;
    uint32_t             version;
    uint64_t             n_tensors;
    uint64_t             n_kv;
    uint32_t             alignment;
    gguf_kv_t           *kv;       /* heap-allocated array[n_kv]      */
    gguf_tensor_info_t  *tensors;  /* heap-allocated array[n_tensors] */
    const uint8_t       *data_base;
} gguf_ctx_t;

/* ── Public API ──────────────────────────────────────────────────── */

/* Load a GGUF file. Returns NULL on error. */
gguf_ctx_t *gguf_load(const char *path);

/* Free all resources (unmaps file, closes fd). */
void gguf_free(gguf_ctx_t *ctx);

/* Look up a metadata key. Returns NULL if not found. */
const gguf_value_t *gguf_get_kv(const gguf_ctx_t *ctx, const char *key);

/* Look up a tensor by name. Returns NULL if not found. */
const gguf_tensor_info_t *gguf_get_tensor(const gguf_ctx_t *ctx, const char *name);

/* Convenience accessors with defaults */
uint32_t   gguf_kv_u32(const gguf_ctx_t *ctx, const char *key, uint32_t def);
float      gguf_kv_f32(const gguf_ctx_t *ctx, const char *key, float    def);
gguf_str_t gguf_kv_str(const gguf_ctx_t *ctx, const char *key);

/* Block sizes for quantized types */
static inline size_t ggml_blksize(ggml_type_t t) {
    switch (t) {
        case GGML_TYPE_Q4_0: return 32;
        case GGML_TYPE_Q8_0: return 32;
        default:             return 1;
    }
}
static inline size_t ggml_type_size(ggml_type_t t) {
    switch (t) {
        case GGML_TYPE_F32:  return 4;
        case GGML_TYPE_F16:  return 2;
        case GGML_TYPE_Q4_0: return 18; /* 2 (fp16 scale) + 16 (nibbles) */
        case GGML_TYPE_Q8_0: return 34; /* 2 (fp16 scale) + 32 (int8)    */
        default:             return 0;
    }
}

#ifdef __cplusplus
}
#endif
#endif /* NIYAH_GGUF_H */
