#include "niyah_runtime.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/*
 * Was a stub: niyah_runtime.h declared create/destroy but nothing defined them.
 *
 * Bump allocator over a single pool. Inference does thousands of short-lived
 * tensor allocations per token; a reset-per-token arena avoids malloc churn
 * and makes peak memory a hard, observable number.
 */

#define NIYAH_ARENA_ALIGN 64u
#define NIYAH_ARENA_DEFAULT (64u * 1024u * 1024u)

typedef struct {
    unsigned char* base;
    size_t         capacity;
    size_t         used;
    bool           owns_memory;
} NiyahArena;

NiyahStatus niyah_runtime_init_inplace(NiyahRuntime* runtime,
                                       const NiyahRuntimeConfig* config)
{
    if (!runtime || runtime->context) {
        return NIYAH_ERR_INVALID_ARG;
    }

    NiyahRuntimeConfig resolved;
    if (config) {
        resolved = *config;
    } else {
        memset(&resolved, 0, sizeof(resolved));
    }

    if (resolved.memory_size == 0) {
        resolved.memory_size = NIYAH_ARENA_DEFAULT;
    }

    if (resolved.use_gpu) {
        resolved.use_gpu = false;
    }

    NiyahArena* arena = (NiyahArena*)calloc(1, sizeof(NiyahArena));
    if (!arena) {
        return NIYAH_ERR_OUT_OF_MEMORY;
    }

    if (resolved.memory_pool) {
        arena->base = (unsigned char*)resolved.memory_pool;
        arena->owns_memory = false;
    } else {
        arena->base = (unsigned char*)malloc(resolved.memory_size);
        if (!arena->base) {
            free(arena);
            return NIYAH_ERR_OUT_OF_MEMORY;
        }
        arena->owns_memory = true;
        resolved.memory_pool = arena->base;
    }

    arena->capacity = resolved.memory_size;
    arena->used = 0;

    runtime->config = resolved;
    runtime->context = arena;
    return NIYAH_OK;
}

void niyah_runtime_deinit_inplace(NiyahRuntime* runtime)
{
    if (!runtime) {
        return;
    }

    NiyahArena* arena = (NiyahArena*)runtime->context;
    if (arena) {
        if (arena->owns_memory) {
            free(arena->base);
        }
        free(arena);
    }

    memset(runtime, 0, sizeof(*runtime));
}

NiyahRuntime* niyah_runtime_create(const NiyahRuntimeConfig* config)
{
    NiyahRuntime* runtime = (NiyahRuntime*)calloc(1, sizeof(NiyahRuntime));
    if (!runtime) {
        return NULL;
    }

    if (niyah_runtime_init_inplace(runtime, config) != NIYAH_OK) {
        free(runtime);
        return NULL;
    }

    return runtime;
}

void niyah_runtime_destroy(NiyahRuntime* runtime)
{
    if (!runtime) {
        return;
    }
    niyah_runtime_deinit_inplace(runtime);
    free(runtime);
}

void* niyah_runtime_alloc(NiyahRuntime* runtime, size_t bytes)
{
    if (!runtime || !runtime->context || bytes == 0) {
        return NULL;
    }

    NiyahArena* arena = (NiyahArena*)runtime->context;

    const uintptr_t current =
        (uintptr_t)(arena->base + arena->used);
    const size_t misalign =
        (size_t)(current % (uintptr_t)NIYAH_ARENA_ALIGN);
    const size_t padding =
        misalign ? (NIYAH_ARENA_ALIGN - misalign) : 0u;

    if (padding > arena->capacity - arena->used) {
        return NULL;
    }
    const size_t offset = arena->used + padding;
    if (bytes > arena->capacity - offset) {
        return NULL;
    }

    void* ptr = arena->base + offset;
    arena->used = offset + bytes;
    return ptr;
}

float* niyah_runtime_alloc_floats(NiyahRuntime* runtime, size_t count)
{
    if (count == 0 || count > SIZE_MAX / sizeof(float)) {
        return NULL;
    }
    float* ptr = (float*)niyah_runtime_alloc(runtime, count * sizeof(float));
    if (ptr) {
        memset(ptr, 0, count * sizeof(float));
    }
    return ptr;
}

void niyah_runtime_reset(NiyahRuntime* runtime)
{
    if (!runtime || !runtime->context) {
        return;
    }
    ((NiyahArena*)runtime->context)->used = 0;
}

NiyahStatus niyah_runtime_rewind(NiyahRuntime* runtime, size_t used)
{
    if (!runtime || !runtime->context) {
        return NIYAH_ERR_INVALID_ARG;
    }

    NiyahArena* arena = (NiyahArena*)runtime->context;
    if (used > arena->used) {
        return NIYAH_ERR_INVALID_ARG;
    }

    arena->used = used;
    return NIYAH_OK;
}

size_t niyah_runtime_used(const NiyahRuntime* runtime)
{
    if (!runtime || !runtime->context) {
        return 0;
    }
    return ((const NiyahArena*)runtime->context)->used;
}

size_t niyah_runtime_capacity(const NiyahRuntime* runtime)
{
    if (!runtime || !runtime->context) {
        return 0;
    }
    return ((const NiyahArena*)runtime->context)->capacity;
}
