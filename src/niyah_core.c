#include "niyah/niyah_core.h"

#include <stdint.h>

NiyahStatus niyah_pool_align_up(size_t size, size_t *aligned)
{
    if (aligned == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    *aligned = 0U;
    if (size == 0U) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    if (size > SIZE_MAX - (NIYAH_POOL_MAX_ALIGN - 1U)) {
        return NIYAH_ERR_OVERFLOW;
    }
    size_t rounded = size + (NIYAH_POOL_MAX_ALIGN - 1U);
    rounded &= ~(NIYAH_POOL_MAX_ALIGN - 1U);
    if (rounded < size) {
        return NIYAH_ERR_OVERFLOW;
    }
    *aligned = rounded;
    return NIYAH_OK;
}

NiyahStatus niyah_pool_init(NiyahPool *pool, void *storage, size_t capacity)
{
    if (pool == NULL || storage == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    pool->base = NULL;
    pool->capacity = 0U;
    pool->offset = 0U;
    if (capacity == 0U) {
        return NIYAH_ERR_INVALID_CONFIG;
    }
    uintptr_t addr = (uintptr_t)storage;
    if ((addr & (NIYAH_POOL_MAX_ALIGN - 1U)) != 0U) {
        return NIYAH_ERR_INVALID_CONFIG;
    }
    pool->base = (unsigned char *)storage;
    pool->capacity = capacity;
    pool->offset = 0U;
    return NIYAH_OK;
}

NiyahStatus niyah_pool_reset(NiyahPool *pool)
{
    if (pool == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    pool->offset = 0U;
    return NIYAH_OK;
}

NiyahStatus niyah_pool_remaining(const NiyahPool *pool, size_t *remaining)
{
    if (pool == NULL || remaining == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    if (pool->offset > pool->capacity) {
        return NIYAH_ERR_OVERFLOW;
    }
    *remaining = pool->capacity - pool->offset;
    return NIYAH_OK;
}

NiyahStatus niyah_pool_alloc(NiyahPool *pool, size_t size, void **out, size_t *out_size)
{
    if (pool == NULL || out == NULL || out_size == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    *out = NULL;
    *out_size = 0U;
    size_t aligned = 0U;
    NiyahStatus status = niyah_pool_align_up(size, &aligned);
    if (status != NIYAH_OK) {
        return status;
    }
    if (pool->base == NULL || pool->capacity == 0U) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    if (pool->offset > pool->capacity) {
        return NIYAH_ERR_OVERFLOW;
    }
    if (aligned > pool->capacity - pool->offset) {
        return NIYAH_ERR_OUT_OF_MEMORY;
    }
    *out = pool->base + pool->offset;
    *out_size = aligned;
    pool->offset += aligned;
    return NIYAH_OK;
}

NiyahStatus niyah_pool_alloc_array(NiyahPool *pool,
                                   size_t count,
                                   size_t elem_size,
                                   void **out,
                                   size_t *out_size)
{
    if (pool == NULL || out == NULL || out_size == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    *out = NULL;
    *out_size = 0U;
    if (count == 0U || elem_size == 0U) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    if (elem_size > SIZE_MAX / count) {
        return NIYAH_ERR_OVERFLOW;
    }
    return niyah_pool_alloc(pool, count * elem_size, out, out_size);
}
