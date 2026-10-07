#ifndef NIYAH_CORE_H
#define NIYAH_CORE_H

#include <stddef.h>
#include <stdint.h>

#include "niyah.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Single-pool memory allocator.
 *
 * All Niyah runtime scratch memory is carved from one caller-owned arena.
 * The allocator performs no runtime malloc/free; ownership stays explicit:
 * the caller provides the backing storage and the pool hands out bump-
 * allocated regions with overflow-checked alignment math.
 *
 * Deterministic by construction: identical allocation sequences on identical
 * pools yield identical addresses; the allocator holds no hidden state.
 */

#define NIYAH_POOL_MAX_ALIGN ((size_t)16U)

typedef struct NiyahPool {
    unsigned char *base;
    size_t capacity;
    size_t offset;
} NiyahPool;

/* Compute the byte size rounded up to the pool's mandatory alignment.
 * Returns NIYAH_ERR_OVERFLOW when the rounded size cannot be represented
 * in size_t, or NIYAH_ERR_INVALID_ARGUMENT when size is zero. */
NiyahStatus niyah_pool_align_up(size_t size, size_t *aligned);

/* Bind a pool to caller-owned storage. The base must be aligned to at least
 * NIYAH_POOL_MAX_ALIGN and capacity must be nonzero. Does not take ownership:
 * the caller retains responsibility for the backing storage's lifetime. */
NiyahStatus niyah_pool_init(NiyahPool *pool, void *storage, size_t capacity);

/* Reset the bump offset to zero. Previously handed-out regions become
 * invalid; the caller must guarantee no live references into the pool. */
NiyahStatus niyah_pool_reset(NiyahPool *pool);

/* Bytes still available for allocation. */
NiyahStatus niyah_pool_remaining(const NiyahPool *pool, size_t *remaining);

/* Bump-allocate an aligned region. On success out holds the region start and
 * out_size the actual aligned byte size reserved. Fails closed with
 * NIYAH_ERR_OUT_OF_MEMORY or NIYAH_ERR_OVERFLOW without mutating the
 * pool. Zero-size allocations are rejected. */
NiyahStatus niyah_pool_alloc(NiyahPool *pool, size_t size, void **out, size_t *out_size);

/* Convenience: allocate an array of count elements of elem_size bytes,
 * rejecting count == 0 and any multiplication overflow. */
NiyahStatus niyah_pool_alloc_array(NiyahPool *pool,
                                   size_t count,
                                   size_t elem_size,
                                   void **out,
                                   size_t *out_size);

#ifdef __cplusplus
}
#endif

#endif
