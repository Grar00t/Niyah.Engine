#include "niyah/niyah_core.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(expr)                                                                       \
    do {                                                                                  \
        if (!(expr)) {                                                                    \
            fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr);     \
            return 1;                                                                     \
        }                                                                                 \
    } while (0)

static int test_align_up(void)
{
    size_t aligned = 0U;
    CHECK(niyah_pool_align_up(1U, &aligned) == NIYAH_OK && aligned == 16U);
    CHECK(niyah_pool_align_up(16U, &aligned) == NIYAH_OK && aligned == 16U);
    CHECK(niyah_pool_align_up(17U, &aligned) == NIYAH_OK && aligned == 32U);
    CHECK(niyah_pool_align_up(0U, &aligned) == NIYAH_ERR_INVALID_ARGUMENT);
    CHECK(niyah_pool_align_up(SIZE_MAX, &aligned) == NIYAH_ERR_OVERFLOW);
    CHECK(niyah_pool_align_up(SIZE_MAX - 8U, &aligned) == NIYAH_ERR_OVERFLOW);
    CHECK(niyah_pool_align_up(1U, NULL) == NIYAH_ERR_INVALID_ARGUMENT);
    return 0;
}

static int test_init_validation(void)
{
    NiyahPool pool;
    unsigned char storage[64];
    CHECK(niyah_pool_init(NULL, storage, sizeof(storage)) == NIYAH_ERR_INVALID_ARGUMENT);
    CHECK(niyah_pool_init(&pool, NULL, sizeof(storage)) == NIYAH_ERR_INVALID_ARGUMENT);
    CHECK(niyah_pool_init(&pool, storage, 0U) == NIYAH_ERR_INVALID_CONFIG);
    unsigned char misaligned[sizeof(storage) + 1];
    unsigned char *bad = misaligned + 1;
    CHECK(niyah_pool_init(&pool, bad, sizeof(storage)) == NIYAH_ERR_INVALID_CONFIG);
    memset(&pool, 0, sizeof(pool));
    CHECK(niyah_pool_init(&pool, storage, sizeof(storage)) == NIYAH_OK);
    CHECK(pool.base == storage && pool.capacity == sizeof(storage) && pool.offset == 0U);
    size_t remaining = 1U;
    CHECK(niyah_pool_remaining(&pool, &remaining) == NIYAH_OK && remaining == sizeof(storage));
    return 0;
}

static int test_alloc_sequence(void)
{
    NiyahPool pool;
    unsigned char storage[96];
    memset(storage, 0xAB, sizeof(storage));
    CHECK(niyah_pool_init(&pool, storage, sizeof(storage)) == NIYAH_OK);
    void *a = NULL;
    void *b = NULL;
    void *c = NULL;
    size_t asz = 0U;
    size_t bsz = 0U;
    size_t csz = 0U;
    CHECK(niyah_pool_alloc(&pool, 1U, &a, &asz) == NIYAH_OK);
    CHECK(a == (void *)storage && asz == 16U);
    CHECK(niyah_pool_alloc(&pool, 16U, &b, &bsz) == NIYAH_OK);
    CHECK(b == (void *)(storage + 16) && bsz == 16U);
    CHECK(niyah_pool_alloc(&pool, 33U, &c, &csz) == NIYAH_OK);
    CHECK(c == (void *)(storage + 32) && csz == 48U);
    size_t remaining = 1U;
    CHECK(niyah_pool_remaining(&pool, &remaining) == NIYAH_OK);
    CHECK(remaining == sizeof(storage) - 80U);
    CHECK(niyah_pool_reset(&pool) == NIYAH_OK);
    CHECK(niyah_pool_remaining(&pool, &remaining) == NIYAH_OK && remaining == sizeof(storage));
    void *d = NULL;
    size_t dsz = 0U;
    CHECK(niyah_pool_alloc(&pool, 1U, &d, &dsz) == NIYAH_OK);
    CHECK(d == (void *)storage);
    return 0;
}

static int test_alloc_failure_paths(void)
{
    NiyahPool pool;
    unsigned char storage[32];
    CHECK(niyah_pool_init(&pool, storage, sizeof(storage)) == NIYAH_OK);
    void *out = (void *)0x1;
    size_t outsz = 1U;
    CHECK(niyah_pool_alloc(&pool, 0U, &out, &outsz) == NIYAH_ERR_INVALID_ARGUMENT);
    CHECK(out == NULL && outsz == 0U);
    out = (void *)0x1;
    outsz = 1U;
    CHECK(niyah_pool_alloc(&pool, 33U, &out, &outsz) == NIYAH_ERR_OUT_OF_MEMORY);
    CHECK(out == NULL && outsz == 0U);
    size_t remaining = 1U;
    CHECK(niyah_pool_remaining(&pool, &remaining) == NIYAH_OK && remaining == sizeof(storage));
    CHECK(niyah_pool_alloc(&pool, SIZE_MAX, &out, &outsz) == NIYAH_ERR_OVERFLOW);
    CHECK(niyah_pool_alloc(&pool, SIZE_MAX - 4U, &out, &outsz) == NIYAH_ERR_OVERFLOW);
    CHECK(niyah_pool_alloc(NULL, 1U, &out, &outsz) == NIYAH_ERR_INVALID_ARGUMENT);
    CHECK(niyah_pool_alloc(&pool, 1U, NULL, &outsz) == NIYAH_ERR_INVALID_ARGUMENT);
    CHECK(niyah_pool_alloc(&pool, 1U, &out, NULL) == NIYAH_ERR_INVALID_ARGUMENT);
    return 0;
}

static int test_alloc_array(void)
{
    NiyahPool pool;
    unsigned char storage[64];
    CHECK(niyah_pool_init(&pool, storage, sizeof(storage)) == NIYAH_OK);
    float *vec = NULL;
    size_t vecsz = 0U;
    CHECK(niyah_pool_alloc_array(&pool, 4U, sizeof(float), (void **)&vec, &vecsz) == NIYAH_OK);
    CHECK(vec == (void *)storage && vecsz == 16U);
    memset(vec, 0, vecsz);
    vec[0] = 1.0f;
    vec[3] = 4.0f;
    CHECK(vec[0] == 1.0f && vec[3] == 4.0f);
    void *out = NULL;
    size_t outsz = 0U;
    CHECK(niyah_pool_alloc_array(&pool, 0U, sizeof(float), &out, &outsz) ==
          NIYAH_ERR_INVALID_ARGUMENT);
    CHECK(niyah_pool_alloc_array(&pool, 4U, 0U, &out, &outsz) == NIYAH_ERR_INVALID_ARGUMENT);
    CHECK(niyah_pool_alloc_array(&pool, SIZE_MAX, 8U, &out, &outsz) == NIYAH_ERR_OVERFLOW);
    CHECK(niyah_pool_alloc_array(&pool, SIZE_MAX / 4U, 8U, &out, &outsz) == NIYAH_ERR_OVERFLOW);
    CHECK(niyah_pool_alloc_array(&pool, 64U, sizeof(float), &out, &outsz) ==
          NIYAH_ERR_OUT_OF_MEMORY);
    size_t remaining = 1U;
    CHECK(niyah_pool_remaining(&pool, &remaining) == NIYAH_OK && remaining == 48U);
    return 0;
}

static int test_determinism(void)
{
    unsigned char s1[128];
    unsigned char s2[128];
    NiyahPool p1;
    NiyahPool p2;
    void *a = NULL;
    void *b = NULL;
    size_t az = 0U;
    size_t bz = 0U;
    CHECK(niyah_pool_init(&p1, s1, sizeof(s1)) == NIYAH_OK);
    CHECK(niyah_pool_init(&p2, s2, sizeof(s2)) == NIYAH_OK);
    for (size_t i = 0U; i < 8U; ++i) {
        CHECK(niyah_pool_alloc(&p1, i + 1U, &a, &az) == NIYAH_OK);
        CHECK(niyah_pool_alloc(&p2, i + 1U, &b, &bz) == NIYAH_OK);
        CHECK(az == bz);
        CHECK((size_t)((unsigned char *)a - s1) == (size_t)((unsigned char *)b - s2));
    }
    return 0;
}

int main(void)
{
    if (test_align_up() != 0) {
        return 1;
    }
    if (test_init_validation() != 0) {
        return 1;
    }
    if (test_alloc_sequence() != 0) {
        return 1;
    }
    if (test_alloc_failure_paths() != 0) {
        return 1;
    }
    if (test_alloc_array() != 0) {
        return 1;
    }
    if (test_determinism() != 0) {
        return 1;
    }
    printf("niyah_core_test passed\n");
    return 0;
}
