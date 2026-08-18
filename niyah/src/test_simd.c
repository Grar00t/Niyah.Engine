/*
 * test_simd.c — Unit tests for niyah_simd primitives
 *
 * Build:
 *   gcc -O2 -std=c11 -Wall -Wextra -Werror -march=native \
 *       -I niyah/include niyah/src/test_simd.c niyah/src/niyah_simd.c \
 *       -o niyah_test_simd -lm
 *   ./niyah_test_simd
 *
 * Expected: all PASS, exit 0.
 */
#include "niyah_simd.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

#define PASS(name) do { printf("[PASS] %s\n", name); pass++; } while(0)
#define FAIL(name, ...) do { printf("[FAIL] %s: ", name); printf(__VA_ARGS__); printf("\n"); fail++; } while(0)
#define NEAR(a,b,eps) (fabsf((a)-(b)) < (eps))

/* ── Q4_0 block layout (must match niyah_simd.c) ─────────────────── */
#define Q4_0_BLOCK_SIZE 32
typedef struct __attribute__((packed)) {
    uint16_t d;       /* fp16 scale */
    uint8_t  qs[16];  /* nibble pairs */
} block_q4_0_t;

/* fp32 → fp16 (round-to-nearest, no inf/nan handling needed for tests) */
static uint16_t fp32_to_fp16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint16_t sign = (x >> 16) & 0x8000u;
    int exp = (int)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = (x & 0x7FFFFFu) >> 13;
    if (exp <= 0)  return sign;
    if (exp >= 31) return sign | 0x7C00u;
    return (uint16_t)(sign | ((uint16_t)exp << 10) | (uint16_t)mant);
}

static void make_q4_block(block_q4_0_t *blk, float scale, int8_t *weights) {
    blk->d = fp32_to_fp16(scale);
    for (int i = 0; i < 16; i++) {
        uint8_t lo = (uint8_t)(weights[i]      + 8);
        uint8_t hi = (uint8_t)(weights[i + 16] + 8);
        blk->qs[i] = (uint8_t)(lo | (hi << 4));
    }
}

static int pass = 0, fail = 0;

/* ── Test 1: Q4_0 dot product — all-ones ─────────────────────────── */
static void test_q4_dot_ones(void) {
    const char *name = "q4_dot_ones";
    const int N = Q4_0_BLOCK_SIZE;
    block_q4_0_t blk;
    int8_t w[N];
    float  b[N];
    for (int i = 0; i < N; i++) { w[i] = 1; b[i] = 1.0f; }
    make_q4_block(&blk, 1.0f, w);
    float got = niyah_dot_q4_0_fp32_scalar(&blk, b, N);
    /* expected: 32 * 1.0 * 1.0 = 32.0 */
    if (NEAR(got, 32.0f, 0.5f)) PASS(name);
    else FAIL(name, "expected ~32.0 got %.4f", got);
}

/* ── Test 2: Q4_0 dot product — zero weights ─────────────────────── */
static void test_q4_dot_zero(void) {
    const char *name = "q4_dot_zero";
    const int N = Q4_0_BLOCK_SIZE;
    block_q4_0_t blk;
    int8_t w[N];
    float  b[N];
    for (int i = 0; i < N; i++) { w[i] = 0; b[i] = 1.0f; }
    make_q4_block(&blk, 1.0f, w);
    float got = niyah_dot_q4_0_fp32_scalar(&blk, b, N);
    if (NEAR(got, 0.0f, 0.01f)) PASS(name);
    else FAIL(name, "expected 0.0 got %.4f", got);
}

/* ── Test 3: SGEMV correctness ───────────────────────────────────── */
static void test_sgemv(void) {
    const char *name = "sgemv_2x3";
    /* A = [[1,2,3],[4,5,6]], x = [1,1,1] => y = [6, 15] */
    float A[] = {1,2,3, 4,5,6};
    float x[] = {1,1,1};
    float y[2] = {0};
    niyah_sgemv_f32(A, x, y, 2, 3);
    if (NEAR(y[0], 6.0f, 0.001f) && NEAR(y[1], 15.0f, 0.001f)) PASS(name);
    else FAIL(name, "y=[%.2f, %.2f] expected [6, 15]", y[0], y[1]);
}

/* ── Test 4: Softmax sums to 1 ───────────────────────────────────── */
static void test_softmax_sum(void) {
    const char *name = "softmax_sum_to_1";
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    niyah_softmax_f32(x, 4);
    float s = x[0]+x[1]+x[2]+x[3];
    if (NEAR(s, 1.0f, 1e-5f)) PASS(name);
    else FAIL(name, "sum=%.6f expected 1.0", s);
}

/* ── Test 5: Softmax argmax preserved ───────────────────────────── */
static void test_softmax_argmax(void) {
    const char *name = "softmax_argmax";
    float x[] = {0.1f, 0.5f, 9.9f, 0.2f};
    niyah_softmax_f32(x, 4);
    int best = 0;
    for (int i = 1; i < 4; i++) if (x[i] > x[best]) best = i;
    if (best == 2) PASS(name);
    else FAIL(name, "argmax=%d expected 2", best);
}

/* ── Test 6: RMSNorm output scale ────────────────────────────────── */
static void test_rmsnorm(void) {
    const char *name = "rmsnorm_unit_weight";
    float x[]   = {3.0f, 4.0f};   /* rms = sqrt((9+16)/2) = sqrt(12.5) */
    float w[]   = {1.0f, 1.0f};
    float out[2] = {0};
    niyah_rmsnorm_f32(out, x, w, 2, 1e-6f);
    float rms = sqrtf((3.0f*3.0f + 4.0f*4.0f) / 2.0f);
    float e0 = 3.0f / rms, e1 = 4.0f / rms;
    if (NEAR(out[0], e0, 1e-4f) && NEAR(out[1], e1, 1e-4f)) PASS(name);
    else FAIL(name, "out=[%.4f,%.4f] expected [%.4f,%.4f]", out[0],out[1],e0,e1);
}

/* ── Test 7: Dispatch wrapper matches scalar ─────────────────────── */
static void test_dispatch_matches_scalar(void) {
    const char *name = "dispatch_matches_scalar";
    const int N = Q4_0_BLOCK_SIZE * 2;
    block_q4_0_t blks[2];
    int8_t w[N];
    float  b[N];
    for (int i = 0; i < N; i++) { w[i] = (int8_t)(i % 7 - 3); b[i] = (float)(i % 5) * 0.1f; }
    make_q4_block(&blks[0], 0.5f, w);
    make_q4_block(&blks[1], 0.5f, w + Q4_0_BLOCK_SIZE);
    float scalar  = niyah_dot_q4_0_fp32_scalar(blks, b, N);
    float dispatch = niyah_dot_q4_0_fp32(blks, b, N);
    if (NEAR(scalar, dispatch, 0.05f)) PASS(name);
    else FAIL(name, "scalar=%.4f dispatch=%.4f", scalar, dispatch);
}

int main(void) {
    printf("=== NIYAH SIMD Unit Tests ===\n");
    test_q4_dot_ones();
    test_q4_dot_zero();
    test_sgemv();
    test_softmax_sum();
    test_softmax_argmax();
    test_rmsnorm();
    test_dispatch_matches_scalar();
    printf("\nResult: %d passed, %d failed\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
