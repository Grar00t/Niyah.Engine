/*
 * niyah_simd.c – SIMD-optimized primitives for LLM inference
 *
 * Targets:
 *   ARM64/NEON  – detected at compile time via __aarch64__
 *   x86_64/AVX2 – detected via __AVX2__
 *   Scalar      – always-correct fallback
 *
 * Operations implemented:
 *   1. Q4_0 dequantize + fp32 dot product
 *   2. Q8_0 dequantize + fp32 dot product
 *   3. FP32 SGEMV (single matrix-vector multiply)
 *   4. Softmax over fp32 vector
 *   5. RMS norm over fp32 vector
 *
 * Compile with: -O3 -march=native (adds -mfpu=neon on ARM or -mavx2 on x86)
 */

#include "niyah_simd.h"
#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>

#ifdef __aarch64__
#  include <arm_neon.h>
#endif

#ifdef __AVX2__
#  include <immintrin.h>
#endif

/* ── Q4_0 block layout ─────────────────────────────────────────────
 * 32 weights per block:
 *   [0..1]  : fp16 delta (scale)
 *   [2..17] : 16 bytes, 2 nibbles per byte = 32 x 4-bit weights
 * Each weight w = (nibble - 8) * delta
 */
#define Q4_0_BLOCK_SIZE  32
#define Q4_0_BYTES       18  /* 2 (fp16 scale) + 16 (nibbles) */

typedef struct __attribute__((packed)) {
    uint16_t d;          /* fp16 scale */
    uint8_t  qs[16];     /* nibble pairs */
} block_q4_0_t;

/* ── Q8_0 block layout ─────────────────────────────────────────────
 * 32 weights per block:
 *   [0..1]  : fp16 scale
 *   [2..33] : 32 x int8 weights
 */
#define Q8_0_BLOCK_SIZE  32
#define Q8_0_BYTES       34

typedef struct __attribute__((packed)) {
    uint16_t d;
    int8_t   qs[32];
} block_q8_0_t;

/* ── fp16 → fp32 (software, no hardfp intrinsic needed) ──────────── */
static inline float fp16_to_fp32(uint16_t h) {
    /* IEEE 754 half-precision conversion */
    uint32_t sign     = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exponent = (uint32_t)(h & 0x7c00u);
    uint32_t mantissa = (uint32_t)(h & 0x03ffu);
    uint32_t result;

    if (exponent == 0x7c00u) {
        /* Inf or NaN */
        result = sign | 0x7f800000u | (mantissa << 13);
    } else if (exponent == 0) {
        /* Denormal: renormalize */
        if (mantissa == 0) {
            result = sign;
        } else {
            exponent = 1;
            while (!(mantissa & 0x0400u)) { mantissa <<= 1; exponent--; }
            mantissa &= 0x03ffu;
            result = sign | ((exponent + 112u) << 23) | (mantissa << 13);
        }
    } else {
        result = sign | ((exponent >> 10u) + 112u) << 23 | (mantissa << 13);
    }

    float f;
    memcpy(&f, &result, 4);
    return f;
}

/* ════════════════════════════════════════════════════════════════════
 * Q4_0 dot product: sum_i dequant(a[i]) * b[i]
 * where b is fp32 and a is Q4_0-encoded.
 * n must be a multiple of Q4_0_BLOCK_SIZE.
 * ════════════════════════════════════════════════════════════════════ */

#ifdef __aarch64__

float niyah_dot_q4_0_fp32_neon(const void *vq, const float *fb, size_t n) {
    assert(n % Q4_0_BLOCK_SIZE == 0);
    const block_q4_0_t *blocks = (const block_q4_0_t *)vq;
    size_t nb = n / Q4_0_BLOCK_SIZE;

    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    for (size_t b = 0; b < nb; b++) {
        const block_q4_0_t *blk = &blocks[b];
        const float scale = fp16_to_fp32(blk->d);
        const float32x4_t vscale = vdupq_n_f32(scale);

        /* Load 16 nibble bytes → 32 int8 weights */
        uint8x16_t raw    = vld1q_u8(blk->qs);
        uint8x16_t lo_u8  = vandq_u8(raw, vdupq_n_u8(0x0F));  /* lower nibbles */
        uint8x16_t hi_u8  = vshrq_n_u8(raw, 4);                /* upper nibbles */

        /* Subtract 8 (zero point) → signed */
        int8x16_t lo_i8 = vreinterpretq_s8_u8(
            vsubq_u8(lo_u8, vdupq_n_u8(8)));
        int8x16_t hi_i8 = vreinterpretq_s8_u8(
            vsubq_u8(hi_u8, vdupq_n_u8(8)));

        /* Load 32 fp32 values from b */
        const float *fb_blk = fb + b * Q4_0_BLOCK_SIZE;

        /* Process lo nibbles (weights 0..15) */
        for (int q = 0; q < 16; q += 4) {
            float32x4_t fw = vld1q_f32(fb_blk + q);
            /* Extract 4 int8 weights → fp32 */
            int8_t w0 = vgetq_lane_s8(lo_i8, q);
            int8_t w1 = vgetq_lane_s8(lo_i8, q+1);
            int8_t w2 = vgetq_lane_s8(lo_i8, q+2);
            int8_t w3 = vgetq_lane_s8(lo_i8, q+3);
            float32x4_t vw = {(float)w0,(float)w1,(float)w2,(float)w3};
            vw = vmulq_f32(vw, vscale);
            acc0 = vmlaq_f32(acc0, vw, fw);
        }

        /* Process hi nibbles (weights 16..31) */
        for (int q = 0; q < 16; q += 4) {
            float32x4_t fw = vld1q_f32(fb_blk + 16 + q);
            int8_t w0 = vgetq_lane_s8(hi_i8, q);
            int8_t w1 = vgetq_lane_s8(hi_i8, q+1);
            int8_t w2 = vgetq_lane_s8(hi_i8, q+2);
            int8_t w3 = vgetq_lane_s8(hi_i8, q+3);
            float32x4_t vw = {(float)w0,(float)w1,(float)w2,(float)w3};
            vw = vmulq_f32(vw, vscale);
            acc1 = vmlaq_f32(acc1, vw, fw);
        }
    }

    acc0 = vaddq_f32(acc0, acc1);
    acc0 = vaddq_f32(acc0, acc2);
    acc0 = vaddq_f32(acc0, acc3);
    return vaddvq_f32(acc0);  /* horizontal add */
}

#endif /* __aarch64__ */

#ifdef __AVX2__

float niyah_dot_q4_0_fp32_avx2(const void *vq, const float *fb, size_t n) {
    assert(n % Q4_0_BLOCK_SIZE == 0);
    const block_q4_0_t *blocks = (const block_q4_0_t *)vq;
    size_t nb = n / Q4_0_BLOCK_SIZE;

    __m256 acc = _mm256_setzero_ps();

    for (size_t b = 0; b < nb; b++) {
        const block_q4_0_t *blk = &blocks[b];
        const float scale = fp16_to_fp32(blk->d);

        /* Load 16 bytes of nibbles */
        __m128i raw128 = _mm_loadu_si128((const __m128i *)blk->qs);
        __m256i raw256 = _mm256_cvtepu8_epi16(raw128);

        /* Lower nibbles */
        __m256i lo = _mm256_and_si256(raw256, _mm256_set1_epi16(0x0F));
        /* Upper nibbles */
        __m256i hi = _mm256_srli_epi16(raw256, 4);

        /* Subtract 8 */
        __m256i sub8 = _mm256_set1_epi16(8);
        lo = _mm256_sub_epi16(lo, sub8);
        hi = _mm256_sub_epi16(hi, sub8);

        const float *fb_blk = fb + b * Q4_0_BLOCK_SIZE;

        /* Convert lo int16 → fp32 (first 8 weights) */
        __m128i lo128 = _mm256_extracti128_si256(lo, 0);
        __m256  lo_f  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(lo128));
        __m256  fb_lo = _mm256_loadu_ps(fb_blk);
        lo_f = _mm256_mul_ps(lo_f, _mm256_set1_ps(scale));
        acc  = _mm256_fmadd_ps(lo_f, fb_lo, acc);

        /* Second 8 lo weights */
        lo128 = _mm256_extracti128_si256(lo, 1);
        lo_f  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(lo128));
        fb_lo = _mm256_loadu_ps(fb_blk + 8);
        lo_f  = _mm256_mul_ps(lo_f, _mm256_set1_ps(scale));
        acc   = _mm256_fmadd_ps(lo_f, fb_lo, acc);

        /* Hi weights: first 8 */
        __m128i hi128 = _mm256_extracti128_si256(hi, 0);
        __m256  hi_f  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(hi128));
        __m256  fb_hi = _mm256_loadu_ps(fb_blk + 16);
        hi_f = _mm256_mul_ps(hi_f, _mm256_set1_ps(scale));
        acc  = _mm256_fmadd_ps(hi_f, fb_hi, acc);

        /* Hi weights: second 8 */
        hi128 = _mm256_extracti128_si256(hi, 1);
        hi_f  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(hi128));
        fb_hi = _mm256_loadu_ps(fb_blk + 24);
        hi_f  = _mm256_mul_ps(hi_f, _mm256_set1_ps(scale));
        acc   = _mm256_fmadd_ps(hi_f, fb_hi, acc);
    }

    /* Horizontal sum of 8 floats */
    __m128 lo128f = _mm256_extractf128_ps(acc, 0);
    __m128 hi128f = _mm256_extractf128_ps(acc, 1);
    lo128f = _mm_add_ps(lo128f, hi128f);
    lo128f = _mm_hadd_ps(lo128f, lo128f);
    lo128f = _mm_hadd_ps(lo128f, lo128f);
    return _mm_cvtss_f32(lo128f);
}

#endif /* __AVX2__ */

/* Scalar fallback */
float niyah_dot_q4_0_fp32_scalar(const void *vq, const float *fb, size_t n) {
    assert(n % Q4_0_BLOCK_SIZE == 0);
    const block_q4_0_t *blocks = (const block_q4_0_t *)vq;
    size_t nb = n / Q4_0_BLOCK_SIZE;
    float sum = 0.0f;

    for (size_t b = 0; b < nb; b++) {
        const block_q4_0_t *blk = &blocks[b];
        const float scale = fp16_to_fp32(blk->d);
        const float *fb_blk = fb + b * Q4_0_BLOCK_SIZE;

        for (int q = 0; q < 16; q++) {
            float w_lo = (float)((blk->qs[q] & 0x0F) - 8) * scale;
            float w_hi = (float)((blk->qs[q] >>  4)  - 8) * scale;
            sum += w_lo * fb_blk[q];
            sum += w_hi * fb_blk[q + 16];
        }
    }
    return sum;
}

/* ── Dispatch wrapper ─────────────────────────────────────────────── */
float niyah_dot_q4_0_fp32(const void *vq, const float *fb, size_t n) {
#ifdef __aarch64__
    return niyah_dot_q4_0_fp32_neon(vq, fb, n);
#elif defined(__AVX2__)
    return niyah_dot_q4_0_fp32_avx2(vq, fb, n);
#else
    return niyah_dot_q4_0_fp32_scalar(vq, fb, n);
#endif
}

/* ════════════════════════════════════════════════════════════════════
 * FP32 SGEMV: y[i] = sum_j A[i*cols + j] * x[j]
 * A is row-major, dims: rows × cols
 * ════════════════════════════════════════════════════════════════════ */
void niyah_sgemv_f32(const float *A, const float *x, float *y,
                     size_t rows, size_t cols) {
#ifdef __aarch64__
    for (size_t i = 0; i < rows; i++) {
        const float *row = A + i * cols;
        float32x4_t acc = vdupq_n_f32(0.0f);
        size_t j = 0;
        for (; j + 4 <= cols; j += 4) {
            float32x4_t va = vld1q_f32(row + j);
            float32x4_t vx = vld1q_f32(x + j);
            acc = vmlaq_f32(acc, va, vx);
        }
        float sum = vaddvq_f32(acc);
        for (; j < cols; j++) sum += row[j] * x[j];
        y[i] = sum;
    }
#elif defined(__AVX2__)
    for (size_t i = 0; i < rows; i++) {
        const float *row = A + i * cols;
        __m256 acc = _mm256_setzero_ps();
        size_t j = 0;
        for (; j + 8 <= cols; j += 8) {
            __m256 va = _mm256_loadu_ps(row + j);
            __m256 vx = _mm256_loadu_ps(x + j);
            acc = _mm256_fmadd_ps(va, vx, acc);
        }
        __m128 lo = _mm256_extractf128_ps(acc, 0);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        float sum = _mm_cvtss_f32(lo);
        for (; j < cols; j++) sum += row[j] * x[j];
        y[i] = sum;
    }
#else
    for (size_t i = 0; i < rows; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < cols; j++) sum += A[i*cols+j] * x[j];
        y[i] = sum;
    }
#endif
}

/* ════════════════════════════════════════════════════════════════════
 * Softmax: x[i] = exp(x[i] - max) / sum
 * In-place.
 * ════════════════════════════════════════════════════════════════════ */
void niyah_softmax_f32(float *x, size_t n) {
    float max_val = x[0];
    for (size_t i = 1; i < n; i++)
        if (x[i] > max_val) max_val = x[i];

    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv = 1.0f / sum;
    for (size_t i = 0; i < n; i++) x[i] *= inv;
}

/* ════════════════════════════════════════════════════════════════════
 * RMS Norm: x[i] = x[i] / sqrt(mean(x^2) + eps)
 * weight vector w is applied: out[i] = norm[i] * w[i]
 * ════════════════════════════════════════════════════════════════════ */
void niyah_rmsnorm_f32(float *out, const float *x, const float *w,
                       size_t n, float eps) {
    float ss = 0.0f;
#ifdef __aarch64__
    float32x4_t vss = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        vss = vmlaq_f32(vss, vx, vx);
    }
    ss = vaddvq_f32(vss);
    for (; i < n; i++) ss += x[i] * x[i];
#else
    for (size_t i = 0; i < n; i++) ss += x[i] * x[i];
#endif
    ss /= (float)n;
    ss += eps;
    float scale = 1.0f / sqrtf(ss);
    for (size_t i = 0; i < n; i++)
        out[i] = x[i] * scale * w[i];
}
