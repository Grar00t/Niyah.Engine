/*
 * niyah_simd.h — SIMD-optimized primitives for NIYAH inference
 *
 * Targets (compile-time selection):
 *   ARM64 NEON  : __aarch64__
 *   x86-64 AVX2 : __AVX2__
 *   Scalar      : always-correct fallback
 *
 * Zero external dependencies. C11 clean. C++17 compatible.
 */
#ifndef NIYAH_SIMD_H
#define NIYAH_SIMD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Q4_0 dot product ────────────────────────────────────────────────
 * Computes sum_i dequant(a[i]) * b[i]
 * where a is Q4_0-encoded and b is fp32.
 * n must be a multiple of 32 (Q4_0_BLOCK_SIZE).
 */
float niyah_dot_q4_0_fp32(const void *q4_0, const float *fp32, size_t n);

/* Platform-specific variants (exposed for testing) */
#ifdef __aarch64__
float niyah_dot_q4_0_fp32_neon  (const void *q4_0, const float *fp32, size_t n);
#endif
#ifdef __AVX2__
float niyah_dot_q4_0_fp32_avx2  (const void *q4_0, const float *fp32, size_t n);
#endif
float niyah_dot_q4_0_fp32_scalar(const void *q4_0, const float *fp32, size_t n);

/* ── FP32 SGEMV ───────────────────────────────────────────────────────
 * y[i] = sum_j A[i*cols + j] * x[j]
 * A is row-major, dims: rows x cols.
 */
void niyah_sgemv_f32(const float *A, const float *x, float *y,
                     size_t rows, size_t cols);

/* ── Softmax (in-place) ───────────────────────────────────────────────
 * x[i] = exp(x[i] - max) / sum
 */
void niyah_softmax_f32(float *x, size_t n);

/* ── RMS Norm ─────────────────────────────────────────────────────────
 * out[i] = (x[i] / sqrt(mean(x^2) + eps)) * w[i]
 */
void niyah_rmsnorm_f32(float *out, const float *x, const float *w,
                       size_t n, float eps);

#ifdef __cplusplus
}
#endif
#endif /* NIYAH_SIMD_H */
