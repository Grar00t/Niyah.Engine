#include "niyah/niyah.h"
#include "niyah_math_internal.h"

#include <math.h>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

float niyah_dot_f32(const float *a, const float *b, size_t n)
{
    size_t i = 0U;
    float sum = 0.0f;

    if (a == NULL || b == NULL) {
        return 0.0f;
    }

#if defined(__AVX2__)
    {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        for (; i + 15U < n; i += 16U) {
#if defined(__FMA__)
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),
                                    _mm256_loadu_ps(b + i), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8U),
                                    _mm256_loadu_ps(b + i + 8U), acc1);
#else
            acc0 = _mm256_add_ps(acc0,
                                 _mm256_mul_ps(_mm256_loadu_ps(a + i),
                                               _mm256_loadu_ps(b + i)));
            acc1 = _mm256_add_ps(acc1,
                                 _mm256_mul_ps(_mm256_loadu_ps(a + i + 8U),
                                               _mm256_loadu_ps(b + i + 8U)));
#endif
        }
        {
            const __m256 acc = _mm256_add_ps(acc0, acc1);
            const __m128 lo = _mm256_castps256_ps128(acc);
            const __m128 hi = _mm256_extractf128_ps(acc, 1);
            const __m128 pair = _mm_add_ps(lo, hi);
            const __m128 sums = _mm_hadd_ps(pair, pair);
            const __m128 total = _mm_hadd_ps(sums, sums);
            sum = _mm_cvtss_f32(total);
        }
    }
#elif defined(__aarch64__) && defined(__ARM_NEON)
    {
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        for (; i + 7U < n; i += 8U) {
            acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
            acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4U), vld1q_f32(b + i + 4U));
        }
        sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    }
#endif
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

void niyah_axpy_f32(float *dst, const float *src, float scale, size_t n)
{
    size_t i = 0U;

    if (dst == NULL || src == NULL) {
        return;
    }

#if defined(__AVX2__)
    {
        const __m256 s = _mm256_set1_ps(scale);
        for (; i + 15U < n; i += 16U) {
            __m256 d0 = _mm256_loadu_ps(dst + i);
            __m256 d1 = _mm256_loadu_ps(dst + i + 8U);
#if defined(__FMA__)
            d0 = _mm256_fmadd_ps(s, _mm256_loadu_ps(src + i), d0);
            d1 = _mm256_fmadd_ps(s, _mm256_loadu_ps(src + i + 8U), d1);
#else
            d0 = _mm256_add_ps(d0, _mm256_mul_ps(s, _mm256_loadu_ps(src + i)));
            d1 = _mm256_add_ps(d1, _mm256_mul_ps(s, _mm256_loadu_ps(src + i + 8U)));
#endif
            _mm256_storeu_ps(dst + i, d0);
            _mm256_storeu_ps(dst + i + 8U, d1);
        }
    }
#elif defined(__aarch64__) && defined(__ARM_NEON)
    {
        const float32x4_t s = vdupq_n_f32(scale);
        for (; i + 7U < n; i += 8U) {
            float32x4_t d0 = vld1q_f32(dst + i);
            float32x4_t d1 = vld1q_f32(dst + i + 4U);
            d0 = vfmaq_f32(d0, vld1q_f32(src + i), s);
            d1 = vfmaq_f32(d1, vld1q_f32(src + i + 4U), s);
            vst1q_f32(dst + i, d0);
            vst1q_f32(dst + i + 4U, d1);
        }
    }
#endif
    for (; i < n; ++i) {
        dst[i] += scale * src[i];
    }
}

void niyah_matvec(float *out,
                  const float *matrix,
                  const float *x,
                  size_t rows,
                  size_t cols)
{
    size_t r;

    if (out == NULL || matrix == NULL || x == NULL) {
        return;
    }

#ifdef NIYAH_ENABLE_OPENMP
#pragma omp parallel for schedule(static) if(rows >= 64U && cols >= 64U)
#endif
    for (r = 0U; r < rows; ++r) {
        out[r] = niyah_dot_f32(matrix + r * cols, x, cols);
    }
}

NiyahStatus niyah_rmsnorm(float *out,
                          const float *x,
                          const float *weight,
                          size_t n,
                          float eps)
{
    size_t i;
    double sum_sq = 0.0;
    float inv_rms;

    if (out == NULL || x == NULL || weight == NULL || n == 0U ||
        !isfinite(eps) || eps <= 0.0f) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    for (i = 0U; i < n; ++i) {
        const double v = (double)x[i];
        sum_sq += v * v;
    }

    inv_rms = 1.0f / sqrtf((float)(sum_sq / (double)n) + eps);
    for (i = 0U; i < n; ++i) {
        out[i] = x[i] * inv_rms * weight[i];
    }
    return NIYAH_OK;
}
