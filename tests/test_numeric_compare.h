#ifndef NIYAH_TEST_NUMERIC_COMPARE_H
#define NIYAH_TEST_NUMERIC_COMPARE_H
#include <math.h>
#include <stddef.h>

/* Infinity is a failing error magnitude, including identical NaN/Inf inputs. */
static float niyah_test_max_abs_diff(const float *a, const float *b, size_t n)
{
    float result = 0.0f;
    size_t i;
    for (i = 0U; i < n; ++i) {
        float d;
        if (!isfinite(a[i]) || !isfinite(b[i])) return INFINITY;
        d = fabsf(a[i] - b[i]);
        if (!isfinite(d)) return INFINITY;
        if (d > result) result = d;
    }
    return result;
}
#endif
