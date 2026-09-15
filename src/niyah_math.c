#include "niyah/niyah.h"

#include <math.h>

void niyah_matvec(float *out,
                  const float *matrix,
                  const float *x,
                  size_t rows,
                  size_t cols)
{
    size_t r;
    size_t c;

    if (out == NULL || matrix == NULL || x == NULL) {
        return;
    }

    for (r = 0U; r < rows; ++r) {
        float sum = 0.0f;
        const float *row = matrix + r * cols;
        for (c = 0U; c < cols; ++c) {
            sum += row[c] * x[c];
        }
        out[r] = sum;
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

    if (out == NULL || x == NULL || weight == NULL || n == 0U || eps <= 0.0f) {
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
