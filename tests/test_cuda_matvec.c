#include "niyah/niyah.h"
#include "niyah_cuda_matvec.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>

#define MAX_ROWS 64U
#define MAX_COLS 64U

static int close_enough(float expected, float actual)
{
    const float diff = fabsf(expected - actual);
    const float scale = 1.0f + fabsf(expected);
    return diff <= 1.0e-4f * scale;
}

static int run_case(size_t rows, size_t cols)
{
    float matrix[MAX_ROWS * MAX_COLS];
    float x[MAX_COLS];
    float cpu[MAX_ROWS];
    float gpu[MAX_ROWS];
    size_t r;
    size_t c;

    if (rows == 0U || cols == 0U ||
        rows > MAX_ROWS || cols > MAX_COLS) {
        return 1;
    }

    for (r = 0U; r < rows; ++r) {
        for (c = 0U; c < cols; ++c) {
            const int value =
                (int)((r * 17U + c * 13U + 3U) % 23U) - 11;
            matrix[r * cols + c] = (float)value * 0.125f;
        }
    }

    for (c = 0U; c < cols; ++c) {
        const int value = (int)((c * 7U + 5U) % 17U) - 8;
        x[c] = (float)value * 0.0625f;
    }

    niyah_matvec(cpu, matrix, x, rows, cols);

    if (niyah_cuda_matvec(gpu, matrix, x, rows, cols) != 0) {
        fprintf(stderr, "CUDA matvec execution failed rows=%zu cols=%zu\n",
                rows, cols);
        return 1;
    }

    for (r = 0U; r < rows; ++r) {
        if (!close_enough(cpu[r], gpu[r])) {
            fprintf(stderr,
                    "parity mismatch row=%zu cpu=%.9g gpu=%.9g\n",
                    r, (double)cpu[r], (double)gpu[r]);
            return 1;
        }
    }

    return 0;
}

int main(void)
{
    if (run_case(1U, 1U) != 0 ||
        run_case(3U, 5U) != 0 ||
        run_case(7U, 11U) != 0 ||
        run_case(32U, 17U) != 0 ||
        run_case(64U, 64U) != 0) {
        return 1;
    }

    puts("P7A_CUDA_MATVEC_PARITY=PASS");
    return 0;
}
