#include "niyah/optimizer.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

static int failures = 0;

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", \
                __FILE__, __LINE__, #expr); \
        failures += 1; \
    } \
} while (0)

static int close_float(float actual, float expected)
{
    const float tolerance = 1.0e-9f;
    return fabsf(actual - expected) <= tolerance;
}

static void test_constant_schedule(void)
{
    float lr = 0.0f;

    CHECK(
        niyah_adamw_linear_warmup_learning_rate(
            1.0e-3f, UINT64_C(1), UINT64_C(0), &lr) == NIYAH_OK);
    CHECK(lr == 1.0e-3f);

    CHECK(
        niyah_adamw_linear_warmup_learning_rate(
            1.0e-3f, UINT64_C(999), UINT64_C(0), &lr) == NIYAH_OK);
    CHECK(lr == 1.0e-3f);
}

static void test_linear_warmup(void)
{
    float lr = 0.0f;

    CHECK(
        niyah_adamw_linear_warmup_learning_rate(
            1.0e-3f, UINT64_C(1), UINT64_C(4), &lr) == NIYAH_OK);
    CHECK(close_float(lr, 2.5e-4f));

    CHECK(
        niyah_adamw_linear_warmup_learning_rate(
            1.0e-3f, UINT64_C(2), UINT64_C(4), &lr) == NIYAH_OK);
    CHECK(close_float(lr, 5.0e-4f));

    CHECK(
        niyah_adamw_linear_warmup_learning_rate(
            1.0e-3f, UINT64_C(3), UINT64_C(4), &lr) == NIYAH_OK);
    CHECK(close_float(lr, 7.5e-4f));

    CHECK(
        niyah_adamw_linear_warmup_learning_rate(
            1.0e-3f, UINT64_C(4), UINT64_C(4), &lr) == NIYAH_OK);
    CHECK(lr == 1.0e-3f);

    CHECK(
        niyah_adamw_linear_warmup_learning_rate(
            1.0e-3f, UINT64_C(5), UINT64_C(4), &lr) == NIYAH_OK);
    CHECK(lr == 1.0e-3f);
}

static void test_invalid_inputs(void)
{
    float lr = 123.0f;

    CHECK(
        niyah_adamw_linear_warmup_learning_rate(
            0.0f, UINT64_C(1), UINT64_C(4), &lr) ==
        NIYAH_ERR_INVALID_CONFIG);
    CHECK(lr == 0.0f);

    lr = 123.0f;
    CHECK(
        niyah_adamw_linear_warmup_learning_rate(
            -1.0f, UINT64_C(1), UINT64_C(4), &lr) ==
        NIYAH_ERR_INVALID_CONFIG);
    CHECK(lr == 0.0f);

    lr = 123.0f;
    CHECK(
        niyah_adamw_linear_warmup_learning_rate(
            NAN, UINT64_C(1), UINT64_C(4), &lr) ==
        NIYAH_ERR_INVALID_CONFIG);
    CHECK(lr == 0.0f);

    lr = 123.0f;
    CHECK(
        niyah_adamw_linear_warmup_learning_rate(
            INFINITY, UINT64_C(1), UINT64_C(4), &lr) ==
        NIYAH_ERR_INVALID_CONFIG);
    CHECK(lr == 0.0f);

    lr = 123.0f;
    CHECK(
        niyah_adamw_linear_warmup_learning_rate(
            1.0e-3f, UINT64_C(0), UINT64_C(4), &lr) ==
        NIYAH_ERR_INVALID_CONFIG);
    CHECK(lr == 0.0f);

    CHECK(
        niyah_adamw_linear_warmup_learning_rate(
            1.0e-3f, UINT64_C(1), UINT64_C(4), NULL) ==
        NIYAH_ERR_INVALID_ARGUMENT);
}

static void test_unrepresentable_effective_rate(void)
{
    float lr = 123.0f;

    CHECK(
        niyah_adamw_linear_warmup_learning_rate(
            FLT_MIN,
            UINT64_C(1),
            UINT64_MAX,
            &lr) == NIYAH_ERR_OVERFLOW);
    CHECK(lr == 0.0f);
}

int main(void)
{
    test_constant_schedule();
    test_linear_warmup();
    test_invalid_inputs();
    test_unrepresentable_effective_rate();

    if (failures != 0) {
        fprintf(stderr, "lr schedule failures=%d\n", failures);
        return 1;
    }

    printf("NIYAH_LR_SCHEDULE_CORE=PASS\n");
    return 0;
}