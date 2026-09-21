#include "niyah/baseline.h"

#include <math.h>
#include <stdio.h>

static int failures = 0;

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", \
                __FILE__, __LINE__, #expr); \
        failures += 1; \
    } \
} while (0)

int main(void)
{
    static const uint32_t tokens[] = {0U, 1U, 0U, 1U};
    static const uint32_t short_tokens[] = {0U};
    static const uint32_t bad_tokens[] = {0U, 4U};
    double got = 0.0;
    double expected;

    /* vocab=4, stream 0,1,0,1 gives transitions 0->1 twice and 1->0 once.
     * P(1|0)=(2+1)/(2+4)=1/2 and P(0|1)=(1+1)/(1+4)=2/5.
     * Mean NLL = [2*(-ln(1/2)) + (-ln(2/5))] / 3.
     */
    expected = (2.0 * log(2.0) + log(2.5)) / 3.0;

    CHECK(niyah_add1_bigram_mean_nll(
              tokens, 4U, 4U, &got) == NIYAH_OK);
    CHECK(fabs(got - expected) <= 1.0e-12);

    CHECK(niyah_add1_bigram_mean_nll(
              short_tokens, 1U, 4U, &got) ==
          NIYAH_ERR_INVALID_CONFIG);
    CHECK(niyah_add1_bigram_mean_nll(
              bad_tokens, 2U, 4U, &got) ==
          NIYAH_ERR_INVALID_ARGUMENT);

    if (failures != 0) {
        fprintf(stderr, "niyah_baseline_test: %d failure(s)\n", failures);
        return 1;
    }

    puts("NIYAH_BASELINE=PASS");
    return 0;
}
