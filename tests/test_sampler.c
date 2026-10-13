#include "niyah/sampler.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "sampler CHECK line=%d: %s\n", __LINE__, #x); return 1; } } while (0)

int main(void)
{
    NiyahSamplerConfig config = {0.0f, UINT64_C(1234)};
    NiyahSampler sampler, before;
    float logits[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    uint32_t token = 99U;
    size_t i;
    unsigned int counts[3] = {0U, 0U, 0U};
    const float invalid[] = {NAN, INFINITY, -INFINITY};
    CHECK(niyah_sampler_init(&sampler, &config) == NIYAH_OK);
    before = sampler;
    CHECK(niyah_sampler_sample(&sampler, logits, 3U, &token) == NIYAH_OK);
    CHECK(token == 0U && sampler.state == before.state);
    for (i = 0U; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
        logits[2] = invalid[i]; token = 99U;
        CHECK(niyah_sampler_sample(&sampler, logits, 3U, &token) == NIYAH_ERR_INVALID_CONFIG);
        CHECK(token == 99U && memcmp(&sampler, &before, sizeof(sampler)) == 0);
        config.temperature = invalid[i];
        CHECK(niyah_sampler_init(&sampler, &config) == NIYAH_ERR_INVALID_CONFIG);
        CHECK(memcmp(&sampler, &before, sizeof(sampler)) == 0);
    }
    config.temperature = FLT_MIN;
    CHECK(niyah_sampler_init(&sampler, &config) == NIYAH_OK);
    logits[0] = -FLT_MAX; logits[1] = FLT_MAX; logits[2] = 0.0f;
    CHECK(niyah_sampler_sample(&sampler, logits, 3U, &token) == NIYAH_OK && token == 1U);
    config.temperature = 1.0f;
    CHECK(niyah_sampler_init(&sampler, &config) == NIYAH_OK);
    /* Independent categorical reference: equal logits imply probability 1/3.
     * This wide fixed-seed bound detects range/bias failures, not performance. */
    logits[0] = logits[1] = logits[2] = 0.0f;
    for (i = 0U; i < 12000U; ++i) {
        CHECK(niyah_sampler_sample(&sampler, logits, 3U, &token) == NIYAH_OK && token < 3U);
        counts[token] += 1U;
    }
    for (i = 0U; i < 3U; ++i) CHECK(counts[i] > 3600U && counts[i] < 4400U);
    puts("SAMPLER_CONTRACT=PASS");
    return 0;
}
