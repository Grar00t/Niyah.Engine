#undef NDEBUG

#include "niyah_mini_model.h"
#include "niyah_mini_vocab.h"
#include "../niyah.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
    NiyahMiniConfig config;
    NiyahMiniModel model;
    NiyahSamplerConfig sampler;
    NiyahStatus status;

    int32_t prompt[1] = { NIYAH_MINI_BOS_TOKEN_ID };
    int32_t output[4];
    int32_t output_len = 0;
    int32_t expected;

    float *flat_logits;

    memset(&model, 0, sizeof(model));
    niyah_mini_config_init(&config, NIYAH_MINI_TINY);

    status = niyah_mini_model_init(&model, &config);
    assert(status == NIYAH_OK);

    /*
     * Force every next-token logit to the same value. This makes
     * stochastic selection observable and prevents learned weights
     * from influencing this contract test.
     */
    memset(
        model.weights.memory_block,
        0,
        model.weights.memory_size
    );

    flat_logits = (float *)calloc(
        (size_t)config.n_vocab,
        sizeof(float)
    );
    assert(flat_logits != NULL);

    sampler.strategy = NIYAH_SAMPLE_TEMPERATURE;
    sampler.temperature = 0.8f;
    sampler.top_k = 0;
    sampler.top_p = 1.0f;

    /*
     * The mini generator must consume the same sampler and seed
     * contract as the engine sampler.
     */
    niyah_sampler_seed(UINT64_C(12345));
    expected = niyah_sample(
        flat_logits,
        config.n_vocab,
        &sampler
    );
    assert(expected >= 0);
    assert(expected < config.n_vocab);

    niyah_sampler_seed(UINT64_C(12345));
    status = niyah_mini_generate(
        &model,
        prompt,
        1,
        1,
        0.8f,
        output,
        &output_len
    );

    assert(status == NIYAH_OK);
    assert(output_len == 1);
    assert(output[0] == expected);

    /*
     * Temperature zero follows the engine sampler's greedy limit.
     * Flat logits therefore select the first vocabulary id.
     */
    output_len = 0;
    status = niyah_mini_generate(
        &model,
        prompt,
        1,
        1,
        0.0f,
        output,
        &output_len
    );

    assert(status == NIYAH_OK);
    assert(output_len == 1);
    assert(output[0] == 0);

    /*
     * Negative and non-finite temperatures remain invalid.
     */
    output_len = 0;
    status = niyah_mini_generate(
        &model,
        prompt,
        1,
        1,
        -0.1f,
        output,
        &output_len
    );
    assert(status == NIYAH_ERR_INVALID_ARG);

    free(flat_logits);
    niyah_mini_model_free(&model);

    return 0;
}
