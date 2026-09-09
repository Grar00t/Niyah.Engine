#undef NDEBUG
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "niyah_mini_model.h"

int main(void)
{
    NiyahMiniConfig cfg;
    NiyahMiniModel model;
    NiyahMiniForwardState state;
    size_t s16, s32, runtime_bytes;
    void *state_mem;
    void *runtime_mem;
    float *logits;

    memset(&model, 0, sizeof(model));
    memset(&state, 0, sizeof(state));

    niyah_mini_config_init(&cfg, NIYAH_MINI_TINY);
    cfg.n_ctx = 32;

    assert(niyah_mini_model_init(&model, &cfg) == NIYAH_OK);

    s16 = niyah_mini_forward_state_memory_size(&cfg, 16);
    s32 = niyah_mini_forward_state_memory_size(&cfg, 32);
    assert(s16 > 0U);
    assert(s32 > s16);
    assert(s32 - s16 == (size_t)(2 * (32 - 16)) * sizeof(float));

    runtime_bytes = niyah_mini_runtime_memory_size(&cfg);
    assert(runtime_bytes > 0U);

    state_mem = malloc(s32);
    runtime_mem = malloc(runtime_bytes);
    logits = (float *)malloc((size_t)cfg.n_vocab * sizeof(float));
    assert(state_mem && runtime_mem && logits);

    assert(niyah_mini_forward_state_bind(
        &state, &cfg, 32, state_mem, s32) == NIYAH_OK);
    assert(state.owns_memory == false);

    assert(niyah_mini_model_bind_runtime(
        &model, runtime_mem, runtime_bytes) == NIYAH_OK);
    assert(model.owns_kv_cache == false);

    assert(niyah_mini_forward_token(
        &model, &state, 1, 0, logits) == NIYAH_OK);

    niyah_mini_forward_state_free(&state);
    niyah_mini_model_free(&model);

    free(state_mem);
    free(runtime_mem);
    free(logits);
    return 0;
}
