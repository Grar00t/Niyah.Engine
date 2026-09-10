/* gate3_heap_harness.c
 * Directly drives niyah_llm_generate() with a real tiny model (same params
 * as niyah_llm_generation_test.c) so the LD_PRELOAD interposer can count
 * exact heap allocation call sites.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "niyah.h"

#define TINY_TOTAL_FLOATS 404

static void fill_config(NiyahModelConfig *c) {
    memset(c, 0, sizeof(*c));
    c->n_vocab      = 8;
    c->n_embd       = 4;
    c->n_head       = 2;
    c->n_kv_head    = 2;
    c->n_ff         = 8;
    c->n_layer      = 2;
    c->n_ctx        = 8;
    c->eos_token_id = 7;
    niyah_model_config_normalize(c);
}

int main(void) {
    NiyahModelConfig config;
    fill_config(&config);

    float *blob = (float *)calloc(TINY_TOTAL_FLOATS, sizeof(float));
    if (!blob) { fputs("OOM\n", stderr); return 1; }
    for (int i = 0; i < TINY_TOTAL_FLOATS; ++i)
        blob[i] = 0.01f * (float)((i % 7) - 3);

    NiyahLLM llm;
    memset(&llm, 0, sizeof(llm));
    llm.model.config       = config;
    llm.model.weights      = blob;
    llm.model.weights_size = (size_t)TINY_TOTAL_FLOATS * sizeof(float);

    /* Run 3 times — collect n_tokens and text for determinism check */
    int32_t ntok[3] = {0};
    char *texts[3]  = {NULL, NULL, NULL};

    for (int run = 0; run < 3; ++run) {
        NiyahLLMOutput out = niyah_llm_generate(&llm, "hello", 4);
        printf("run=%d status=%d n_tokens=%d text=%s\n",
               run, out.status, out.n_tokens,
               out.text ? out.text : "(null)");
        ntok[run]   = out.n_tokens;
        texts[run]  = out.text ? strdup(out.text) : NULL;
        /* logits kept alive inside output until free */
        niyah_llm_output_free(&out);
    }

    /* Determinism: same n_tokens and same text for all 3 runs */
    int det = 1;
    for (int r = 1; r < 3; ++r) {
        if (ntok[r] != ntok[0]) { det = 0; break; }
        if (texts[r] && texts[0] && strcmp(texts[r], texts[0]) != 0)
            { det = 0; break; }
    }
    printf("DETERMINISTIC=%s\n", det ? "YES" : "NO");

    for (int r = 0; r < 3; ++r) free(texts[r]);
    free(blob);
    return 0;
}
