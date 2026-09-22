#include "niyah/niyah.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int check(int cond, const char *msg) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", msg); return 0; }
    return 1;
}

int main(void) {
    const uint8_t data[] = "abababababab";
    niyah_model m;
    double bpb = 0.0, ppl = 0.0;
    uint8_t a[32], b[32];
    size_t as = 0, bs = 0;

    niyah_model_init(&m);
    if (!check(niyah_model_train_bytes(&m, data, sizeof(data) - 1) == NIYAH_OK, "train")) return 1;
    if (!check(m.bytes_seen == sizeof(data) - 1, "bytes_seen")) return 1;
    if (!check(m.bigram['a']['b'] == 6, "a->b count")) return 1;
    if (!check(m.bigram['b']['a'] == 5, "b->a count")) return 1;
    if (!check(niyah_model_eval_bytes(&m, data, sizeof(data) - 1, &bpb, &ppl) == NIYAH_OK, "eval")) return 1;
    if (!check(isfinite(bpb) && bpb > 0.0, "finite bpb")) return 1;
    if (!check(isfinite(ppl) && ppl > 1.0, "finite ppl")) return 1;
    if (!check(niyah_model_generate(&m, (const uint8_t *)"a", 1, sizeof(a), 123, a, sizeof(a), &as) == NIYAH_OK, "gen a")) return 1;
    if (!check(niyah_model_generate(&m, (const uint8_t *)"a", 1, sizeof(b), 123, b, sizeof(b), &bs) == NIYAH_OK, "gen b")) return 1;
    if (!check(as == bs && memcmp(a, b, as) == 0, "deterministic seed")) return 1;
    puts("PASS");
    return 0;
}
