#ifndef NIYAH_PPL_H
#define NIYAH_PPL_H
#include <math.h>

typedef struct { double nll_sum; long tokens; } niyah_ppl_acc;

static void niyah_ppl_reset(niyah_ppl_acc *a) { a->nll_sum = 0.0; a->tokens = 0; }

static void niyah_ppl_add(niyah_ppl_acc *a, double mean_loss, long n_tokens) {
    a->nll_sum += mean_loss * (double)n_tokens;
    a->tokens += n_tokens;
}

static double niyah_ppl_val_loss(const niyah_ppl_acc *a) {
    return a->tokens > 0 ? a->nll_sum / (double)a->tokens : 0.0;
}

static double niyah_ppl_value(const niyah_ppl_acc *a) {
    double l = niyah_ppl_val_loss(a);
    return l > 700.0 ? INFINITY : exp(l);
}
#endif
