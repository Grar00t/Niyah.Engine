#include "niyah/dataset.h"
#include "niyah/eval.h"
#include <math.h>
int main(void) {
    const uint32_t t[]={'a','b','a','b','a'};
    niyah_dataset ds; niyah_dataset_init(&ds);
    if(niyah_dataset_from_tokens(t,5u,&ds)!=NIYAH_OK) return 1;
    niyah_eval_result r;
    if(niyah_eval_add1_bigram_baseline(&ds,&r)!=NIYAH_OK) return 2;
    niyah_dataset_free(&ds);
    return r.transitions==4u && isfinite(r.perplexity) && r.perplexity>0.0 ? 0 : 3;
}
