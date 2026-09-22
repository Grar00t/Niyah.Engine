#include "niyah/eval.h"
#include <math.h>

niyah_status niyah_eval_model(const niyah_model *model, const niyah_dataset *dataset, niyah_eval_result *out) {
    if(!model || !dataset || !out) return NIYAH_ERR_INVALID;
    if(dataset->token_count < 2u) return NIYAH_ERR_STATE;
    double nll=0.0;
    const uint64_t transitions=dataset->token_count-1u;
    for(uint64_t i=0u;i<transitions;++i) {
        double lp=niyah_model_logprob_add1(model,dataset->tokens[i],dataset->tokens[i+1u]);
        if(!isfinite(lp)) return NIYAH_ERR_STATE;
        nll -= lp;
    }
    out->transitions=transitions;
    out->nll=nll;
    out->avg_nll=nll/(double)transitions;
    out->perplexity=exp(out->avg_nll);
    return NIYAH_OK;
}

niyah_status niyah_eval_add1_bigram_baseline(const niyah_dataset *dataset, niyah_eval_result *out) {
    if(!dataset || !out) return NIYAH_ERR_INVALID;
    if(dataset->token_count < 2u) return NIYAH_ERR_STATE;
    niyah_model model; niyah_model_init(&model);
    for(uint64_t i=0u;i+1u<dataset->token_count;++i) {
        niyah_status st=niyah_model_observe(&model,dataset->tokens[i],dataset->tokens[i+1u]);
        if(st!=NIYAH_OK) return st;
    }
    return niyah_eval_model(&model,dataset,out);
}
