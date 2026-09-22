#include "niyah/train.h"
#include <string.h>

static niyah_status apply_updates(
    const niyah_dataset *dataset,
    niyah_checkpoint *ckpt,
    uint64_t updates,
    niyah_train_result *result) {
    if(!dataset || !ckpt || !result) return NIYAH_ERR_INVALID;
    if(dataset->token_count < 2u) return NIYAH_ERR_STATE;
    const uint64_t transitions=dataset->token_count-1u;
    result->updates_requested=updates;
    result->updates_applied=0u;
    result->cursor_before=ckpt->cursor;
    for(uint64_t u=0u;u<updates;++u) {
        uint64_t pos=ckpt->cursor % transitions;
        niyah_status st=niyah_model_observe(&ckpt->model,dataset->tokens[pos],dataset->tokens[pos+1u]);
        if(st!=NIYAH_OK) return st;
        ckpt->cursor++;
        result->updates_applied++;
    }
    result->cursor_after=ckpt->cursor;
    return NIYAH_OK;
}

niyah_status niyah_train_new(
    const niyah_dataset *dataset,
    uint64_t updates,
    niyah_checkpoint *out,
    niyah_train_result *result) {
    if(!dataset || !out || !result) return NIYAH_ERR_INVALID;
    niyah_checkpoint_init(out);
    memcpy(out->dataset_sha256,dataset->content_sha256,32u);
    return apply_updates(dataset,out,updates,result);
}

niyah_status niyah_train_resume(
    const niyah_dataset *dataset,
    niyah_checkpoint *checkpoint,
    uint64_t updates,
    niyah_train_result *result) {
    if(!dataset || !checkpoint || !result) return NIYAH_ERR_INVALID;
    niyah_status st=niyah_checkpoint_require_dataset(checkpoint,dataset->content_sha256);
    if(st!=NIYAH_OK) return st;
    return apply_updates(dataset,checkpoint,updates,result);
}
