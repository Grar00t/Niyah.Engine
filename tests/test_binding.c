#include "niyah/dataset.h"
#include "niyah/train.h"
int main(void) {
    const uint32_t a[]={'a','b'}; const uint32_t b[]={'a','c'};
    niyah_dataset da,db; niyah_dataset_init(&da); niyah_dataset_init(&db);
    niyah_dataset_from_tokens(a,2u,&da); niyah_dataset_from_tokens(b,2u,&db);
    niyah_checkpoint ck; niyah_checkpoint_init(&ck); niyah_train_result r;
    if(niyah_train_new(&da,1u,&ck,&r)!=NIYAH_OK) return 1;
    niyah_status st=niyah_train_resume(&db,&ck,1u,&r);
    niyah_dataset_free(&da); niyah_dataset_free(&db);
    return st==NIYAH_ERR_MISMATCH ? 0 : 2;
}
