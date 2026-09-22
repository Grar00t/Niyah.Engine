#include "niyah/dataset.h"
#include "niyah/train.h"
#include <string.h>
int main(void) {
    const uint32_t t[]={'a','b','a','c','a','b'};
    niyah_dataset ds; niyah_dataset_init(&ds);
    if(niyah_dataset_from_tokens(t,6u,&ds)!=NIYAH_OK) return 1;
    niyah_checkpoint full,split; niyah_checkpoint_init(&full); niyah_checkpoint_init(&split);
    niyah_train_result r;
    if(niyah_train_new(&ds,17u,&full,&r)!=NIYAH_OK) return 2;
    if(niyah_train_new(&ds,7u,&split,&r)!=NIYAH_OK) return 3;
    if(niyah_train_resume(&ds,&split,10u,&r)!=NIYAH_OK) return 4;
    int ok=full.cursor==split.cursor && memcmp(&full.model,&split.model,sizeof(full.model))==0 && memcmp(full.dataset_sha256,split.dataset_sha256,32u)==0;
    niyah_dataset_free(&ds); return ok?0:5;
}
