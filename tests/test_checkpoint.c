#include "niyah/checkpoint.h"
#include <stdio.h>
#include <string.h>
int main(void) {
    niyah_checkpoint a,b; niyah_checkpoint_init(&a); niyah_checkpoint_init(&b);
    a.cursor=7u; a.dataset_sha256[0]=42u;
    if(niyah_model_observe(&a.model,'a','b')!=NIYAH_OK) return 1;
    if(niyah_checkpoint_save(&a,"test.ckpt")!=NIYAH_OK) return 2;
    if(niyah_checkpoint_load("test.ckpt",&b)!=NIYAH_OK) return 3;
    remove("test.ckpt");
    return b.cursor==7u && b.dataset_sha256[0]==42u && b.model.counts['a']['b']==1u && b.model.transitions_seen==1u ? 0 : 4;
}
