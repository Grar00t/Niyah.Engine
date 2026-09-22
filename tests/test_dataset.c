#include "niyah/dataset.h"
#include <stdio.h>
#include <string.h>
int main(void) {
    const uint32_t t[]={1u,2u,3u,2u,1u};
    niyah_dataset a,b; niyah_dataset_init(&a); niyah_dataset_init(&b);
    if(niyah_dataset_from_tokens(t,5u,&a)!=NIYAH_OK) return 1;
    const char *p="test_dataset.srd";
    if(niyah_dataset_save(&a,p)!=NIYAH_OK) return 2;
    if(niyah_dataset_load(p,&b)!=NIYAH_OK) return 3;
    int ok=b.token_count==5u && memcmp(a.content_sha256,b.content_sha256,32u)==0 && memcmp(a.tokens,b.tokens,5u*sizeof(uint32_t))==0;
    remove(p); niyah_dataset_free(&a); niyah_dataset_free(&b); return ok?0:4;
}
