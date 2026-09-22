#include "niyah/sha256.h"
#include <stdio.h>
#include <string.h>
int main(void) {
    uint8_t d[32]; char h[65];
    niyah_sha256("abc",3u,d); niyah_sha256_hex(d,h);
    const char *exp="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    if(strcmp(h,exp)!=0) { fprintf(stderr,"got=%s\n",h); return 1; }
    return 0;
}
