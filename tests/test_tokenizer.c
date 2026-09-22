#include "niyah/tokenizer.h"
#include <string.h>
int main(void) {
    const uint8_t src[]="مرحبا Niyah";
    uint32_t tokens[sizeof(src)-1u]; uint8_t dst[sizeof(src)-1u];
    size_t nt=0u,nb=0u; niyah_tokenizer tok;
    if(niyah_tokenizer_init_byte(&tok)!=NIYAH_OK) return 1;
    if(niyah_tokenizer_encode(&tok,src,sizeof(src)-1u,tokens,sizeof(tokens)/sizeof(tokens[0]),&nt)!=NIYAH_OK) return 2;
    if(niyah_tokenizer_decode(&tok,tokens,nt,dst,sizeof(dst),&nb)!=NIYAH_OK) return 3;
    return nb==sizeof(src)-1u && memcmp(src,dst,nb)==0 ? 0 : 4;
}
