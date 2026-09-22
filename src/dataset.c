#include "niyah/dataset.h"
#include "niyah/sha256.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const uint8_t k_magic[8] = {'N','I','Y','D','S','1',0,0};
static const uint32_t k_version = 1u;

static void put_u32_le(uint8_t out[4], uint32_t v) {
    out[0]=(uint8_t)v; out[1]=(uint8_t)(v>>8u); out[2]=(uint8_t)(v>>16u); out[3]=(uint8_t)(v>>24u);
}
static void put_u64_le(uint8_t out[8], uint64_t v) {
    for (size_t i=0;i<8u;++i) out[i]=(uint8_t)(v>>(8u*i));
}
static uint32_t get_u32_le(const uint8_t in[4]) {
    return (uint32_t)in[0] | ((uint32_t)in[1]<<8u) | ((uint32_t)in[2]<<16u) | ((uint32_t)in[3]<<24u);
}
static uint64_t get_u64_le(const uint8_t in[8]) {
    uint64_t v=0u; for(size_t i=0;i<8u;++i) v|=((uint64_t)in[i])<<(8u*i); return v;
}

static void compute_hash(const uint32_t *tokens, uint64_t count, uint8_t out[32]) {
    niyah_sha256_ctx ctx;
    niyah_sha256_init(&ctx);
    uint8_t b[4];
    for (uint64_t i=0;i<count;++i) {
        put_u32_le(b, tokens[i]);
        niyah_sha256_update(&ctx,b,4u);
    }
    niyah_sha256_final(&ctx,out);
}

void niyah_dataset_init(niyah_dataset *dataset) {
    if (!dataset) return;
    dataset->tokens=NULL; dataset->token_count=0u; memset(dataset->content_sha256,0,32u);
}

void niyah_dataset_free(niyah_dataset *dataset) {
    if (!dataset) return;
    free(dataset->tokens); niyah_dataset_init(dataset);
}

niyah_status niyah_dataset_from_tokens(const uint32_t *tokens, uint64_t token_count, niyah_dataset *out) {
    if ((!tokens && token_count != 0u) || !out) return NIYAH_ERR_INVALID;
    if (token_count > (uint64_t)(SIZE_MAX / sizeof(uint32_t))) return NIYAH_ERR_NOMEM;
    niyah_dataset tmp; niyah_dataset_init(&tmp);
    if (token_count != 0u) {
        tmp.tokens=(uint32_t*)malloc((size_t)token_count*sizeof(uint32_t));
        if (!tmp.tokens) return NIYAH_ERR_NOMEM;
        memcpy(tmp.tokens,tokens,(size_t)token_count*sizeof(uint32_t));
    }
    tmp.token_count=token_count;
    compute_hash(tokens,token_count,tmp.content_sha256);
    niyah_dataset_free(out);
    *out=tmp;
    return NIYAH_OK;
}

niyah_status niyah_dataset_save(const niyah_dataset *dataset, const char *path) {
    if (!dataset || !path || (!dataset->tokens && dataset->token_count != 0u)) return NIYAH_ERR_INVALID;
    FILE *f=fopen(path,"wb"); if(!f) return NIYAH_ERR_IO;
    uint8_t u32[4],u64[8]; put_u32_le(u32,k_version); put_u64_le(u64,dataset->token_count);
    int ok=1;
    ok &= fwrite(k_magic,1,8u,f)==8u;
    ok &= fwrite(u32,1,4u,f)==4u;
    ok &= fwrite(u64,1,8u,f)==8u;
    ok &= fwrite(dataset->content_sha256,1,32u,f)==32u;
    for(uint64_t i=0; ok && i<dataset->token_count; ++i) {
        put_u32_le(u32,dataset->tokens[i]);
        ok &= fwrite(u32,1,4u,f)==4u;
    }
    if (fclose(f)!=0) ok=0;
    return ok ? NIYAH_OK : NIYAH_ERR_IO;
}

niyah_status niyah_dataset_load(const char *path, niyah_dataset *out) {
    if(!path || !out) return NIYAH_ERR_INVALID;
    FILE *f=fopen(path,"rb"); if(!f) return NIYAH_ERR_IO;
    uint8_t magic[8],u32[4],u64[8],stored_hash[32];
    niyah_status st=NIYAH_OK;
    if(fread(magic,1,8u,f)!=8u || memcmp(magic,k_magic,8u)!=0) st=NIYAH_ERR_FORMAT;
    if(st==NIYAH_OK && fread(u32,1,4u,f)!=4u) st=NIYAH_ERR_FORMAT;
    if(st==NIYAH_OK && get_u32_le(u32)!=k_version) st=NIYAH_ERR_FORMAT;
    if(st==NIYAH_OK && fread(u64,1,8u,f)!=8u) st=NIYAH_ERR_FORMAT;
    uint64_t count= st==NIYAH_OK ? get_u64_le(u64) : 0u;
    if(st==NIYAH_OK && fread(stored_hash,1,32u,f)!=32u) st=NIYAH_ERR_FORMAT;
    if(st==NIYAH_OK && count > (uint64_t)(SIZE_MAX/sizeof(uint32_t))) st=NIYAH_ERR_NOMEM;
    uint32_t *tokens=NULL;
    if(st==NIYAH_OK && count!=0u) {
        tokens=(uint32_t*)malloc((size_t)count*sizeof(uint32_t));
        if(!tokens) st=NIYAH_ERR_NOMEM;
    }
    for(uint64_t i=0; st==NIYAH_OK && i<count; ++i) {
        if(fread(u32,1,4u,f)!=4u) { st=NIYAH_ERR_FORMAT; break; }
        tokens[i]=get_u32_le(u32);
        if(tokens[i] >= 256u) st=NIYAH_ERR_FORMAT;
    }
    if(st==NIYAH_OK) {
        int extra=fgetc(f);
        if(extra!=EOF) st=NIYAH_ERR_FORMAT;
    }
    fclose(f);
    if(st==NIYAH_OK) {
        uint8_t actual[32]; compute_hash(tokens,count,actual);
        if(memcmp(actual,stored_hash,32u)!=0) st=NIYAH_ERR_MISMATCH;
        else {
            niyah_dataset_free(out);
            out->tokens=tokens; out->token_count=count; memcpy(out->content_sha256,actual,32u);
            tokens=NULL;
        }
    }
    free(tokens);
    return st;
}
