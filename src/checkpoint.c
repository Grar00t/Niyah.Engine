#include "niyah/checkpoint.h"
#include "niyah/sha256.h"
#include <stdio.h>
#include <string.h>

static const uint8_t k_magic[8] = {'N','I','Y','C','K','1',0,0};
static const uint32_t k_version = 1u;

static void put_u32_le(uint8_t out[4], uint32_t v) {
    out[0]=(uint8_t)v; out[1]=(uint8_t)(v>>8u); out[2]=(uint8_t)(v>>16u); out[3]=(uint8_t)(v>>24u);
}
static void put_u64_le(uint8_t out[8], uint64_t v) {
    for(size_t i=0;i<8u;++i) out[i]=(uint8_t)(v>>(8u*i));
}
static uint32_t get_u32_le(const uint8_t in[4]) {
    return (uint32_t)in[0] | ((uint32_t)in[1]<<8u) | ((uint32_t)in[2]<<16u) | ((uint32_t)in[3]<<24u);
}
static uint64_t get_u64_le(const uint8_t in[8]) {
    uint64_t v=0u; for(size_t i=0;i<8u;++i) v|=((uint64_t)in[i])<<(8u*i); return v;
}

void niyah_checkpoint_init(niyah_checkpoint *ckpt) {
    if(!ckpt) return;
    niyah_model_init(&ckpt->model);
    ckpt->cursor=0u;
    memset(ckpt->dataset_sha256,0,32u);
}

static void hash_payload(const niyah_checkpoint *ckpt, uint8_t out[32]) {
    niyah_sha256_ctx ctx;
    uint8_t b[8];
    niyah_sha256_init(&ctx);
    put_u64_le(b,ckpt->cursor); niyah_sha256_update(&ctx,b,8u);
    niyah_sha256_update(&ctx,ckpt->dataset_sha256,32u);
    put_u64_le(b,ckpt->model.transitions_seen); niyah_sha256_update(&ctx,b,8u);
    for(uint32_t i=0u;i<NIYAH_MODEL_VOCAB;++i) {
        for(uint32_t j=0u;j<NIYAH_MODEL_VOCAB;++j) {
            put_u64_le(b,ckpt->model.counts[i][j]); niyah_sha256_update(&ctx,b,8u);
        }
    }
    niyah_sha256_final(&ctx,out);
}

niyah_status niyah_checkpoint_save(const niyah_checkpoint *ckpt, const char *path) {
    if(!ckpt || !path) return NIYAH_ERR_INVALID;
    FILE *f=fopen(path,"wb"); if(!f) return NIYAH_ERR_IO;
    uint8_t u32[4],u64[8],digest[32];
    put_u32_le(u32,k_version); hash_payload(ckpt,digest);
    int ok=1;
    ok &= fwrite(k_magic,1,8u,f)==8u;
    ok &= fwrite(u32,1,4u,f)==4u;
    put_u64_le(u64,ckpt->cursor); ok &= fwrite(u64,1,8u,f)==8u;
    ok &= fwrite(ckpt->dataset_sha256,1,32u,f)==32u;
    put_u64_le(u64,ckpt->model.transitions_seen); ok &= fwrite(u64,1,8u,f)==8u;
    ok &= fwrite(digest,1,32u,f)==32u;
    for(uint32_t i=0u; ok && i<NIYAH_MODEL_VOCAB; ++i) {
        for(uint32_t j=0u; ok && j<NIYAH_MODEL_VOCAB; ++j) {
            put_u64_le(u64,ckpt->model.counts[i][j]);
            ok &= fwrite(u64,1,8u,f)==8u;
        }
    }
    if(fclose(f)!=0) ok=0;
    return ok ? NIYAH_OK : NIYAH_ERR_IO;
}

niyah_status niyah_checkpoint_load(const char *path, niyah_checkpoint *out) {
    if(!path || !out) return NIYAH_ERR_INVALID;
    FILE *f=fopen(path,"rb"); if(!f) return NIYAH_ERR_IO;
    niyah_checkpoint tmp; niyah_checkpoint_init(&tmp);
    uint8_t magic[8],u32[4],u64[8],stored_digest[32];
    niyah_status st=NIYAH_OK;
    if(fread(magic,1,8u,f)!=8u || memcmp(magic,k_magic,8u)!=0) st=NIYAH_ERR_FORMAT;
    if(st==NIYAH_OK && fread(u32,1,4u,f)!=4u) st=NIYAH_ERR_FORMAT;
    if(st==NIYAH_OK && get_u32_le(u32)!=k_version) st=NIYAH_ERR_FORMAT;
    if(st==NIYAH_OK && fread(u64,1,8u,f)!=8u) st=NIYAH_ERR_FORMAT;
    if(st==NIYAH_OK) tmp.cursor=get_u64_le(u64);
    if(st==NIYAH_OK && fread(tmp.dataset_sha256,1,32u,f)!=32u) st=NIYAH_ERR_FORMAT;
    if(st==NIYAH_OK && fread(u64,1,8u,f)!=8u) st=NIYAH_ERR_FORMAT;
    if(st==NIYAH_OK) tmp.model.transitions_seen=get_u64_le(u64);
    if(st==NIYAH_OK && fread(stored_digest,1,32u,f)!=32u) st=NIYAH_ERR_FORMAT;
    for(uint32_t i=0u; st==NIYAH_OK && i<NIYAH_MODEL_VOCAB; ++i) {
        for(uint32_t j=0u; st==NIYAH_OK && j<NIYAH_MODEL_VOCAB; ++j) {
            if(fread(u64,1,8u,f)!=8u) { st=NIYAH_ERR_FORMAT; break; }
            tmp.model.counts[i][j]=get_u64_le(u64);
        }
    }
    if(st==NIYAH_OK && fgetc(f)!=EOF) st=NIYAH_ERR_FORMAT;
    fclose(f);
    if(st==NIYAH_OK) {
        uint8_t actual[32]; hash_payload(&tmp,actual);
        if(memcmp(actual,stored_digest,32u)!=0) st=NIYAH_ERR_MISMATCH;
    }
    if(st==NIYAH_OK) *out=tmp;
    return st;
}

niyah_status niyah_checkpoint_require_dataset(const niyah_checkpoint *ckpt, const uint8_t dataset_sha256[32]) {
    if(!ckpt || !dataset_sha256) return NIYAH_ERR_INVALID;
    return memcmp(ckpt->dataset_sha256,dataset_sha256,32u)==0 ? NIYAH_OK : NIYAH_ERR_MISMATCH;
}
