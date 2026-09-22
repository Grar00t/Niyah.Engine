#include "niyah/sha256.h"
#include <string.h>

#define ROTR(x,n) (((x) >> (n)) | ((x) << (32u - (n))))
#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTR((x),2u) ^ ROTR((x),13u) ^ ROTR((x),22u))
#define EP1(x) (ROTR((x),6u) ^ ROTR((x),11u) ^ ROTR((x),25u))
#define SIG0(x) (ROTR((x),7u) ^ ROTR((x),18u) ^ ((x) >> 3u))
#define SIG1(x) (ROTR((x),17u) ^ ROTR((x),19u) ^ ((x) >> 10u))

static const uint32_t k[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};

static void transform(niyah_sha256_ctx *ctx, const uint8_t data[64]) {
    uint32_t m[64];
    uint32_t a,b,c,d,e,f,g,h;
    for (size_t i = 0; i < 16u; ++i) {
        size_t j = i * 4u;
        m[i] = ((uint32_t)data[j] << 24u) |
               ((uint32_t)data[j+1u] << 16u) |
               ((uint32_t)data[j+2u] << 8u) |
               ((uint32_t)data[j+3u]);
    }
    for (size_t i = 16u; i < 64u; ++i) {
        m[i] = SIG1(m[i-2u]) + m[i-7u] + SIG0(m[i-15u]) + m[i-16u];
    }
    a=ctx->state[0]; b=ctx->state[1]; c=ctx->state[2]; d=ctx->state[3];
    e=ctx->state[4]; f=ctx->state[5]; g=ctx->state[6]; h=ctx->state[7];
    for (size_t i = 0; i < 64u; ++i) {
        uint32_t t1 = h + EP1(e) + CH(e,f,g) + k[i] + m[i];
        uint32_t t2 = EP0(a) + MAJ(a,b,c);
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    ctx->state[0]+=a; ctx->state[1]+=b; ctx->state[2]+=c; ctx->state[3]+=d;
    ctx->state[4]+=e; ctx->state[5]+=f; ctx->state[6]+=g; ctx->state[7]+=h;
}

void niyah_sha256_init(niyah_sha256_ctx *ctx) {
    if (!ctx) return;
    ctx->datalen = 0u;
    ctx->bitlen = 0u;
    ctx->state[0]=0x6a09e667u; ctx->state[1]=0xbb67ae85u; ctx->state[2]=0x3c6ef372u; ctx->state[3]=0xa54ff53au;
    ctx->state[4]=0x510e527fu; ctx->state[5]=0x9b05688cu; ctx->state[6]=0x1f83d9abu; ctx->state[7]=0x5be0cd19u;
}

void niyah_sha256_update(niyah_sha256_ctx *ctx, const void *data_ptr, size_t len) {
    if (!ctx || (!data_ptr && len != 0u)) return;
    const uint8_t *data = (const uint8_t *)data_ptr;
    for (size_t i = 0; i < len; ++i) {
        ctx->data[ctx->datalen++] = data[i];
        if (ctx->datalen == 64u) {
            transform(ctx, ctx->data);
            ctx->bitlen += 512u;
            ctx->datalen = 0u;
        }
    }
}

void niyah_sha256_final(niyah_sha256_ctx *ctx, uint8_t out[32]) {
    if (!ctx || !out) return;
    size_t i = ctx->datalen;
    ctx->data[i++] = 0x80u;
    if (i > 56u) {
        while (i < 64u) ctx->data[i++] = 0u;
        transform(ctx, ctx->data);
        i = 0u;
    }
    while (i < 56u) ctx->data[i++] = 0u;
    ctx->bitlen += (uint64_t)ctx->datalen * 8u;
    for (size_t j = 0; j < 8u; ++j) {
        ctx->data[63u-j] = (uint8_t)(ctx->bitlen >> (8u*j));
    }
    transform(ctx, ctx->data);
    for (size_t j = 0; j < 8u; ++j) {
        out[j*4u]     = (uint8_t)(ctx->state[j] >> 24u);
        out[j*4u+1u] = (uint8_t)(ctx->state[j] >> 16u);
        out[j*4u+2u] = (uint8_t)(ctx->state[j] >> 8u);
        out[j*4u+3u] = (uint8_t)ctx->state[j];
    }
}

void niyah_sha256(const void *data, size_t len, uint8_t out[32]) {
    niyah_sha256_ctx ctx;
    niyah_sha256_init(&ctx);
    niyah_sha256_update(&ctx, data, len);
    niyah_sha256_final(&ctx, out);
}

void niyah_sha256_hex(const uint8_t digest[32], char out_hex[65]) {
    static const char hex[] = "0123456789abcdef";
    if (!digest || !out_hex) return;
    for (size_t i = 0; i < 32u; ++i) {
        out_hex[i*2u] = hex[(digest[i] >> 4u) & 0x0fu];
        out_hex[i*2u+1u] = hex[digest[i] & 0x0fu];
    }
    out_hex[64] = '\0';
}
