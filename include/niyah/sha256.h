#ifndef NIYAH_SHA256_H
#define NIYAH_SHA256_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct niyah_sha256_ctx {
    uint32_t state[8];
    uint64_t bitlen;
    uint8_t data[64];
    size_t datalen;
} niyah_sha256_ctx;

void niyah_sha256_init(niyah_sha256_ctx *ctx);
void niyah_sha256_update(niyah_sha256_ctx *ctx, const void *data, size_t len);
void niyah_sha256_final(niyah_sha256_ctx *ctx, uint8_t out[32]);
void niyah_sha256(const void *data, size_t len, uint8_t out[32]);
void niyah_sha256_hex(const uint8_t digest[32], char out_hex[65]);

#ifdef __cplusplus
}
#endif

#endif
