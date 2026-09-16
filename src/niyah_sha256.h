#ifndef NIYAH_SHA256_H
#define NIYAH_SHA256_H

#include <stddef.h>
#include <stdint.h>

typedef struct NiyahSha256 {
    uint32_t h[8];
    uint64_t bit_count;
    unsigned char block[64];
    size_t block_used;
} NiyahSha256;

void niyah_sha256_init(NiyahSha256 *state);
int niyah_sha256_update(NiyahSha256 *state,
                        const unsigned char *data,
                        size_t size);
void niyah_sha256_final(NiyahSha256 *state,
                        unsigned char out[32]);

#endif
