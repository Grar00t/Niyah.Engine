#include "niyah_sha256.h"

#include <string.h>

static uint32_t niyah_sha256_rotr(uint32_t x, unsigned n)
{
    return (x >> n) | (x << (32U - n));
}

static void niyah_sha256_transform(NiyahSha256 *s,
                                   const unsigned char block[64])
{
    static const uint32_t k[64] = {
        UINT32_C(0x428a2f98), UINT32_C(0x71374491), UINT32_C(0xb5c0fbcf), UINT32_C(0xe9b5dba5),
        UINT32_C(0x3956c25b), UINT32_C(0x59f111f1), UINT32_C(0x923f82a4), UINT32_C(0xab1c5ed5),
        UINT32_C(0xd807aa98), UINT32_C(0x12835b01), UINT32_C(0x243185be), UINT32_C(0x550c7dc3),
        UINT32_C(0x72be5d74), UINT32_C(0x80deb1fe), UINT32_C(0x9bdc06a7), UINT32_C(0xc19bf174),
        UINT32_C(0xe49b69c1), UINT32_C(0xefbe4786), UINT32_C(0x0fc19dc6), UINT32_C(0x240ca1cc),
        UINT32_C(0x2de92c6f), UINT32_C(0x4a7484aa), UINT32_C(0x5cb0a9dc), UINT32_C(0x76f988da),
        UINT32_C(0x983e5152), UINT32_C(0xa831c66d), UINT32_C(0xb00327c8), UINT32_C(0xbf597fc7),
        UINT32_C(0xc6e00bf3), UINT32_C(0xd5a79147), UINT32_C(0x06ca6351), UINT32_C(0x14292967),
        UINT32_C(0x27b70a85), UINT32_C(0x2e1b2138), UINT32_C(0x4d2c6dfc), UINT32_C(0x53380d13),
        UINT32_C(0x650a7354), UINT32_C(0x766a0abb), UINT32_C(0x81c2c92e), UINT32_C(0x92722c85),
        UINT32_C(0xa2bfe8a1), UINT32_C(0xa81a664b), UINT32_C(0xc24b8b70), UINT32_C(0xc76c51a3),
        UINT32_C(0xd192e819), UINT32_C(0xd6990624), UINT32_C(0xf40e3585), UINT32_C(0x106aa070),
        UINT32_C(0x19a4c116), UINT32_C(0x1e376c08), UINT32_C(0x2748774c), UINT32_C(0x34b0bcb5),
        UINT32_C(0x391c0cb3), UINT32_C(0x4ed8aa4a), UINT32_C(0x5b9cca4f), UINT32_C(0x682e6ff3),
        UINT32_C(0x748f82ee), UINT32_C(0x78a5636f), UINT32_C(0x84c87814), UINT32_C(0x8cc70208),
        UINT32_C(0x90befffa), UINT32_C(0xa4506ceb), UINT32_C(0xbef9a3f7), UINT32_C(0xc67178f2)
    };
    uint32_t w[64];
    uint32_t a, b, c, d, e, f, g, h;
    size_t i;

    for (i = 0U; i < 16U; ++i) {
        const size_t j = i * 4U;
        w[i] = ((uint32_t)block[j] << 24) |
               ((uint32_t)block[j + 1U] << 16) |
               ((uint32_t)block[j + 2U] << 8) |
               (uint32_t)block[j + 3U];
    }
    for (i = 16U; i < 64U; ++i) {
        const uint32_t x = w[i - 15U];
        const uint32_t y = w[i - 2U];
        const uint32_t s0 = niyah_sha256_rotr(x, 7U) ^
                            niyah_sha256_rotr(x, 18U) ^ (x >> 3U);
        const uint32_t s1 = niyah_sha256_rotr(y, 17U) ^
                            niyah_sha256_rotr(y, 19U) ^ (y >> 10U);
        w[i] = w[i - 16U] + s0 + w[i - 7U] + s1;
    }

    a = s->h[0]; b = s->h[1]; c = s->h[2]; d = s->h[3];
    e = s->h[4]; f = s->h[5]; g = s->h[6]; h = s->h[7];

    for (i = 0U; i < 64U; ++i) {
        const uint32_t s1 = niyah_sha256_rotr(e, 6U) ^
                            niyah_sha256_rotr(e, 11U) ^
                            niyah_sha256_rotr(e, 25U);
        const uint32_t ch = (e & f) ^ ((~e) & g);
        const uint32_t t1 = h + s1 + ch + k[i] + w[i];
        const uint32_t s0 = niyah_sha256_rotr(a, 2U) ^
                            niyah_sha256_rotr(a, 13U) ^
                            niyah_sha256_rotr(a, 22U);
        const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t t2 = s0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    s->h[0] += a; s->h[1] += b; s->h[2] += c; s->h[3] += d;
    s->h[4] += e; s->h[5] += f; s->h[6] += g; s->h[7] += h;
}

void niyah_sha256_init(NiyahSha256 *s)
{
    static const uint32_t initial[8] = {
        UINT32_C(0x6a09e667), UINT32_C(0xbb67ae85),
        UINT32_C(0x3c6ef372), UINT32_C(0xa54ff53a),
        UINT32_C(0x510e527f), UINT32_C(0x9b05688c),
        UINT32_C(0x1f83d9ab), UINT32_C(0x5be0cd19)
    };
    memcpy(s->h, initial, sizeof(initial));
    s->bit_count = UINT64_C(0);
    s->block_used = 0U;
}

int niyah_sha256_update(NiyahSha256 *s,
                        const unsigned char *data,
                        size_t size)
{
    size_t offset = 0U;

    if (size > (size_t)(UINT64_MAX / UINT64_C(8)) ||
        s->bit_count > UINT64_MAX - (uint64_t)size * UINT64_C(8)) {
        return 0;
    }
    s->bit_count += (uint64_t)size * UINT64_C(8);

    while (offset < size) {
        const size_t available = 64U - s->block_used;
        const size_t remaining = size - offset;
        const size_t take = remaining < available ? remaining : available;

        memcpy(s->block + s->block_used, data + offset, take);
        s->block_used += take;
        offset += take;

        if (s->block_used == 64U) {
            niyah_sha256_transform(s, s->block);
            s->block_used = 0U;
        }
    }
    return 1;
}

void niyah_sha256_final(NiyahSha256 *s, unsigned char out[32])
{
    size_t i;
    uint64_t bits = s->bit_count;

    s->block[s->block_used++] = 0x80U;
    if (s->block_used > 56U) {
        memset(s->block + s->block_used, 0, 64U - s->block_used);
        niyah_sha256_transform(s, s->block);
        s->block_used = 0U;
    }
    memset(s->block + s->block_used, 0, 56U - s->block_used);

    for (i = 0U; i < 8U; ++i) {
        s->block[63U - i] = (unsigned char)(bits & UINT64_C(0xff));
        bits >>= 8U;
    }
    niyah_sha256_transform(s, s->block);

    for (i = 0U; i < 8U; ++i) {
        out[i * 4U + 0U] = (unsigned char)(s->h[i] >> 24);
        out[i * 4U + 1U] = (unsigned char)(s->h[i] >> 16);
        out[i * 4U + 2U] = (unsigned char)(s->h[i] >> 8);
        out[i * 4U + 3U] = (unsigned char)s->h[i];
    }
}
