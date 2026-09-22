#include "niyah_sha256.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int test_abc_kat(void)
{
    static const unsigned char expected[32] = {
        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
        0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
        0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
        0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad
    };
    static const unsigned char input[] = {'a', 'b', 'c'};
    NiyahSha256 state;
    unsigned char digest[32];

    niyah_sha256_init(&state);
    if (!niyah_sha256_update(&state, input, sizeof(input))) {
        fprintf(stderr, "abc update unexpectedly failed\n");
        return 0;
    }
    niyah_sha256_final(&state, digest);
    if (memcmp(digest, expected, sizeof(expected)) != 0) {
        fprintf(stderr, "abc SHA-256 KAT mismatch\n");
        return 0;
    }
    return 1;
}

static int test_update_overflow(void)
{
    NiyahSha256 state;
    const unsigned char byte = 0U;

    niyah_sha256_init(&state);
    state.bit_count = UINT64_MAX - UINT64_C(7);
    if (niyah_sha256_update(&state, &byte, 1U) != 0) {
        fprintf(stderr, "overflow update unexpectedly succeeded\n");
        return 0;
    }
    if (state.bit_count != UINT64_MAX - UINT64_C(7)) {
        fprintf(stderr, "overflow update mutated bit_count\n");
        return 0;
    }
    return 1;
}

int main(void)
{
    if (!test_abc_kat()) {
        return 1;
    }
    if (!test_update_overflow()) {
        return 1;
    }
    return 0;
}
