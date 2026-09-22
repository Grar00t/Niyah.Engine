#include "../src/niyah_sha256.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int test_abc_known_answer(void)
{
    static const unsigned char expected[32] = {
        0xbaU, 0x78U, 0x16U, 0xbfU, 0x8fU, 0x01U, 0xcfU, 0xeaU,
        0x41U, 0x41U, 0x40U, 0xdeU, 0x5dU, 0xaeU, 0x22U, 0x23U,
        0xb0U, 0x03U, 0x61U, 0xa3U, 0x96U, 0x17U, 0x7aU, 0x9cU,
        0xb4U, 0x10U, 0xffU, 0x61U, 0xf2U, 0x00U, 0x15U, 0xadU
    };
    static const unsigned char abc[] = {'a', 'b', 'c'};
    NiyahSha256 sha;
    unsigned char digest[32];

    niyah_sha256_init(&sha);
    if (!niyah_sha256_update(&sha, abc, sizeof(abc))) {
        fprintf(stderr, "SHA-256 update rejected abc\n");
        return 1;
    }
    niyah_sha256_final(&sha, digest);
    if (memcmp(digest, expected, sizeof(expected)) != 0) {
        fprintf(stderr, "SHA-256 abc KAT mismatch\n");
        return 1;
    }
    return 0;
}

static int test_bit_count_overflow(void)
{
    static const unsigned char byte = 0U;
    const uint64_t before = UINT64_MAX - UINT64_C(7);
    NiyahSha256 sha;

    niyah_sha256_init(&sha);
    sha.bit_count = before;
    if (niyah_sha256_update(&sha, &byte, 1U) != 0) {
        fprintf(stderr, "SHA-256 overflow guard did not fire\n");
        return 1;
    }
    if (sha.bit_count != before) {
        fprintf(stderr, "SHA-256 overflow guard mutated bit_count\n");
        return 1;
    }
    return 0;
}

int main(void)
{
    if (test_abc_known_answer() != 0) {
        return 1;
    }
    return test_bit_count_overflow();
}
