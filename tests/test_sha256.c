#include "niyah_sha256.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int check(int condition, const char *message)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        return 0;
    }
    return 1;
}

static int test_abc_known_answer(void)
{
    static const unsigned char input[] = {'a', 'b', 'c'};
    static const unsigned char expected[32] = {
        0xbaU, 0x78U, 0x16U, 0xbfU, 0x8fU, 0x01U, 0xcfU, 0xeaU,
        0x41U, 0x41U, 0x40U, 0xdeU, 0x5dU, 0xaeU, 0x22U, 0x23U,
        0xb0U, 0x03U, 0x61U, 0xa3U, 0x96U, 0x17U, 0x7aU, 0x9cU,
        0xb4U, 0x10U, 0xffU, 0x61U, 0xf2U, 0x00U, 0x15U, 0xadU
    };
    NiyahSha256 sha;
    unsigned char digest[32];

    niyah_sha256_init(&sha);
    if (!check(niyah_sha256_update(&sha, input, sizeof(input)) == 1,
               "abc update")) {
        return 0;
    }
    niyah_sha256_final(&sha, digest);
    return check(memcmp(digest, expected, sizeof(expected)) == 0,
                 "abc digest");
}

static int test_update_bit_count_overflow(void)
{
    static const unsigned char byte = 0U;
    NiyahSha256 sha;

    niyah_sha256_init(&sha);
    sha.bit_count = UINT64_MAX - UINT64_C(7);
    return check(niyah_sha256_update(&sha, &byte, 1U) == 0,
                 "bit_count overflow rejection");
}

int main(void)
{
    if (!test_abc_known_answer()) return 1;
    if (!test_update_bit_count_overflow()) return 1;
    puts("PASS");
    return 0;
}
