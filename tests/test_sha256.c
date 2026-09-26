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

static int test_abc(void)
{
    static const unsigned char expected[32] = {
        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
        0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
        0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
        0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad
    };
    static const unsigned char input[] = "abc";
    NiyahSha256 sha;
    unsigned char digest[32];

    niyah_sha256_init(&sha);
    if (!check(niyah_sha256_update(&sha, input, 3U) != 0,
               "abc update")) {
        return 0;
    }
    niyah_sha256_final(&sha, digest);
    return check(memcmp(digest, expected, sizeof(expected)) == 0,
                 "abc digest");
}

static int test_overflow_guard(void)
{
    const unsigned char byte = 0U;
    NiyahSha256 sha;

    niyah_sha256_init(&sha);
    sha.bit_count = UINT64_MAX - UINT64_C(7);
    return check(niyah_sha256_update(&sha, &byte, 1U) == 0,
                 "bit count overflow guard");
}

int main(void)
{
    if (!test_abc()) {
        return 1;
    }
    if (!test_overflow_guard()) {
        return 1;
    }
    puts("PASS");
    return 0;
}
