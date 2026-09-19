#include "niyah/receipt.h"

#include <string.h>

#define CHECK(x) do { if (!(x)) return __LINE__; } while (0)

int main(void)
{
    NiyahExecutionReceipt r;
    char buf[256];
    size_t n = 0U;

    const char expected[] =
        "NIYAH_RECEIPT_V1\n"
        "intent=ADD\n"
        "lhs=2\n"
        "rhs=7\n"
        "result=9\n"
        "ir=ADD|2|7\n";

    memset(&r, 0, sizeof(r));

    r.ir.op = NIYAH_IR_OP_ADD;
    r.ir.lhs = 2;
    r.ir.rhs = 7;
    r.result = 9;

    CHECK(
        niyah_receipt_format(
            &r, NULL, 0U, &n) ==
        NIYAH_ERR_BUFFER_TOO_SMALL);

    CHECK(n == strlen(expected));

    CHECK(
        niyah_receipt_format(
            &r, buf, sizeof(buf), &n) ==
        NIYAH_OK);

    CHECK(strcmp(buf, expected) == 0);
    CHECK(n == strlen(expected));

    r.ir.op = NIYAH_IR_OP_INVALID;

    CHECK(
        niyah_receipt_format(
            &r, buf, sizeof(buf), &n) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    {
        static const uint8_t expected_hash[32] = {
            0x30, 0x7d, 0x9e, 0x6b,
            0x0d, 0xa2, 0xbf, 0x9f,
            0x67, 0x7d, 0x76, 0x24,
            0x41, 0x84, 0x4e, 0x8b,
            0xc9, 0x0c, 0x4a, 0x76,
            0xb6, 0xcb, 0x38, 0x3a,
            0xa0, 0x6f, 0x5f, 0xeb,
            0xe4, 0x0c, 0x8a, 0x14
        };

        uint8_t digest[NIYAH_RECEIPT_SHA256_SIZE];

        r.ir.op = NIYAH_IR_OP_ADD;

        CHECK(
            niyah_receipt_sha256(
                &r, digest) ==
            NIYAH_OK);

        CHECK(
            memcmp(
                digest,
                expected_hash,
                sizeof(digest)) == 0);

        CHECK(
            niyah_receipt_sha256(
                NULL, digest) ==
            NIYAH_ERR_INVALID_ARGUMENT);

        CHECK(
            niyah_receipt_sha256(
                &r, NULL) ==
            NIYAH_ERR_INVALID_ARGUMENT);
    }

    return 0;
}
