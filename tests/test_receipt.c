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

    return 0;
}
