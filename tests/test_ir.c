#include "niyah/ir.h"

#include <stdint.h>

#define CHECK(x) \
    do { \
        if (!(x)) return __LINE__; \
    } while (0)

static int check_valid(
    const char *text,
    NiyahIrOp op,
    int64_t lhs,
    int64_t rhs,
    int64_t expected)
{
    NiyahIr ir;
    int64_t result = 0;

    CHECK(niyah_ir_parse(text, &ir) == NIYAH_OK);
    CHECK(ir.op == op);
    CHECK(ir.lhs == lhs);
    CHECK(ir.rhs == rhs);

    CHECK(
        niyah_ir_execute(&ir, &result) ==
        NIYAH_OK);

    CHECK(result == expected);

    return 0;
}

int main(void)
{
    NiyahIr ir;
    int64_t result = 0;

    CHECK(
        check_valid(
            "ADD|2|7",
            NIYAH_IR_OP_ADD,
            2, 7, 9) == 0);

    CHECK(
        check_valid(
            "SUB|9|4",
            NIYAH_IR_OP_SUB,
            9, 4, 5) == 0);

    CHECK(
        check_valid(
            "SUB|2|7",
            NIYAH_IR_OP_SUB,
            2, 7, -5) == 0);

    CHECK(
        check_valid(
            "MUL|4|5",
            NIYAH_IR_OP_MUL,
            4, 5, 20) == 0);

    CHECK(
        check_valid(
            "ADD|0|0",
            NIYAH_IR_OP_ADD,
            0, 0, 0) == 0);

    CHECK(
        niyah_ir_parse(
            NULL, &ir) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    CHECK(
        niyah_ir_parse(
            "", &ir) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    CHECK(
        niyah_ir_parse(
            "ADD|2", &ir) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    CHECK(
        niyah_ir_parse(
            "ADD|2|7|9", &ir) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    CHECK(
        niyah_ir_parse(
            "DIV|8|2", &ir) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    CHECK(
        niyah_ir_parse(
            "ADD|x|2", &ir) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    CHECK(
        niyah_ir_parse(
            "ADD|-1|2", &ir) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    CHECK(
        niyah_ir_parse(
            "ADD|2| 7", &ir) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    CHECK(
        niyah_ir_parse(
            "ADD|9223372036854775808|1",
            &ir) ==
        NIYAH_ERR_OVERFLOW);

    ir.op = NIYAH_IR_OP_MUL;
    ir.lhs = INT64_MAX;
    ir.rhs = 2;

    CHECK(
        niyah_ir_execute(
            &ir, &result) ==
        NIYAH_ERR_OVERFLOW);

    ir.op = NIYAH_IR_OP_ADD;
    ir.lhs = INT64_MAX;
    ir.rhs = 1;

    CHECK(
        niyah_ir_execute(
            &ir, &result) ==
        NIYAH_ERR_OVERFLOW);

    ir.op = NIYAH_IR_OP_INVALID;
    ir.lhs = 1;
    ir.rhs = 1;

    CHECK(
        niyah_ir_execute(
            &ir, &result) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    return 0;
}
