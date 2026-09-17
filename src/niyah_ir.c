#include "niyah/ir.h"

#include <stdint.h>
#include <string.h>

static NiyahStatus parse_operand(
    const char *begin,
    const char *end,
    int64_t *out)
{
    uint64_t value = 0U;
    const uint64_t limit = (uint64_t)INT64_MAX;
    const char *p;

    if (begin == NULL ||
        end == NULL ||
        out == NULL ||
        begin >= end) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    for (p = begin; p < end; ++p) {
        uint64_t digit;

        if (*p < '0' || *p > '9') {
            return NIYAH_ERR_INVALID_ARGUMENT;
        }

        digit = (uint64_t)(*p - '0');

        if (value > (limit - digit) / 10U) {
            return NIYAH_ERR_OVERFLOW;
        }

        value = value * 10U + digit;
    }

    *out = (int64_t)value;
    return NIYAH_OK;
}

NiyahStatus niyah_ir_parse(
    const char *text,
    NiyahIr *out_ir)
{
    NiyahIr parsed;
    const char *first;
    const char *second;
    const char *end;
    size_t op_len;
    NiyahStatus status;

    if (text == NULL || out_ir == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    memset(out_ir, 0, sizeof(*out_ir));
    memset(&parsed, 0, sizeof(parsed));

    first = strchr(text, '|');
    if (first == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    second = strchr(first + 1, '|');
    if (second == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    if (strchr(second + 1, '|') != NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    end = text + strlen(text);

    if (first == text ||
        first + 1 == second ||
        second + 1 == end) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    op_len = (size_t)(first - text);

    if (op_len == 3U && memcmp(text, "ADD", 3U) == 0) {
        parsed.op = NIYAH_IR_OP_ADD;
    } else if (
        op_len == 3U &&
        memcmp(text, "SUB", 3U) == 0) {
        parsed.op = NIYAH_IR_OP_SUB;
    } else if (
        op_len == 3U &&
        memcmp(text, "MUL", 3U) == 0) {
        parsed.op = NIYAH_IR_OP_MUL;
    } else {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    status = parse_operand(
        first + 1,
        second,
        &parsed.lhs);

    if (status != NIYAH_OK) {
        return status;
    }

    status = parse_operand(
        second + 1,
        end,
        &parsed.rhs);

    if (status != NIYAH_OK) {
        return status;
    }

    *out_ir = parsed;
    return NIYAH_OK;
}

NiyahStatus niyah_ir_execute(
    const NiyahIr *ir,
    int64_t *out_result)
{
    if (ir == NULL || out_result == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    if (ir->lhs < 0 || ir->rhs < 0) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    switch (ir->op) {
        case NIYAH_IR_OP_ADD:
            if (ir->rhs > INT64_MAX - ir->lhs) {
                return NIYAH_ERR_OVERFLOW;
            }

            *out_result = ir->lhs + ir->rhs;
            return NIYAH_OK;

        case NIYAH_IR_OP_SUB:
            *out_result = ir->lhs - ir->rhs;
            return NIYAH_OK;

        case NIYAH_IR_OP_MUL:
            if (ir->lhs != 0 &&
                ir->rhs > INT64_MAX / ir->lhs) {
                return NIYAH_ERR_OVERFLOW;
            }

            *out_result = ir->lhs * ir->rhs;
            return NIYAH_OK;

        default:
            return NIYAH_ERR_INVALID_ARGUMENT;
    }
}
