#include "niyah/receipt.h"
#include "niyah_sha256.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

static const char *op_name(NiyahIrOp op)
{
    switch (op) {
        case NIYAH_IR_OP_ADD: return "ADD";
        case NIYAH_IR_OP_SUB: return "SUB";
        case NIYAH_IR_OP_MUL: return "MUL";
        default: return NULL;
    }
}

NiyahStatus niyah_receipt_format(
    const NiyahExecutionReceipt *receipt,
    char *out,
    size_t out_size,
    size_t *out_length)
{
    const char *op;
    int needed;

    if (receipt == NULL || out_length == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    op = op_name(receipt->ir.op);
    if (op == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    needed = snprintf(
        NULL,
        0,
        "NIYAH_RECEIPT_V1\n"
        "intent=%s\n"
        "lhs=%" PRId64 "\n"
        "rhs=%" PRId64 "\n"
        "result=%" PRId64 "\n"
        "ir=%s|%" PRId64 "|%" PRId64 "\n",
        op,
        receipt->ir.lhs,
        receipt->ir.rhs,
        receipt->result,
        op,
        receipt->ir.lhs,
        receipt->ir.rhs);

    if (needed < 0) {
        return NIYAH_ERR_IO;
    }

    *out_length = (size_t)needed;

    if (out == NULL || out_size <= (size_t)needed) {
        return NIYAH_ERR_BUFFER_TOO_SMALL;
    }

    if (snprintf(
            out,
            out_size,
            "NIYAH_RECEIPT_V1\n"
            "intent=%s\n"
            "lhs=%" PRId64 "\n"
            "rhs=%" PRId64 "\n"
            "result=%" PRId64 "\n"
            "ir=%s|%" PRId64 "|%" PRId64 "\n",
            op,
            receipt->ir.lhs,
            receipt->ir.rhs,
            receipt->result,
            op,
            receipt->ir.lhs,
            receipt->ir.rhs) != needed) {
        return NIYAH_ERR_IO;
    }

    return NIYAH_OK;
}


NiyahStatus niyah_receipt_sha256(
    const NiyahExecutionReceipt *receipt,
    uint8_t out_hash[NIYAH_RECEIPT_SHA256_SIZE])
{
    char canonical[256];
    size_t canonical_length = 0U;
    NiyahSha256 sha;
    NiyahStatus status;

    if (receipt == NULL || out_hash == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    status = niyah_receipt_format(
        receipt,
        canonical,
        sizeof(canonical),
        &canonical_length);

    if (status != NIYAH_OK) {
        return status;
    }

    niyah_sha256_init(&sha);

    if (!niyah_sha256_update(
            &sha,
            (const unsigned char *)canonical,
            canonical_length)) {
        return NIYAH_ERR_OVERFLOW;
    }

    niyah_sha256_final(&sha, out_hash);
    return NIYAH_OK;
}
