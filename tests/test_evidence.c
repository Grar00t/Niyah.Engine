#include "niyah/evidence.h"

#include <stdint.h>
#include <string.h>

#define CHECK(x) \
    do { \
        if (!(x)) return __LINE__; \
    } while (0)

int main(void)
{
    uint8_t receipt[32];
    uint8_t checkpoint[32];
    uint8_t tokenizer[32];

    uint8_t root[32];
    uint8_t changed[32];

    static const uint8_t expected[32] = {
        0xb1, 0x52, 0x9b, 0xe1,
        0x10, 0x49, 0x99, 0x33,
        0xc3, 0x0b, 0xb2, 0x65,
        0xba, 0xce, 0x27, 0xdb,
        0xde, 0x7d, 0xcc, 0xdc,
        0x42, 0xae, 0x1e, 0xcf,
        0x4f, 0x9f, 0xe9, 0x2a,
        0xb0, 0x79, 0xbd, 0xdc
    };

    size_t i;

    for (i = 0U; i < 32U; ++i) {
        receipt[i] = (uint8_t)i;
        checkpoint[i] = 0x11U;
        tokenizer[i] = 0x22U;
    }

    CHECK(
        niyah_evidence_root_sha256(
            receipt,
            checkpoint,
            tokenizer,
            root) == NIYAH_OK);

    CHECK(memcmp(root, expected, 32U) == 0);

    receipt[0] ^= 1U;

    CHECK(
        niyah_evidence_root_sha256(
            receipt,
            checkpoint,
            tokenizer,
            changed) == NIYAH_OK);

    CHECK(memcmp(root, changed, 32U) != 0);

    receipt[0] ^= 1U;
    checkpoint[0] ^= 1U;

    CHECK(
        niyah_evidence_root_sha256(
            receipt,
            checkpoint,
            tokenizer,
            changed) == NIYAH_OK);

    CHECK(memcmp(root, changed, 32U) != 0);

    checkpoint[0] ^= 1U;
    tokenizer[0] ^= 1U;

    CHECK(
        niyah_evidence_root_sha256(
            receipt,
            checkpoint,
            tokenizer,
            changed) == NIYAH_OK);

    CHECK(memcmp(root, changed, 32U) != 0);

    CHECK(
        niyah_evidence_root_sha256(
            NULL,
            checkpoint,
            tokenizer,
            root) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    /*
     * Verify the entire chain from the canonical receipt,
     * not from a caller-supplied receipt hash.
     */
    {
        NiyahExecutionReceipt exec_receipt;
        uint8_t exec_receipt_hash[
            NIYAH_RECEIPT_SHA256_SIZE];

        uint8_t valid_root[
            NIYAH_EVIDENCE_ROOT_SHA256_SIZE];

        size_t j;

        for (j = 0U; j < 32U; ++j) {
            checkpoint[j] = 0x11U;
            tokenizer[j] = 0x22U;
        }

        memset(&exec_receipt, 0, sizeof(exec_receipt));

        exec_receipt.ir.op = NIYAH_IR_OP_ADD;
        exec_receipt.ir.lhs = 2;
        exec_receipt.ir.rhs = 7;
        exec_receipt.result = 9;

        CHECK(
            niyah_receipt_sha256(
                &exec_receipt,
                exec_receipt_hash) ==
            NIYAH_OK);

        CHECK(
            niyah_evidence_root_sha256(
                exec_receipt_hash,
                checkpoint,
                tokenizer,
                valid_root) ==
            NIYAH_OK);

        CHECK(
            niyah_evidence_verify_root(
                &exec_receipt,
                checkpoint,
                tokenizer,
                valid_root) ==
            NIYAH_OK);

        /*
         * Receipt tamper.
         */
        exec_receipt.result = 10;

        CHECK(
            niyah_evidence_verify_root(
                &exec_receipt,
                checkpoint,
                tokenizer,
                valid_root) ==
            NIYAH_ERR_CORRUPT_DATA);

        exec_receipt.result = 9;

        /*
         * Checkpoint identity tamper.
         */
        checkpoint[0] ^= 1U;

        CHECK(
            niyah_evidence_verify_root(
                &exec_receipt,
                checkpoint,
                tokenizer,
                valid_root) ==
            NIYAH_ERR_CORRUPT_DATA);

        checkpoint[0] ^= 1U;

        /*
         * Tokenizer identity tamper.
         */
        tokenizer[0] ^= 1U;

        CHECK(
            niyah_evidence_verify_root(
                &exec_receipt,
                checkpoint,
                tokenizer,
                valid_root) ==
            NIYAH_ERR_CORRUPT_DATA);

        tokenizer[0] ^= 1U;

        /*
         * Claimed root tamper.
         */
        valid_root[0] ^= 1U;

        CHECK(
            niyah_evidence_verify_root(
                &exec_receipt,
                checkpoint,
                tokenizer,
                valid_root) ==
            NIYAH_ERR_CORRUPT_DATA);

        valid_root[0] ^= 1U;

        /*
         * Argument contract.
         */
        CHECK(
            niyah_evidence_verify_root(
                NULL,
                checkpoint,
                tokenizer,
                valid_root) ==
            NIYAH_ERR_INVALID_ARGUMENT);

        CHECK(
            niyah_evidence_verify_root(
                &exec_receipt,
                NULL,
                tokenizer,
                valid_root) ==
            NIYAH_ERR_INVALID_ARGUMENT);

        CHECK(
            niyah_evidence_verify_root(
                &exec_receipt,
                checkpoint,
                NULL,
                valid_root) ==
            NIYAH_ERR_INVALID_ARGUMENT);

        CHECK(
            niyah_evidence_verify_root(
                &exec_receipt,
                checkpoint,
                tokenizer,
                NULL) ==
            NIYAH_ERR_INVALID_ARGUMENT);
    }

    return 0;
}
