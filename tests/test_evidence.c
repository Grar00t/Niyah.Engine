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

    {
        static const char artifact[] =
            "NIYAH_EVIDENCE_V1\n"
            "receipt_sha256="
            "307d9e6b0da2bf9f677d762441844e8b"
            "c90c4a76b6cb383aa06f5febe40c8a14\n"
            "checkpoint_sha256="
            "f4271a4698163d2c0b9915d097230e7a"
            "5d72d2ddef094a008af48d4e5ff496cc\n"
            "tokenizer_sha256="
            "be7834ed2615b31887f3297a644861f8"
            "2e7e4ede739e1baf34c393a6fc8dd210\n"
            "evidence_root_sha256="
            "d1983841bd47ac32e90b2ff0f3a3ac65"
            "631c90bdf698e207656e6f3277472aaf\n"
            "NIYAH_RECEIPT_V1\n"
            "intent=ADD\n"
            "lhs=2\n"
            "rhs=7\n"
            "result=9\n"
            "ir=ADD|2|7\n";

        NiyahEvidenceDocument document;
        uint8_t wrong_checkpoint[32];

        CHECK(
            niyah_evidence_parse_document(
                artifact,
                sizeof(artifact) - 1U,
                &document) ==
            NIYAH_OK);

        CHECK(document.receipt.ir.op ==
              NIYAH_IR_OP_ADD);
        CHECK(document.receipt.ir.lhs == 2);
        CHECK(document.receipt.ir.rhs == 7);
        CHECK(document.receipt.result == 9);

        CHECK(
            niyah_evidence_verify_document(
                &document,
                document.checkpoint_sha256,
                document.tokenizer_sha256) ==
            NIYAH_OK);

        memcpy(
            wrong_checkpoint,
            document.checkpoint_sha256,
            sizeof(wrong_checkpoint));

        wrong_checkpoint[0] ^= 1U;

        CHECK(
            niyah_evidence_verify_document(
                &document,
                wrong_checkpoint,
                document.tokenizer_sha256) ==
            NIYAH_ERR_CORRUPT_DATA);

        document.receipt.result = 8;

        CHECK(
            niyah_evidence_verify_document(
                &document,
                document.checkpoint_sha256,
                document.tokenizer_sha256) ==
            NIYAH_ERR_CORRUPT_DATA);

        CHECK(
            niyah_evidence_parse_document(
                "NIYAH_EVIDENCE_V1\n",
                sizeof("NIYAH_EVIDENCE_V1\n") - 1U,
                &document) ==
            NIYAH_ERR_CORRUPT_DATA);

        CHECK(
            niyah_evidence_parse_document(
                NULL,
                0U,
                &document) ==
            NIYAH_ERR_INVALID_ARGUMENT);
    }

    return 0;
}
