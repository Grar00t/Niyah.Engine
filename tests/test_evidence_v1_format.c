#include "niyah/evidence.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(x) \
    do { \
        if (!(x)) return __LINE__; \
    } while (0)

static const char valid_artifact[] =
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

static int replace_once(
    const char *source,
    const char *from,
    const char *to,
    char *out,
    size_t out_capacity,
    size_t *out_length)
{
    const char *match;
    size_t source_length;
    size_t from_length;
    size_t to_length;
    size_t prefix_length;
    size_t needed;

    match = strstr(source, from);
    if (match == NULL) {
        return 0;
    }

    source_length = strlen(source);
    from_length = strlen(from);
    to_length = strlen(to);
    prefix_length = (size_t)(match - source);

    if (source_length < from_length) {
        return 0;
    }

    needed =
        source_length -
        from_length +
        to_length;

    if (needed + 1U > out_capacity) {
        return 0;
    }

    memcpy(
        out,
        source,
        prefix_length);

    memcpy(
        out + prefix_length,
        to,
        to_length);

    memcpy(
        out + prefix_length + to_length,
        match + from_length,
        source_length -
            prefix_length -
            from_length);

    out[needed] = '\0';

    if (out_length != NULL) {
        *out_length = needed;
    }

    return 1;
}

static int expect_corrupt(
    const char *text,
    size_t size)
{
    NiyahEvidenceDocument document;

    memset(&document, 0, sizeof(document));

    return niyah_evidence_parse_document(
               text,
               size,
               &document) ==
           NIYAH_ERR_CORRUPT_DATA;
}

int main(void)
{
    NiyahEvidenceDocument document;
    char mutated[2048];
    size_t mutated_length = 0U;
    size_t valid_length =
        sizeof(valid_artifact) - 1U;

    memset(&document, 0, sizeof(document));

    /* Exact V1 golden artifact. */
    CHECK(
        niyah_evidence_parse_document(
            valid_artifact,
            valid_length,
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

    /*
     * Final LF is part of canonical V1.
     */
    CHECK(
        expect_corrupt(
            valid_artifact,
            valid_length - 1U));

    /*
     * Truncation must never parse.
     */
    CHECK(
        expect_corrupt(
            valid_artifact,
            valid_length / 2U));

    /*
     * Trailing data forbidden.
     */
    CHECK(valid_length + 6U <
          sizeof(mutated));

    memcpy(
        mutated,
        valid_artifact,
        valid_length);

    memcpy(
        mutated + valid_length,
        "EXTRA\n",
        6U);

    mutated_length = valid_length + 6U;

    CHECK(
        expect_corrupt(
            mutated,
            mutated_length));

    /*
     * CRLF is not canonical V1.
     */
    CHECK(
        replace_once(
            valid_artifact,
            "NIYAH_EVIDENCE_V1\n",
            "NIYAH_EVIDENCE_V1\r\n",
            mutated,
            sizeof(mutated),
            &mutated_length));

    CHECK(
        expect_corrupt(
            mutated,
            mutated_length));

    /*
     * Hex encoding is canonical lowercase.
     */
    CHECK(
        replace_once(
            valid_artifact,
            "receipt_sha256=307d",
            "receipt_sha256=307D",
            mutated,
            sizeof(mutated),
            &mutated_length));

    CHECK(
        expect_corrupt(
            mutated,
            mutated_length));

    /*
     * Duplicate/wrong field name rejected.
     */
    CHECK(
        replace_once(
            valid_artifact,
            "checkpoint_sha256=",
            "receipt_sha256=",
            mutated,
            sizeof(mutated),
            &mutated_length));

    CHECK(
        expect_corrupt(
            mutated,
            mutated_length));

    /*
     * Unknown field rejected.
     */
    CHECK(
        replace_once(
            valid_artifact,
            "tokenizer_sha256=",
            "tokenizer_sha257=",
            mutated,
            sizeof(mutated),
            &mutated_length));

    CHECK(
        expect_corrupt(
            mutated,
            mutated_length));

    /*
     * Non-canonical numeric representation rejected.
     */
    CHECK(
        replace_once(
            valid_artifact,
            "lhs=2\n",
            "lhs=02\n",
            mutated,
            sizeof(mutated),
            &mutated_length));

    CHECK(
        expect_corrupt(
            mutated,
            mutated_length));

    /*
     * Receipt fields and IR must agree.
     */
    CHECK(
        replace_once(
            valid_artifact,
            "ir=ADD|2|7\n",
            "ir=ADD|2|8\n",
            mutated,
            sizeof(mutated),
            &mutated_length));

    CHECK(
        expect_corrupt(
            mutated,
            mutated_length));

    CHECK(
        replace_once(
            valid_artifact,
            "intent=ADD\n",
            "intent=MUL\n",
            mutated,
            sizeof(mutated),
            &mutated_length));

    CHECK(
        expect_corrupt(
            mutated,
            mutated_length));

    /*
     * Parse may succeed only for structurally valid artifacts;
     * cryptographic/content mismatches belong to verify.
     */
    CHECK(
        niyah_evidence_parse_document(
            valid_artifact,
            valid_length,
            &document) ==
        NIYAH_OK);

    document.receipt_sha256[0] ^= 1U;

    CHECK(
        niyah_evidence_verify_document(
            &document,
            document.checkpoint_sha256,
            document.tokenizer_sha256) ==
        NIYAH_ERR_CORRUPT_DATA);

    document.receipt_sha256[0] ^= 1U;
    document.evidence_root_sha256[0] ^= 1U;

    CHECK(
        niyah_evidence_verify_document(
            &document,
            document.checkpoint_sha256,
            document.tokenizer_sha256) ==
        NIYAH_ERR_CORRUPT_DATA);

    /*
     * API argument contract.
     */
    CHECK(
        niyah_evidence_parse_document(
            NULL,
            valid_length,
            &document) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    CHECK(
        niyah_evidence_parse_document(
            valid_artifact,
            valid_length,
            NULL) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    return 0;
}
