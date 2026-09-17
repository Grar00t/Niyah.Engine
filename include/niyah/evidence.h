#ifndef NIYAH_EVIDENCE_H
#define NIYAH_EVIDENCE_H

#include "niyah/checkpoint.h"
#include "niyah/receipt.h"
#include "niyah/tokenizer.h"

#include <stdint.h>

#define NIYAH_EVIDENCE_ROOT_SHA256_SIZE 32U

#ifdef __cplusplus
extern "C" {
#endif

NiyahStatus niyah_evidence_root_sha256(
    const uint8_t
        receipt_sha256[NIYAH_RECEIPT_SHA256_SIZE],
    const uint8_t
        checkpoint_sha256[
            NIYAH_CHECKPOINT_IDENTITY_SHA256_SIZE],
    const uint8_t
        tokenizer_sha256[
            NIYAH_TOKENIZER_IDENTITY_SHA256_SIZE],
    uint8_t
        out_root[NIYAH_EVIDENCE_ROOT_SHA256_SIZE]);

NiyahStatus niyah_evidence_verify_root(
    const NiyahExecutionReceipt *receipt,
    const uint8_t
        checkpoint_sha256[
            NIYAH_CHECKPOINT_IDENTITY_SHA256_SIZE],
    const uint8_t
        tokenizer_sha256[
            NIYAH_TOKENIZER_IDENTITY_SHA256_SIZE],
    const uint8_t
        claimed_root[
            NIYAH_EVIDENCE_ROOT_SHA256_SIZE]);

typedef struct NiyahEvidenceDocument {
    uint8_t receipt_sha256[NIYAH_RECEIPT_SHA256_SIZE];

    uint8_t checkpoint_sha256[
        NIYAH_CHECKPOINT_IDENTITY_SHA256_SIZE];

    uint8_t tokenizer_sha256[
        NIYAH_TOKENIZER_IDENTITY_SHA256_SIZE];

    uint8_t evidence_root_sha256[
        NIYAH_EVIDENCE_ROOT_SHA256_SIZE];

    NiyahExecutionReceipt receipt;
} NiyahEvidenceDocument;

NiyahStatus niyah_evidence_parse_document(
    const char *text,
    size_t text_size,
    NiyahEvidenceDocument *out_document);

NiyahStatus niyah_evidence_verify_document(
    const NiyahEvidenceDocument *document,
    const uint8_t
        actual_checkpoint_sha256[
            NIYAH_CHECKPOINT_IDENTITY_SHA256_SIZE],
    const uint8_t
        actual_tokenizer_sha256[
            NIYAH_TOKENIZER_IDENTITY_SHA256_SIZE]);

#ifdef __cplusplus
}
#endif

#endif
