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

#ifdef __cplusplus
}
#endif

#endif
