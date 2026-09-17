#include "niyah/evidence.h"
#include "niyah_sha256.h"

#include <stddef.h>

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
        out_root[NIYAH_EVIDENCE_ROOT_SHA256_SIZE])
{
    static const unsigned char domain[] =
        "NIYAH_EVIDENCE_V1";

    NiyahSha256 sha;

    if (receipt_sha256 == NULL ||
        checkpoint_sha256 == NULL ||
        tokenizer_sha256 == NULL ||
        out_root == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    niyah_sha256_init(&sha);

    if (!niyah_sha256_update(
            &sha,
            domain,
            sizeof(domain) - 1U) ||
        !niyah_sha256_update(
            &sha,
            receipt_sha256,
            NIYAH_RECEIPT_SHA256_SIZE) ||
        !niyah_sha256_update(
            &sha,
            checkpoint_sha256,
            NIYAH_CHECKPOINT_IDENTITY_SHA256_SIZE) ||
        !niyah_sha256_update(
            &sha,
            tokenizer_sha256,
            NIYAH_TOKENIZER_IDENTITY_SHA256_SIZE)) {
        return NIYAH_ERR_OVERFLOW;
    }

    niyah_sha256_final(&sha, out_root);
    return NIYAH_OK;
}
