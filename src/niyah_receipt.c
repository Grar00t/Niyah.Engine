#include "niyah/receipt.h"

#include "niyah_sha256.h"

#include <stddef.h>
#include <string.h>


static int digest_is_zero(
    const NiyahCoordinationDigest *digest)
{
    size_t i;

    if (digest == NULL)
        return 1;

    for (i = 0U;
         i < NIYAH_COORDINATION_DIGEST_BYTES;
         ++i) {
        if (digest->bytes[i] != 0U)
            return 0;
    }

    return 1;
}


static int valid_lobe(
    NiyahCoordinationLobe lobe)
{
    return
        lobe == NIYAH_LOBE_COMPUTE ||
        lobe == NIYAH_LOBE_CHRONICLE ||
        lobe == NIYAH_LOBE_VERIFIER;
}


static int valid_state(
    NiyahReceiptClaimState state)
{
    return
        state == NIYAH_RECEIPT_STATE_UNVERIFIED ||
        state == NIYAH_RECEIPT_STATE_VERIFIED ||
        state == NIYAH_RECEIPT_STATE_STALE ||
        state == NIYAH_RECEIPT_STATE_CONFLICT ||
        state == NIYAH_RECEIPT_STATE_REJECTED;
}


static int valid_action(
    NiyahCoordinationAction action)
{
    return
        action == NIYAH_COORDINATION_EXECUTE ||
        action == NIYAH_COORDINATION_VERIFY ||
        action == NIYAH_COORDINATION_PROMOTE;
}


static void store_u32_be(
    uint8_t out[4],
    uint32_t value)
{
    out[0] = (uint8_t)(value >> 24);
    out[1] = (uint8_t)(value >> 16);
    out[2] = (uint8_t)(value >> 8);
    out[3] = (uint8_t)value;
}


static void append_u32(
    uint8_t *out,
    size_t *offset,
    uint32_t value)
{
    store_u32_be(
        out + *offset,
        value);

    *offset += 4U;
}


static void append_digest(
    uint8_t *out,
    size_t *offset,
    const NiyahCoordinationDigest *digest)
{
    memcpy(
        out + *offset,
        digest->bytes,
        NIYAH_COORDINATION_DIGEST_BYTES);

    *offset +=
        NIYAH_COORDINATION_DIGEST_BYTES;
}


NiyahReceiptStatus niyah_receipt_v1_validate(
    const NiyahReceiptV1 *receipt)
{
    const uint32_t known_flags =
        NIYAH_RECEIPT_FLAG_GENESIS;

    const int genesis =
        receipt != NULL &&
        (receipt->flags &
         NIYAH_RECEIPT_FLAG_GENESIS) != 0U;

    if (receipt == NULL)
        return NIYAH_RECEIPT_INVALID_ARGUMENT;

    if (receipt->version !=
        NIYAH_RECEIPT_V1_VERSION) {
        return NIYAH_RECEIPT_INVALID_VERSION;
    }

    if ((receipt->flags & ~known_flags) != 0U)
        return NIYAH_RECEIPT_INVALID_ARGUMENT;

    if (digest_is_zero(&receipt->task_id) ||
        digest_is_zero(&receipt->base_revision) ||
        digest_is_zero(&receipt->result_revision) ||
        digest_is_zero(&receipt->delta_digest) ||
        digest_is_zero(&receipt->evidence_root)) {
        return NIYAH_RECEIPT_INVALID_IDENTITY;
    }

    if (!valid_lobe(receipt->producer_lobe))
        return NIYAH_RECEIPT_INVALID_LOBE;

    if (receipt->project_id == 0U ||
        receipt->scope == 0U) {
        return NIYAH_RECEIPT_INVALID_SCOPE;
    }

    if (!valid_state(receipt->state_before) ||
        !valid_state(receipt->state_after) ||
        !valid_action(receipt->next_action)) {
        return NIYAH_RECEIPT_INVALID_STATE;
    }

    /*
     * Genesis must have no parent.
     * Every non-genesis receipt must be chained.
     */
    if (genesis) {
        if (!digest_is_zero(
                &receipt->parent_receipt_sha256)) {
            return NIYAH_RECEIPT_INVALID_PARENT;
        }
    } else {
        if (digest_is_zero(
                &receipt->parent_receipt_sha256)) {
            return NIYAH_RECEIPT_INVALID_PARENT;
        }
    }

    /*
     * Contradictions are explicit state, never silently hidden
     * inside an otherwise VERIFIED receipt.
     */
    if (receipt->conflict_count != 0U &&
        receipt->state_after !=
            NIYAH_RECEIPT_STATE_CONFLICT) {
        return
            NIYAH_RECEIPT_INVALID_CONFLICT_STATE;
    }

    if (receipt->state_after ==
            NIYAH_RECEIPT_STATE_CONFLICT &&
        receipt->conflict_count == 0U) {
        return
            NIYAH_RECEIPT_INVALID_CONFLICT_STATE;
    }

    return NIYAH_RECEIPT_OK;
}


NiyahReceiptStatus niyah_receipt_v1_encode(
    const NiyahReceiptV1 *receipt,
    uint8_t out_bytes[
        NIYAH_RECEIPT_V1_CANONICAL_BYTES])
{
    NiyahReceiptStatus status;
    size_t offset = 0U;

    if (out_bytes == NULL)
        return NIYAH_RECEIPT_INVALID_ARGUMENT;

    status =
        niyah_receipt_v1_validate(receipt);

    if (status != NIYAH_RECEIPT_OK)
        return status;

    /*
     * Magic: NRC1
     */
    out_bytes[offset++] = (uint8_t)'N';
    out_bytes[offset++] = (uint8_t)'R';
    out_bytes[offset++] = (uint8_t)'C';
    out_bytes[offset++] = (uint8_t)'1';

    append_u32(
        out_bytes,
        &offset,
        receipt->version);

    append_digest(
        out_bytes,
        &offset,
        &receipt->task_id);

    append_u32(
        out_bytes,
        &offset,
        (uint32_t)receipt->producer_lobe);

    append_u32(
        out_bytes,
        &offset,
        receipt->project_id);

    append_u32(
        out_bytes,
        &offset,
        receipt->scope);

    append_digest(
        out_bytes,
        &offset,
        &receipt->base_revision);

    append_digest(
        out_bytes,
        &offset,
        &receipt->result_revision);

    append_digest(
        out_bytes,
        &offset,
        &receipt->parent_receipt_sha256);

    append_digest(
        out_bytes,
        &offset,
        &receipt->delta_digest);

    append_digest(
        out_bytes,
        &offset,
        &receipt->evidence_root);

    append_u32(
        out_bytes,
        &offset,
        (uint32_t)receipt->state_before);

    append_u32(
        out_bytes,
        &offset,
        (uint32_t)receipt->state_after);

    append_u32(
        out_bytes,
        &offset,
        receipt->conflict_count);

    append_u32(
        out_bytes,
        &offset,
        (uint32_t)receipt->next_action);

    append_u32(
        out_bytes,
        &offset,
        receipt->flags);

    if (offset !=
        NIYAH_RECEIPT_V1_CANONICAL_BYTES) {
        return NIYAH_RECEIPT_INVALID_ARGUMENT;
    }

    return NIYAH_RECEIPT_OK;
}


NiyahReceiptStatus niyah_receipt_v1_sha256(
    const NiyahReceiptV1 *receipt,
    NiyahCoordinationDigest *out_digest)
{
    static const unsigned char domain[] =
        "NIYAH_CONTEXT_RECEIPT_SHA256_V1";

    uint8_t encoded[
        NIYAH_RECEIPT_V1_CANONICAL_BYTES];

    NiyahSha256 sha;
    NiyahReceiptStatus status;

    if (out_digest == NULL)
        return NIYAH_RECEIPT_INVALID_ARGUMENT;

    status =
        niyah_receipt_v1_encode(
            receipt,
            encoded);

    if (status != NIYAH_RECEIPT_OK)
        return status;

    niyah_sha256_init(&sha);

    if (!niyah_sha256_update(
            &sha,
            domain,
            sizeof(domain) - 1U) ||
        !niyah_sha256_update(
            &sha,
            encoded,
            sizeof(encoded))) {
        return NIYAH_RECEIPT_HASH_FAILURE;
    }

    niyah_sha256_final(
        &sha,
        out_digest->bytes);

    return NIYAH_RECEIPT_OK;
}



NiyahReceiptStatus
niyah_receipt_v1_coordination_gate(
    const NiyahReceiptV1 *receipt,
    const NiyahCoordinationDigest *observed_revision,
    const NiyahCoordinationDigest *
        expected_parent_receipt_sha256,
    uint32_t expected_project_id,
    uint32_t allowed_scope,
    NiyahCoordinationLobe verifier_lobe,
    NiyahCoordinationLobe writer_lobe,
    NiyahCoordinationAction action,
    NiyahCoordinationLobe actor,
    NiyahCoordinationGate *out_gate)
{
    NiyahCoordinationEnvelope envelope;
    NiyahReceiptStatus status;

    if (out_gate == NULL ||
        observed_revision == NULL ||
        expected_parent_receipt_sha256 == NULL) {
        return NIYAH_RECEIPT_INVALID_ARGUMENT;
    }

    *out_gate =
        NIYAH_COORDINATION_INVALID_ENVELOPE;

    status =
        niyah_receipt_v1_validate(receipt);

    if (status != NIYAH_RECEIPT_OK)
        return status;

    /*
     * A receipt may be internally valid but still belong to an
     * older handoff chain. Never silently rebase that context.
     */
    if (!niyah_coordination_digest_equal(
            &receipt->parent_receipt_sha256,
            expected_parent_receipt_sha256)) {
        return NIYAH_RECEIPT_CHAIN_MISMATCH;
    }

    /*
     * Prevent a receipt prepared for VERIFY from being reused as
     * authorization for EXECUTE/PROMOTE, or vice versa.
     */
    if (receipt->next_action != action)
        return NIYAH_RECEIPT_ACTION_MISMATCH;

    memset(
        &envelope,
        0,
        sizeof(envelope));

    envelope.base_revision =
        receipt->base_revision;

    envelope.observed_revision =
        *observed_revision;

    envelope.project_id =
        receipt->project_id;

    envelope.expected_project_id =
        expected_project_id;

    envelope.requested_scope =
        receipt->scope;

    envelope.allowed_scope =
        allowed_scope;

    envelope.producer_lobe =
        receipt->producer_lobe;

    envelope.verifier_lobe =
        verifier_lobe;

    envelope.writer_lobe =
        writer_lobe;

    /*
     * A receipt is proof-bearing for promotion only when its
     * resulting claim state is explicitly VERIFIED.
     */
    envelope.proof_present =
        receipt->state_after ==
        NIYAH_RECEIPT_STATE_VERIFIED;

    envelope.contradiction_present =
        receipt->conflict_count != 0U ||
        receipt->state_after ==
            NIYAH_RECEIPT_STATE_CONFLICT;

    *out_gate =
        niyah_coordination_gate(
            &envelope,
            action,
            actor);

    return NIYAH_RECEIPT_OK;
}


const char *niyah_receipt_status_name(
    NiyahReceiptStatus status)
{
    switch (status) {
        case NIYAH_RECEIPT_OK:
            return "OK";

        case NIYAH_RECEIPT_INVALID_ARGUMENT:
            return "INVALID_ARGUMENT";

        case NIYAH_RECEIPT_INVALID_VERSION:
            return "INVALID_VERSION";

        case NIYAH_RECEIPT_INVALID_IDENTITY:
            return "INVALID_IDENTITY";

        case NIYAH_RECEIPT_INVALID_LOBE:
            return "INVALID_LOBE";

        case NIYAH_RECEIPT_INVALID_SCOPE:
            return "INVALID_SCOPE";

        case NIYAH_RECEIPT_INVALID_STATE:
            return "INVALID_STATE";

        case NIYAH_RECEIPT_INVALID_PARENT:
            return "INVALID_PARENT";

        case NIYAH_RECEIPT_INVALID_CONFLICT_STATE:
            return "INVALID_CONFLICT_STATE";

        case NIYAH_RECEIPT_CHAIN_MISMATCH:
            return "CHAIN_MISMATCH";

        case NIYAH_RECEIPT_ACTION_MISMATCH:
            return "ACTION_MISMATCH";

        case NIYAH_RECEIPT_HASH_FAILURE:
            return "HASH_FAILURE";

        default:
            return "UNKNOWN";
    }
}
