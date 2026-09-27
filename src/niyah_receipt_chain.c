#include "niyah/receipt_chain.h"

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


static int chain_state_valid(
    const NiyahReceiptChainV1 *state)
{
    if (state == NULL)
        return 0;

    if (state->version !=
        NIYAH_RECEIPT_CHAIN_V1_VERSION) {
        return 0;
    }

    if (state->project_id == 0U ||
        state->allowed_scope == 0U ||
        digest_is_zero(&state->current_revision)) {
        return 0;
    }

    if (state->has_head != 0 &&
        state->has_head != 1) {
        return 0;
    }

    if (state->has_head == 0) {
        if (state->sequence != UINT64_C(0) ||
            !digest_is_zero(
                &state->head_receipt_sha256)) {
            return 0;
        }
    } else {
        if (state->sequence == UINT64_C(0) ||
            digest_is_zero(
                &state->head_receipt_sha256)) {
            return 0;
        }
    }

    return 1;
}


NiyahReceiptChainStatus niyah_receipt_chain_v1_init(
    NiyahReceiptChainV1 *state,
    uint32_t project_id,
    uint32_t allowed_scope,
    const NiyahCoordinationDigest *current_revision)
{
    if (state == NULL ||
        current_revision == NULL) {
        return NIYAH_RECEIPT_CHAIN_INVALID_ARGUMENT;
    }

    if (project_id == 0U ||
        allowed_scope == 0U ||
        digest_is_zero(current_revision)) {
        return NIYAH_RECEIPT_CHAIN_INVALID_STATE;
    }

    memset(
        state,
        0,
        sizeof(*state));

    state->version =
        NIYAH_RECEIPT_CHAIN_V1_VERSION;

    state->project_id =
        project_id;

    state->allowed_scope =
        allowed_scope;

    state->sequence =
        UINT64_C(0);

    state->current_revision =
        *current_revision;

    state->has_head = 0;

    return NIYAH_RECEIPT_CHAIN_OK;
}


NiyahReceiptChainStatus niyah_receipt_chain_v1_apply(
    NiyahReceiptChainV1 *state,
    const NiyahReceiptV1 *receipt,
    NiyahCoordinationLobe verifier_lobe,
    NiyahCoordinationLobe writer_lobe,
    NiyahCoordinationAction action,
    NiyahCoordinationLobe actor,
    NiyahCoordinationGate *out_gate,
    NiyahReceiptStatus *out_receipt_status)
{
    NiyahReceiptChainV1 next;
    NiyahCoordinationDigest expected_parent;
    NiyahCoordinationDigest receipt_hash;

    NiyahCoordinationGate gate =
        NIYAH_COORDINATION_INVALID_ENVELOPE;

    NiyahReceiptStatus receipt_status;

    if (state == NULL ||
        receipt == NULL ||
        out_gate == NULL ||
        out_receipt_status == NULL) {
        return
            NIYAH_RECEIPT_CHAIN_INVALID_ARGUMENT;
    }

    *out_gate =
        NIYAH_COORDINATION_INVALID_ENVELOPE;

    *out_receipt_status =
        NIYAH_RECEIPT_INVALID_ARGUMENT;

    if (!chain_state_valid(state))
        return NIYAH_RECEIPT_CHAIN_INVALID_STATE;

    /*
     * Sequence exhaustion must never wrap back to zero.
     */
    if (state->sequence == UINT64_MAX)
        return NIYAH_RECEIPT_CHAIN_SEQUENCE_OVERFLOW;

    memset(
        &expected_parent,
        0,
        sizeof(expected_parent));

    if (state->has_head != 0) {
        expected_parent =
            state->head_receipt_sha256;
    }

    /*
     * Do not allow a non-genesis first handoff even if some
     * future receipt validator becomes more permissive.
     */
    if (state->has_head == 0 &&
        (receipt->flags &
         NIYAH_RECEIPT_FLAG_GENESIS) == 0U) {
        *out_receipt_status =
            NIYAH_RECEIPT_INVALID_PARENT;

        return
            NIYAH_RECEIPT_CHAIN_RECEIPT_REJECTED;
    }

    receipt_status =
        niyah_receipt_v1_coordination_gate(
            receipt,
            &state->current_revision,
            &expected_parent,
            state->project_id,
            state->allowed_scope,
            verifier_lobe,
            writer_lobe,
            action,
            actor,
            &gate);

    *out_receipt_status =
        receipt_status;

    *out_gate =
        gate;

    if (receipt_status != NIYAH_RECEIPT_OK ||
        gate != NIYAH_COORDINATION_PASS) {
        return
            NIYAH_RECEIPT_CHAIN_RECEIPT_REJECTED;
    }

    receipt_status =
        niyah_receipt_v1_sha256(
            receipt,
            &receipt_hash);

    *out_receipt_status =
        receipt_status;

    if (receipt_status != NIYAH_RECEIPT_OK)
        return NIYAH_RECEIPT_CHAIN_HASH_FAILURE;

    /*
     * Stage all mutations in a local copy.
     * The caller-visible state changes only after every gate passes.
     */
    next = *state;

    next.sequence += UINT64_C(1);

    next.current_revision =
        receipt->result_revision;

    next.head_receipt_sha256 =
        receipt_hash;

    next.has_head = 1;

    if (!chain_state_valid(&next))
        return NIYAH_RECEIPT_CHAIN_INVALID_STATE;

    *state = next;

    return NIYAH_RECEIPT_CHAIN_OK;
}


const char *niyah_receipt_chain_status_name(
    NiyahReceiptChainStatus status)
{
    switch (status) {
        case NIYAH_RECEIPT_CHAIN_OK:
            return "OK";

        case NIYAH_RECEIPT_CHAIN_INVALID_ARGUMENT:
            return "INVALID_ARGUMENT";

        case NIYAH_RECEIPT_CHAIN_INVALID_VERSION:
            return "INVALID_VERSION";

        case NIYAH_RECEIPT_CHAIN_INVALID_STATE:
            return "INVALID_STATE";

        case NIYAH_RECEIPT_CHAIN_RECEIPT_REJECTED:
            return "RECEIPT_REJECTED";

        case NIYAH_RECEIPT_CHAIN_HASH_FAILURE:
            return "HASH_FAILURE";

        case NIYAH_RECEIPT_CHAIN_SEQUENCE_OVERFLOW:
            return "SEQUENCE_OVERFLOW";

        default:
            return "UNKNOWN";
    }
}
