#ifndef NIYAH_RECEIPT_H
#define NIYAH_RECEIPT_H

#include <stddef.h>
#include <stdint.h>

#include "niyah/coordination.h"

#ifdef __cplusplus
extern "C" {
#endif


#define NIYAH_RECEIPT_V1_VERSION 1U

/*
 * Canonical on-wire representation.
 *
 * No pointers, strings, timestamps, padding or native-endian fields
 * are included in the canonical encoding.
 */
#define NIYAH_RECEIPT_V1_CANONICAL_BYTES 232U

#define NIYAH_RECEIPT_FLAG_GENESIS (1U << 0)


typedef enum NiyahReceiptClaimState {
    NIYAH_RECEIPT_STATE_UNVERIFIED = 1,
    NIYAH_RECEIPT_STATE_VERIFIED = 2,
    NIYAH_RECEIPT_STATE_STALE = 3,
    NIYAH_RECEIPT_STATE_CONFLICT = 4,
    NIYAH_RECEIPT_STATE_REJECTED = 5
} NiyahReceiptClaimState;


typedef enum NiyahReceiptStatus {
    NIYAH_RECEIPT_OK = 0,

    NIYAH_RECEIPT_INVALID_ARGUMENT,
    NIYAH_RECEIPT_INVALID_VERSION,
    NIYAH_RECEIPT_INVALID_IDENTITY,
    NIYAH_RECEIPT_INVALID_LOBE,
    NIYAH_RECEIPT_INVALID_SCOPE,
    NIYAH_RECEIPT_INVALID_STATE,
    NIYAH_RECEIPT_INVALID_PARENT,
    NIYAH_RECEIPT_INVALID_CONFLICT_STATE,
    NIYAH_RECEIPT_CHAIN_MISMATCH,
    NIYAH_RECEIPT_ACTION_MISMATCH,
    NIYAH_RECEIPT_HASH_FAILURE
} NiyahReceiptStatus;


/*
 * Fixed-width context handoff.
 *
 * A receipt carries state transition metadata and cryptographic
 * references to evidence. It deliberately does not carry prompts,
 * prose summaries, raw logs or arbitrary strings.
 */
typedef struct NiyahReceiptV1 {
    uint32_t version;

    NiyahCoordinationDigest task_id;

    NiyahCoordinationLobe producer_lobe;

    uint32_t project_id;
    uint32_t scope;

    NiyahCoordinationDigest base_revision;
    NiyahCoordinationDigest result_revision;

    NiyahCoordinationDigest parent_receipt_sha256;

    NiyahCoordinationDigest delta_digest;
    NiyahCoordinationDigest evidence_root;

    NiyahReceiptClaimState state_before;
    NiyahReceiptClaimState state_after;

    uint32_t conflict_count;

    NiyahCoordinationAction next_action;

    uint32_t flags;
} NiyahReceiptV1;


NiyahReceiptStatus niyah_receipt_v1_validate(
    const NiyahReceiptV1 *receipt);


NiyahReceiptStatus niyah_receipt_v1_encode(
    const NiyahReceiptV1 *receipt,
    uint8_t out_bytes[NIYAH_RECEIPT_V1_CANONICAL_BYTES]);


NiyahReceiptStatus niyah_receipt_v1_sha256(
    const NiyahReceiptV1 *receipt,
    NiyahCoordinationDigest *out_digest);


/*
 * Validate a receipt against the receiver's current state and
 * feed it through the deterministic coordination gate.
 *
 * expected_parent_receipt_sha256:
 *   Hash of the last accepted receipt. A zero digest is used
 *   only when the incoming receipt is a genesis receipt.
 *
 * observed_revision:
 *   Receiver's current source/state revision. This prevents a
 *   valid old receipt from authorizing work on a newer base.
 *
 * No prose context is consulted by this operation.
 */
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
    NiyahCoordinationGate *out_gate);


const char *niyah_receipt_status_name(
    NiyahReceiptStatus status);


#ifdef __cplusplus
}
#endif

#endif
