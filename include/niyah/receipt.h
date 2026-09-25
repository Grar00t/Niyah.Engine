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


const char *niyah_receipt_status_name(
    NiyahReceiptStatus status);


#ifdef __cplusplus
}
#endif

#endif
