#ifndef NIYAH_RECEIPT_CHAIN_H
#define NIYAH_RECEIPT_CHAIN_H

#include <stdint.h>

#include "niyah/receipt.h"

#ifdef __cplusplus
extern "C" {
#endif


#define NIYAH_RECEIPT_CHAIN_V1_VERSION 1U


typedef enum NiyahReceiptChainStatus {
    NIYAH_RECEIPT_CHAIN_OK = 0,

    NIYAH_RECEIPT_CHAIN_INVALID_ARGUMENT,
    NIYAH_RECEIPT_CHAIN_INVALID_VERSION,
    NIYAH_RECEIPT_CHAIN_INVALID_STATE,
    NIYAH_RECEIPT_CHAIN_RECEIPT_REJECTED,
    NIYAH_RECEIPT_CHAIN_HASH_FAILURE,
    NIYAH_RECEIPT_CHAIN_SEQUENCE_OVERFLOW
} NiyahReceiptChainStatus;


/*
 * Minimal deterministic inter-lobe state.
 *
 * No prompt text.
 * No prose summary.
 * No raw logs.
 * No timestamps.
 * No heap-owned context.
 *
 * The receipt hash chain carries history.
 */
typedef struct NiyahReceiptChainV1 {
    uint32_t version;

    uint32_t project_id;
    uint32_t allowed_scope;

    uint64_t sequence;

    NiyahCoordinationDigest current_revision;
    NiyahCoordinationDigest head_receipt_sha256;

    int has_head;
} NiyahReceiptChainV1;


/*
 * Initialize an empty chain against a known current revision.
 *
 * The first accepted receipt must be a valid genesis receipt.
 */
NiyahReceiptChainStatus niyah_receipt_chain_v1_init(
    NiyahReceiptChainV1 *state,
    uint32_t project_id,
    uint32_t allowed_scope,
    const NiyahCoordinationDigest *current_revision);


/*
 * Atomically accept one handoff receipt.
 *
 * State advances only when:
 *
 *   - parent hash matches the chain head
 *   - base revision matches current revision
 *   - project/scope are authorized
 *   - requested action matches
 *   - coordination gate returns PASS
 *   - receipt hash succeeds
 *
 * Rejection leaves state byte-for-byte unchanged.
 */
NiyahReceiptChainStatus niyah_receipt_chain_v1_apply(
    NiyahReceiptChainV1 *state,
    const NiyahReceiptV1 *receipt,
    NiyahCoordinationLobe verifier_lobe,
    NiyahCoordinationLobe writer_lobe,
    NiyahCoordinationAction action,
    NiyahCoordinationLobe actor,
    NiyahCoordinationGate *out_gate,
    NiyahReceiptStatus *out_receipt_status);


const char *niyah_receipt_chain_status_name(
    NiyahReceiptChainStatus status);


#ifdef __cplusplus
}
#endif

#endif
