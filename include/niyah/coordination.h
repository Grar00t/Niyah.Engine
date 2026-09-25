#ifndef NIYAH_COORDINATION_H
#define NIYAH_COORDINATION_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NIYAH_COORDINATION_DIGEST_BYTES 32U

typedef struct NiyahCoordinationDigest {
    uint8_t bytes[NIYAH_COORDINATION_DIGEST_BYTES];
} NiyahCoordinationDigest;


/*
 * The three lobes are roles, not model copies.
 *
 * COMPUTE:
 *   Core/runtime/training/checkpoint mutation.
 *
 * CHRONICLE:
 *   Provenance/graph/receipt state.
 *
 * VERIFIER:
 *   Independent verification and integration gates.
 *
 * Promotion policy is deterministic coordination logic rather than
 * a fourth autonomous model.
 */
typedef enum NiyahCoordinationLobe {
    NIYAH_LOBE_NONE = 0,
    NIYAH_LOBE_COMPUTE = 1,
    NIYAH_LOBE_CHRONICLE = 2,
    NIYAH_LOBE_VERIFIER = 3
} NiyahCoordinationLobe;


typedef enum NiyahCoordinationAction {
    NIYAH_COORDINATION_EXECUTE = 1,
    NIYAH_COORDINATION_VERIFY = 2,
    NIYAH_COORDINATION_PROMOTE = 3
} NiyahCoordinationAction;


/*
 * Scope is explicit so evidence from one subsystem cannot silently
 * authorize mutation in another.
 */
typedef enum NiyahCoordinationScope {
    NIYAH_SCOPE_ENGINE_CORE = 1U << 0,
    NIYAH_SCOPE_GRAPH       = 1U << 1,
    NIYAH_SCOPE_DATA        = 1U << 2,
    NIYAH_SCOPE_CHECKPOINT  = 1U << 3,
    NIYAH_SCOPE_EVIDENCE    = 1U << 4
} NiyahCoordinationScope;


/*
 * These gates encode failure modes observed during real development.
 */
typedef enum NiyahCoordinationGate {
    NIYAH_COORDINATION_PASS = 0,

    NIYAH_COORDINATION_INVALID_ENVELOPE,
    NIYAH_COORDINATION_STALE_BASE,
    NIYAH_COORDINATION_WRONG_SCOPE,
    NIYAH_COORDINATION_WRITER_VIOLATION,
    NIYAH_COORDINATION_SELF_VERIFICATION,
    NIYAH_COORDINATION_CONTRADICTION,
    NIYAH_COORDINATION_PROMOTION_WITHOUT_PROOF,
    NIYAH_COORDINATION_UNAUTHORIZED_ACTOR
} NiyahCoordinationGate;


/*
 * A compact handoff envelope.
 *
 * base_revision:
 *   State the task was authorized against.
 *
 * observed_revision:
 *   State immediately before the requested action.
 *
 * project_id / expected_project_id:
 *   Caller-owned deterministic project identifiers.
 *
 * requested_scope / allowed_scope:
 *   Capability boundary.
 *
 * writer_lobe:
 *   Enforces ONE ARTIFACT = ONE WRITER.
 */
typedef struct NiyahCoordinationEnvelope {
    NiyahCoordinationDigest base_revision;
    NiyahCoordinationDigest observed_revision;

    uint32_t project_id;
    uint32_t expected_project_id;

    uint32_t requested_scope;
    uint32_t allowed_scope;

    NiyahCoordinationLobe producer_lobe;
    NiyahCoordinationLobe verifier_lobe;
    NiyahCoordinationLobe writer_lobe;

    int proof_present;
    int contradiction_present;
} NiyahCoordinationEnvelope;


int niyah_coordination_digest_equal(
    const NiyahCoordinationDigest *a,
    const NiyahCoordinationDigest *b);


NiyahCoordinationGate niyah_coordination_gate(
    const NiyahCoordinationEnvelope *envelope,
    NiyahCoordinationAction action,
    NiyahCoordinationLobe actor);


const char *niyah_coordination_gate_name(
    NiyahCoordinationGate gate);


#ifdef __cplusplus
}
#endif

#endif
