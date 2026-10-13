#ifndef NIYAH_ORACLE_H
#define NIYAH_ORACLE_H

#include <stddef.h>
#include <stdint.h>

#include "niyah/coordination.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ORACLE_ABI_V1 1U

#define ORACLE_POLICY_REQUIRE_EVIDENCE (1U << 0)
#define ORACLE_POLICY_REQUIRE_TARGET   (1U << 1)
#define ORACLE_POLICY_DENY_UNKNOWN     (1U << 2)

typedef enum OracleLobe {
    ORACLE_LOBE_NONE = 0,
    ORACLE_LOBE_SENSORY = 1,
    ORACLE_LOBE_COGNITIVE = 2,
    ORACLE_LOBE_EXECUTIVE = 3
} OracleLobe;

typedef enum OracleObservationType {
    ORACLE_OBSERVATION_BYTES = 1,
    ORACLE_OBSERVATION_TEXT = 2,
    ORACLE_OBSERVATION_FILE = 3,
    ORACLE_OBSERVATION_NETWORK = 4,
    ORACLE_OBSERVATION_TOOL_RESULT = 5
} OracleObservationType;

typedef enum OracleTrust {
    ORACLE_TRUST_UNTRUSTED = 1,
    ORACLE_TRUST_OBSERVED = 2,
    ORACLE_TRUST_VERIFIED = 3
} OracleTrust;

typedef enum OracleCapability {
    ORACLE_CAP_RESPOND = 1U << 0,
    ORACLE_CAP_SENSE = 1U << 1,
    ORACLE_CAP_FILE_WRITE = 1U << 2,
    ORACLE_CAP_NETWORK = 1U << 3,
    ORACLE_CAP_TOOL = 1U << 4,
    ORACLE_CAP_AUDIT = 1U << 5
} OracleCapability;

typedef enum OracleAction {
    ORACLE_ACTION_NONE = 0,
    ORACLE_ACTION_RESPOND = 1,
    ORACLE_ACTION_SENSE = 2,
    ORACLE_ACTION_FILE_WRITE = 3,
    ORACLE_ACTION_NETWORK_CONNECT = 4,
    ORACLE_ACTION_TOOL_EXECUTE = 5,
    ORACLE_ACTION_AUDIT_APPEND = 6
} OracleAction;

typedef enum OracleDecisionStatus {
    ORACLE_DECISION_REJECT = 0,
    ORACLE_DECISION_ALLOW = 1
} OracleDecisionStatus;

typedef enum OracleDecisionReason {
    ORACLE_REASON_OK = 0,
    ORACLE_REASON_INVALID_ARGUMENT,
    ORACLE_REASON_INVALID_VERSION,
    ORACLE_REASON_INVALID_ACTION,
    ORACLE_REASON_CAPABILITY_MISMATCH,
    ORACLE_REASON_CAPABILITY_DENIED,
    ORACLE_REASON_TOKEN_BUDGET,
    ORACLE_REASON_BYTE_BUDGET,
    ORACLE_REASON_SIDE_EFFECT_BUDGET,
    ORACLE_REASON_EVIDENCE_REQUIRED,
    ORACLE_REASON_TARGET_REQUIRED
} OracleDecisionReason;

typedef struct OracleObservationV1 {
    uint32_t version;
    OracleObservationType type;
    OracleTrust trust;
    uint32_t flags;
    uint64_t object_id;
    uint64_t byte_length;
    NiyahCoordinationDigest content_sha256;
} OracleObservationV1;

typedef struct OracleCognitiveProposalV1 {
    uint32_t version;
    OracleAction action;
    uint32_t capability;
    uint32_t flags;
    uint64_t subject_id;
    uint64_t target_id;
    uint64_t token_budget;
    uint64_t byte_budget;
    NiyahCoordinationDigest evidence_root;
} OracleCognitiveProposalV1;

typedef struct OracleExecutivePolicyV1 {
    uint32_t version;
    uint32_t allowed_capabilities;
    uint64_t max_token_budget;
    uint64_t max_byte_budget;
    uint32_t max_side_effects;
    uint32_t flags;
} OracleExecutivePolicyV1;

typedef struct OracleExecutiveDecisionV1 {
    uint32_t version;
    OracleDecisionStatus status;
    OracleDecisionReason reason;
    uint32_t granted_capability;
    uint64_t subject_id;
    uint64_t target_id;
    NiyahCoordinationDigest proposal_sha256;
} OracleExecutiveDecisionV1;

/*
 * ORACLE core is intentionally content-agnostic. It authorizes capabilities,
 * budgets and provenance only. No political, ideological, geographic or
 * vendor identity taxonomy is encoded in this ABI.
 */

int oracle_digest_is_zero(const NiyahCoordinationDigest *digest);
uint32_t oracle_action_required_capability(OracleAction action);
int oracle_action_is_side_effect(OracleAction action);

int oracle_observation_v1_validate(const OracleObservationV1 *observation);
int oracle_proposal_v1_validate(const OracleCognitiveProposalV1 *proposal);
int oracle_policy_v1_validate(const OracleExecutivePolicyV1 *policy);

int oracle_observation_v1_sha256(
    const OracleObservationV1 *observation,
    NiyahCoordinationDigest *out_digest);

int oracle_proposal_v1_sha256(
    const OracleCognitiveProposalV1 *proposal,
    NiyahCoordinationDigest *out_digest);

OracleDecisionReason oracle_executive_authorize_v1(
    const OracleExecutivePolicyV1 *policy,
    const OracleCognitiveProposalV1 *proposal,
    uint32_t side_effects_used,
    OracleExecutiveDecisionV1 *out_decision);

const char *oracle_decision_reason_name(OracleDecisionReason reason);

#ifdef __cplusplus
}
#endif

#endif
