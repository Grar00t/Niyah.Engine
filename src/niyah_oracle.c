#include "niyah/oracle.h"
#include "niyah_sha256.h"

#include <string.h>

#define ORACLE_KNOWN_CAPABILITIES \
    (ORACLE_CAP_RESPOND | ORACLE_CAP_SENSE | ORACLE_CAP_FILE_WRITE | \
     ORACLE_CAP_NETWORK | ORACLE_CAP_TOOL | ORACLE_CAP_AUDIT)
#define ORACLE_KNOWN_POLICY_FLAGS \
    (ORACLE_POLICY_REQUIRE_EVIDENCE | ORACLE_POLICY_REQUIRE_TARGET | \
     ORACLE_POLICY_DENY_UNKNOWN)

static void oracle_put_u32_le(unsigned char out[4], uint32_t value) {
    out[0] = (unsigned char)(value & 0xffU);
    out[1] = (unsigned char)((value >> 8U) & 0xffU);
    out[2] = (unsigned char)((value >> 16U) & 0xffU);
    out[3] = (unsigned char)((value >> 24U) & 0xffU);
}

static void oracle_put_u64_le(unsigned char out[8], uint64_t value) {
    size_t i;
    for (i = 0U; i < 8U; ++i) {
        out[i] = (unsigned char)((value >> (8U * i)) & 0xffU);
    }
}

static int oracle_sha_u32(NiyahSha256 *sha, uint32_t value) {
    unsigned char bytes[4];
    oracle_put_u32_le(bytes, value);
    return niyah_sha256_update(sha, bytes, sizeof(bytes));
}

static int oracle_sha_u64(NiyahSha256 *sha, uint64_t value) {
    unsigned char bytes[8];
    oracle_put_u64_le(bytes, value);
    return niyah_sha256_update(sha, bytes, sizeof(bytes));
}

int oracle_digest_is_zero(const NiyahCoordinationDigest *digest) {
    size_t i;
    if (digest == NULL) {
        return 1;
    }
    for (i = 0U; i < NIYAH_COORDINATION_DIGEST_BYTES; ++i) {
        if (digest->bytes[i] != 0U) {
            return 0;
        }
    }
    return 1;
}
uint32_t oracle_action_required_capability(OracleAction action) {
    switch (action) {
        case ORACLE_ACTION_RESPOND: return ORACLE_CAP_RESPOND;
        case ORACLE_ACTION_SENSE: return ORACLE_CAP_SENSE;
        case ORACLE_ACTION_FILE_WRITE: return ORACLE_CAP_FILE_WRITE;
        case ORACLE_ACTION_NETWORK_CONNECT: return ORACLE_CAP_NETWORK;
        case ORACLE_ACTION_TOOL_EXECUTE: return ORACLE_CAP_TOOL;
        case ORACLE_ACTION_AUDIT_APPEND: return ORACLE_CAP_AUDIT;
        default: return 0U;
    }
}

int oracle_action_is_side_effect(OracleAction action) {
    return action == ORACLE_ACTION_FILE_WRITE ||
           action == ORACLE_ACTION_NETWORK_CONNECT ||
           action == ORACLE_ACTION_TOOL_EXECUTE ||
           action == ORACLE_ACTION_AUDIT_APPEND;
}

int oracle_observation_v1_validate(const OracleObservationV1 *observation) {
    if (observation == NULL || observation->version != ORACLE_ABI_V1) {
        return 0;
    }
    if (observation->type < ORACLE_OBSERVATION_BYTES ||
        observation->type > ORACLE_OBSERVATION_TOOL_RESULT) {
        return 0;
    }
    if (observation->trust < ORACLE_TRUST_UNTRUSTED ||
        observation->trust > ORACLE_TRUST_VERIFIED) {
        return 0;
    }
    return 1;
}

int oracle_proposal_v1_validate(const OracleCognitiveProposalV1 *proposal) {
    uint32_t required;
    if (proposal == NULL || proposal->version != ORACLE_ABI_V1) {
        return 0;
    }
    required = oracle_action_required_capability(proposal->action);
    if (required == 0U || proposal->capability != required) {
        return 0;
    }
    return 1;
}

int oracle_policy_v1_validate(const OracleExecutivePolicyV1 *policy) {
    if (policy == NULL || policy->version != ORACLE_ABI_V1) {
        return 0;
    }
    if ((policy->allowed_capabilities & ~ORACLE_KNOWN_CAPABILITIES) != 0U) {
        return 0;
    }
    if ((policy->flags & ~ORACLE_KNOWN_POLICY_FLAGS) != 0U) {
        return 0;
    }
    return 1;
}

int oracle_observation_v1_sha256(
    const OracleObservationV1 *observation,
    NiyahCoordinationDigest *out_digest) {
    static const unsigned char domain[] = "ORACLE-OBSERVATION-V1";
    NiyahSha256 sha;
    if (!oracle_observation_v1_validate(observation) || out_digest == NULL) {
        return -1;
    }
    niyah_sha256_init(&sha);
    if (niyah_sha256_update(&sha, domain, sizeof(domain) - 1U) == 0 ||
        oracle_sha_u32(&sha, observation->version) == 0 ||
        oracle_sha_u32(&sha, (uint32_t)observation->type) == 0 ||
        oracle_sha_u32(&sha, (uint32_t)observation->trust) == 0 ||
        oracle_sha_u32(&sha, observation->flags) == 0 ||
        oracle_sha_u64(&sha, observation->object_id) == 0 ||
        oracle_sha_u64(&sha, observation->byte_length) == 0 ||
        niyah_sha256_update(&sha, observation->content_sha256.bytes,
                            NIYAH_COORDINATION_DIGEST_BYTES) == 0) {
        return -1;
    }
    niyah_sha256_final(&sha, out_digest->bytes);
    return 0;
}
int oracle_proposal_v1_sha256(
    const OracleCognitiveProposalV1 *proposal,
    NiyahCoordinationDigest *out_digest) {
    static const unsigned char domain[] = "ORACLE-PROPOSAL-V1";
    NiyahSha256 sha;
    if (!oracle_proposal_v1_validate(proposal) || out_digest == NULL) {
        return -1;
    }
    niyah_sha256_init(&sha);
    if (niyah_sha256_update(&sha, domain, sizeof(domain) - 1U) == 0 ||
        oracle_sha_u32(&sha, proposal->version) == 0 ||
        oracle_sha_u32(&sha, (uint32_t)proposal->action) == 0 ||
        oracle_sha_u32(&sha, proposal->capability) == 0 ||
        oracle_sha_u32(&sha, proposal->flags) == 0 ||
        oracle_sha_u64(&sha, proposal->subject_id) == 0 ||
        oracle_sha_u64(&sha, proposal->target_id) == 0 ||
        oracle_sha_u64(&sha, proposal->token_budget) == 0 ||
        oracle_sha_u64(&sha, proposal->byte_budget) == 0 ||
        niyah_sha256_update(&sha, proposal->evidence_root.bytes,
                            NIYAH_COORDINATION_DIGEST_BYTES) == 0) {
        return -1;
    }
    niyah_sha256_final(&sha, out_digest->bytes);
    return 0;
}

static OracleDecisionReason oracle_reject(
    OracleExecutiveDecisionV1 *out_decision,
    OracleDecisionReason reason) {
    out_decision->status = ORACLE_DECISION_REJECT;
    out_decision->reason = reason;
    out_decision->granted_capability = 0U;
    return reason;
}

OracleDecisionReason oracle_executive_authorize_v1(
    const OracleExecutivePolicyV1 *policy,
    const OracleCognitiveProposalV1 *proposal,
    uint32_t side_effects_used,
    OracleExecutiveDecisionV1 *out_decision) {
    uint32_t required;

    if (out_decision == NULL) {
        return ORACLE_REASON_INVALID_ARGUMENT;
    }

    memset(out_decision, 0, sizeof(*out_decision));
    out_decision->version = ORACLE_ABI_V1;

    if (policy == NULL || proposal == NULL) {
        return oracle_reject(out_decision, ORACLE_REASON_INVALID_ARGUMENT);
    }

    out_decision->subject_id = proposal->subject_id;
    out_decision->target_id = proposal->target_id;

    if (policy->version != ORACLE_ABI_V1 ||
        proposal->version != ORACLE_ABI_V1) {
        return oracle_reject(out_decision, ORACLE_REASON_INVALID_VERSION);
    }

    if (!oracle_policy_v1_validate(policy)) {
        return oracle_reject(out_decision, ORACLE_REASON_INVALID_ARGUMENT);
    }

    required = oracle_action_required_capability(proposal->action);
    if (required == 0U) {
        return oracle_reject(out_decision, ORACLE_REASON_INVALID_ACTION);
    }

    if (proposal->capability != required) {
        return oracle_reject(out_decision, ORACLE_REASON_CAPABILITY_MISMATCH);
    }

    if ((policy->allowed_capabilities & required) == 0U) {
        return oracle_reject(out_decision, ORACLE_REASON_CAPABILITY_DENIED);
    }

    if (proposal->token_budget > policy->max_token_budget) {
        return oracle_reject(out_decision, ORACLE_REASON_TOKEN_BUDGET);
    }

    if (proposal->byte_budget > policy->max_byte_budget) {
        return oracle_reject(out_decision, ORACLE_REASON_BYTE_BUDGET);
    }
    if (oracle_action_is_side_effect(proposal->action) &&
        side_effects_used >= policy->max_side_effects) {
        return oracle_reject(out_decision, ORACLE_REASON_SIDE_EFFECT_BUDGET);
    }

    if ((policy->flags & ORACLE_POLICY_REQUIRE_EVIDENCE) != 0U &&
        oracle_digest_is_zero(&proposal->evidence_root)) {
        return oracle_reject(out_decision, ORACLE_REASON_EVIDENCE_REQUIRED);
    }

    if ((policy->flags & ORACLE_POLICY_REQUIRE_TARGET) != 0U &&
        proposal->target_id == 0U) {
        return oracle_reject(out_decision, ORACLE_REASON_TARGET_REQUIRED);
    }

    if (oracle_proposal_v1_sha256(proposal,
                                  &out_decision->proposal_sha256) != 0) {
        return oracle_reject(out_decision, ORACLE_REASON_INVALID_ARGUMENT);
    }

    out_decision->status = ORACLE_DECISION_ALLOW;
    out_decision->reason = ORACLE_REASON_OK;
    out_decision->granted_capability = required;
    return ORACLE_REASON_OK;
}

const char *oracle_decision_reason_name(OracleDecisionReason reason) {
    switch (reason) {
        case ORACLE_REASON_OK: return "OK";
        case ORACLE_REASON_INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case ORACLE_REASON_INVALID_VERSION: return "INVALID_VERSION";
        case ORACLE_REASON_INVALID_ACTION: return "INVALID_ACTION";
        case ORACLE_REASON_CAPABILITY_MISMATCH: return "CAPABILITY_MISMATCH";
        case ORACLE_REASON_CAPABILITY_DENIED: return "CAPABILITY_DENIED";
        case ORACLE_REASON_TOKEN_BUDGET: return "TOKEN_BUDGET";
        case ORACLE_REASON_BYTE_BUDGET: return "BYTE_BUDGET";
        case ORACLE_REASON_SIDE_EFFECT_BUDGET: return "SIDE_EFFECT_BUDGET";
        case ORACLE_REASON_EVIDENCE_REQUIRED: return "EVIDENCE_REQUIRED";
        case ORACLE_REASON_TARGET_REQUIRED: return "TARGET_REQUIRED";
        default: return "UNKNOWN";
    }
}
