#include "niyah/coordination.h"
#include "niyah/oracle.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++failures; \
    } \
} while (0)

static NiyahCoordinationDigest digest(uint8_t seed) {
    NiyahCoordinationDigest value;
    size_t i;

    for (i = 0U; i < NIYAH_COORDINATION_DIGEST_BYTES; ++i) {
        value.bytes[i] = (uint8_t)(seed + (uint8_t)i);
    }

    return value;
}

static OracleExecutivePolicyV1 policy_allow_network(void) {
    OracleExecutivePolicyV1 policy;

    memset(&policy, 0, sizeof(policy));
    policy.version = ORACLE_ABI_V1;
    policy.allowed_capabilities = ORACLE_CAP_NETWORK;
    policy.max_token_budget = 1024U;
    policy.max_byte_budget = 4096U;
    policy.max_side_effects = 1U;
    policy.flags =
        ORACLE_POLICY_REQUIRE_EVIDENCE |
        ORACLE_POLICY_REQUIRE_TARGET;

    return policy;
}

static OracleCognitiveProposalV1 network_proposal(void) {
    OracleCognitiveProposalV1 proposal;

    memset(&proposal, 0, sizeof(proposal));
    proposal.version = ORACLE_ABI_V1;
    proposal.action = ORACLE_ACTION_NETWORK_CONNECT;
    proposal.capability = ORACLE_CAP_NETWORK;
    proposal.subject_id = 7U;
    proposal.target_id = 42U;
    proposal.token_budget = 32U;
    proposal.byte_budget = 256U;
    proposal.evidence_root = digest(0x20U);

    return proposal;
}

static NiyahCoordinationEnvelope good_envelope(void) {
    NiyahCoordinationEnvelope envelope;

    memset(&envelope, 0, sizeof(envelope));
    envelope.base_revision = digest(0x40U);
    envelope.observed_revision = envelope.base_revision;
    envelope.project_id = UINT32_C(0x4e495941);
    envelope.expected_project_id = envelope.project_id;
    envelope.requested_scope = NIYAH_SCOPE_EVIDENCE;
    envelope.allowed_scope = NIYAH_SCOPE_EVIDENCE;
    envelope.producer_lobe = NIYAH_LOBE_CHRONICLE;
    envelope.verifier_lobe = NIYAH_LOBE_VERIFIER;
    envelope.writer_lobe = NIYAH_LOBE_CHRONICLE;
    envelope.proof_present = 1;
    envelope.contradiction_present = 0;

    return envelope;
}

static int authorize_and_promote(
    const OracleExecutivePolicyV1 *policy,
    const OracleCognitiveProposalV1 *proposal,
    uint32_t side_effects_used,
    const NiyahCoordinationEnvelope *envelope,
    OracleDecisionReason *out_oracle,
    NiyahCoordinationGate *out_coordination) {
    OracleExecutiveDecisionV1 decision;
    OracleDecisionReason oracle_reason;
    NiyahCoordinationGate coordination_gate;

    if (out_oracle == NULL || out_coordination == NULL) {
        return 0;
    }

    oracle_reason = oracle_executive_authorize_v1(
        policy,
        proposal,
        side_effects_used,
        &decision);

    *out_oracle = oracle_reason;

    if (oracle_reason != ORACLE_REASON_OK ||
        decision.status != ORACLE_DECISION_ALLOW) {
        *out_coordination = NIYAH_COORDINATION_INVALID_ENVELOPE;
        return 0;
    }

    coordination_gate = niyah_coordination_gate(
        envelope,
        NIYAH_COORDINATION_PROMOTE,
        NIYAH_LOBE_VERIFIER);

    *out_coordination = coordination_gate;

    return coordination_gate == NIYAH_COORDINATION_PASS;
}

static void test_allow_only_when_both_pass(void) {
    OracleExecutivePolicyV1 policy = policy_allow_network();
    OracleCognitiveProposalV1 proposal = network_proposal();
    NiyahCoordinationEnvelope envelope = good_envelope();
    OracleDecisionReason oracle_reason;
    NiyahCoordinationGate coordination_gate;

    CHECK(authorize_and_promote(
        &policy,
        &proposal,
        0U,
        &envelope,
        &oracle_reason,
        &coordination_gate));
    CHECK(oracle_reason == ORACLE_REASON_OK);
    CHECK(coordination_gate == NIYAH_COORDINATION_PASS);
}

static void test_oracle_denial_blocks_promotion(void) {
    OracleExecutivePolicyV1 policy = policy_allow_network();
    OracleCognitiveProposalV1 proposal = network_proposal();
    NiyahCoordinationEnvelope envelope = good_envelope();
    OracleDecisionReason oracle_reason;
    NiyahCoordinationGate coordination_gate;

    policy.allowed_capabilities = 0U;

    CHECK(!authorize_and_promote(
        &policy,
        &proposal,
        0U,
        &envelope,
        &oracle_reason,
        &coordination_gate));
    CHECK(oracle_reason == ORACLE_REASON_CAPABILITY_DENIED);
    CHECK(coordination_gate == NIYAH_COORDINATION_INVALID_ENVELOPE);
}

static void test_stale_state_blocks_promotion(void) {
    OracleExecutivePolicyV1 policy = policy_allow_network();
    OracleCognitiveProposalV1 proposal = network_proposal();
    NiyahCoordinationEnvelope envelope = good_envelope();
    OracleDecisionReason oracle_reason;
    NiyahCoordinationGate coordination_gate;

    envelope.observed_revision.bytes[0] ^= UINT8_C(0x01);

    CHECK(!authorize_and_promote(
        &policy,
        &proposal,
        0U,
        &envelope,
        &oracle_reason,
        &coordination_gate));
    CHECK(oracle_reason == ORACLE_REASON_OK);
    CHECK(coordination_gate == NIYAH_COORDINATION_STALE_BASE);
}

static void test_missing_proof_blocks_promotion(void) {
    OracleExecutivePolicyV1 policy = policy_allow_network();
    OracleCognitiveProposalV1 proposal = network_proposal();
    NiyahCoordinationEnvelope envelope = good_envelope();
    OracleDecisionReason oracle_reason;
    NiyahCoordinationGate coordination_gate;

    envelope.proof_present = 0;

    CHECK(!authorize_and_promote(
        &policy,
        &proposal,
        0U,
        &envelope,
        &oracle_reason,
        &coordination_gate));
    CHECK(oracle_reason == ORACLE_REASON_OK);
    CHECK(coordination_gate == NIYAH_COORDINATION_PROMOTION_WITHOUT_PROOF);
}

static void test_contradiction_blocks_promotion(void) {
    OracleExecutivePolicyV1 policy = policy_allow_network();
    OracleCognitiveProposalV1 proposal = network_proposal();
    NiyahCoordinationEnvelope envelope = good_envelope();
    OracleDecisionReason oracle_reason;
    NiyahCoordinationGate coordination_gate;

    envelope.contradiction_present = 1;

    CHECK(!authorize_and_promote(
        &policy,
        &proposal,
        0U,
        &envelope,
        &oracle_reason,
        &coordination_gate));
    CHECK(oracle_reason == ORACLE_REASON_OK);
    CHECK(coordination_gate == NIYAH_COORDINATION_CONTRADICTION);
}

int main(void) {
    test_allow_only_when_both_pass();
    test_oracle_denial_blocks_promotion();
    test_stale_state_blocks_promotion();
    test_missing_proof_blocks_promotion();
    test_contradiction_blocks_promotion();

    if (failures != 0) {
        fprintf(stderr, "ORACLE_COORDINATION_FAILURES=%d\n", failures);
        return 1;
    }

    puts("ORACLE_COORDINATION_INTEGRATION=PASS");
    return 0;
}
