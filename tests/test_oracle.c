#include "niyah/oracle.h"

#include <stdio.h>
#include <string.h>

static int failures = 0;

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++failures; \
    } \
} while (0)

static void fill_digest(NiyahCoordinationDigest *digest, uint8_t seed) {
    size_t i;
    for (i = 0U; i < NIYAH_COORDINATION_DIGEST_BYTES; ++i) {
        digest->bytes[i] = (uint8_t)(seed + (uint8_t)i);
    }
}

static OracleExecutivePolicyV1 base_policy(void) {
    OracleExecutivePolicyV1 policy;
    memset(&policy, 0, sizeof(policy));
    policy.version = ORACLE_ABI_V1;
    policy.allowed_capabilities =
        ORACLE_CAP_RESPOND |
        ORACLE_CAP_SENSE |
        ORACLE_CAP_FILE_WRITE |
        ORACLE_CAP_NETWORK |
        ORACLE_CAP_TOOL |
        ORACLE_CAP_AUDIT;
    policy.max_token_budget = 4096U;
    policy.max_byte_budget = 1U << 20;
    policy.max_side_effects = 4U;
    policy.flags =
        ORACLE_POLICY_REQUIRE_EVIDENCE |
        ORACLE_POLICY_REQUIRE_TARGET |
        ORACLE_POLICY_DENY_UNKNOWN;
    return policy;
}

static OracleCognitiveProposalV1 base_proposal(void) {
    OracleCognitiveProposalV1 proposal;
    memset(&proposal, 0, sizeof(proposal));
    proposal.version = ORACLE_ABI_V1;
    proposal.action = ORACLE_ACTION_NETWORK_CONNECT;
    proposal.capability = ORACLE_CAP_NETWORK;
    proposal.subject_id = 7U;
    proposal.target_id = 42U;
    proposal.token_budget = 128U;
    proposal.byte_budget = 4096U;
    fill_digest(&proposal.evidence_root, 0x20U);
    return proposal;
}
static void test_allow_and_hash(void) {
    OracleExecutivePolicyV1 policy = base_policy();
    OracleCognitiveProposalV1 proposal = base_proposal();
    OracleExecutiveDecisionV1 decision;
    NiyahCoordinationDigest a;
    NiyahCoordinationDigest b;

    CHECK(oracle_policy_v1_validate(&policy));
    CHECK(oracle_proposal_v1_validate(&proposal));
    CHECK(oracle_proposal_v1_sha256(&proposal, &a) == 0);
    CHECK(oracle_proposal_v1_sha256(&proposal, &b) == 0);
    CHECK(memcmp(a.bytes, b.bytes, sizeof(a.bytes)) == 0);

    CHECK(oracle_executive_authorize_v1(
              &policy, &proposal, 0U, &decision) == ORACLE_REASON_OK);
    CHECK(decision.status == ORACLE_DECISION_ALLOW);
    CHECK(decision.granted_capability == ORACLE_CAP_NETWORK);
    CHECK(decision.subject_id == proposal.subject_id);
    CHECK(decision.target_id == proposal.target_id);
    CHECK(memcmp(decision.proposal_sha256.bytes, a.bytes, sizeof(a.bytes)) == 0);

    proposal.target_id = 43U;
    CHECK(oracle_proposal_v1_sha256(&proposal, &b) == 0);
    CHECK(memcmp(a.bytes, b.bytes, sizeof(a.bytes)) != 0);
}

static void test_fail_closed(void) {
    OracleExecutivePolicyV1 policy = base_policy();
    OracleCognitiveProposalV1 proposal = base_proposal();
    OracleExecutiveDecisionV1 decision;

    proposal.capability = ORACLE_CAP_TOOL;
    CHECK(oracle_executive_authorize_v1(
              &policy, &proposal, 0U, &decision) ==
          ORACLE_REASON_CAPABILITY_MISMATCH);
    CHECK(decision.status == ORACLE_DECISION_REJECT);

    proposal = base_proposal();
    policy.allowed_capabilities &= ~ORACLE_CAP_NETWORK;
    CHECK(oracle_executive_authorize_v1(
              &policy, &proposal, 0U, &decision) ==
          ORACLE_REASON_CAPABILITY_DENIED);

    policy = base_policy();
    proposal.token_budget = policy.max_token_budget + 1U;
    CHECK(oracle_executive_authorize_v1(
              &policy, &proposal, 0U, &decision) ==
          ORACLE_REASON_TOKEN_BUDGET);

    proposal = base_proposal();
    proposal.byte_budget = policy.max_byte_budget + 1U;
    CHECK(oracle_executive_authorize_v1(
              &policy, &proposal, 0U, &decision) ==
          ORACLE_REASON_BYTE_BUDGET);
}
static void test_policy_gates(void) {
    OracleExecutivePolicyV1 policy = base_policy();
    OracleCognitiveProposalV1 proposal = base_proposal();
    OracleExecutiveDecisionV1 decision;

    CHECK(oracle_executive_authorize_v1(
              &policy, &proposal, policy.max_side_effects, &decision) ==
          ORACLE_REASON_SIDE_EFFECT_BUDGET);

    proposal = base_proposal();
    memset(&proposal.evidence_root, 0, sizeof(proposal.evidence_root));
    CHECK(oracle_executive_authorize_v1(
              &policy, &proposal, 0U, &decision) ==
          ORACLE_REASON_EVIDENCE_REQUIRED);

    proposal = base_proposal();
    proposal.target_id = 0U;
    CHECK(oracle_executive_authorize_v1(
              &policy, &proposal, 0U, &decision) ==
          ORACLE_REASON_TARGET_REQUIRED);

    proposal = base_proposal();
    proposal.action = (OracleAction)99;
    CHECK(oracle_executive_authorize_v1(
              &policy, &proposal, 0U, &decision) ==
          ORACLE_REASON_INVALID_ACTION);

    proposal = base_proposal();
    proposal.version = ORACLE_ABI_V1 + 1U;
    CHECK(oracle_executive_authorize_v1(
              &policy, &proposal, 0U, &decision) ==
          ORACLE_REASON_INVALID_VERSION);
}

static void test_observation(void) {
    OracleObservationV1 observation;
    NiyahCoordinationDigest a;
    NiyahCoordinationDigest b;

    memset(&observation, 0, sizeof(observation));
    observation.version = ORACLE_ABI_V1;
    observation.type = ORACLE_OBSERVATION_NETWORK;
    observation.trust = ORACLE_TRUST_OBSERVED;
    observation.object_id = 88U;
    observation.byte_length = 512U;
    fill_digest(&observation.content_sha256, 0x70U);

    CHECK(oracle_observation_v1_validate(&observation));
    CHECK(oracle_observation_v1_sha256(&observation, &a) == 0);
    CHECK(oracle_observation_v1_sha256(&observation, &b) == 0);
    CHECK(memcmp(a.bytes, b.bytes, sizeof(a.bytes)) == 0);

    observation.byte_length++;
    CHECK(oracle_observation_v1_sha256(&observation, &b) == 0);
    CHECK(memcmp(a.bytes, b.bytes, sizeof(a.bytes)) != 0);
}

int main(void) {
    test_allow_and_hash();
    test_fail_closed();
    test_policy_gates();
    test_observation();

    CHECK(strcmp(
              oracle_decision_reason_name(ORACLE_REASON_CAPABILITY_DENIED),
              "CAPABILITY_DENIED") == 0);

    if (failures != 0) {
        fprintf(stderr, "ORACLE TEST FAILURES=%d\n", failures);
        return 1;
    }

    puts("ORACLE THREE-LOBE ABI PASS");
    return 0;
}
