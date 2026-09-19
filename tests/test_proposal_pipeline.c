#include "niyah/proposal_pipeline.h"

#include <stdint.h>
#include <string.h>

#define CHECK(x) \
    do { \
        if (!(x)) return __LINE__; \
    } while (0)

static int run_ok(
    const char *user_text,
    const char *proposal_text,
    const NiyahProposalPolicy *policy,
    NiyahGuardedProposalState expected_state,
    NiyahGuardedProposalResult *out_result)
{
    NiyahGuardedProposalResult local;
    NiyahGuardedProposalResult *result =
        out_result != NULL ?
            out_result :
            &local;

    CHECK(
        niyah_guarded_proposal_run(
            user_text,
            proposal_text,
            policy,
            result) ==
        NIYAH_OK);

    CHECK(result->state == expected_state);

    return 0;
}

int main(void)
{
    NiyahProposalPolicy policy;
    NiyahGuardedProposalResult result;

    /*
     * Malformed model prose fails closed at strict parse.
     */
    memset(&policy, 0, sizeof(policy));
    policy.allow_arithmetic = 1;
    policy.allow_network = 1;

    CHECK(
        run_ok(
            "What is 2+7?",
            "The answer is ADD|2|7",
            &policy,
            NIYAH_GUARDED_PROPOSAL_PARSE_REJECTED,
            &result) == 0);

    CHECK(
        result.proposal_status ==
        NIYAH_ERR_INVALID_ARGUMENT);

    CHECK(
        result.grounding.state ==
        NIYAH_PROPOSAL_GROUNDING_INVALID);

    CHECK(result.execution.executed == 0);

    /*
     * Parser overflow is also a safe proposal rejection.
     */
    CHECK(
        run_ok(
            "What is 1+1?",
            "ADD|9223372036854775808|1",
            &policy,
            NIYAH_GUARDED_PROPOSAL_PARSE_REJECTED,
            &result) == 0);

    CHECK(
        result.proposal_status ==
        NIYAH_ERR_OVERFLOW);

    CHECK(result.execution.executed == 0);

    /*
     * Valid network syntax but wrong proposal values:
     * grounding blocks before policy/execution.
     */
    memset(&policy, 0, sizeof(policy));
    policy.allow_network = 1;

    CHECK(
        run_ok(
            "Is 192.168.1.42 inside "
            "192.168.1.0/24?",
            "IP_IN_CIDR|"
            "8.8.8.8|"
            "192.168.1.0|"
            "24",
            &policy,
            NIYAH_GUARDED_PROPOSAL_GROUNDING_REJECTED,
            &result) == 0);

    CHECK(
        result.proposal_status ==
        NIYAH_OK);

    CHECK(
        result.grounding.state ==
        NIYAH_PROPOSAL_GROUNDING_VALUE_MISMATCH);

    CHECK(
        result.execution.decision ==
        NIYAH_PROPOSAL_DECISION_INVALID);

    CHECK(result.execution.executed == 0);

    /*
     * No trusted reference means STOP even when domain is
     * policy-allowed.
     */
    CHECK(
        run_ok(
            "Compare 192.168.1.42 with "
            "192.168.1.0/24.",
            "IP_IN_CIDR|"
            "192.168.1.42|"
            "192.168.1.0|"
            "24",
            &policy,
            NIYAH_GUARDED_PROPOSAL_GROUNDING_REJECTED,
            &result) == 0);

    CHECK(
        result.grounding.state ==
        NIYAH_PROPOSAL_GROUNDING_NO_REFERENCE);

    CHECK(result.execution.executed == 0);

    /*
     * Arithmetic currently cannot cross the guarded
     * boundary because V1 has no trusted arithmetic
     * natural-language grounding route.
     */
    memset(&policy, 0, sizeof(policy));
    policy.allow_arithmetic = 1;

    CHECK(
        run_ok(
            "What is 2+7?",
            "ADD|2|7",
            &policy,
            NIYAH_GUARDED_PROPOSAL_GROUNDING_REJECTED,
            &result) == 0);

    CHECK(
        result.grounding.state ==
        NIYAH_PROPOSAL_GROUNDING_NO_REFERENCE);

    CHECK(result.execution.executed == 0);

    /*
     * A typed arithmetic proposal against a concrete network
     * request is a kind mismatch.
     */
    policy.allow_network = 1;

    CHECK(
        run_ok(
            "Is 192.168.1.42 inside "
            "192.168.1.0/24?",
            "ADD|2|7",
            &policy,
            NIYAH_GUARDED_PROPOSAL_GROUNDING_REJECTED,
            &result) == 0);

    CHECK(
        result.grounding.state ==
        NIYAH_PROPOSAL_GROUNDING_KIND_MISMATCH);

    CHECK(result.execution.executed == 0);

    /*
     * Exact grounding does not imply policy authorization.
     */
    memset(&policy, 0, sizeof(policy));

    CHECK(
        run_ok(
            "Is 192.168.1.42 inside "
            "192.168.1.0/24?",
            "IP_IN_CIDR|"
            "192.168.1.42|"
            "192.168.1.0|"
            "24",
            &policy,
            NIYAH_GUARDED_PROPOSAL_POLICY_DENIED,
            &result) == 0);

    CHECK(
        result.grounding.state ==
        NIYAH_PROPOSAL_GROUNDING_MATCH);

    CHECK(
        result.execution.decision ==
        NIYAH_PROPOSAL_DECISION_DENY);

    CHECK(result.execution.executed == 0);

    /*
     * Exact grounding + explicit policy ALLOW permits
     * deterministic execution.
     */
    memset(&policy, 0, sizeof(policy));
    policy.allow_network = 1;

    CHECK(
        run_ok(
            "Is 192.168.1.42 inside "
            "192.168.1.0/24?",
            "IP_IN_CIDR|"
            "192.168.1.42|"
            "192.168.1.0|"
            "24",
            &policy,
            NIYAH_GUARDED_PROPOSAL_EXECUTED,
            &result) == 0);

    CHECK(
        result.grounding.state ==
        NIYAH_PROPOSAL_GROUNDING_MATCH);

    CHECK(
        result.execution.decision ==
        NIYAH_PROPOSAL_DECISION_ALLOW);

    CHECK(result.execution.executed == 1);
    CHECK(result.execution.network_match == 1);

    CHECK(
        result.execution.proposal.network_ir.address ==
        UINT32_C(0xC0A8012A));

    /*
     * A grounded negative membership result is still a
     * successful deterministic execution.
     */
    CHECK(
        run_ok(
            "Is 192.168.2.42 inside "
            "192.168.1.0/24?",
            "IP_IN_CIDR|"
            "192.168.2.42|"
            "192.168.1.0|"
            "24",
            &policy,
            NIYAH_GUARDED_PROPOSAL_EXECUTED,
            &result) == 0);

    CHECK(result.execution.executed == 1);
    CHECK(result.execution.network_match == 0);

    /*
     * Stage-order proof:
     *
     * Invalid policy must NOT be evaluated if grounding
     * already rejects the proposal.
     */
    memset(&policy, 0, sizeof(policy));
    policy.allow_network = 2;

    CHECK(
        run_ok(
            "Is 192.168.1.42 inside "
            "192.168.1.0/24?",
            "IP_IN_CIDR|"
            "8.8.8.8|"
            "192.168.1.0|"
            "24",
            &policy,
            NIYAH_GUARDED_PROPOSAL_GROUNDING_REJECTED,
            &result) == 0);

    CHECK(
        result.grounding.state ==
        NIYAH_PROPOSAL_GROUNDING_VALUE_MISMATCH);

    CHECK(result.execution.executed == 0);

    /*
     * Once grounding MATCH is reached, the same invalid
     * policy becomes a real configuration error.
     */
    CHECK(
        niyah_guarded_proposal_run(
            "Is 192.168.1.42 inside "
            "192.168.1.0/24?",
            "IP_IN_CIDR|"
            "192.168.1.42|"
            "192.168.1.0|"
            "24",
            &policy,
            &result) ==
        NIYAH_ERR_INVALID_CONFIG);

    CHECK(
        result.grounding.state ==
        NIYAH_PROPOSAL_GROUNDING_MATCH);

    CHECK(result.execution.executed == 0);

    /*
     * API misuse remains a real error.
     */
    memset(&policy, 0, sizeof(policy));

    CHECK(
        niyah_guarded_proposal_run(
            NULL,
            "ADD|2|7",
            &policy,
            &result) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    CHECK(
        niyah_guarded_proposal_run(
            "What is 2+7?",
            NULL,
            &policy,
            &result) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    CHECK(
        niyah_guarded_proposal_run(
            "What is 2+7?",
            "ADD|2|7",
            NULL,
            &result) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    CHECK(
        niyah_guarded_proposal_run(
            "What is 2+7?",
            "ADD|2|7",
            &policy,
            NULL) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    return 0;
}
