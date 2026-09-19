#include "niyah/proposal.h"
#include "niyah/proposal_execute.h"
#include "niyah/proposal_policy.h"

#include <stdint.h>
#include <string.h>

#define CHECK(x) \
    do { \
        if (!(x)) return __LINE__; \
    } while (0)

static int parse(
    const char *text,
    NiyahProposal *proposal)
{
    return niyah_proposal_parse(
        text,
        proposal) == NIYAH_OK;
}

int main(void)
{
    NiyahProposal arithmetic;
    NiyahProposal network_inside;
    NiyahProposal network_outside;
    NiyahProposalPolicy policy;
    NiyahProposalExecutionResult result;

    CHECK(
        parse(
            "ADD|2|7",
            &arithmetic));

    CHECK(
        parse(
            "IP_IN_CIDR|"
            "192.168.1.42|"
            "192.168.1.0|"
            "24",
            &network_inside));

    CHECK(
        parse(
            "IP_IN_CIDR|"
            "192.168.2.42|"
            "192.168.1.0|"
            "24",
            &network_outside));

    /*
     * Default policy denies and must not execute.
     */
    memset(&policy, 0, sizeof(policy));

    CHECK(
        niyah_proposal_execute_authorized(
            &policy,
            &arithmetic,
            &result) ==
        NIYAH_OK);

    CHECK(
        result.decision ==
        NIYAH_PROPOSAL_DECISION_DENY);

    CHECK(result.executed == 0);

    CHECK(
        result.proposal.kind ==
        NIYAH_PROPOSAL_ARITHMETIC);

    CHECK(
        niyah_proposal_execute_authorized(
            &policy,
            &network_inside,
            &result) ==
        NIYAH_OK);

    CHECK(
        result.decision ==
        NIYAH_PROPOSAL_DECISION_DENY);

    CHECK(result.executed == 0);

    /*
     * Arithmetic-only policy.
     */
    memset(&policy, 0, sizeof(policy));
    policy.allow_arithmetic = 1;

    CHECK(
        niyah_proposal_execute_authorized(
            &policy,
            &arithmetic,
            &result) ==
        NIYAH_OK);

    CHECK(
        result.decision ==
        NIYAH_PROPOSAL_DECISION_ALLOW);

    CHECK(result.executed == 1);

    CHECK(result.arithmetic_result == 9);

    /*
     * Same policy still denies network.
     */
    CHECK(
        niyah_proposal_execute_authorized(
            &policy,
            &network_inside,
            &result) ==
        NIYAH_OK);

    CHECK(
        result.decision ==
        NIYAH_PROPOSAL_DECISION_DENY);

    CHECK(result.executed == 0);

    /*
     * Network-only policy.
     */
    memset(&policy, 0, sizeof(policy));
    policy.allow_network = 1;

    CHECK(
        niyah_proposal_execute_authorized(
            &policy,
            &network_inside,
            &result) ==
        NIYAH_OK);

    CHECK(
        result.decision ==
        NIYAH_PROPOSAL_DECISION_ALLOW);

    CHECK(result.executed == 1);
    CHECK(result.network_match == 1);

    CHECK(
        result.proposal.network_ir.address ==
        UINT32_C(0xC0A8012A));

    CHECK(
        niyah_proposal_execute_authorized(
            &policy,
            &network_outside,
            &result) ==
        NIYAH_OK);

    CHECK(
        result.decision ==
        NIYAH_PROPOSAL_DECISION_ALLOW);

    CHECK(result.executed == 1);
    CHECK(result.network_match == 0);

    /*
     * ALLOW does not guarantee successful execution.
     *
     * Parsing is valid because each operand fits int64.
     * Execution overflows.
     */
    {
        NiyahProposal overflow;

        CHECK(
            parse(
                "ADD|9223372036854775807|1",
                &overflow));

        memset(&policy, 0, sizeof(policy));
        policy.allow_arithmetic = 1;

        CHECK(
            niyah_proposal_execute_authorized(
                &policy,
                &overflow,
                &result) ==
            NIYAH_ERR_OVERFLOW);

        CHECK(
            result.decision ==
            NIYAH_PROPOSAL_DECISION_ALLOW);

        CHECK(result.executed == 0);
    }

    /*
     * Invalid policy is a configuration error,
     * not a denial.
     */
    {
        memset(&policy, 0, sizeof(policy));
        policy.allow_network = 2;

        CHECK(
            niyah_proposal_execute_authorized(
                &policy,
                &network_inside,
                &result) ==
            NIYAH_ERR_INVALID_CONFIG);

        CHECK(
            result.decision ==
            NIYAH_PROPOSAL_DECISION_INVALID);

        CHECK(result.executed == 0);
    }

    /*
     * Corrupt proposal kind is rejected before execution.
     */
    {
        NiyahProposal invalid;

        memset(&invalid, 0, sizeof(invalid));
        memset(&policy, 0, sizeof(policy));
        policy.allow_arithmetic = 1;
        policy.allow_network = 1;

        CHECK(
            niyah_proposal_execute_authorized(
                &policy,
                &invalid,
                &result) ==
            NIYAH_ERR_INVALID_ARGUMENT);

        CHECK(
            result.decision ==
            NIYAH_PROPOSAL_DECISION_INVALID);

        CHECK(result.executed == 0);
    }

    /*
     * A typed kind with invalid inner IR cannot become
     * executable merely because policy allows its domain.
     */
    {
        NiyahProposal invalid_ir;

        memset(&invalid_ir, 0, sizeof(invalid_ir));

        invalid_ir.kind =
            NIYAH_PROPOSAL_NETWORK;

        invalid_ir.network_ir.op =
            NIYAH_NETWORK_IR_OP_INVALID;

        memset(&policy, 0, sizeof(policy));
        policy.allow_network = 1;

        CHECK(
            niyah_proposal_execute_authorized(
                &policy,
                &invalid_ir,
                &result) ==
            NIYAH_ERR_INVALID_ARGUMENT);

        CHECK(
            result.decision ==
            NIYAH_PROPOSAL_DECISION_ALLOW);

        CHECK(result.executed == 0);
    }

    /* API misuse. */
    {
        memset(&policy, 0, sizeof(policy));

        CHECK(
            niyah_proposal_execute_authorized(
                NULL,
                &arithmetic,
                &result) ==
            NIYAH_ERR_INVALID_ARGUMENT);

        CHECK(
            niyah_proposal_execute_authorized(
                &policy,
                NULL,
                &result) ==
            NIYAH_ERR_INVALID_ARGUMENT);

        CHECK(
            niyah_proposal_execute_authorized(
                &policy,
                &arithmetic,
                NULL) ==
            NIYAH_ERR_INVALID_ARGUMENT);
    }

    return 0;
}
