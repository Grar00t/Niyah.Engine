#include "niyah/proposal.h"
#include "niyah/proposal_policy.h"

#include <string.h>

#define CHECK(x) \
    do { \
        if (!(x)) return __LINE__; \
    } while (0)

static int parse_arithmetic(
    NiyahProposal *proposal)
{
    return niyah_proposal_parse(
        "ADD|2|7",
        proposal) == NIYAH_OK;
}

static int parse_network(
    NiyahProposal *proposal)
{
    return niyah_proposal_parse(
        "IP_IN_CIDR|"
        "192.168.1.42|"
        "192.168.1.0|"
        "24",
        proposal) == NIYAH_OK;
}

static int expect_decision(
    const NiyahProposalPolicy *policy,
    const NiyahProposal *proposal,
    NiyahProposalDecision expected)
{
    NiyahProposalDecision decision =
        NIYAH_PROPOSAL_DECISION_INVALID;

    CHECK(
        niyah_proposal_policy_decide(
            policy,
            proposal,
            &decision) ==
        NIYAH_OK);

    CHECK(decision == expected);

    return 0;
}

int main(void)
{
    NiyahProposal arithmetic;
    NiyahProposal network;
    NiyahProposalPolicy policy;

    CHECK(parse_arithmetic(&arithmetic));
    CHECK(parse_network(&network));

    /*
     * Default-zero policy is fail-closed.
     */
    memset(&policy, 0, sizeof(policy));

    CHECK(
        expect_decision(
            &policy,
            &arithmetic,
            NIYAH_PROPOSAL_DECISION_DENY) == 0);

    CHECK(
        expect_decision(
            &policy,
            &network,
            NIYAH_PROPOSAL_DECISION_DENY) == 0);

    /*
     * Arithmetic-only allowlist.
     */
    memset(&policy, 0, sizeof(policy));
    policy.allow_arithmetic = 1;

    CHECK(
        expect_decision(
            &policy,
            &arithmetic,
            NIYAH_PROPOSAL_DECISION_ALLOW) == 0);

    CHECK(
        expect_decision(
            &policy,
            &network,
            NIYAH_PROPOSAL_DECISION_DENY) == 0);

    /*
     * Network-only allowlist.
     */
    memset(&policy, 0, sizeof(policy));
    policy.allow_network = 1;

    CHECK(
        expect_decision(
            &policy,
            &arithmetic,
            NIYAH_PROPOSAL_DECISION_DENY) == 0);

    CHECK(
        expect_decision(
            &policy,
            &network,
            NIYAH_PROPOSAL_DECISION_ALLOW) == 0);

    /*
     * Explicitly allow both known V1 domains.
     */
    memset(&policy, 0, sizeof(policy));
    policy.allow_arithmetic = 1;
    policy.allow_network = 1;

    CHECK(
        expect_decision(
            &policy,
            &arithmetic,
            NIYAH_PROPOSAL_DECISION_ALLOW) == 0);

    CHECK(
        expect_decision(
            &policy,
            &network,
            NIYAH_PROPOSAL_DECISION_ALLOW) == 0);

    /*
     * Invalid configuration is not a policy denial.
     */
    {
        NiyahProposalDecision decision =
            NIYAH_PROPOSAL_DECISION_ALLOW;

        memset(&policy, 0, sizeof(policy));
        policy.allow_arithmetic = 2;

        CHECK(
            niyah_proposal_policy_decide(
                &policy,
                &arithmetic,
                &decision) ==
            NIYAH_ERR_INVALID_CONFIG);

        CHECK(
            decision ==
            NIYAH_PROPOSAL_DECISION_INVALID);
    }

    /*
     * Unknown/corrupt proposal kind is an API/data error,
     * not an authorization denial.
     */
    {
        NiyahProposal invalid;
        NiyahProposalDecision decision =
            NIYAH_PROPOSAL_DECISION_ALLOW;

        memset(&invalid, 0, sizeof(invalid));
        memset(&policy, 0, sizeof(policy));

        CHECK(
            niyah_proposal_policy_decide(
                &policy,
                &invalid,
                &decision) ==
            NIYAH_ERR_INVALID_ARGUMENT);

        CHECK(
            decision ==
            NIYAH_PROPOSAL_DECISION_INVALID);
    }

    /*
     * Policy does not replace strict parsing.
     */
    CHECK(
        niyah_proposal_parse(
            "The answer is ADD|2|7",
            &arithmetic) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    CHECK(
        niyah_proposal_parse(
            "IP_IN_CIDR|"
            "192.168.1.42|"
            "192.168.1.7|"
            "24",
            &network) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    /*
     * API misuse.
     */
    {
        NiyahProposalDecision decision;
        NiyahProposal proposal;

        CHECK(parse_arithmetic(&proposal));

        memset(&policy, 0, sizeof(policy));

        CHECK(
            niyah_proposal_policy_decide(
                NULL,
                &proposal,
                &decision) ==
            NIYAH_ERR_INVALID_ARGUMENT);

        CHECK(
            niyah_proposal_policy_decide(
                &policy,
                NULL,
                &decision) ==
            NIYAH_ERR_INVALID_ARGUMENT);

        CHECK(
            niyah_proposal_policy_decide(
                &policy,
                &proposal,
                NULL) ==
            NIYAH_ERR_INVALID_ARGUMENT);
    }

    return 0;
}
