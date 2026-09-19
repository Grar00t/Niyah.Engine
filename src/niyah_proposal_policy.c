#include "niyah/proposal_policy.h"

static int valid_flag(int value)
{
    return value == 0 || value == 1;
}

NiyahStatus niyah_proposal_policy_decide(
    const NiyahProposalPolicy *policy,
    const NiyahProposal *proposal,
    NiyahProposalDecision *out_decision)
{
    if (policy == NULL ||
        proposal == NULL ||
        out_decision == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    *out_decision =
        NIYAH_PROPOSAL_DECISION_INVALID;

    if (!valid_flag(policy->allow_arithmetic) ||
        !valid_flag(policy->allow_network)) {
        return NIYAH_ERR_INVALID_CONFIG;
    }

    switch (proposal->kind) {
        case NIYAH_PROPOSAL_ARITHMETIC:
            *out_decision =
                policy->allow_arithmetic ?
                    NIYAH_PROPOSAL_DECISION_ALLOW :
                    NIYAH_PROPOSAL_DECISION_DENY;
            return NIYAH_OK;

        case NIYAH_PROPOSAL_NETWORK:
            *out_decision =
                policy->allow_network ?
                    NIYAH_PROPOSAL_DECISION_ALLOW :
                    NIYAH_PROPOSAL_DECISION_DENY;
            return NIYAH_OK;

        default:
            return NIYAH_ERR_INVALID_ARGUMENT;
    }
}
