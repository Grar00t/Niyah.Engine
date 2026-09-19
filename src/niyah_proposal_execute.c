#include "niyah/proposal_execute.h"

#include <string.h>

NiyahStatus niyah_proposal_execute_authorized(
    const NiyahProposalPolicy *policy,
    const NiyahProposal *proposal,
    NiyahProposalExecutionResult *out_result)
{
    NiyahProposalDecision decision;
    NiyahStatus status;

    if (policy == NULL ||
        proposal == NULL ||
        out_result == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    memset(
        out_result,
        0,
        sizeof(*out_result));

    status = niyah_proposal_policy_decide(
        policy,
        proposal,
        &decision);

    if (status != NIYAH_OK) {
        return status;
    }

    out_result->proposal = *proposal;
    out_result->decision = decision;

    if (decision ==
        NIYAH_PROPOSAL_DECISION_DENY) {
        return NIYAH_OK;
    }

    if (decision !=
        NIYAH_PROPOSAL_DECISION_ALLOW) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    switch (proposal->kind) {
        case NIYAH_PROPOSAL_ARITHMETIC:
            status = niyah_ir_execute(
                &proposal->arithmetic_ir,
                &out_result->arithmetic_result);

            if (status != NIYAH_OK) {
                return status;
            }

            out_result->executed = 1;
            return NIYAH_OK;

        case NIYAH_PROPOSAL_NETWORK:
            status = niyah_network_ir_execute(
                &proposal->network_ir,
                &out_result->network_match);

            if (status != NIYAH_OK) {
                return status;
            }

            out_result->executed = 1;
            return NIYAH_OK;

        default:
            return NIYAH_ERR_INVALID_ARGUMENT;
    }
}
