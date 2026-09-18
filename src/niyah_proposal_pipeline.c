#include "niyah/proposal_pipeline.h"

#include <string.h>

static int is_expected_parse_rejection(
    NiyahStatus status)
{
    return
        status == NIYAH_ERR_INVALID_ARGUMENT ||
        status == NIYAH_ERR_OVERFLOW;
}

NiyahStatus niyah_guarded_proposal_run(
    const char *user_text,
    const char *proposal_text,
    const NiyahProposalPolicy *policy,
    NiyahGuardedProposalResult *out_result)
{
    NiyahStatus status;

    if (user_text == NULL ||
        proposal_text == NULL ||
        policy == NULL ||
        out_result == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    memset(
        out_result,
        0,
        sizeof(*out_result));

    status = niyah_proposal_parse(
        proposal_text,
        &out_result->proposal);

    out_result->proposal_status =
        status;

    if (status != NIYAH_OK) {
        if (is_expected_parse_rejection(status)) {
            out_result->state =
                NIYAH_GUARDED_PROPOSAL_PARSE_REJECTED;

            return NIYAH_OK;
        }

        return status;
    }

    status = niyah_proposal_ground_text(
        user_text,
        &out_result->proposal,
        &out_result->grounding);

    if (status != NIYAH_OK) {
        return status;
    }

    switch (out_result->grounding.state) {
        case NIYAH_PROPOSAL_GROUNDING_NO_REFERENCE:
        case NIYAH_PROPOSAL_GROUNDING_KIND_MISMATCH:
        case NIYAH_PROPOSAL_GROUNDING_VALUE_MISMATCH:
            out_result->state =
                NIYAH_GUARDED_PROPOSAL_GROUNDING_REJECTED;

            return NIYAH_OK;

        case NIYAH_PROPOSAL_GROUNDING_MATCH:
            break;

        default:
            return NIYAH_ERR_CORRUPT_DATA;
    }

    status = niyah_proposal_execute_authorized(
        policy,
        &out_result->proposal,
        &out_result->execution);

    if (status != NIYAH_OK) {
        return status;
    }

    if (out_result->execution.decision ==
        NIYAH_PROPOSAL_DECISION_DENY) {

        if (out_result->execution.executed != 0) {
            return NIYAH_ERR_CORRUPT_DATA;
        }

        out_result->state =
            NIYAH_GUARDED_PROPOSAL_POLICY_DENIED;

        return NIYAH_OK;
    }

    if (out_result->execution.decision ==
            NIYAH_PROPOSAL_DECISION_ALLOW &&
        out_result->execution.executed == 1) {

        out_result->state =
            NIYAH_GUARDED_PROPOSAL_EXECUTED;

        return NIYAH_OK;
    }

    return NIYAH_ERR_CORRUPT_DATA;
}
