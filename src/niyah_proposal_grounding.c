#include "niyah/proposal_grounding.h"

#include <string.h>

static int network_ir_equal(
    const NiyahNetworkIr *lhs,
    const NiyahNetworkIr *rhs)
{
    if (lhs == NULL || rhs == NULL) {
        return 0;
    }

    return
        lhs->op == rhs->op &&
        lhs->address == rhs->address &&
        lhs->network == rhs->network &&
        lhs->prefix_length == rhs->prefix_length;
}

NiyahStatus niyah_proposal_ground_text(
    const char *user_text,
    const NiyahProposal *proposal,
    NiyahProposalGroundingResult *out_result)
{
    NiyahRoute reference;
    NiyahStatus status;

    if (user_text == NULL ||
        proposal == NULL ||
        out_result == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    memset(
        out_result,
        0,
        sizeof(*out_result));

    if (proposal->kind !=
            NIYAH_PROPOSAL_ARITHMETIC &&
        proposal->kind !=
            NIYAH_PROPOSAL_NETWORK) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    status = niyah_route_text(
        user_text,
        &reference);

    if (status != NIYAH_OK) {
        return status;
    }

    out_result->reference_route =
        reference;

    if (reference.kind ==
        NIYAH_ROUTE_NONE) {
        out_result->state =
            NIYAH_PROPOSAL_GROUNDING_NO_REFERENCE;

        return NIYAH_OK;
    }

    switch (proposal->kind) {
        case NIYAH_PROPOSAL_ARITHMETIC:
            /*
             * Current trusted router has no arithmetic
             * route kind. A concrete non-NONE reference is
             * therefore a different domain.
             */
            out_result->state =
                NIYAH_PROPOSAL_GROUNDING_KIND_MISMATCH;

            return NIYAH_OK;

        case NIYAH_PROPOSAL_NETWORK:
            if (reference.kind !=
                NIYAH_ROUTE_NETWORK_IP_IN_CIDR) {
                out_result->state =
                    NIYAH_PROPOSAL_GROUNDING_KIND_MISMATCH;

                return NIYAH_OK;
            }

            if (network_ir_equal(
                    &proposal->network_ir,
                    &reference.network_ir)) {
                out_result->state =
                    NIYAH_PROPOSAL_GROUNDING_MATCH;
            } else {
                out_result->state =
                    NIYAH_PROPOSAL_GROUNDING_VALUE_MISMATCH;
            }

            return NIYAH_OK;

        default:
            return NIYAH_ERR_INVALID_ARGUMENT;
    }
}
