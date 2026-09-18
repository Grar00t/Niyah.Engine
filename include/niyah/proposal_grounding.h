#ifndef NIYAH_PROPOSAL_GROUNDING_H
#define NIYAH_PROPOSAL_GROUNDING_H

#include "niyah/proposal.h"
#include "niyah/router.h"
#include "niyah/niyah.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum NiyahProposalGroundingState {
    NIYAH_PROPOSAL_GROUNDING_INVALID = 0,

    /*
     * No deterministic trusted reference could be derived
     * from the user text.
     */
    NIYAH_PROPOSAL_GROUNDING_NO_REFERENCE = 1,

    /*
     * A trusted reference exists, but its domain does not
     * match the proposal domain.
     */
    NIYAH_PROPOSAL_GROUNDING_KIND_MISMATCH = 2,

    /*
     * Proposal and trusted reference share a domain but
     * their typed values differ.
     */
    NIYAH_PROPOSAL_GROUNDING_VALUE_MISMATCH = 3,

    /*
     * Proposal exactly matches the trusted typed reference.
     */
    NIYAH_PROPOSAL_GROUNDING_MATCH = 4
} NiyahProposalGroundingState;

typedef struct NiyahProposalGroundingResult {
    NiyahProposalGroundingState state;

    /*
     * Deterministic route derived directly from user text.
     *
     * kind == NIYAH_ROUTE_NONE when no reference exists.
     */
    NiyahRoute reference_route;
} NiyahProposalGroundingResult;

/*
 * Compare an already parsed model proposal against a trusted
 * deterministic interpretation of the original user text.
 *
 * V1 trusted grounding source:
 *   niyah_route_text()
 *
 * Therefore V1 can positively ground the currently supported
 * native network route.
 *
 * Arithmetic proposals currently receive NO_REFERENCE for
 * ordinary arithmetic user text because no deterministic
 * arithmetic natural-language router exists yet.
 *
 * This function:
 *   - does NOT execute the proposal
 *   - does NOT authorize the proposal
 *   - does NOT treat NO_REFERENCE or mismatch as API errors
 */
NiyahStatus niyah_proposal_ground_text(
    const char *user_text,
    const NiyahProposal *proposal,
    NiyahProposalGroundingResult *out_result);

#ifdef __cplusplus
}
#endif

#endif
