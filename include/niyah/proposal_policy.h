#ifndef NIYAH_PROPOSAL_POLICY_H
#define NIYAH_PROPOSAL_POLICY_H

#include "niyah/proposal.h"
#include "niyah/niyah.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum NiyahProposalDecision {
    NIYAH_PROPOSAL_DECISION_INVALID = 0,
    NIYAH_PROPOSAL_DECISION_DENY = 1,
    NIYAH_PROPOSAL_DECISION_ALLOW = 2
} NiyahProposalDecision;

/*
 * V1 policy is an explicit domain allowlist.
 *
 * Zero-initialized policy is fail-closed:
 *   arithmetic denied
 *   network denied
 *
 * Each field must be exactly 0 or 1.
 */
typedef struct NiyahProposalPolicy {
    int allow_arithmetic;
    int allow_network;
} NiyahProposalPolicy;

/*
 * Decide whether a previously parsed typed proposal is
 * authorized for a later execution stage.
 *
 * This function:
 *   - does NOT parse model text
 *   - does NOT execute IR
 *   - does NOT imply that authorization equals execution
 *
 * ALLOW means only that the proposal domain is permitted by
 * this policy.
 *
 * DENY is a normal policy decision and returns NIYAH_OK.
 */
NiyahStatus niyah_proposal_policy_decide(
    const NiyahProposalPolicy *policy,
    const NiyahProposal *proposal,
    NiyahProposalDecision *out_decision);

#ifdef __cplusplus
}
#endif

#endif
