#ifndef NIYAH_PROPOSAL_EXECUTE_H
#define NIYAH_PROPOSAL_EXECUTE_H

#include "niyah/proposal.h"
#include "niyah/proposal_policy.h"
#include "niyah/niyah.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NiyahProposalExecutionResult {
    /*
     * Exact typed proposal evaluated by policy.
     */
    NiyahProposal proposal;

    /*
     * Policy decision that preceded execution.
     */
    NiyahProposalDecision decision;

    /*
     * 0 => no executor completed successfully
     * 1 => authorized typed executor completed successfully
     */
    int executed;

    /*
     * Valid only when:
     * executed == 1 &&
     * proposal.kind == NIYAH_PROPOSAL_ARITHMETIC
     */
    int64_t arithmetic_result;

    /*
     * Valid only when:
     * executed == 1 &&
     * proposal.kind == NIYAH_PROPOSAL_NETWORK
     *
     * 0 => address outside network
     * 1 => address inside network
     */
    int network_match;
} NiyahProposalExecutionResult;

/*
 * Apply policy to an already parsed typed proposal and,
 * only on an explicit ALLOW decision, dispatch to the
 * corresponding deterministic typed executor.
 *
 * DENY:
 *   returns NIYAH_OK
 *   decision == DENY
 *   executed == 0
 *
 * ALLOW + execution success:
 *   returns NIYAH_OK
 *   decision == ALLOW
 *   executed == 1
 *
 * ALLOW + execution failure:
 *   propagates executor error
 *   decision == ALLOW
 *   executed == 0
 *
 * This function does NOT parse model text.
 */
NiyahStatus niyah_proposal_execute_authorized(
    const NiyahProposalPolicy *policy,
    const NiyahProposal *proposal,
    NiyahProposalExecutionResult *out_result);

#ifdef __cplusplus
}
#endif

#endif
