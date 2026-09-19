#ifndef NIYAH_PROPOSAL_PIPELINE_H
#define NIYAH_PROPOSAL_PIPELINE_H

#include "niyah/proposal.h"
#include "niyah/proposal_execute.h"
#include "niyah/proposal_grounding.h"
#include "niyah/proposal_policy.h"
#include "niyah/niyah.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum NiyahGuardedProposalState {
    NIYAH_GUARDED_PROPOSAL_INVALID = 0,

    /*
     * Proposal text failed strict typed parsing.
     */
    NIYAH_GUARDED_PROPOSAL_PARSE_REJECTED = 1,

    /*
     * Proposal parsed, but did not exactly match a trusted
     * deterministic reference derived from user text.
     */
    NIYAH_GUARDED_PROPOSAL_GROUNDING_REJECTED = 2,

    /*
     * Proposal was grounded, but policy denied execution.
     */
    NIYAH_GUARDED_PROPOSAL_POLICY_DENIED = 3,

    /*
     * Proposal was parsed, grounded, authorized and
     * successfully executed.
     */
    NIYAH_GUARDED_PROPOSAL_EXECUTED = 4
} NiyahGuardedProposalState;

typedef struct NiyahGuardedProposalResult {
    NiyahGuardedProposalState state;

    /*
     * Exact strict parser status.
     *
     * For PARSE_REJECTED this preserves whether rejection
     * was INVALID_ARGUMENT, OVERFLOW, etc.
     */
    NiyahStatus proposal_status;

    /*
     * Valid after strict parse succeeds.
     */
    NiyahProposal proposal;

    /*
     * Valid after strict parse succeeds and grounding runs.
     */
    NiyahProposalGroundingResult grounding;

    /*
     * Valid only after grounding MATCH allows the policy /
     * execution stage to run.
     */
    NiyahProposalExecutionResult execution;
} NiyahGuardedProposalResult;

/*
 * Guarded model-to-symbolic pipeline.
 *
 * Required order:
 *
 *   strict proposal parse
 *       ->
 *   deterministic grounding against original user text
 *       ->
 *   grounding must equal MATCH
 *       ->
 *   explicit proposal policy
 *       ->
 *   authorized deterministic typed execution
 *
 * Expected malformed model proposal text is fail-closed:
 *
 *   NIYAH_OK
 *   state == PARSE_REJECTED
 *   proposal_status contains parser rejection reason
 *
 * Grounding rejection and policy denial are also normal
 * fail-closed outcomes and return NIYAH_OK.
 *
 * Real API/config/execution failures remain NiyahStatus
 * errors.
 */
NiyahStatus niyah_guarded_proposal_run(
    const char *user_text,
    const char *proposal_text,
    const NiyahProposalPolicy *policy,
    NiyahGuardedProposalResult *out_result);

#ifdef __cplusplus
}
#endif

#endif
