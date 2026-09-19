#ifndef NIYAH_PROPOSAL_PIPELINE_FORMAT_H
#define NIYAH_PROPOSAL_PIPELINE_FORMAT_H

#include "niyah/proposal_pipeline.h"
#include "niyah/niyah.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Canonically format a completed guarded-proposal result.
 *
 * Query mode:
 *   out_text == NULL
 *   out_capacity == 0
 *
 * out_length excludes the terminating NUL.
 *
 * This function:
 *   - does NOT parse
 *   - does NOT ground
 *   - does NOT authorize
 *   - does NOT execute
 */
NiyahStatus niyah_guarded_proposal_format(
    const NiyahGuardedProposalResult *result,
    char *out_text,
    size_t out_capacity,
    size_t *out_length);

#ifdef __cplusplus
}
#endif

#endif
