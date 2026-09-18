#ifndef NIYAH_PROPOSAL_H
#define NIYAH_PROPOSAL_H

#include "niyah/ir.h"
#include "niyah/network_ir.h"
#include "niyah/niyah.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum NiyahProposalKind {
    NIYAH_PROPOSAL_INVALID = 0,
    NIYAH_PROPOSAL_ARITHMETIC = 1,
    NIYAH_PROPOSAL_NETWORK = 2
} NiyahProposalKind;

/*
 * Strict typed proposal produced at the model/symbolic boundary.
 *
 * This is NOT natural-language routing.
 * The proposal text must already conform to one supported
 * canonical IR grammar.
 */
typedef struct NiyahProposal {
    NiyahProposalKind kind;

    /*
     * Valid only when:
     * kind == NIYAH_PROPOSAL_ARITHMETIC
     */
    NiyahIr arithmetic_ir;

    /*
     * Valid only when:
     * kind == NIYAH_PROPOSAL_NETWORK
     */
    NiyahNetworkIr network_ir;
} NiyahProposal;

/*
 * Supported V1 proposal grammars:
 *
 *   ADD|2|7
 *   SUB|9|4
 *   MUL|4|5
 *
 *   IP_IN_CIDR|192.168.1.42|192.168.1.0|24
 *
 * Natural language, formatted execution output, unknown
 * operations, and malformed IR are rejected.
 */
NiyahStatus niyah_proposal_parse(
    const char *text,
    NiyahProposal *out_proposal);

#ifdef __cplusplus
}
#endif

#endif
