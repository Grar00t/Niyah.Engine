#include "niyah/proposal_pipeline_format.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static const char *parse_status_name(
    NiyahStatus status)
{
    switch (status) {
        case NIYAH_ERR_INVALID_ARGUMENT:
            return "NIYAH_ERR_INVALID_ARGUMENT";

        case NIYAH_ERR_OVERFLOW:
            return "NIYAH_ERR_OVERFLOW";

        default:
            return NULL;
    }
}

static const char *grounding_name(
    NiyahProposalGroundingState state)
{
    switch (state) {
        case NIYAH_PROPOSAL_GROUNDING_NO_REFERENCE:
            return "NO_REFERENCE";

        case NIYAH_PROPOSAL_GROUNDING_KIND_MISMATCH:
            return "KIND_MISMATCH";

        case NIYAH_PROPOSAL_GROUNDING_VALUE_MISMATCH:
            return "VALUE_MISMATCH";

        case NIYAH_PROPOSAL_GROUNDING_MATCH:
            return "MATCH";

        default:
            return NULL;
    }
}

static const char *arithmetic_op_name(
    NiyahIrOp op)
{
    switch (op) {
        case NIYAH_IR_OP_ADD:
            return "ADD";

        case NIYAH_IR_OP_SUB:
            return "SUB";

        case NIYAH_IR_OP_MUL:
            return "MUL";

        default:
            return NULL;
    }
}

static int proposal_equal(
    const NiyahProposal *lhs,
    const NiyahProposal *rhs)
{
    if (lhs == NULL || rhs == NULL) {
        return 0;
    }

    if (lhs->kind != rhs->kind) {
        return 0;
    }

    switch (lhs->kind) {
        case NIYAH_PROPOSAL_ARITHMETIC:
            return
                lhs->arithmetic_ir.op ==
                    rhs->arithmetic_ir.op &&
                lhs->arithmetic_ir.lhs ==
                    rhs->arithmetic_ir.lhs &&
                lhs->arithmetic_ir.rhs ==
                    rhs->arithmetic_ir.rhs;

        case NIYAH_PROPOSAL_NETWORK:
            return
                lhs->network_ir.op ==
                    rhs->network_ir.op &&
                lhs->network_ir.address ==
                    rhs->network_ir.address &&
                lhs->network_ir.network ==
                    rhs->network_ir.network &&
                lhs->network_ir.prefix_length ==
                    rhs->network_ir.prefix_length;

        default:
            return 0;
    }
}

static uint32_t prefix_mask(
    uint8_t prefix_length)
{
    if (prefix_length == 0U) {
        return UINT32_C(0);
    }

    return
        UINT32_MAX <<
        (32U - prefix_length);
}

static int valid_network_ir(
    const NiyahNetworkIr *ir)
{
    uint32_t mask;

    if (ir == NULL ||
        ir->op !=
            NIYAH_NETWORK_IR_OP_IP_IN_CIDR ||
        ir->prefix_length > 32U) {
        return 0;
    }

    mask = prefix_mask(ir->prefix_length);

    return
        (ir->network & ~mask) == 0U;
}

static void ipv4_octets(
    uint32_t address,
    unsigned int *a,
    unsigned int *b,
    unsigned int *c,
    unsigned int *d)
{
    *a = (unsigned int)(
        (address >> 24U) & UINT32_C(0xff));

    *b = (unsigned int)(
        (address >> 16U) & UINT32_C(0xff));

    *c = (unsigned int)(
        (address >> 8U) & UINT32_C(0xff));

    *d = (unsigned int)(
        address & UINT32_C(0xff));
}

NiyahStatus niyah_guarded_proposal_format(
    const NiyahGuardedProposalResult *result,
    char *out_text,
    size_t out_capacity,
    size_t *out_length)
{
    char text[320];
    int needed;

    if (result == NULL ||
        out_length == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    if (out_text == NULL &&
        out_capacity != 0U) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    switch (result->state) {
        case NIYAH_GUARDED_PROPOSAL_PARSE_REJECTED:
        {
            const char *status_name =
                parse_status_name(
                    result->proposal_status);

            if (status_name == NULL) {
                return NIYAH_ERR_INVALID_ARGUMENT;
            }

            needed = snprintf(
                text,
                sizeof(text),
                "state=PARSE_REJECTED\n"
                "proposal_status=%s\n",
                status_name);

            break;
        }

        case NIYAH_GUARDED_PROPOSAL_GROUNDING_REJECTED:
        {
            const char *name;

            if (result->proposal_status !=
                NIYAH_OK) {
                return NIYAH_ERR_INVALID_ARGUMENT;
            }

            name = grounding_name(
                result->grounding.state);

            if (name == NULL ||
                result->grounding.state ==
                    NIYAH_PROPOSAL_GROUNDING_MATCH) {
                return NIYAH_ERR_INVALID_ARGUMENT;
            }

            needed = snprintf(
                text,
                sizeof(text),
                "state=GROUNDING_REJECTED\n"
                "grounding=%s\n",
                name);

            break;
        }

        case NIYAH_GUARDED_PROPOSAL_POLICY_DENIED:
            if (result->proposal_status !=
                    NIYAH_OK ||
                result->grounding.state !=
                    NIYAH_PROPOSAL_GROUNDING_MATCH ||
                result->execution.decision !=
                    NIYAH_PROPOSAL_DECISION_DENY ||
                result->execution.executed != 0) {
                return NIYAH_ERR_INVALID_ARGUMENT;
            }

            needed = snprintf(
                text,
                sizeof(text),
                "state=POLICY_DENIED\n");

            break;

        case NIYAH_GUARDED_PROPOSAL_EXECUTED:
            if (result->proposal_status !=
                    NIYAH_OK ||
                result->grounding.state !=
                    NIYAH_PROPOSAL_GROUNDING_MATCH ||
                result->execution.decision !=
                    NIYAH_PROPOSAL_DECISION_ALLOW ||
                result->execution.executed != 1 ||
                !proposal_equal(
                    &result->proposal,
                    &result->execution.proposal)) {
                return NIYAH_ERR_INVALID_ARGUMENT;
            }

            if (result->proposal.kind ==
                NIYAH_PROPOSAL_NETWORK) {
                unsigned int aa;
                unsigned int ab;
                unsigned int ac;
                unsigned int ad;
                unsigned int na;
                unsigned int nb;
                unsigned int nc;
                unsigned int nd;

                if (!valid_network_ir(
                        &result->proposal.network_ir) ||
                    (result->execution.network_match != 0 &&
                     result->execution.network_match != 1)) {
                    return NIYAH_ERR_INVALID_ARGUMENT;
                }

                ipv4_octets(
                    result->proposal.network_ir.address,
                    &aa, &ab, &ac, &ad);

                ipv4_octets(
                    result->proposal.network_ir.network,
                    &na, &nb, &nc, &nd);

                needed = snprintf(
                    text,
                    sizeof(text),
                    "state=EXECUTED\n"
                    "proposal_kind=NETWORK\n"
                    "address=%u.%u.%u.%u\n"
                    "network=%u.%u.%u.%u/%u\n"
                    "match=%s\n",
                    aa, ab, ac, ad,
                    na, nb, nc, nd,
                    (unsigned int)
                        result->proposal
                            .network_ir
                            .prefix_length,
                    result->execution.network_match ?
                        "true" :
                        "false");

                break;
            }

            if (result->proposal.kind ==
                NIYAH_PROPOSAL_ARITHMETIC) {
                const char *op =
                    arithmetic_op_name(
                        result->proposal
                            .arithmetic_ir
                            .op);

                if (op == NULL ||
                    result->proposal
                        .arithmetic_ir
                        .lhs < 0 ||
                    result->proposal
                        .arithmetic_ir
                        .rhs < 0) {
                    return NIYAH_ERR_INVALID_ARGUMENT;
                }

                needed = snprintf(
                    text,
                    sizeof(text),
                    "state=EXECUTED\n"
                    "proposal_kind=ARITHMETIC\n"
                    "op=%s\n"
                    "lhs=%" PRId64 "\n"
                    "rhs=%" PRId64 "\n"
                    "result=%" PRId64 "\n",
                    op,
                    result->proposal
                        .arithmetic_ir
                        .lhs,
                    result->proposal
                        .arithmetic_ir
                        .rhs,
                    result->execution
                        .arithmetic_result);

                break;
            }

            return NIYAH_ERR_INVALID_ARGUMENT;

        default:
            return NIYAH_ERR_INVALID_ARGUMENT;
    }

    if (needed < 0) {
        return NIYAH_ERR_IO;
    }

    if ((size_t)needed >= sizeof(text)) {
        return NIYAH_ERR_OVERFLOW;
    }

    *out_length = (size_t)needed;

    if (out_text == NULL &&
        out_capacity == 0U) {
        return NIYAH_OK;
    }

    if (out_capacity <= (size_t)needed) {
        return NIYAH_ERR_BUFFER_TOO_SMALL;
    }

    memcpy(
        out_text,
        text,
        (size_t)needed + 1U);

    return NIYAH_OK;
}
