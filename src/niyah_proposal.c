#include "niyah/proposal.h"

#include <string.h>

static int starts_with(
    const char *text,
    const char *prefix)
{
    size_t prefix_length;

    if (text == NULL || prefix == NULL) {
        return 0;
    }

    prefix_length = strlen(prefix);

    return strncmp(
        text,
        prefix,
        prefix_length) == 0;
}

NiyahStatus niyah_proposal_parse(
    const char *text,
    NiyahProposal *out_proposal)
{
    NiyahStatus status;

    if (text == NULL ||
        out_proposal == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    memset(
        out_proposal,
        0,
        sizeof(*out_proposal));

    if (starts_with(text, "ADD|") ||
        starts_with(text, "SUB|") ||
        starts_with(text, "MUL|")) {

        status = niyah_ir_parse(
            text,
            &out_proposal->arithmetic_ir);

        if (status != NIYAH_OK) {
            return status;
        }

        out_proposal->kind =
            NIYAH_PROPOSAL_ARITHMETIC;

        return NIYAH_OK;
    }

    if (starts_with(
            text,
            "IP_IN_CIDR|")) {

        status = niyah_network_ir_parse(
            text,
            &out_proposal->network_ir);

        if (status != NIYAH_OK) {
            return status;
        }

        out_proposal->kind =
            NIYAH_PROPOSAL_NETWORK;

        return NIYAH_OK;
    }

    return NIYAH_ERR_INVALID_ARGUMENT;
}
