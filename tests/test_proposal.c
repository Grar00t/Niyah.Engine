#include "niyah/proposal.h"

#include <stdint.h>

#define CHECK(x) \
    do { \
        if (!(x)) return __LINE__; \
    } while (0)

int main(void)
{
    NiyahProposal proposal;

    /* Arithmetic proposal. */
    CHECK(
        niyah_proposal_parse(
            "ADD|2|7",
            &proposal) ==
        NIYAH_OK);

    CHECK(
        proposal.kind ==
        NIYAH_PROPOSAL_ARITHMETIC);

    CHECK(
        proposal.arithmetic_ir.op ==
        NIYAH_IR_OP_ADD);

    CHECK(
        proposal.arithmetic_ir.lhs == 2);

    CHECK(
        proposal.arithmetic_ir.rhs == 7);

    /* Another arithmetic operation. */
    CHECK(
        niyah_proposal_parse(
            "MUL|4|5",
            &proposal) ==
        NIYAH_OK);

    CHECK(
        proposal.kind ==
        NIYAH_PROPOSAL_ARITHMETIC);

    CHECK(
        proposal.arithmetic_ir.op ==
        NIYAH_IR_OP_MUL);

    /* Network typed proposal. */
    CHECK(
        niyah_proposal_parse(
            "IP_IN_CIDR|"
            "192.168.1.42|"
            "192.168.1.0|"
            "24",
            &proposal) ==
        NIYAH_OK);

    CHECK(
        proposal.kind ==
        NIYAH_PROPOSAL_NETWORK);

    CHECK(
        proposal.network_ir.op ==
        NIYAH_NETWORK_IR_OP_IP_IN_CIDR);

    CHECK(
        proposal.network_ir.address ==
        UINT32_C(0xC0A8012A));

    CHECK(
        proposal.network_ir.network ==
        UINT32_C(0xC0A80100));

    CHECK(
        proposal.network_ir.prefix_length ==
        24U);

    /* /0 and /32 remain valid strict network proposals. */
    CHECK(
        niyah_proposal_parse(
            "IP_IN_CIDR|"
            "203.0.113.9|"
            "0.0.0.0|"
            "0",
            &proposal) ==
        NIYAH_OK);

    CHECK(
        proposal.kind ==
        NIYAH_PROPOSAL_NETWORK);

    CHECK(
        proposal.network_ir.prefix_length ==
        0U);

    CHECK(
        niyah_proposal_parse(
            "IP_IN_CIDR|"
            "10.0.0.1|"
            "10.0.0.1|"
            "32",
            &proposal) ==
        NIYAH_OK);

    CHECK(
        proposal.network_ir.prefix_length ==
        32U);

    /*
     * Malformed arithmetic stays malformed.
     */
    CHECK(
        niyah_proposal_parse(
            "ADD|2",
            &proposal) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    CHECK(
        niyah_proposal_parse(
            "ADD|2|7|9",
            &proposal) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    /*
     * Preserve arithmetic parser overflow semantics.
     */
    CHECK(
        niyah_proposal_parse(
            "ADD|9223372036854775808|1",
            &proposal) ==
        NIYAH_ERR_OVERFLOW);

    /*
     * Network grammar remains strict.
     */
    CHECK(
        niyah_proposal_parse(
            "IP_IN_CIDR|"
            "192.168.1.42|"
            "192.168.1.7|"
            "24",
            &proposal) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    CHECK(
        niyah_proposal_parse(
            "IP_IN_CIDR|"
            "192.168.1.42|"
            "192.168.1.0|"
            "33",
            &proposal) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    /*
     * Model prose is not executable symbolic syntax.
     */
    CHECK(
        niyah_proposal_parse(
            "The answer is ADD|2|7",
            &proposal) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    CHECK(
        niyah_proposal_parse(
            "Is 192.168.1.42 inside "
            "192.168.1.0/24?",
            &proposal) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    /*
     * Canonical execution output is also not a proposal.
     */
    CHECK(
        niyah_proposal_parse(
            "route=NETWORK_IP_IN_CIDR",
            &proposal) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    /*
     * Unknown proposal domains fail closed.
     */
    CHECK(
        niyah_proposal_parse(
            "PING|192.168.1.1",
            &proposal) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    /* API misuse. */
    CHECK(
        niyah_proposal_parse(
            NULL,
            &proposal) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    CHECK(
        niyah_proposal_parse(
            "ADD|2|7",
            NULL) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    return 0;
}
