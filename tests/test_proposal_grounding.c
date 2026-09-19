#include "niyah/proposal.h"
#include "niyah/proposal_grounding.h"

#include <stdint.h>
#include <string.h>

#define CHECK(x) \
    do { \
        if (!(x)) return __LINE__; \
    } while (0)

static int parse(
    const char *text,
    NiyahProposal *proposal)
{
    return niyah_proposal_parse(
        text,
        proposal) == NIYAH_OK;
}

static int ground(
    const char *user_text,
    const NiyahProposal *proposal,
    NiyahProposalGroundingState expected,
    NiyahProposalGroundingResult *out_result)
{
    NiyahProposalGroundingResult local;
    NiyahProposalGroundingResult *result =
        out_result != NULL ?
            out_result :
            &local;

    CHECK(
        niyah_proposal_ground_text(
            user_text,
            proposal,
            result) ==
        NIYAH_OK);

    CHECK(result->state == expected);

    return 0;
}

int main(void)
{
    NiyahProposal proposal;
    NiyahProposalGroundingResult result;

    /*
     * Exact English network grounding.
     */
    CHECK(
        parse(
            "IP_IN_CIDR|"
            "192.168.1.42|"
            "192.168.1.0|"
            "24",
            &proposal));

    CHECK(
        ground(
            "Is 192.168.1.42 inside "
            "192.168.1.0/24?",
            &proposal,
            NIYAH_PROPOSAL_GROUNDING_MATCH,
            &result) == 0);

    CHECK(
        result.reference_route.kind ==
        NIYAH_ROUTE_NETWORK_IP_IN_CIDR);

    CHECK(
        result.reference_route.network_ir.address ==
        UINT32_C(0xC0A8012A));

    CHECK(
        result.reference_route.network_ir.network ==
        UINT32_C(0xC0A80100));

    CHECK(
        result.reference_route.network_ir.prefix_length ==
        24U);

    /*
     * Exact Arabic network grounding.
     */
    CHECK(
        parse(
            "IP_IN_CIDR|"
            "172.16.5.9|"
            "172.16.0.0|"
            "16",
            &proposal));

    CHECK(
        ground(
            "هل 172.16.5.9 داخل الشبكة "
            "172.16.0.0/16؟",
            &proposal,
            NIYAH_PROPOSAL_GROUNDING_MATCH,
            NULL) == 0);

    /*
     * Same domain, wrong address.
     */
    CHECK(
        parse(
            "IP_IN_CIDR|"
            "8.8.8.8|"
            "192.168.1.0|"
            "24",
            &proposal));

    CHECK(
        ground(
            "Is 192.168.1.42 inside "
            "192.168.1.0/24?",
            &proposal,
            NIYAH_PROPOSAL_GROUNDING_VALUE_MISMATCH,
            &result) == 0);

    CHECK(
        result.reference_route.network_ir.address ==
        UINT32_C(0xC0A8012A));

    /*
     * Same domain, wrong network.
     */
    CHECK(
        parse(
            "IP_IN_CIDR|"
            "192.168.1.42|"
            "10.0.0.0|"
            "8",
            &proposal));

    CHECK(
        ground(
            "Is 192.168.1.42 inside "
            "192.168.1.0/24?",
            &proposal,
            NIYAH_PROPOSAL_GROUNDING_VALUE_MISMATCH,
            NULL) == 0);

    /*
     * Unsupported/non-membership text provides no trusted
     * network reference.
     */
    CHECK(
        parse(
            "IP_IN_CIDR|"
            "192.168.1.42|"
            "192.168.1.0|"
            "24",
            &proposal));

    CHECK(
        ground(
            "What is 2+2?",
            &proposal,
            NIYAH_PROPOSAL_GROUNDING_NO_REFERENCE,
            NULL) == 0);

    CHECK(
        ground(
            "Compare 192.168.1.42 with "
            "192.168.1.0/24.",
            &proposal,
            NIYAH_PROPOSAL_GROUNDING_NO_REFERENCE,
            NULL) == 0);

    CHECK(
        ground(
            "Ping 192.168.1.42 from "
            "192.168.1.0/24.",
            &proposal,
            NIYAH_PROPOSAL_GROUNDING_NO_REFERENCE,
            NULL) == 0);

    /*
     * Arithmetic V1 has no deterministic natural-language
     * grounding reference yet.
     */
    CHECK(
        parse(
            "ADD|2|7",
            &proposal));

    CHECK(
        ground(
            "What is 2+7?",
            &proposal,
            NIYAH_PROPOSAL_GROUNDING_NO_REFERENCE,
            &result) == 0);

    CHECK(
        result.reference_route.kind ==
        NIYAH_ROUTE_NONE);

    /*
     * Arithmetic proposal against a concrete network request
     * is a domain mismatch, not a grounded proposal.
     */
    CHECK(
        ground(
            "Is 192.168.1.42 inside "
            "192.168.1.0/24?",
            &proposal,
            NIYAH_PROPOSAL_GROUNDING_KIND_MISMATCH,
            &result) == 0);

    CHECK(
        result.reference_route.kind ==
        NIYAH_ROUTE_NETWORK_IP_IN_CIDR);

    /*
     * Noncanonical user network text cannot manufacture a
     * grounding reference.
     */
    CHECK(
        parse(
            "IP_IN_CIDR|"
            "192.168.1.42|"
            "192.168.1.0|"
            "24",
            &proposal));

    CHECK(
        ground(
            "Is 192.168.1.42 inside "
            "192.168.1.7/24?",
            &proposal,
            NIYAH_PROPOSAL_GROUNDING_NO_REFERENCE,
            NULL) == 0);

    /*
     * Unknown proposal kind is corrupt typed input.
     */
    {
        NiyahProposal invalid;

        memset(&invalid, 0, sizeof(invalid));

        CHECK(
            niyah_proposal_ground_text(
                "hello",
                &invalid,
                &result) ==
            NIYAH_ERR_INVALID_ARGUMENT);

        CHECK(
            result.state ==
            NIYAH_PROPOSAL_GROUNDING_INVALID);
    }

    /*
     * API misuse.
     */
    CHECK(
        parse(
            "ADD|2|7",
            &proposal));

    CHECK(
        niyah_proposal_ground_text(
            NULL,
            &proposal,
            &result) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    CHECK(
        niyah_proposal_ground_text(
            "What is 2+7?",
            NULL,
            &result) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    CHECK(
        niyah_proposal_ground_text(
            "What is 2+7?",
            &proposal,
            NULL) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    return 0;
}
