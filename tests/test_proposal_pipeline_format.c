#include "niyah/proposal_pipeline.h"
#include "niyah/proposal_pipeline_format.h"

#include <string.h>

#define CHECK(x) \
    do { \
        if (!(x)) return __LINE__; \
    } while (0)

static int format_equals(
    const NiyahGuardedProposalResult *result,
    const char *expected)
{
    char buffer[320];
    size_t required = 0U;
    size_t written = 0U;

    CHECK(
        niyah_guarded_proposal_format(
            result,
            NULL,
            0U,
            &required) ==
        NIYAH_OK);

    CHECK(required == strlen(expected));

    CHECK(
        niyah_guarded_proposal_format(
            result,
            buffer,
            sizeof(buffer),
            &written) ==
        NIYAH_OK);

    CHECK(written == required);
    CHECK(strcmp(buffer, expected) == 0);

    return 0;
}

int main(void)
{
    NiyahProposalPolicy policy;
    NiyahGuardedProposalResult result;

    /*
     * Strict parse rejection.
     */
    memset(&policy, 0, sizeof(policy));
    policy.allow_network = 1;

    CHECK(
        niyah_guarded_proposal_run(
            "hello",
            "not typed IR",
            &policy,
            &result) ==
        NIYAH_OK);

    CHECK(
        format_equals(
            &result,
            "state=PARSE_REJECTED\n"
            "proposal_status="
            "NIYAH_ERR_INVALID_ARGUMENT\n") ==
        0);

    /*
     * Parser overflow remains distinguishable.
     */
    CHECK(
        niyah_guarded_proposal_run(
            "hello",
            "ADD|9223372036854775808|1",
            &policy,
            &result) ==
        NIYAH_OK);

    CHECK(
        format_equals(
            &result,
            "state=PARSE_REJECTED\n"
            "proposal_status="
            "NIYAH_ERR_OVERFLOW\n") ==
        0);

    /*
     * No trusted grounding reference.
     */
    CHECK(
        niyah_guarded_proposal_run(
            "Compare 192.168.1.42 with "
            "192.168.1.0/24.",
            "IP_IN_CIDR|"
            "192.168.1.42|"
            "192.168.1.0|"
            "24",
            &policy,
            &result) ==
        NIYAH_OK);

    CHECK(
        format_equals(
            &result,
            "state=GROUNDING_REJECTED\n"
            "grounding=NO_REFERENCE\n") ==
        0);

    /*
     * Grounded domain but wrong values.
     */
    CHECK(
        niyah_guarded_proposal_run(
            "Is 192.168.1.42 inside "
            "192.168.1.0/24?",
            "IP_IN_CIDR|"
            "8.8.8.8|"
            "192.168.1.0|"
            "24",
            &policy,
            &result) ==
        NIYAH_OK);

    CHECK(
        format_equals(
            &result,
            "state=GROUNDING_REJECTED\n"
            "grounding=VALUE_MISMATCH\n") ==
        0);

    /*
     * Grounded but explicitly denied.
     */
    memset(&policy, 0, sizeof(policy));

    CHECK(
        niyah_guarded_proposal_run(
            "Is 192.168.1.42 inside "
            "192.168.1.0/24?",
            "IP_IN_CIDR|"
            "192.168.1.42|"
            "192.168.1.0|"
            "24",
            &policy,
            &result) ==
        NIYAH_OK);

    CHECK(
        format_equals(
            &result,
            "state=POLICY_DENIED\n") ==
        0);

    /*
     * Grounded + allowed + executed true.
     */
    memset(&policy, 0, sizeof(policy));
    policy.allow_network = 1;

    CHECK(
        niyah_guarded_proposal_run(
            "Is 192.168.1.42 inside "
            "192.168.1.0/24?",
            "IP_IN_CIDR|"
            "192.168.1.42|"
            "192.168.1.0|"
            "24",
            &policy,
            &result) ==
        NIYAH_OK);

    CHECK(
        format_equals(
            &result,
            "state=EXECUTED\n"
            "proposal_kind=NETWORK\n"
            "address=192.168.1.42\n"
            "network=192.168.1.0/24\n"
            "match=true\n") ==
        0);

    /*
     * Grounded + allowed + executed false membership.
     */
    CHECK(
        niyah_guarded_proposal_run(
            "Is 192.168.2.42 inside "
            "192.168.1.0/24?",
            "IP_IN_CIDR|"
            "192.168.2.42|"
            "192.168.1.0|"
            "24",
            &policy,
            &result) ==
        NIYAH_OK);

    CHECK(
        format_equals(
            &result,
            "state=EXECUTED\n"
            "proposal_kind=NETWORK\n"
            "address=192.168.2.42\n"
            "network=192.168.1.0/24\n"
            "match=false\n") ==
        0);

    /*
     * Exact-capacity buffer cannot hold terminating NUL.
     */
    {
        char buffer[320];
        size_t required = 0U;
        size_t written = 0U;

        CHECK(
            niyah_guarded_proposal_format(
                &result,
                NULL,
                0U,
                &required) ==
            NIYAH_OK);

        CHECK(required < sizeof(buffer));

        CHECK(
            niyah_guarded_proposal_format(
                &result,
                buffer,
                required,
                &written) ==
            NIYAH_ERR_BUFFER_TOO_SMALL);

        CHECK(written == required);
    }

    /*
     * API misuse.
     */
    {
        size_t length = 0U;
        char buffer[8];

        CHECK(
            niyah_guarded_proposal_format(
                NULL,
                buffer,
                sizeof(buffer),
                &length) ==
            NIYAH_ERR_INVALID_ARGUMENT);

        CHECK(
            niyah_guarded_proposal_format(
                &result,
                NULL,
                1U,
                &length) ==
            NIYAH_ERR_INVALID_ARGUMENT);

        CHECK(
            niyah_guarded_proposal_format(
                &result,
                buffer,
                sizeof(buffer),
                NULL) ==
            NIYAH_ERR_INVALID_ARGUMENT);
    }

    return 0;
}
