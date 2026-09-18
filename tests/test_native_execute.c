#include "niyah/native_execute.h"

#define CHECK(x) \
    do { \
        if (!(x)) return __LINE__; \
    } while (0)

static int check_network(
    const char *text,
    int expected_match)
{
    NiyahNativeExecutionResult result;

    CHECK(
        niyah_native_execute_text(
            text,
            &result) ==
        NIYAH_OK);

    CHECK(
        result.route_kind ==
        NIYAH_ROUTE_NETWORK_IP_IN_CIDR);

    CHECK(
        result.network_match ==
        expected_match);

    return 0;
}

static int check_none(
    const char *text)
{
    NiyahNativeExecutionResult result;

    CHECK(
        niyah_native_execute_text(
            text,
            &result) ==
        NIYAH_OK);

    CHECK(
        result.route_kind ==
        NIYAH_ROUTE_NONE);

    return 0;
}

int main(void)
{
    /* English positive membership. */
    CHECK(
        check_network(
            "Is 192.168.1.42 inside 192.168.1.0/24?",
            1) == 0);

    /* English negative membership. */
    CHECK(
        check_network(
            "Is 192.168.2.42 inside 192.168.1.0/24?",
            0) == 0);

    /* Arabic positive membership. */
    CHECK(
        check_network(
            "هل 172.16.5.9 داخل الشبكة 172.16.0.0/16؟",
            1) == 0);

    /* Arabic negative membership. */
    CHECK(
        check_network(
            "هل 172.17.5.9 داخل الشبكة 172.16.0.0/16؟",
            0) == 0);

    /* /0 always contains valid IPv4 addresses. */
    CHECK(
        check_network(
            "Check whether 203.0.113.9 is in 0.0.0.0/0.",
            1) == 0);

    /* /32 exact match. */
    CHECK(
        check_network(
            "Check whether 10.0.0.1 is in 10.0.0.1/32.",
            1) == 0);

    CHECK(
        check_network(
            "Check whether 10.0.0.2 is in 10.0.0.1/32.",
            0) == 0);

    /*
     * Unsupported domains and non-membership text
     * are valid API calls with no native route.
     */
    CHECK(
        check_none(
            "What is 2+2?") == 0);

    CHECK(
        check_none(
            "Explain TCP congestion control.") == 0);

    CHECK(
        check_none(
            "Compare 192.168.1.42 with 192.168.1.0/24.") == 0);

    CHECK(
        check_none(
            "Ping 192.168.1.42 from 192.168.1.0/24.") == 0);

    /*
     * Invalid network geometry is not admitted as
     * an executable native route.
     */
    CHECK(
        check_none(
            "Is 192.168.1.42 inside 192.168.1.7/24?") == 0);

    /* API misuse is still a real error. */
    {
        NiyahNativeExecutionResult result;

        CHECK(
            niyah_native_execute_text(
                NULL,
                &result) ==
            NIYAH_ERR_INVALID_ARGUMENT);

        CHECK(
            niyah_native_execute_text(
                "hello",
                NULL) ==
            NIYAH_ERR_INVALID_ARGUMENT);
    }

    return 0;
}
