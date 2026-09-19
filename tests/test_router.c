#include "niyah/router.h"

#include <stdint.h>

#define CHECK(x) \
    do { \
        if (!(x)) return __LINE__; \
    } while (0)

static int check_network(
    const char *text,
    uint32_t address,
    uint32_t network,
    uint8_t prefix)
{
    NiyahRoute route;

    CHECK(
        niyah_route_text(
            text,
            &route) ==
        NIYAH_OK);

    CHECK(
        route.kind ==
        NIYAH_ROUTE_NETWORK_IP_IN_CIDR);

    CHECK(
        route.network_ir.op ==
        NIYAH_NETWORK_IR_OP_IP_IN_CIDR);

    CHECK(route.network_ir.address == address);
    CHECK(route.network_ir.network == network);
    CHECK(route.network_ir.prefix_length == prefix);

    return 0;
}

static int check_none(
    const char *text)
{
    NiyahRoute route;

    CHECK(
        niyah_route_text(
            text,
            &route) ==
        NIYAH_OK);

    CHECK(
        route.kind ==
        NIYAH_ROUTE_NONE);

    return 0;
}

int main(void)
{
    /* English network membership. */
    CHECK(
        check_network(
            "Is 192.168.1.42 inside 192.168.1.0/24?",
            UINT32_C(0xC0A8012A),
            UINT32_C(0xC0A80100),
            24U) == 0);

    /* Arabic network membership. */
    CHECK(
        check_network(
            "هل 172.16.5.9 داخل الشبكة 172.16.0.0/16؟",
            UINT32_C(0xAC100509),
            UINT32_C(0xAC100000),
            16U) == 0);

    /*
     * Valid network-looking structure without membership intent
     * must not route.
     */
    CHECK(
        check_none(
            "Compare 192.168.1.42 with 192.168.1.0/24.") == 0);

    CHECK(
        check_none(
            "Ping 192.168.1.42 from 192.168.1.0/24.") == 0);

    /*
     * Unsupported domains are not API errors.
     * They simply have no deterministic route in V1.
     */
    CHECK(
        check_none(
            "What is 2+2?") == 0);

    CHECK(
        check_none(
            "Explain TCP congestion control.") == 0);

    CHECK(
        check_none(
            "مرحبا كيف حالك؟") == 0);

    /*
     * Membership language with invalid operands also does
     * not produce a native route.
     */
    CHECK(
        check_none(
            "Is 192.168.1.42 inside 192.168.1.7/24?") == 0);

    /*
     * P9T-O RED:
     * deterministic routing must not authorize a
     * negated membership request.
     */
    CHECK(
        check_none(
            "Do not test whether "
            "198.51.100.77 belongs to "
            "198.51.100.0/24; just print them.") == 0);

    /*
     * Meta-mention of a membership cue is not a
     * membership request.
     */
    CHECK(
        check_none(
            "Explain what belongs to means using "
            "198.51.100.77 and "
            "198.51.100.0/24.") == 0);

    CHECK(
        check_none(
            "لا تختبر ما إذا كان "
            "172.16.5.9 داخل الشبكة "
            "172.16.0.0/16؛ فقط اطبع القيم.") == 0);

    /*
     * Unrelated negation must not suppress a real
     * affirmative membership request.
     */
    CHECK(
        check_network(
            "Do not rewrite the values; "
            "does 198.51.100.77 belong to "
            "198.51.100.0/24?",
            UINT32_C(0xC633644D),
            UINT32_C(0xC6336400),
            24U) == 0);

    /* API misuse remains an actual error. */
    {
        NiyahRoute route;

        CHECK(
            niyah_route_text(
                NULL,
                &route) ==
            NIYAH_ERR_INVALID_ARGUMENT);

        CHECK(
            niyah_route_text(
                "hello",
                NULL) ==
            NIYAH_ERR_INVALID_ARGUMENT);
    }


    /*
     * P9T-A RED:
     * ambiguous membership text must not produce a route.
     */
    CHECK(
        check_none(
            "Is 192.168.1.42 inside "
            "192.168.1.0/24 or 10.0.0.0/8?") == 0);

    CHECK(
        check_none(
            "Is 192.168.1.42 or 192.168.1.43 "
            "inside 192.168.1.0/24?") == 0);

    CHECK(
        check_none(
            "Is 192.168.1.42 inside "
            "192.168.1.0/24 or "
            "10.0.0.1 inside 10.0.0.0/8?") == 0);

    CHECK(
        check_none(
            "Is 1.1.1.1 inside 0.0.0.0/0 "
            "and also consider 1.1.1.0/24?") == 0);

    CHECK(
        check_none(
            "Is 192.168.1.42 inside "
            "192.168.1.0/24? Reference 8.8.8.8.") == 0);

    return 0;
}
