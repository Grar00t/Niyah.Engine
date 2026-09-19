#include "niyah/network_slots.h"

#define CHECK(x) \
    do { \
        if (!(x)) return __LINE__; \
    } while (0)

static int run_case(
    const char *text,
    int expected_match)
{
    NiyahNetworkIr ir;
    int match = -1;

    CHECK(
        niyah_network_slots_extract_ip_in_cidr(
            text,
            &ir) ==
        NIYAH_OK);

    CHECK(
        ir.op ==
        NIYAH_NETWORK_IR_OP_IP_IN_CIDR);

    CHECK(
        niyah_network_ir_execute(
            &ir,
            &match) ==
        NIYAH_OK);

    CHECK(match == expected_match);

    return 0;
}

int main(void)
{
    NiyahNetworkIr ir;

    /* English: inside. */
    CHECK(
        run_case(
            "Is 192.168.1.42 inside 192.168.1.0/24?",
            1) == 0);

    /* English: outside. */
    CHECK(
        run_case(
            "Is 192.168.2.42 inside 192.168.1.0/24?",
            0) == 0);

    /* Arabic: inside. */
    CHECK(
        run_case(
            "هل 172.16.5.9 داخل الشبكة 172.16.0.0/16؟",
            1) == 0);

    /* Arabic: outside. */
    CHECK(
        run_case(
            "هل 172.17.5.9 داخل الشبكة 172.16.0.0/16؟",
            0) == 0);

    /* /0 boundary. */
    CHECK(
        run_case(
            "Check whether 203.0.113.9 is in 0.0.0.0/0.",
            1) == 0);

    /* /32 positive boundary. */
    CHECK(
        run_case(
            "Check whether 10.0.0.1 is in 10.0.0.1/32.",
            1) == 0);

    /* /32 negative boundary. */
    CHECK(
        run_case(
            "Check whether 10.0.0.2 is in 10.0.0.1/32.",
            0) == 0);

    /*
     * Natural-language pipeline must refuse an incomplete
     * network operand instead of inventing a prefix.
     */
    CHECK(
        niyah_network_slots_extract_ip_in_cidr(
            "Is 192.168.1.42 inside 192.168.1.0?",
            &ir) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    /*
     * Strict IR contract still applies through the
     * natural-language extraction path.
     */
    CHECK(
        niyah_network_slots_extract_ip_in_cidr(
            "Is 192.168.1.42 inside 192.168.1.7/24?",
            &ir) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    return 0;
}
