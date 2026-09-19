#include "niyah/network_ir.h"

#include <stdint.h>

#define CHECK(x) \
    do { \
        if (!(x)) return __LINE__; \
    } while (0)

static int test_parse_and_execute(
    const char *text,
    int expected_match)
{
    NiyahNetworkIr ir;
    int match = -1;

    CHECK(
        niyah_network_ir_parse(
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
    int match;

    /* Positive membership. */
    CHECK(
        test_parse_and_execute(
            "IP_IN_CIDR|"
            "192.168.1.42|"
            "192.168.1.0|24",
            1) == 0);

    /* Negative membership. */
    CHECK(
        test_parse_and_execute(
            "IP_IN_CIDR|"
            "192.168.2.42|"
            "192.168.1.0|24",
            0) == 0);

    /* Boundary /32. */
    CHECK(
        test_parse_and_execute(
            "IP_IN_CIDR|"
            "10.0.0.1|"
            "10.0.0.1|32",
            1) == 0);

    CHECK(
        test_parse_and_execute(
            "IP_IN_CIDR|"
            "10.0.0.2|"
            "10.0.0.1|32",
            0) == 0);

    /* Boundary /0. */
    CHECK(
        test_parse_and_execute(
            "IP_IN_CIDR|"
            "203.0.113.9|"
            "0.0.0.0|0",
            1) == 0);

    /* Network and broadcast are still members. */
    CHECK(
        test_parse_and_execute(
            "IP_IN_CIDR|"
            "192.168.1.0|"
            "192.168.1.0|24",
            1) == 0);

    CHECK(
        test_parse_and_execute(
            "IP_IN_CIDR|"
            "192.168.1.255|"
            "192.168.1.0|24",
            1) == 0);

    /* Strict malformed inputs. */
    CHECK(
        niyah_network_ir_parse(
            "",
            &ir) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    CHECK(
        niyah_network_ir_parse(
            "IP_IN_CIDR",
            &ir) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    CHECK(
        niyah_network_ir_parse(
            "IP_IN_CIDR|"
            "192.168.1.42|"
            "192.168.1.0",
            &ir) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    CHECK(
        niyah_network_ir_parse(
            "IP_IN_CIDR|"
            "256.1.1.1|"
            "192.168.1.0|24",
            &ir) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    CHECK(
        niyah_network_ir_parse(
            "IP_IN_CIDR|"
            "192.168.1.42|"
            "192.168.1.0|33",
            &ir) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    /* Non-canonical octet. */
    CHECK(
        niyah_network_ir_parse(
            "IP_IN_CIDR|"
            "192.168.001.42|"
            "192.168.1.0|24",
            &ir) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    /* Non-canonical prefix. */
    CHECK(
        niyah_network_ir_parse(
            "IP_IN_CIDR|"
            "192.168.1.42|"
            "192.168.1.0|024",
            &ir) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    /* Host bits set in network operand. */
    CHECK(
        niyah_network_ir_parse(
            "IP_IN_CIDR|"
            "192.168.1.42|"
            "192.168.1.7|24",
            &ir) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    /* Wrong opcode. */
    CHECK(
        niyah_network_ir_parse(
            "IP_OUTSIDE_CIDR|"
            "192.168.1.42|"
            "192.168.1.0|24",
            &ir) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    /* Whitespace forbidden. */
    CHECK(
        niyah_network_ir_parse(
            "IP_IN_CIDR|"
            "192.168.1.42 |"
            "192.168.1.0|24",
            &ir) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    /* Extra field forbidden. */
    CHECK(
        niyah_network_ir_parse(
            "IP_IN_CIDR|"
            "192.168.1.42|"
            "192.168.1.0|24|X",
            &ir) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    /* API argument contract. */
    CHECK(
        niyah_network_ir_parse(
            NULL,
            &ir) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    CHECK(
        niyah_network_ir_parse(
            "IP_IN_CIDR|"
            "1.1.1.1|"
            "0.0.0.0|0",
            NULL) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    ir.op =
        NIYAH_NETWORK_IR_OP_INVALID;

    CHECK(
        niyah_network_ir_execute(
            &ir,
            &match) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    CHECK(
        niyah_network_ir_execute(
            NULL,
            &match) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    CHECK(
        niyah_network_ir_execute(
            &ir,
            NULL) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    return 0;
}
