#include "niyah/network_slots.h"

#define CHECK(x) \
    do { \
        if (!(x)) return __LINE__; \
    } while (0)

static int check_case(
    const char *text,
    uint32_t address,
    uint32_t network,
    uint8_t prefix)
{
    NiyahNetworkIr ir;

    CHECK(
        niyah_network_slots_extract_ip_in_cidr(
            text,
            &ir) ==
        NIYAH_OK);

    CHECK(
        ir.op ==
        NIYAH_NETWORK_IR_OP_IP_IN_CIDR);

    CHECK(ir.address == address);
    CHECK(ir.network == network);
    CHECK(ir.prefix_length == prefix);

    return 0;
}

int main(void)
{
    CHECK(
        check_case(
            "Is 192.168.1.42 inside 192.168.1.0/24?",
            UINT32_C(0xC0A8012A),
            UINT32_C(0xC0A80100),
            24U) == 0);

    CHECK(
        check_case(
            "Determine whether 10.20.30.40 "
            "is inside network 10.20.0.0/16.",
            UINT32_C(0x0A141E28),
            UINT32_C(0x0A140000),
            16U) == 0);

    CHECK(
        check_case(
            "هل 172.16.5.9 داخل الشبكة "
            "172.16.0.0/16؟",
            UINT32_C(0xAC100509),
            UINT32_C(0xAC100000),
            16U) == 0);

    CHECK(
        check_case(
            "تحقق هل العنوان 203.0.113.14 "
            "ينتمي إلى الشبكة 203.0.113.0/28.",
            UINT32_C(0xCB00710E),
            UINT32_C(0xCB007100),
            28U) == 0);

    CHECK(
        check_case(
            "1.1.1.1 in 0.0.0.0/0",
            UINT32_C(0x01010101),
            UINT32_C(0),
            0U) == 0);

    {
        NiyahNetworkIr ir;

        CHECK(
            niyah_network_slots_extract_ip_in_cidr(
                "no address here",
                &ir) ==
            NIYAH_ERR_INVALID_ARGUMENT);

        CHECK(
            niyah_network_slots_extract_ip_in_cidr(
                "192.168.1.1 and 192.168.1.0",
                &ir) ==
            NIYAH_ERR_INVALID_ARGUMENT);

        CHECK(
            niyah_network_slots_extract_ip_in_cidr(
                "192.168.1.1 in "
                "192.168.1.7/24",
                &ir) ==
            NIYAH_ERR_INVALID_ARGUMENT);

        CHECK(
            niyah_network_slots_extract_ip_in_cidr(
                NULL,
                &ir) ==
            NIYAH_ERR_INVALID_ARGUMENT);

        CHECK(
            niyah_network_slots_extract_ip_in_cidr(
                "1.1.1.1 in 0.0.0.0/0",
                NULL) ==
            NIYAH_ERR_INVALID_ARGUMENT);
    }


    /*
     * P9T-A RED:
     * ambiguous operand sets must fail closed.
     *
     * Current first-match extraction is expected to make
     * several of these assertions FAIL before the fix.
     */
    {
        NiyahNetworkIr ir;

        CHECK(
            niyah_network_slots_extract_ip_in_cidr(
                "Is 192.168.1.42 inside "
                "192.168.1.0/24 or 10.0.0.0/8?",
                &ir) ==
            NIYAH_ERR_INVALID_ARGUMENT);

        CHECK(
            niyah_network_slots_extract_ip_in_cidr(
                "Is 192.168.1.42 or 192.168.1.43 "
                "inside 192.168.1.0/24?",
                &ir) ==
            NIYAH_ERR_INVALID_ARGUMENT);

        CHECK(
            niyah_network_slots_extract_ip_in_cidr(
                "Is 192.168.1.42 inside "
                "192.168.1.0/24 or "
                "10.0.0.1 inside 10.0.0.0/8?",
                &ir) ==
            NIYAH_ERR_INVALID_ARGUMENT);

        CHECK(
            niyah_network_slots_extract_ip_in_cidr(
                "Is 1.1.1.1 inside 0.0.0.0/0 "
                "and also consider 1.1.1.0/24?",
                &ir) ==
            NIYAH_ERR_INVALID_ARGUMENT);

        CHECK(
            niyah_network_slots_extract_ip_in_cidr(
                "Is 192.168.1.42 inside "
                "192.168.1.0/24? Reference 8.8.8.8.",
                &ir) ==
            NIYAH_ERR_INVALID_ARGUMENT);
    }

    return 0;
}
