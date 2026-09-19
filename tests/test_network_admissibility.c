#include "niyah/network_admissibility.h"

#include <stdint.h>

#define CHECK(x) \
    do { \
        if (!(x)) return __LINE__; \
    } while (0)

static int expect_accept(
    const char *text,
    uint32_t address,
    uint32_t network,
    uint8_t prefix)
{
    NiyahNetworkIr ir;

    CHECK(
        niyah_network_admit_ip_in_cidr(
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

static int expect_reject(
    const char *text)
{
    NiyahNetworkIr ir;

    CHECK(
        niyah_network_admit_ip_in_cidr(
            text,
            &ir) ==
        NIYAH_ERR_INVALID_ARGUMENT);

    return 0;
}

int main(void)
{
    /* English membership forms. */
    CHECK(
        expect_accept(
            "Is 192.168.1.42 inside 192.168.1.0/24?",
            UINT32_C(0xC0A8012A),
            UINT32_C(0xC0A80100),
            24U) == 0);

    CHECK(
        expect_accept(
            "Check whether 10.0.0.1 is in 10.0.0.1/32.",
            UINT32_C(0x0A000001),
            UINT32_C(0x0A000001),
            32U) == 0);

    CHECK(
        expect_accept(
            "Does 203.0.113.9 belong to 0.0.0.0/0?",
            UINT32_C(0xCB007109),
            UINT32_C(0),
            0U) == 0);

    /* Arabic membership forms. */
    CHECK(
        expect_accept(
            "هل 172.16.5.9 داخل الشبكة 172.16.0.0/16؟",
            UINT32_C(0xAC100509),
            UINT32_C(0xAC100000),
            16U) == 0);

    CHECK(
        expect_accept(
            "هل 203.0.113.14 ينتمي إلى 203.0.113.0/28؟",
            UINT32_C(0xCB00710E),
            UINT32_C(0xCB007100),
            28U) == 0);

    /*
     * Structurally valid IPv4 + CIDR pairs,
     * but not membership requests.
     */
    CHECK(
        expect_reject(
            "Compare 192.168.1.42 with 192.168.1.0/24.") == 0);

    CHECK(
        expect_reject(
            "Write 192.168.1.42 and 192.168.1.0/24 to a file.") == 0);

    CHECK(
        expect_reject(
            "Ping 192.168.1.42 from 192.168.1.0/24.") == 0);

    CHECK(
        expect_reject(
            "سجل 192.168.1.42 و 192.168.1.0/24 في الملف.") == 0);

    /*
     * Membership wording is not enough:
     * structural validation must still fail closed.
     */
    CHECK(
        expect_reject(
            "Is 192.168.1.42 inside 192.168.1.0?") == 0);

    CHECK(
        expect_reject(
            "Is 192.168.1.42 inside 192.168.1.7/24?") == 0);

    /*
     * P9T-O RED:
     * membership cues under explicit negation or
     * meta-linguistic mention must fail closed.
     */

    CHECK(
        expect_reject(
            "Do not test whether "
            "198.51.100.77 belongs to "
            "198.51.100.0/24; just print them.") == 0);

    CHECK(
        expect_reject(
            "Explain what belongs to means using "
            "198.51.100.77 and "
            "198.51.100.0/24.") == 0);

    CHECK(
        expect_reject(
            "لا تختبر ما إذا كان "
            "172.16.5.9 داخل الشبكة "
            "172.16.0.0/16؛ فقط اطبع القيم.") == 0);

    CHECK(
        expect_reject(
            "اشرح عبارة داخل الشبكة باستخدام "
            "172.16.5.9 و "
            "172.16.0.0/16.") == 0);

    /*
     * Negation elsewhere in the sentence must not
     * become a global veto when the membership
     * predicate itself is affirmative.
     */
    CHECK(
        expect_accept(
            "Do not rewrite the values; "
            "does 198.51.100.77 belong to "
            "198.51.100.0/24?",
            UINT32_C(0xC633644D),
            UINT32_C(0xC6336400),
            24U) == 0);

    CHECK(
        expect_accept(
            "لا تغيّر القيم؛ هل "
            "172.16.5.9 داخل الشبكة "
            "172.16.0.0/16؟",
            UINT32_C(0xAC100509),
            UINT32_C(0xAC100000),
            16U) == 0);

    /* API contract. */
    {
        NiyahNetworkIr ir;

        CHECK(
            niyah_network_admit_ip_in_cidr(
                NULL,
                &ir) ==
            NIYAH_ERR_INVALID_ARGUMENT);

        CHECK(
            niyah_network_admit_ip_in_cidr(
                "Is 1.1.1.1 inside 0.0.0.0/0?",
                NULL) ==
            NIYAH_ERR_INVALID_ARGUMENT);
    }

    return 0;
}
