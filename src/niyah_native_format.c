#include "niyah/native_format.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static unsigned ipv4_octet(
    uint32_t address,
    unsigned shift)
{
    return (unsigned)(
        (address >> shift) &
        UINT32_C(0xff));
}

NiyahStatus niyah_native_execution_format(
    const NiyahNativeExecutionResult *result,
    char *out_text,
    size_t out_capacity,
    size_t *out_length)
{
    char text[160];
    int written;

    if (result == NULL || out_length == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    if (result->route_kind == NIYAH_ROUTE_NONE) {
        written = snprintf(
            text,
            sizeof(text),
            "route=NONE\n");
    } else if (
        result->route_kind ==
        NIYAH_ROUTE_NETWORK_IP_IN_CIDR) {

        const NiyahNetworkIr *ir =
            &result->network_ir;

        if (ir->op !=
                NIYAH_NETWORK_IR_OP_IP_IN_CIDR ||
            ir->prefix_length > 32U ||
            (result->network_match != 0 &&
             result->network_match != 1)) {
            return NIYAH_ERR_INVALID_ARGUMENT;
        }

        written = snprintf(
            text,
            sizeof(text),
            "route=NETWORK_IP_IN_CIDR\n"
            "address=%u.%u.%u.%u\n"
            "network=%u.%u.%u.%u/%u\n"
            "match=%s\n",
            ipv4_octet(ir->address, 24U),
            ipv4_octet(ir->address, 16U),
            ipv4_octet(ir->address, 8U),
            ipv4_octet(ir->address, 0U),
            ipv4_octet(ir->network, 24U),
            ipv4_octet(ir->network, 16U),
            ipv4_octet(ir->network, 8U),
            ipv4_octet(ir->network, 0U),
            (unsigned)ir->prefix_length,
            result->network_match ?
                "true" :
                "false");
    } else {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    if (written < 0 ||
        (size_t)written >= sizeof(text)) {
        return NIYAH_ERR_OVERFLOW;
    }

    *out_length = (size_t)written;

    if (out_text == NULL) {
        if (out_capacity != 0U) {
            return NIYAH_ERR_INVALID_ARGUMENT;
        }

        return NIYAH_OK;
    }

    if (out_capacity <= (size_t)written) {
        return NIYAH_ERR_BUFFER_TOO_SMALL;
    }

    memcpy(
        out_text,
        text,
        (size_t)written + 1U);

    return NIYAH_OK;
}
