#include "niyah/network_ir.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

static int parse_decimal_component(
    const char *text,
    size_t size,
    uint32_t maximum,
    uint32_t *out_value)
{
    uint32_t value = 0U;
    size_t i;

    if (text == NULL ||
        out_value == NULL ||
        size == 0U) {
        return 0;
    }

    /*
     * Canonical decimal representation:
     * 0 is valid, 00 / 01 / 001 are not.
     */
    if (size > 1U && text[0] == '0') {
        return 0;
    }

    for (i = 0U; i < size; ++i) {
        uint32_t digit;

        if (text[i] < '0' || text[i] > '9') {
            return 0;
        }

        digit = (uint32_t)(text[i] - '0');

        if (value > (maximum - digit) / 10U) {
            return 0;
        }

        value = value * 10U + digit;
    }

    *out_value = value;
    return 1;
}

static int parse_ipv4(
    const char *text,
    size_t size,
    uint32_t *out_address)
{
    uint32_t octets[4];
    size_t component_start = 0U;
    size_t component_index = 0U;
    size_t i;

    if (text == NULL ||
        out_address == NULL ||
        size == 0U) {
        return 0;
    }

    for (i = 0U; i <= size; ++i) {
        if (i == size || text[i] == '.') {
            size_t component_size;

            if (component_index >= 4U) {
                return 0;
            }

            component_size =
                i - component_start;

            if (!parse_decimal_component(
                    text + component_start,
                    component_size,
                    255U,
                    &octets[component_index])) {
                return 0;
            }

            ++component_index;
            component_start = i + 1U;
        }
    }

    if (component_index != 4U) {
        return 0;
    }

    *out_address =
        (octets[0] << 24U) |
        (octets[1] << 16U) |
        (octets[2] << 8U) |
        octets[3];

    return 1;
}

static uint32_t cidr_mask(uint8_t prefix_length)
{
    if (prefix_length == 0U) {
        return UINT32_C(0);
    }

    return UINT32_MAX <<
           (32U - (uint32_t)prefix_length);
}

NiyahStatus niyah_network_ir_parse(
    const char *text,
    NiyahNetworkIr *out_ir)
{
    static const char prefix[] =
        "IP_IN_CIDR|";

    const char *address_start;
    const char *network_start;
    const char *prefix_start;

    const char *separator;
    size_t address_size;
    size_t network_size;
    size_t prefix_size;

    uint32_t address;
    uint32_t network;
    uint32_t prefix_value;
    uint32_t mask;

    if (text == NULL || out_ir == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    memset(out_ir, 0, sizeof(*out_ir));

    if (strncmp(
            text,
            prefix,
            sizeof(prefix) - 1U) != 0) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    address_start =
        text + sizeof(prefix) - 1U;

    separator = strchr(
        address_start,
        '|');

    if (separator == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    address_size =
        (size_t)(separator - address_start);

    network_start = separator + 1;

    separator = strchr(
        network_start,
        '|');

    if (separator == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    network_size =
        (size_t)(separator - network_start);

    prefix_start = separator + 1;

    /*
     * A fourth delimiter means extra fields.
     */
    if (strchr(prefix_start, '|') != NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    prefix_size = strlen(prefix_start);

    if (!parse_ipv4(
            address_start,
            address_size,
            &address) ||
        !parse_ipv4(
            network_start,
            network_size,
            &network) ||
        !parse_decimal_component(
            prefix_start,
            prefix_size,
            32U,
            &prefix_value)) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    mask = cidr_mask(
        (uint8_t)prefix_value);

    /*
     * Typed IR V1 requires a canonical network,
     * not an arbitrary host address plus prefix.
     */
    if ((network & mask) != network) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    out_ir->op =
        NIYAH_NETWORK_IR_OP_IP_IN_CIDR;

    out_ir->address = address;
    out_ir->network = network;
    out_ir->prefix_length =
        (uint8_t)prefix_value;

    return NIYAH_OK;
}

NiyahStatus niyah_network_ir_execute(
    const NiyahNetworkIr *ir,
    int *out_match)
{
    uint32_t mask;

    if (ir == NULL || out_match == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    if (ir->op !=
            NIYAH_NETWORK_IR_OP_IP_IN_CIDR ||
        ir->prefix_length > 32U) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    mask = cidr_mask(
        ir->prefix_length);

    /*
     * Refuse non-canonical in-memory IR too.
     */
    if ((ir->network & mask) != ir->network) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    *out_match =
        ((ir->address & mask) ==
         ir->network)
            ? 1
            : 0;

    return NIYAH_OK;
}
