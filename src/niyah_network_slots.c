#include "niyah/network_slots.h"

#include <ctype.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

static int is_ipv4_char(char c)
{
    return (c >= '0' && c <= '9') || c == '.';
}

static int is_ascii_word_char(char c)
{
    const unsigned char u = (unsigned char)c;

    return
        (u >= (unsigned char)'0' && u <= (unsigned char)'9') ||
        (u >= (unsigned char)'A' && u <= (unsigned char)'Z') ||
        (u >= (unsigned char)'a' && u <= (unsigned char)'z') ||
        c == '_';
}

static int copy_span(
    const char *text,
    size_t begin,
    size_t end,
    char *buffer,
    size_t capacity)
{
    size_t n;

    if (text == NULL ||
        buffer == NULL ||
        end < begin) {
        return 0;
    }

    n = end - begin;

    if (n + 1U > capacity) {
        return 0;
    }

    memcpy(buffer, text + begin, n);
    buffer[n] = '\0';
    return 1;
}

static int ipv4_span_is_valid(
    const char *text,
    size_t begin,
    size_t end)
{
    char address[32];
    char ir_text[96];
    NiyahNetworkIr ir;
    int written;

    if (!copy_span(
            text,
            begin,
            end,
            address,
            sizeof(address))) {
        return 0;
    }

    written = snprintf(
        ir_text,
        sizeof(ir_text),
        "IP_IN_CIDR|%s|0.0.0.0|0",
        address);

    if (written < 0 ||
        (size_t)written >= sizeof(ir_text)) {
        return 0;
    }

    return
        niyah_network_ir_parse(
            ir_text,
            &ir) == NIYAH_OK;
}

static int network_span_is_valid(
    const char *text,
    size_t network_begin,
    size_t network_end,
    size_t prefix_begin,
    size_t prefix_end)
{
    char network[32];
    char prefix[8];
    char ir_text[96];
    NiyahNetworkIr ir;
    int written;

    if (!copy_span(
            text,
            network_begin,
            network_end,
            network,
            sizeof(network)) ||
        !copy_span(
            text,
            prefix_begin,
            prefix_end,
            prefix,
            sizeof(prefix))) {
        return 0;
    }

    written = snprintf(
        ir_text,
        sizeof(ir_text),
        "IP_IN_CIDR|0.0.0.0|%s|%s",
        network,
        prefix);

    if (written < 0 ||
        (size_t)written >= sizeof(ir_text)) {
        return 0;
    }

    return
        niyah_network_ir_parse(
            ir_text,
            &ir) == NIYAH_OK;
}

NiyahStatus niyah_network_slots_extract_ip_in_cidr(
    const char *text,
    NiyahNetworkIr *out_ir)
{
    size_t i;
    size_t address_count = 0U;
    size_t network_count = 0U;
    size_t address_begin = 0U;
    size_t address_end = 0U;
    size_t network_begin = 0U;
    size_t network_end = 0U;
    size_t prefix_begin = 0U;
    size_t prefix_end = 0U;
    char address[32];
    char network[32];
    char prefix[8];
    char ir_text[96];
    int written;

    if (text == NULL || out_ir == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    /*
     * Scan the complete input before constructing IR.
     *
     * V1 accepts only:
     *
     *   exactly one standalone valid IPv4 operand
     *   exactly one valid canonical IPv4/prefix operand
     *
     * Additional valid operands make the text ambiguous and
     * therefore fail closed.
     */
    i = 0U;
    while (text[i] != '\0') {
        size_t begin;
        size_t end;

        if (!isdigit((unsigned char)text[i])) {
            ++i;
            continue;
        }

        /*
         * Do not begin in the middle of an IPv4-like span or
         * an ASCII word token.
         */
        if (i > 0U &&
            (is_ipv4_char(text[i - 1U]) ||
             is_ascii_word_char(text[i - 1U]))) {
            ++i;
            continue;
        }

        begin = i;
        end = i;

        while (text[end] != '\0' &&
               is_ipv4_char(text[end])) {
            ++end;
        }

        /*
         * A sentence-ending period is punctuation, not part
         * of the IPv4 operand. Trim trailing dots before
         * classifying the candidate.
         *
         * Invalid incomplete forms such as 192.168.1. remain
         * invalid after strict IPv4 parsing.
         */
        while (end > begin &&
               text[end - 1U] == '.') {
            --end;
        }

        /*
         * IPv4 immediately followed by '/' is a CIDR candidate
         * and must never also count as the standalone address.
         */
        if (text[end] == '/') {
            size_t p;
            size_t p_begin;

            p_begin = end + 1U;
            p = p_begin;

            while (text[p] >= '0' &&
                   text[p] <= '9') {
                ++p;
            }

            /*
             * A slash without a decimal prefix is malformed.
             */
            if (p == p_begin) {
                return NIYAH_ERR_INVALID_ARGUMENT;
            }

            /*
             * The decimal prefix must terminate cleanly rather
             * than continue into another numeric/word token.
             */
            if (is_ascii_word_char(text[p]) ||
                (text[p] == '.' &&
                 isdigit((unsigned char)text[p + 1U]))) {
                return NIYAH_ERR_INVALID_ARGUMENT;
            }

            /*
             * Existing strict network IR parsing remains the
             * authority for:
             *
             *   IPv4 validity
             *   prefix 0..32
             *   leading-zero rules
             *   canonical network host bits
             */
            if (!network_span_is_valid(
                    text,
                    begin,
                    end,
                    p_begin,
                    p)) {
                return NIYAH_ERR_INVALID_ARGUMENT;
            }

            ++network_count;
            if (network_count > 1U) {
                return NIYAH_ERR_INVALID_ARGUMENT;
            }

            network_begin = begin;
            network_end = end;
            prefix_begin = p_begin;
            prefix_end = p;
            i = p;
            continue;
        }

        /*
         * A standalone IPv4 token must also terminate outside
         * an ASCII word. Invalid numeric/dotted spans are not
         * accepted as operands.
         */
        if (!is_ascii_word_char(text[end]) &&
            ipv4_span_is_valid(
                text,
                begin,
                end)) {
            ++address_count;
            if (address_count > 1U) {
                return NIYAH_ERR_INVALID_ARGUMENT;
            }

            address_begin = begin;
            address_end = end;
        }

        i = end;
    }

    if (address_count != 1U ||
        network_count != 1U) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    /*
     * Preserve the existing V1 relation shape:
     * standalone address precedes the CIDR operand.
     */
    if (address_begin >= network_begin) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    if (!copy_span(
            text,
            address_begin,
            address_end,
            address,
            sizeof(address)) ||
        !copy_span(
            text,
            network_begin,
            network_end,
            network,
            sizeof(network)) ||
        !copy_span(
            text,
            prefix_begin,
            prefix_end,
            prefix,
            sizeof(prefix))) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    written = snprintf(
        ir_text,
        sizeof(ir_text),
        "IP_IN_CIDR|%s|%s|%s",
        address,
        network,
        prefix);

    if (written < 0 ||
        (size_t)written >= sizeof(ir_text)) {
        return NIYAH_ERR_OVERFLOW;
    }

    return niyah_network_ir_parse(
        ir_text,
        out_ir);
}
