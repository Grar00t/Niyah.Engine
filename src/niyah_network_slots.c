#include "niyah/network_slots.h"

#include <ctype.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

static int is_ipv4_char(char c)
{
    return (c >= '0' && c <= '9') || c == '.';
}

static int find_ipv4_token(
    const char *text,
    size_t start,
    size_t *out_begin,
    size_t *out_end)
{
    size_t i;

    for (i = start; text[i] != '\0'; ++i) {
        size_t j;

        if (!isdigit((unsigned char)text[i])) {
            continue;
        }

        if (i > 0U && is_ipv4_char(text[i - 1U])) {
            continue;
        }

        j = i;

        while (text[j] != '\0' &&
               is_ipv4_char(text[j])) {
            ++j;
        }

        if (j > i &&
            (text[j] == '\0' ||
             !is_ipv4_char(text[j]))) {
            *out_begin = i;
            *out_end = j;
            return 1;
        }
    }

    return 0;
}

static int copy_span(
    const char *text,
    size_t begin,
    size_t end,
    char *buffer,
    size_t capacity)
{
    size_t n;

    if (end < begin) {
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

NiyahStatus niyah_network_slots_extract_ip_in_cidr(
    const char *text,
    NiyahNetworkIr *out_ir)
{
    size_t first_begin;
    size_t first_end;
    size_t second_begin;
    size_t second_end;
    size_t prefix_begin;
    size_t prefix_end;

    char address[32];
    char network[32];
    char prefix[8];
    char ir_text[96];

    int written;

    if (text == NULL || out_ir == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    if (!find_ipv4_token(
            text,
            0U,
            &first_begin,
            &first_end)) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    if (!find_ipv4_token(
            text,
            first_end,
            &second_begin,
            &second_end)) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    if (text[second_end] != '/') {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    prefix_begin = second_end + 1U;
    prefix_end = prefix_begin;

    while (text[prefix_end] >= '0' &&
           text[prefix_end] <= '9') {
        ++prefix_end;
    }

    if (prefix_end == prefix_begin) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    if (!copy_span(
            text,
            first_begin,
            first_end,
            address,
            sizeof(address)) ||
        !copy_span(
            text,
            second_begin,
            second_end,
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
