#include "niyah/network_admissibility.h"
#include "niyah/network_slots.h"

#include <stddef.h>
#include <string.h>

static unsigned char ascii_lower(unsigned char c)
{
    if (c >= (unsigned char)'A' &&
        c <= (unsigned char)'Z') {
        return (unsigned char)(
            c - (unsigned char)'A' +
            (unsigned char)'a');
    }

    return c;
}

static int contains_ascii_case_insensitive(
    const char *text,
    const char *needle)
{
    size_t i;
    size_t needle_size;

    if (text == NULL || needle == NULL) {
        return 0;
    }

    needle_size = strlen(needle);

    if (needle_size == 0U) {
        return 0;
    }

    for (i = 0U; text[i] != '\0'; ++i) {
        size_t j = 0U;

        while (j < needle_size &&
               text[i + j] != '\0' &&
               ascii_lower(
                   (unsigned char)text[i + j]) ==
               ascii_lower(
                   (unsigned char)needle[j])) {
            ++j;
        }

        if (j == needle_size) {
            return 1;
        }
    }

    return 0;
}

static int has_membership_intent(
    const char *text)
{
    /*
     * English V1:
     * use explicit relational phrases rather than generic
     * words such as "in", which would admit unrelated text.
     */
    if (contains_ascii_case_insensitive(
            text,
            " inside ") ||
        contains_ascii_case_insensitive(
            text,
            " is in ") ||
        contains_ascii_case_insensitive(
            text,
            " belongs to ") ||
        contains_ascii_case_insensitive(
            text,
            " belong to ")) {
        return 1;
    }

    /*
     * Arabic UTF-8 V1.
     *
     * strstr is byte-safe here because these are exact UTF-8
     * phrase markers and no character-level transformation is
     * attempted.
     */
    if (strstr(text, " داخل ") != NULL ||
        strstr(text, " ينتمي إلى ") != NULL ||
        strstr(text, " ينتمي الى ") != NULL) {
        return 1;
    }

    return 0;
}

NiyahStatus niyah_network_admit_ip_in_cidr(
    const char *text,
    NiyahNetworkIr *out_ir)
{
    if (text == NULL || out_ir == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    if (!has_membership_intent(text)) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    return niyah_network_slots_extract_ip_in_cidr(
        text,
        out_ir);
}
