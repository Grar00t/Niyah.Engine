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

static int ascii_matches_at(
    const char *text,
    size_t position,
    const char *needle)
{
    size_t j;

    if (text == NULL || needle == NULL) {
        return 0;
    }

    for (j = 0U; needle[j] != '\0'; ++j) {
        if (text[position + j] == '\0' ||
            ascii_lower(
                (unsigned char)text[position + j]) !=
            ascii_lower(
                (unsigned char)needle[j])) {
            return 0;
        }
    }

    return j > 0U;
}

static int range_contains_ascii_case_insensitive(
    const char *text,
    size_t begin,
    size_t end,
    const char *needle)
{
    size_t i;
    size_t needle_size;

    if (text == NULL ||
        needle == NULL ||
        end < begin) {
        return 0;
    }

    needle_size = strlen(needle);

    if (needle_size == 0U ||
        needle_size > end - begin) {
        return 0;
    }

    for (i = begin;
         i + needle_size <= end;
         ++i) {
        if (ascii_matches_at(
                text,
                i,
                needle)) {
            return 1;
        }
    }

    return 0;
}

static int range_contains_exact(
    const char *text,
    size_t begin,
    size_t end,
    const char *needle)
{
    size_t i;
    size_t needle_size;

    if (text == NULL ||
        needle == NULL ||
        end < begin) {
        return 0;
    }

    needle_size = strlen(needle);

    if (needle_size == 0U ||
        needle_size > end - begin) {
        return 0;
    }

    for (i = begin;
         i + needle_size <= end;
         ++i) {
        if (memcmp(
                text + i,
                needle,
                needle_size) == 0) {
            return 1;
        }
    }

    return 0;
}

static int is_ascii_clause_boundary(
    unsigned char c)
{
    return
        c == (unsigned char)';' ||
        c == (unsigned char)'?' ||
        c == (unsigned char)'!' ||
        c == (unsigned char)'\n' ||
        c == (unsigned char)'\r';
}

static int is_arabic_clause_boundary_at(
    const char *text,
    size_t position)
{
    const unsigned char first =
        (unsigned char)text[position];

    const unsigned char second =
        (unsigned char)text[position + 1U];

    /*
     * U+061B ARABIC SEMICOLON:
     *   UTF-8 D8 9B
     *
     * U+061F ARABIC QUESTION MARK:
     *   UTF-8 D8 9F
     */
    return
        first == UINT8_C(0xD8) &&
        (second == UINT8_C(0x9B) ||
         second == UINT8_C(0x9F));
}

static size_t clause_begin_before(
    const char *text,
    size_t position)
{
    size_t begin = 0U;
    size_t i = 0U;

    while (i < position &&
           text[i] != '\0') {
        if (is_ascii_clause_boundary(
                (unsigned char)text[i])) {
            begin = i + 1U;
            ++i;
            continue;
        }

        if (i + 1U < position &&
            is_arabic_clause_boundary_at(
                text,
                i)) {
            begin = i + 2U;
            i += 2U;
            continue;
        }

        ++i;
    }

    return begin;
}

static size_t clause_end_after(
    const char *text,
    size_t position)
{
    size_t i = position;

    while (text[i] != '\0') {
        if (is_ascii_clause_boundary(
                (unsigned char)text[i])) {
            return i;
        }

        if (is_arabic_clause_boundary_at(
                text,
                i)) {
            return i;
        }

        ++i;
    }

    return i;
}

static int cue_is_scoped_out(
    const char *text,
    size_t cue_begin,
    size_t cue_size)
{
    size_t clause_begin;
    size_t clause_end;
    const size_t cue_end =
        cue_begin + cue_size;

    clause_begin =
        clause_begin_before(
            text,
            cue_begin);

    clause_end =
        clause_end_after(
            text,
            cue_end);

    /*
     * V1 scoped English negation.
     *
     * This deliberately targets negation of the membership
     * predicate rather than treating any "not" anywhere in
     * the complete sentence as a global veto.
     */
    if (range_contains_ascii_case_insensitive(
            text,
            clause_begin,
            cue_begin,
            "do not test whether") ||
        range_contains_ascii_case_insensitive(
            text,
            clause_begin,
            cue_begin,
            "do not check whether") ||
        range_contains_ascii_case_insensitive(
            text,
            clause_begin,
            cue_begin,
            "do not determine whether") ||
        range_contains_ascii_case_insensitive(
            text,
            clause_begin,
            cue_begin,
            "don't test whether") ||
        range_contains_ascii_case_insensitive(
            text,
            clause_begin,
            cue_begin,
            "don't check whether") ||
        range_contains_ascii_case_insensitive(
            text,
            clause_begin,
            cue_begin,
            "don't determine whether")) {
        return 1;
    }

    /*
     * V1 scoped Arabic negation.
     */
    if (range_contains_exact(
            text,
            clause_begin,
            cue_begin,
            "لا تختبر ما إذا كان") ||
        range_contains_exact(
            text,
            clause_begin,
            cue_begin,
            "لا تختبر ما اذا كان") ||
        range_contains_exact(
            text,
            clause_begin,
            cue_begin,
            "لا تتحقق مما إذا كان") ||
        range_contains_exact(
            text,
            clause_begin,
            cue_begin,
            "لا تتحقق مما اذا كان")) {
        return 1;
    }

    /*
     * Meta-linguistic mention is not an execution request.
     *
     * English:
     *   "Explain what belongs to means ..."
     *   "Explain the phrase inside ..."
     */
    if ((range_contains_ascii_case_insensitive(
             text,
             clause_begin,
             cue_begin,
             "explain what") &&
         range_contains_ascii_case_insensitive(
             text,
             cue_end,
             clause_end,
             "means")) ||
        range_contains_ascii_case_insensitive(
            text,
            clause_begin,
            cue_begin,
            "explain the phrase")) {
        return 1;
    }

    /*
     * Arabic:
     *   "اشرح عبارة داخل الشبكة ..."
     */
    if (range_contains_exact(
            text,
            clause_begin,
            cue_begin,
            "اشرح عبارة") ||
        range_contains_exact(
            text,
            clause_begin,
            cue_begin,
            "فسر عبارة")) {
        return 1;
    }

    return 0;
}

static int has_ascii_membership_cue(
    const char *text,
    const char *cue)
{
    size_t i;
    size_t cue_size;

    cue_size = strlen(cue);

    for (i = 0U;
         text[i] != '\0';
         ++i) {
        if (!ascii_matches_at(
                text,
                i,
                cue)) {
            continue;
        }

        if (!cue_is_scoped_out(
                text,
                i,
                cue_size)) {
            return 1;
        }
    }

    return 0;
}

static int has_exact_membership_cue(
    const char *text,
    const char *cue)
{
    const char *cursor;
    const char *match;
    size_t cue_size;

    cue_size = strlen(cue);
    cursor = text;

    while ((match = strstr(
                cursor,
                cue)) != NULL) {
        const size_t position =
            (size_t)(match - text);

        if (!cue_is_scoped_out(
                text,
                position,
                cue_size)) {
            return 1;
        }

        cursor = match + 1;
    }

    return 0;
}

static int has_membership_intent(
    const char *text)
{
    /*
     * English V1:
     * explicit relational phrases only.
     *
     * A cue is ignored when the cue itself is scoped by a
     * recognized negation or meta-linguistic construction.
     */
    if (has_ascii_membership_cue(
            text,
            " inside ") ||
        has_ascii_membership_cue(
            text,
            " is in ") ||
        has_ascii_membership_cue(
            text,
            " belongs to ") ||
        has_ascii_membership_cue(
            text,
            " belong to ")) {
        return 1;
    }

    /*
     * Arabic UTF-8 V1.
     */
    if (has_exact_membership_cue(
            text,
            " داخل ") ||
        has_exact_membership_cue(
            text,
            " ينتمي إلى ") ||
        has_exact_membership_cue(
            text,
            " ينتمي الى ")) {
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
