#include "niyah/evidence.h"
#include "niyah_sha256.h"

#include <stddef.h>
#include <string.h>

NiyahStatus niyah_evidence_root_sha256(
    const uint8_t
        receipt_sha256[NIYAH_RECEIPT_SHA256_SIZE],
    const uint8_t
        checkpoint_sha256[
            NIYAH_CHECKPOINT_IDENTITY_SHA256_SIZE],
    const uint8_t
        tokenizer_sha256[
            NIYAH_TOKENIZER_IDENTITY_SHA256_SIZE],
    uint8_t
        out_root[NIYAH_EVIDENCE_ROOT_SHA256_SIZE])
{
    static const unsigned char domain[] =
        "NIYAH_EVIDENCE_V1";

    NiyahSha256 sha;

    if (receipt_sha256 == NULL ||
        checkpoint_sha256 == NULL ||
        tokenizer_sha256 == NULL ||
        out_root == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    niyah_sha256_init(&sha);

    if (!niyah_sha256_update(
            &sha,
            domain,
            sizeof(domain) - 1U) ||
        !niyah_sha256_update(
            &sha,
            receipt_sha256,
            NIYAH_RECEIPT_SHA256_SIZE) ||
        !niyah_sha256_update(
            &sha,
            checkpoint_sha256,
            NIYAH_CHECKPOINT_IDENTITY_SHA256_SIZE) ||
        !niyah_sha256_update(
            &sha,
            tokenizer_sha256,
            NIYAH_TOKENIZER_IDENTITY_SHA256_SIZE)) {
        return NIYAH_ERR_OVERFLOW;
    }

    niyah_sha256_final(&sha, out_root);
    return NIYAH_OK;
}


NiyahStatus niyah_evidence_verify_root(
    const NiyahExecutionReceipt *receipt,
    const uint8_t
        checkpoint_sha256[
            NIYAH_CHECKPOINT_IDENTITY_SHA256_SIZE],
    const uint8_t
        tokenizer_sha256[
            NIYAH_TOKENIZER_IDENTITY_SHA256_SIZE],
    const uint8_t
        claimed_root[
            NIYAH_EVIDENCE_ROOT_SHA256_SIZE])
{
    uint8_t receipt_hash[NIYAH_RECEIPT_SHA256_SIZE];
    uint8_t computed_root[NIYAH_EVIDENCE_ROOT_SHA256_SIZE];
    NiyahStatus status;
    unsigned difference = 0U;
    size_t i;

    if (receipt == NULL ||
        checkpoint_sha256 == NULL ||
        tokenizer_sha256 == NULL ||
        claimed_root == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    status = niyah_receipt_sha256(
        receipt,
        receipt_hash);

    if (status != NIYAH_OK) {
        return status;
    }

    status = niyah_evidence_root_sha256(
        receipt_hash,
        checkpoint_sha256,
        tokenizer_sha256,
        computed_root);

    if (status != NIYAH_OK) {
        return status;
    }

    for (i = 0U;
         i < NIYAH_EVIDENCE_ROOT_SHA256_SIZE;
         ++i) {
        difference |=
            (unsigned)(
                computed_root[i] ^
                claimed_root[i]);
    }

    if (difference != 0U) {
        return NIYAH_ERR_CORRUPT_DATA;
    }

    return NIYAH_OK;
}


static int evidence_next_line(
    const char *text,
    size_t text_size,
    size_t *offset,
    const char **out_line,
    size_t *out_length)
{
    size_t i;
    size_t start;

    if (text == NULL ||
        offset == NULL ||
        out_line == NULL ||
        out_length == NULL ||
        *offset >= text_size) {
        return 0;
    }

    start = *offset;

    for (i = start; i < text_size; ++i) {
        if (text[i] == '\0') {
            return 0;
        }

        if (text[i] == '\n') {
            *out_line = text + start;
            *out_length = i - start;
            *offset = i + 1U;
            return 1;
        }
    }

    return 0;
}

static int evidence_line_equals(
    const char *line,
    size_t line_length,
    const char *expected)
{
    size_t expected_length;

    if (line == NULL || expected == NULL) {
        return 0;
    }

    expected_length = strlen(expected);

    return line_length == expected_length &&
           memcmp(
               line,
               expected,
               expected_length) == 0;
}

static int evidence_hex_nibble(
    char c,
    uint8_t *out)
{
    if (c >= '0' && c <= '9') {
        *out = (uint8_t)(c - '0');
        return 1;
    }

    if (c >= 'a' && c <= 'f') {
        *out = (uint8_t)(c - 'a' + 10);
        return 1;
    }

    return 0;
}

static int evidence_parse_hex_field(
    const char *line,
    size_t line_length,
    const char *prefix,
    uint8_t out[32])
{
    size_t prefix_length;
    size_t i;

    prefix_length = strlen(prefix);

    if (line_length != prefix_length + 64U ||
        memcmp(
            line,
            prefix,
            prefix_length) != 0) {
        return 0;
    }

    for (i = 0U; i < 32U; ++i) {
        uint8_t high;
        uint8_t low;

        if (!evidence_hex_nibble(
                line[prefix_length + i * 2U],
                &high) ||
            !evidence_hex_nibble(
                line[prefix_length + i * 2U + 1U],
                &low)) {
            return 0;
        }

        out[i] = (uint8_t)(
            (uint8_t)(high << 4U) | low);
    }

    return 1;
}

static int evidence_parse_u64_span(
    const char *text,
    size_t size,
    uint64_t limit,
    uint64_t *out)
{
    uint64_t value = 0U;
    size_t i;

    if (text == NULL ||
        out == NULL ||
        size == 0U) {
        return 0;
    }

    for (i = 0U; i < size; ++i) {
        uint64_t digit;

        if (text[i] < '0' ||
            text[i] > '9') {
            return 0;
        }

        digit = (uint64_t)(text[i] - '0');

        if (value > (limit - digit) / 10U) {
            return 0;
        }

        value = value * 10U + digit;
    }

    *out = value;
    return 1;
}

static int evidence_parse_nonnegative_i64_field(
    const char *line,
    size_t line_length,
    const char *prefix,
    int64_t *out)
{
    size_t prefix_length;
    uint64_t value;

    prefix_length = strlen(prefix);

    if (line_length <= prefix_length ||
        memcmp(
            line,
            prefix,
            prefix_length) != 0) {
        return 0;
    }

    if (!evidence_parse_u64_span(
            line + prefix_length,
            line_length - prefix_length,
            (uint64_t)INT64_MAX,
            &value)) {
        return 0;
    }

    *out = (int64_t)value;
    return 1;
}

static int evidence_parse_i64_field(
    const char *line,
    size_t line_length,
    const char *prefix,
    int64_t *out)
{
    size_t prefix_length;
    const char *value_text;
    size_t value_size;
    uint64_t magnitude;

    prefix_length = strlen(prefix);

    if (line_length <= prefix_length ||
        memcmp(
            line,
            prefix,
            prefix_length) != 0) {
        return 0;
    }

    value_text = line + prefix_length;
    value_size = line_length - prefix_length;

    if (value_text[0] == '-') {
        uint64_t limit =
            (uint64_t)INT64_MAX + UINT64_C(1);

        if (value_size <= 1U ||
            !evidence_parse_u64_span(
                value_text + 1U,
                value_size - 1U,
                limit,
                &magnitude)) {
            return 0;
        }

        if (magnitude == limit) {
            *out = INT64_MIN;
        } else {
            *out = -(int64_t)magnitude;
        }

        return 1;
    }

    if (!evidence_parse_u64_span(
            value_text,
            value_size,
            (uint64_t)INT64_MAX,
            &magnitude)) {
        return 0;
    }

    *out = (int64_t)magnitude;
    return 1;
}

static int evidence_equal_bytes(
    const uint8_t *a,
    const uint8_t *b,
    size_t size)
{
    unsigned difference = 0U;
    size_t i;

    for (i = 0U; i < size; ++i) {
        difference |=
            (unsigned)(a[i] ^ b[i]);
    }

    return difference == 0U;
}

NiyahStatus niyah_evidence_parse_document(
    const char *text,
    size_t text_size,
    NiyahEvidenceDocument *out_document)
{
    NiyahEvidenceDocument parsed;
    const char *line;
    size_t line_length;
    size_t offset = 0U;
    size_t receipt_start;
    char canonical[256];
    size_t canonical_length = 0U;
    char ir_text[128];
    NiyahIr parsed_ir;
    NiyahStatus status;

    if (text == NULL ||
        out_document == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    memset(&parsed, 0, sizeof(parsed));

#define NEXT_LINE_OR_CORRUPT() \
    do { \
        if (!evidence_next_line( \
                text, \
                text_size, \
                &offset, \
                &line, \
                &line_length)) { \
            return NIYAH_ERR_CORRUPT_DATA; \
        } \
    } while (0)

    NEXT_LINE_OR_CORRUPT();

    if (!evidence_line_equals(
            line,
            line_length,
            "NIYAH_EVIDENCE_V1")) {
        return NIYAH_ERR_CORRUPT_DATA;
    }

    NEXT_LINE_OR_CORRUPT();

    if (!evidence_parse_hex_field(
            line,
            line_length,
            "receipt_sha256=",
            parsed.receipt_sha256)) {
        return NIYAH_ERR_CORRUPT_DATA;
    }

    NEXT_LINE_OR_CORRUPT();

    if (!evidence_parse_hex_field(
            line,
            line_length,
            "checkpoint_sha256=",
            parsed.checkpoint_sha256)) {
        return NIYAH_ERR_CORRUPT_DATA;
    }

    NEXT_LINE_OR_CORRUPT();

    if (!evidence_parse_hex_field(
            line,
            line_length,
            "tokenizer_sha256=",
            parsed.tokenizer_sha256)) {
        return NIYAH_ERR_CORRUPT_DATA;
    }

    NEXT_LINE_OR_CORRUPT();

    if (!evidence_parse_hex_field(
            line,
            line_length,
            "evidence_root_sha256=",
            parsed.evidence_root_sha256)) {
        return NIYAH_ERR_CORRUPT_DATA;
    }

    receipt_start = offset;

    NEXT_LINE_OR_CORRUPT();

    if (!evidence_line_equals(
            line,
            line_length,
            "NIYAH_RECEIPT_V1")) {
        return NIYAH_ERR_CORRUPT_DATA;
    }

    NEXT_LINE_OR_CORRUPT();

    if (evidence_line_equals(
            line,
            line_length,
            "intent=ADD")) {
        parsed.receipt.ir.op =
            NIYAH_IR_OP_ADD;
    } else if (evidence_line_equals(
                   line,
                   line_length,
                   "intent=SUB")) {
        parsed.receipt.ir.op =
            NIYAH_IR_OP_SUB;
    } else if (evidence_line_equals(
                   line,
                   line_length,
                   "intent=MUL")) {
        parsed.receipt.ir.op =
            NIYAH_IR_OP_MUL;
    } else {
        return NIYAH_ERR_CORRUPT_DATA;
    }

    NEXT_LINE_OR_CORRUPT();

    if (!evidence_parse_nonnegative_i64_field(
            line,
            line_length,
            "lhs=",
            &parsed.receipt.ir.lhs)) {
        return NIYAH_ERR_CORRUPT_DATA;
    }

    NEXT_LINE_OR_CORRUPT();

    if (!evidence_parse_nonnegative_i64_field(
            line,
            line_length,
            "rhs=",
            &parsed.receipt.ir.rhs)) {
        return NIYAH_ERR_CORRUPT_DATA;
    }

    NEXT_LINE_OR_CORRUPT();

    if (!evidence_parse_i64_field(
            line,
            line_length,
            "result=",
            &parsed.receipt.result)) {
        return NIYAH_ERR_CORRUPT_DATA;
    }

    NEXT_LINE_OR_CORRUPT();

    if (line_length <= 3U ||
        memcmp(line, "ir=", 3U) != 0 ||
        line_length - 3U >=
            sizeof(ir_text)) {
        return NIYAH_ERR_CORRUPT_DATA;
    }

    memcpy(
        ir_text,
        line + 3U,
        line_length - 3U);

    ir_text[line_length - 3U] = '\0';

    status = niyah_ir_parse(
        ir_text,
        &parsed_ir);

    if (status != NIYAH_OK ||
        parsed_ir.op != parsed.receipt.ir.op ||
        parsed_ir.lhs != parsed.receipt.ir.lhs ||
        parsed_ir.rhs != parsed.receipt.ir.rhs) {
        return NIYAH_ERR_CORRUPT_DATA;
    }

    if (offset != text_size) {
        return NIYAH_ERR_CORRUPT_DATA;
    }

    status = niyah_receipt_format(
        &parsed.receipt,
        canonical,
        sizeof(canonical),
        &canonical_length);

    if (status != NIYAH_OK) {
        return NIYAH_ERR_CORRUPT_DATA;
    }

    if (canonical_length !=
            text_size - receipt_start ||
        memcmp(
            canonical,
            text + receipt_start,
            canonical_length) != 0) {
        return NIYAH_ERR_CORRUPT_DATA;
    }

#undef NEXT_LINE_OR_CORRUPT

    *out_document = parsed;
    return NIYAH_OK;
}

NiyahStatus niyah_evidence_verify_document(
    const NiyahEvidenceDocument *document,
    const uint8_t
        actual_checkpoint_sha256[
            NIYAH_CHECKPOINT_IDENTITY_SHA256_SIZE],
    const uint8_t
        actual_tokenizer_sha256[
            NIYAH_TOKENIZER_IDENTITY_SHA256_SIZE])
{
    uint8_t computed_receipt[
        NIYAH_RECEIPT_SHA256_SIZE];

    NiyahStatus status;

    if (document == NULL ||
        actual_checkpoint_sha256 == NULL ||
        actual_tokenizer_sha256 == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    status = niyah_receipt_sha256(
        &document->receipt,
        computed_receipt);

    if (status != NIYAH_OK) {
        return NIYAH_ERR_CORRUPT_DATA;
    }

    if (!evidence_equal_bytes(
            computed_receipt,
            document->receipt_sha256,
            NIYAH_RECEIPT_SHA256_SIZE) ||
        !evidence_equal_bytes(
            actual_checkpoint_sha256,
            document->checkpoint_sha256,
            NIYAH_CHECKPOINT_IDENTITY_SHA256_SIZE) ||
        !evidence_equal_bytes(
            actual_tokenizer_sha256,
            document->tokenizer_sha256,
            NIYAH_TOKENIZER_IDENTITY_SHA256_SIZE)) {
        return NIYAH_ERR_CORRUPT_DATA;
    }

    return niyah_evidence_verify_root(
        &document->receipt,
        actual_checkpoint_sha256,
        actual_tokenizer_sha256,
        document->evidence_root_sha256);
}
