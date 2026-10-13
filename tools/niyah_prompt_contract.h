#ifndef NIYAH_PROMPT_CONTRACT_INTERNAL_H
#define NIYAH_PROMPT_CONTRACT_INTERNAL_H

#include "../src/niyah_sha256.h"
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* CLI metadata only; checkpoint serialization is unchanged. Each digest is
 * lowercase hex. V2 binds checkpoint bytes and each template component.
 * NULL component means an explicit CLI override, whose bytes take precedence.
 * A legacy 64-hex checkpoint binding remains readable but returns legacy=1;
 * callers must disclose that template-content integrity was not verified. */
static int niyah_prompt_digest_matches(const char *hex, const uint8_t digest[32])
{
    static const char digits[] = "0123456789abcdef";
    size_t i;
    for (i = 0U; i < 32U; ++i)
        if (hex[i * 2U] != digits[digest[i] >> 4U] ||
            hex[i * 2U + 1U] != digits[digest[i] & 15U]) return 0;
    return 1;
}

static int niyah_prompt_component_matches(const char *hex, const char *text)
{
    NiyahSha256 hash;
    uint8_t digest[32];
    size_t i;
    for (i = 0U; i < 64U; ++i)
        if (!((hex[i] >= '0' && hex[i] <= '9') ||
              (hex[i] >= 'a' && hex[i] <= 'f'))) return 0;
    if (text == NULL) return 1;
    niyah_sha256_init(&hash);
    if (!niyah_sha256_update(&hash, (const uint8_t *)text, strlen(text))) return 0;
    niyah_sha256_final(&hash, digest);
    return niyah_prompt_digest_matches(hex, digest);
}

static int niyah_prompt_contract_validate(const char *metadata,
    const uint8_t checkpoint[32], const char *prefix, const char *suffix,
    int *legacy)
{
    const char *p;
    char normalized[211];
    size_t size, i, used = 0U;
    if (metadata == NULL || checkpoint == NULL || legacy == NULL) return 0;
    *legacy = 0;
    size = strlen(metadata);
    while (size != 0U && (metadata[size - 1U] == '\r' || metadata[size - 1U] == '\n')) --size;
    if (size == 64U) {
        if (!niyah_prompt_digest_matches(metadata, checkpoint)) return 0;
        *legacy = 1;
        return 1;
    }
    /* Accept text-file LF or CRLF metadata. Template bytes themselves are never
     * normalized and their SHA256 digests still bind the exact file content. */
    for (i = 0U; i < size; ++i) {
        if (metadata[i] == '\r' && i + 1U < size && metadata[i + 1U] == '\n') continue;
        if (used >= sizeof(normalized) - 1U) return 0;
        normalized[used++] = metadata[i];
    }
    normalized[used] = '\0';
    if (used != 16U + 64U + 1U + 64U + 1U + 64U ||
        memcmp(normalized, "NIYAH_PROMPT_V2\n", 16U) != 0) return 0;
    p = normalized + 16U;
    if (p[64] != '\n' || p[129] != '\n' ||
        !niyah_prompt_digest_matches(p, checkpoint) ||
        !niyah_prompt_component_matches(p + 65U, prefix) ||
        !niyah_prompt_component_matches(p + 130U, suffix)) return 0;
    return 1;
}
#endif
