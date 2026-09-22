#include "niyah/tokenizer.h"
#include "niyah/sha256.h"
#include <string.h>

niyah_status niyah_tokenizer_init_byte(niyah_tokenizer *tok) {
    static const char identity[] = "niyah.byte-tokenizer.v1:vocab=256";
    if (!tok) return NIYAH_ERR_INVALID;
    niyah_sha256(identity, sizeof(identity)-1u, tok->identity_sha256);
    return NIYAH_OK;
}

niyah_status niyah_tokenizer_encode(
    const niyah_tokenizer *tok,
    const uint8_t *bytes,
    size_t byte_count,
    uint32_t *tokens,
    size_t token_capacity,
    size_t *token_count) {
    if (!tok || (!bytes && byte_count != 0u) || !token_count) return NIYAH_ERR_INVALID;
    *token_count = byte_count;
    if (!tokens) return NIYAH_OK;
    if (token_capacity < byte_count) return NIYAH_ERR_INVALID;
    for (size_t i = 0; i < byte_count; ++i) tokens[i] = (uint32_t)bytes[i];
    return NIYAH_OK;
}

niyah_status niyah_tokenizer_decode(
    const niyah_tokenizer *tok,
    const uint32_t *tokens,
    size_t token_count,
    uint8_t *bytes,
    size_t byte_capacity,
    size_t *byte_count) {
    if (!tok || (!tokens && token_count != 0u) || !byte_count) return NIYAH_ERR_INVALID;
    *byte_count = token_count;
    if (!bytes) return NIYAH_OK;
    if (byte_capacity < token_count) return NIYAH_ERR_INVALID;
    for (size_t i = 0; i < token_count; ++i) {
        if (tokens[i] >= NIYAH_BYTE_VOCAB) return NIYAH_ERR_FORMAT;
        bytes[i] = (uint8_t)tokens[i];
    }
    return NIYAH_OK;
}

const uint8_t *niyah_tokenizer_identity_sha256(const niyah_tokenizer *tok) {
    return tok ? tok->identity_sha256 : NULL;
}
