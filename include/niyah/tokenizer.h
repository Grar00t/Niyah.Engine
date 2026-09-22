#ifndef NIYAH_TOKENIZER_H
#define NIYAH_TOKENIZER_H

#include "niyah/common.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NIYAH_BYTE_VOCAB 256u

typedef struct niyah_tokenizer {
    uint8_t identity_sha256[32];
} niyah_tokenizer;

niyah_status niyah_tokenizer_init_byte(niyah_tokenizer *tok);
niyah_status niyah_tokenizer_encode(
    const niyah_tokenizer *tok,
    const uint8_t *bytes,
    size_t byte_count,
    uint32_t *tokens,
    size_t token_capacity,
    size_t *token_count);
niyah_status niyah_tokenizer_decode(
    const niyah_tokenizer *tok,
    const uint32_t *tokens,
    size_t token_count,
    uint8_t *bytes,
    size_t byte_capacity,
    size_t *byte_count);
const uint8_t *niyah_tokenizer_identity_sha256(const niyah_tokenizer *tok);

#ifdef __cplusplus
}
#endif

#endif
