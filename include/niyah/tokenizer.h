#ifndef NIYAH_TOKENIZER_H
#define NIYAH_TOKENIZER_H

#include "niyah/niyah.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NIYAH_TOKENIZER_BYTE_VOCAB_SIZE 256U
#define NIYAH_TOKEN_BOS 256U
#define NIYAH_TOKEN_EOS 257U
#define NIYAH_TOKENIZER_BASE_VOCAB_SIZE 258U
#define NIYAH_TOKENIZER_IDENTITY_SHA256_SIZE 32U

typedef struct NiyahTokenizer NiyahTokenizer;

typedef struct NiyahTokenizerTrainConfig {
    uint32_t target_vocab_size;
    uint32_t min_pair_frequency;
} NiyahTokenizerTrainConfig;

NiyahStatus niyah_tokenizer_train(const uint8_t *corpus,
                                  size_t corpus_size,
                                  const NiyahTokenizerTrainConfig *config,
                                  NiyahTokenizer **out_tokenizer);

void niyah_tokenizer_destroy(NiyahTokenizer *tokenizer);

size_t niyah_tokenizer_vocab_size(const NiyahTokenizer *tokenizer);
size_t niyah_tokenizer_merge_count(const NiyahTokenizer *tokenizer);

NiyahStatus niyah_tokenizer_merge_at(const NiyahTokenizer *tokenizer,
                                     size_t merge_index,
                                     uint32_t *left_token,
                                     uint32_t *right_token,
                                     uint32_t *output_token);

NiyahStatus niyah_tokenizer_identity_sha256(
    const NiyahTokenizer *tokenizer,
    uint8_t out_identity[NIYAH_TOKENIZER_IDENTITY_SHA256_SIZE]);

NiyahStatus niyah_tokenizer_save(const NiyahTokenizer *tokenizer,
                                 const char *path);

NiyahStatus niyah_tokenizer_load(const char *path,
                                 NiyahTokenizer **out_tokenizer);

NiyahStatus niyah_tokenizer_encode(const NiyahTokenizer *tokenizer,
                                   const uint8_t *input,
                                   size_t input_size,
                                   uint32_t *output_tokens,
                                   size_t output_capacity,
                                   size_t *output_count);

NiyahStatus niyah_tokenizer_decode(const NiyahTokenizer *tokenizer,
                                   const uint32_t *tokens,
                                   size_t token_count,
                                   uint8_t *output_bytes,
                                   size_t output_capacity,
                                   size_t *output_size);

#ifdef __cplusplus
}
#endif

#endif
