#include "niyah/tokenizer.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

typedef struct NiyahTokenBytes {
    uint8_t *data;
    size_t length;
} NiyahTokenBytes;

typedef struct NiyahMerge {
    uint32_t left;
    uint32_t right;
    uint32_t output;
} NiyahMerge;

struct NiyahTokenizer {
    NiyahTokenBytes *vocab;
    size_t vocab_size;
    size_t vocab_capacity;
    NiyahMerge *merges;
    size_t merge_count;
    size_t merge_capacity;
};

static int niyah_size_mul_ok(size_t a, size_t b, size_t *out)
{
    if (out == NULL || (a != 0U && b > SIZE_MAX / a)) {
        return 0;
    }
    *out = a * b;
    return 1;
}

static int niyah_size_add_ok(size_t a, size_t b, size_t *out)
{
    if (out == NULL || a > SIZE_MAX - b) {
        return 0;
    }
    *out = a + b;
    return 1;
}

static int niyah_u64_compare(const void *a, const void *b)
{
    const uint64_t lhs = *(const uint64_t *)a;
    const uint64_t rhs = *(const uint64_t *)b;
    if (lhs < rhs) {
        return -1;
    }
    if (lhs > rhs) {
        return 1;
    }
    return 0;
}

static uint64_t niyah_pair_key(uint32_t left, uint32_t right)
{
    return ((uint64_t)left << 32U) | (uint64_t)right;
}

static NiyahStatus niyah_tokenizer_alloc_base(uint32_t target_vocab_size,
                                               NiyahTokenizer **out_tokenizer)
{
    NiyahTokenizer *tokenizer;
    size_t vocab_bytes = 0U;
    size_t merge_capacity;
    size_t merge_bytes = 0U;
    uint32_t i;

    if (out_tokenizer == NULL || target_vocab_size < NIYAH_TOKENIZER_BASE_VOCAB_SIZE) {
        return NIYAH_ERR_INVALID_CONFIG;
    }
    if (!niyah_size_mul_ok((size_t)target_vocab_size, sizeof(NiyahTokenBytes), &vocab_bytes)) {
        return NIYAH_ERR_OVERFLOW;
    }

    tokenizer = (NiyahTokenizer *)calloc(1U, sizeof(*tokenizer));
    if (tokenizer == NULL) {
        return NIYAH_ERR_OUT_OF_MEMORY;
    }

    tokenizer->vocab = (NiyahTokenBytes *)calloc(1U, vocab_bytes);
    if (tokenizer->vocab == NULL) {
        free(tokenizer);
        return NIYAH_ERR_OUT_OF_MEMORY;
    }

    merge_capacity = (size_t)target_vocab_size - (size_t)NIYAH_TOKENIZER_BASE_VOCAB_SIZE;
    if (merge_capacity != 0U) {
        if (!niyah_size_mul_ok(merge_capacity, sizeof(NiyahMerge), &merge_bytes)) {
            free(tokenizer->vocab);
            free(tokenizer);
            return NIYAH_ERR_OVERFLOW;
        }
        tokenizer->merges = (NiyahMerge *)calloc(1U, merge_bytes);
        if (tokenizer->merges == NULL) {
            free(tokenizer->vocab);
            free(tokenizer);
            return NIYAH_ERR_OUT_OF_MEMORY;
        }
    }

    tokenizer->vocab_capacity = (size_t)target_vocab_size;
    tokenizer->merge_capacity = merge_capacity;
    tokenizer->vocab_size = (size_t)NIYAH_TOKENIZER_BASE_VOCAB_SIZE;

    for (i = 0U; i < NIYAH_TOKENIZER_BYTE_VOCAB_SIZE; ++i) {
        tokenizer->vocab[i].data = (uint8_t *)malloc(1U);
        if (tokenizer->vocab[i].data == NULL) {
            niyah_tokenizer_destroy(tokenizer);
            return NIYAH_ERR_OUT_OF_MEMORY;
        }
        tokenizer->vocab[i].data[0] = (uint8_t)i;
        tokenizer->vocab[i].length = 1U;
    }

    *out_tokenizer = tokenizer;
    return NIYAH_OK;
}

static NiyahStatus niyah_tokenizer_add_merge(NiyahTokenizer *tokenizer,
                                             uint32_t left,
                                             uint32_t right,
                                             uint32_t *out_token)
{
    NiyahTokenBytes *left_bytes;
    NiyahTokenBytes *right_bytes;
    NiyahTokenBytes *new_bytes;
    size_t total_length = 0U;
    uint32_t token_id;

    if (tokenizer == NULL || out_token == NULL ||
        (size_t)left >= tokenizer->vocab_size ||
        (size_t)right >= tokenizer->vocab_size ||
        tokenizer->vocab_size >= tokenizer->vocab_capacity ||
        tokenizer->merge_count >= tokenizer->merge_capacity ||
        tokenizer->vocab_size > (size_t)UINT32_MAX) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    left_bytes = &tokenizer->vocab[left];
    right_bytes = &tokenizer->vocab[right];
    if (!niyah_size_add_ok(left_bytes->length, right_bytes->length, &total_length)) {
        return NIYAH_ERR_OVERFLOW;
    }
    if (total_length == 0U) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    new_bytes = &tokenizer->vocab[tokenizer->vocab_size];
    new_bytes->data = (uint8_t *)malloc(total_length);
    if (new_bytes->data == NULL) {
        return NIYAH_ERR_OUT_OF_MEMORY;
    }
    memcpy(new_bytes->data, left_bytes->data, left_bytes->length);
    memcpy(new_bytes->data + left_bytes->length, right_bytes->data, right_bytes->length);
    new_bytes->length = total_length;

    token_id = (uint32_t)tokenizer->vocab_size;
    tokenizer->merges[tokenizer->merge_count].left = left;
    tokenizer->merges[tokenizer->merge_count].right = right;
    tokenizer->merges[tokenizer->merge_count].output = token_id;
    tokenizer->merge_count += 1U;
    tokenizer->vocab_size += 1U;
    *out_token = token_id;
    return NIYAH_OK;
}

static void niyah_apply_merge(uint32_t *tokens,
                              size_t *token_count,
                              uint32_t left,
                              uint32_t right,
                              uint32_t output)
{
    size_t read_index = 0U;
    size_t write_index = 0U;
    size_t count;

    if (tokens == NULL || token_count == NULL) {
        return;
    }

    count = *token_count;
    while (read_index < count) {
        if (read_index + 1U < count &&
            tokens[read_index] == left &&
            tokens[read_index + 1U] == right) {
            tokens[write_index++] = output;
            read_index += 2U;
        } else {
            tokens[write_index++] = tokens[read_index++];
        }
    }
    *token_count = write_index;
}

NiyahStatus niyah_tokenizer_train(const uint8_t *corpus,
                                  size_t corpus_size,
                                  const NiyahTokenizerTrainConfig *config,
                                  NiyahTokenizer **out_tokenizer)
{
    NiyahTokenizer *tokenizer = NULL;
    uint32_t *tokens = NULL;
    size_t token_bytes = 0U;
    size_t token_count;
    size_t i;
    NiyahStatus status;

    if (corpus == NULL || corpus_size == 0U || config == NULL || out_tokenizer == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    *out_tokenizer = NULL;
    if (config->target_vocab_size < NIYAH_TOKENIZER_BASE_VOCAB_SIZE ||
        config->min_pair_frequency == 0U) {
        return NIYAH_ERR_INVALID_CONFIG;
    }
    if (!niyah_size_mul_ok(corpus_size, sizeof(uint32_t), &token_bytes)) {
        return NIYAH_ERR_OVERFLOW;
    }

    status = niyah_tokenizer_alloc_base(config->target_vocab_size, &tokenizer);
    if (status != NIYAH_OK) {
        return status;
    }

    tokens = (uint32_t *)malloc(token_bytes);
    if (tokens == NULL) {
        niyah_tokenizer_destroy(tokenizer);
        return NIYAH_ERR_OUT_OF_MEMORY;
    }
    for (i = 0U; i < corpus_size; ++i) {
        tokens[i] = (uint32_t)corpus[i];
    }
    token_count = corpus_size;

    while (tokenizer->vocab_size < tokenizer->vocab_capacity && token_count > 1U) {
        uint64_t *pairs;
        size_t pair_count = token_count - 1U;
        size_t pair_bytes = 0U;
        uint64_t best_key = 0U;
        size_t best_frequency = 0U;
        size_t run_start;
        uint32_t left;
        uint32_t right;
        uint32_t output;

        if (!niyah_size_mul_ok(pair_count, sizeof(uint64_t), &pair_bytes)) {
            status = NIYAH_ERR_OVERFLOW;
            goto fail;
        }
        pairs = (uint64_t *)malloc(pair_bytes);
        if (pairs == NULL) {
            status = NIYAH_ERR_OUT_OF_MEMORY;
            goto fail;
        }
        for (i = 0U; i < pair_count; ++i) {
            pairs[i] = niyah_pair_key(tokens[i], tokens[i + 1U]);
        }
        qsort(pairs, pair_count, sizeof(uint64_t), niyah_u64_compare);

        run_start = 0U;
        while (run_start < pair_count) {
            size_t run_end = run_start + 1U;
            size_t frequency;
            while (run_end < pair_count && pairs[run_end] == pairs[run_start]) {
                run_end += 1U;
            }
            frequency = run_end - run_start;
            if (frequency > best_frequency) {
                best_frequency = frequency;
                best_key = pairs[run_start];
            }
            run_start = run_end;
        }
        free(pairs);

        if (best_frequency < (size_t)config->min_pair_frequency) {
            break;
        }

        left = (uint32_t)(best_key >> 32U);
        right = (uint32_t)(best_key & UINT64_C(0xffffffff));
        status = niyah_tokenizer_add_merge(tokenizer, left, right, &output);
        if (status != NIYAH_OK) {
            goto fail;
        }
        niyah_apply_merge(tokens, &token_count, left, right, output);
    }

    free(tokens);
    *out_tokenizer = tokenizer;
    return NIYAH_OK;

fail:
    free(tokens);
    niyah_tokenizer_destroy(tokenizer);
    return status;
}

void niyah_tokenizer_destroy(NiyahTokenizer *tokenizer)
{
    size_t i;
    if (tokenizer == NULL) {
        return;
    }
    if (tokenizer->vocab != NULL) {
        for (i = 0U; i < tokenizer->vocab_size; ++i) {
            free(tokenizer->vocab[i].data);
        }
    }
    free(tokenizer->merges);
    free(tokenizer->vocab);
    free(tokenizer);
}

size_t niyah_tokenizer_vocab_size(const NiyahTokenizer *tokenizer)
{
    return tokenizer == NULL ? 0U : tokenizer->vocab_size;
}

size_t niyah_tokenizer_merge_count(const NiyahTokenizer *tokenizer)
{
    return tokenizer == NULL ? 0U : tokenizer->merge_count;
}

NiyahStatus niyah_tokenizer_merge_at(const NiyahTokenizer *tokenizer,
                                     size_t merge_index,
                                     uint32_t *left_token,
                                     uint32_t *right_token,
                                     uint32_t *output_token)
{
    const NiyahMerge *merge;
    if (tokenizer == NULL || left_token == NULL || right_token == NULL ||
        output_token == NULL || merge_index >= tokenizer->merge_count) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    merge = &tokenizer->merges[merge_index];
    *left_token = merge->left;
    *right_token = merge->right;
    *output_token = merge->output;
    return NIYAH_OK;
}

NiyahStatus niyah_tokenizer_encode(const NiyahTokenizer *tokenizer,
                                   const uint8_t *input,
                                   size_t input_size,
                                   uint32_t *output_tokens,
                                   size_t output_capacity,
                                   size_t *output_count)
{
    uint32_t *work = NULL;
    size_t work_bytes = 0U;
    size_t count;
    size_t i;

    if (tokenizer == NULL || output_count == NULL || (input == NULL && input_size != 0U)) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    if (output_tokens == NULL && output_capacity != 0U) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    if (input_size == 0U) {
        *output_count = 0U;
        return NIYAH_OK;
    }
    if (!niyah_size_mul_ok(input_size, sizeof(uint32_t), &work_bytes)) {
        return NIYAH_ERR_OVERFLOW;
    }
    work = (uint32_t *)malloc(work_bytes);
    if (work == NULL) {
        return NIYAH_ERR_OUT_OF_MEMORY;
    }
    for (i = 0U; i < input_size; ++i) {
        work[i] = (uint32_t)input[i];
    }
    count = input_size;

    for (i = 0U; i < tokenizer->merge_count; ++i) {
        const NiyahMerge *merge = &tokenizer->merges[i];
        niyah_apply_merge(work, &count, merge->left, merge->right, merge->output);
    }

    *output_count = count;
    if (output_tokens == NULL) {
        free(work);
        return output_capacity == 0U ? NIYAH_OK : NIYAH_ERR_INVALID_ARGUMENT;
    }
    if (output_capacity < count) {
        free(work);
        return NIYAH_ERR_BUFFER_TOO_SMALL;
    }
    memcpy(output_tokens, work, count * sizeof(uint32_t));
    free(work);
    return NIYAH_OK;
}

NiyahStatus niyah_tokenizer_decode(const NiyahTokenizer *tokenizer,
                                   const uint32_t *tokens,
                                   size_t token_count,
                                   uint8_t *output_bytes,
                                   size_t output_capacity,
                                   size_t *output_size)
{
    size_t required = 0U;
    size_t cursor = 0U;
    size_t i;

    if (tokenizer == NULL || output_size == NULL || (tokens == NULL && token_count != 0U)) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }
    if (output_bytes == NULL && output_capacity != 0U) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    for (i = 0U; i < token_count; ++i) {
        const uint32_t token = tokens[i];
        if (token == NIYAH_TOKEN_BOS || token == NIYAH_TOKEN_EOS) {
            continue;
        }
        if ((size_t)token >= tokenizer->vocab_size) {
            return NIYAH_ERR_INVALID_ARGUMENT;
        }
        if (!niyah_size_add_ok(required, tokenizer->vocab[token].length, &required)) {
            return NIYAH_ERR_OVERFLOW;
        }
    }

    *output_size = required;
    if (output_bytes == NULL) {
        return output_capacity == 0U ? NIYAH_OK : NIYAH_ERR_INVALID_ARGUMENT;
    }
    if (output_capacity < required) {
        return NIYAH_ERR_BUFFER_TOO_SMALL;
    }

    for (i = 0U; i < token_count; ++i) {
        const uint32_t token = tokens[i];
        const NiyahTokenBytes *token_bytes;
        if (token == NIYAH_TOKEN_BOS || token == NIYAH_TOKEN_EOS) {
            continue;
        }
        token_bytes = &tokenizer->vocab[token];
        memcpy(output_bytes + cursor, token_bytes->data, token_bytes->length);
        cursor += token_bytes->length;
    }
    return NIYAH_OK;
}
