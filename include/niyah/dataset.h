#ifndef NIYAH_DATASET_H
#define NIYAH_DATASET_H

#include "niyah/niyah.h"
#include "niyah/tokenizer.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NIYAH_DATASET_IDENTITY_SHA256_SIZE 32U
#define NIYAH_DATASET_CHECKPOINT_IDENTITY_SHA256_SIZE 32U

typedef struct NiyahDatasetCursor {
    size_t sample_count;
    size_t position;
    uint64_t seed;
    uint64_t epoch;
    uint8_t dataset_identity[NIYAH_DATASET_IDENTITY_SHA256_SIZE];
    int has_dataset_identity;
    uint8_t checkpoint_identity[NIYAH_DATASET_CHECKPOINT_IDENTITY_SHA256_SIZE];
    int has_checkpoint_identity;
    size_t *order;
} NiyahDatasetCursor;

NiyahStatus niyah_dataset_cursor_init(NiyahDatasetCursor *cursor,
                                      size_t sample_count,
                                      uint64_t seed);
void niyah_dataset_cursor_destroy(NiyahDatasetCursor *cursor);
NiyahStatus niyah_dataset_cursor_next(NiyahDatasetCursor *cursor,
                                      size_t *out_sample_index);

/* Rebuild deterministic order for an explicit epoch and position.
 * Higher-level training code uses this to roll a cursor back when a
 * selected sample fails before an optimizer step commits.
 */
NiyahStatus niyah_dataset_cursor_seek(NiyahDatasetCursor *cursor,
                                      uint64_t epoch,
                                      size_t position);

/* Bind sequencing state to an exact dataset content identity. Bound cursors
 * persist as NIYAHDST V2; unbound cursors retain V1 persistence.
 */
NiyahStatus niyah_dataset_cursor_bind_identity(
    NiyahDatasetCursor *cursor,
    const uint8_t identity[NIYAH_DATASET_IDENTITY_SHA256_SIZE]);

/* Bind a dataset-bound cursor to the exact checkpoint bytes that form the
 * other half of one persisted training-state pair. Such cursors persist as
 * NIYAHDST V3. Dataset-only cursors retain V2 persistence.
 */
NiyahStatus niyah_dataset_cursor_bind_checkpoint_identity(
    NiyahDatasetCursor *cursor,
    const uint8_t identity[NIYAH_DATASET_CHECKPOINT_IDENTITY_SHA256_SIZE]);

NiyahStatus niyah_dataset_cursor_save(const NiyahDatasetCursor *cursor,
                                      const char *path);
NiyahStatus niyah_dataset_cursor_load(const char *path,
                                      NiyahDatasetCursor *out_cursor);

#define NIYAH_DATASET_TOKENIZER_IDENTITY_SIZE 32U
#define NIYAH_DATASET_SHARD_IDENTITY_SHA256_SIZE 32U
#define NIYAH_DATASET_COLLECTION_IDENTITY_SHA256_SIZE 32U

typedef struct NiyahDatasetShard {
    uint32_t *tokens;
    size_t token_count;
    size_t sequence_length;
    size_t sample_count;
    uint8_t tokenizer_identity[NIYAH_DATASET_TOKENIZER_IDENTITY_SIZE];
} NiyahDatasetShard;

/* Deterministic text preprocessing for causal-LM training.
 *
 * The encoded stream is stored as:
 *   BOS, tokenizer(text), EOS
 *
 * sample i is a non-overlapping target span of at most sequence_length
 * transitions. Inputs and targets are zero-copy shifted views over the same
 * token storage: targets == tokens + 1.
 */
NiyahStatus niyah_dataset_shard_build_text(
    const NiyahTokenizer *tokenizer,
    const uint8_t *text,
    size_t text_size,
    size_t sequence_length,
    NiyahDatasetShard *out_shard);

void niyah_dataset_shard_destroy(NiyahDatasetShard *shard);

NiyahStatus niyah_dataset_shard_sample(
    const NiyahDatasetShard *shard,
    size_t sample_index,
    const uint32_t **out_tokens,
    const uint32_t **out_targets,
    size_t *out_token_count);

/* Stable content identity over tokenizer identity, sample geometry, and the
 * canonical token stream. This proves content identity, not semantic truth
 * or authenticity.
 */
NiyahStatus niyah_dataset_shard_identity_sha256(
    const NiyahDatasetShard *shard,
    uint8_t out_identity[NIYAH_DATASET_SHARD_IDENTITY_SHA256_SIZE]);

/* Stable identity for an ordered collection of loaded shards.
 * Reordering, adding, removing, or changing a shard changes the identity.
 * All shards must carry the same tokenizer identity.
 */
NiyahStatus niyah_dataset_collection_identity_sha256(
    const NiyahDatasetShard *shards,
    size_t shard_count,
    uint8_t out_identity[NIYAH_DATASET_COLLECTION_IDENTITY_SHA256_SIZE]);

/* Binary NIYAHSRD V1 persistence. The tokenizer SHA-256 identity is required
 * and checked on load. CRC32 detects accidental corruption only; it is not an
 * authenticity mechanism.
 */
NiyahStatus niyah_dataset_shard_save(
    const NiyahDatasetShard *shard,
    const NiyahTokenizer *tokenizer,
    const char *path);

NiyahStatus niyah_dataset_shard_load(
    const char *path,
    const NiyahTokenizer *tokenizer,
    NiyahDatasetShard *out_shard);

#ifdef __cplusplus
}
#endif

#endif
