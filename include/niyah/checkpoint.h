#ifndef NIYAH_CHECKPOINT_H
#define NIYAH_CHECKPOINT_H

#include "niyah/optimizer.h"
#include "niyah/tokenizer.h"

#include <stdint.h>

#define NIYAH_CHECKPOINT_IDENTITY_SHA256_SIZE 32U

#ifdef __cplusplus
extern "C" {
#endif

NiyahStatus niyah_checkpoint_save(const char *path,
                                  const NiyahModel *model,
                                  const NiyahAdamWState *optimizer_state,
                                  const NiyahAdamWConfig *optimizer_config);

NiyahStatus niyah_checkpoint_load(const char *path,
                                  NiyahModel *out_model,
                                  NiyahAdamWState *out_optimizer_state,
                                  NiyahAdamWConfig *out_optimizer_config);

NiyahStatus niyah_checkpoint_save_with_tokenizer(
    const char *path,
    const NiyahModel *model,
    const NiyahAdamWState *optimizer_state,
    const NiyahAdamWConfig *optimizer_config,
    const NiyahTokenizer *tokenizer);

NiyahStatus niyah_checkpoint_load_with_tokenizer(
    const char *path,
    const NiyahTokenizer *tokenizer,
    NiyahModel *out_model,
    NiyahAdamWState *out_optimizer_state,
    NiyahAdamWConfig *out_optimizer_config);

/* SHA-256 identity of the exact persisted checkpoint bytes. This provides
 * content identity for checkpoint/cursor pairing; it does not authenticate
 * the checkpoint or establish semantic truth.
 */
NiyahStatus niyah_checkpoint_identity_sha256(
    const char *path,
    uint8_t out_identity[NIYAH_CHECKPOINT_IDENTITY_SHA256_SIZE]);

#ifdef __cplusplus
}
#endif

#endif
