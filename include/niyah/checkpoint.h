#ifndef NIYAH_CHECKPOINT_H
#define NIYAH_CHECKPOINT_H

#include "niyah/common.h"
#include "niyah/model.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct niyah_checkpoint {
    niyah_model model;
    uint64_t cursor;
    uint8_t dataset_sha256[32];
} niyah_checkpoint;

void niyah_checkpoint_init(niyah_checkpoint *ckpt);
niyah_status niyah_checkpoint_save(const niyah_checkpoint *ckpt, const char *path);
niyah_status niyah_checkpoint_load(const char *path, niyah_checkpoint *out);
niyah_status niyah_checkpoint_require_dataset(
    const niyah_checkpoint *ckpt,
    const uint8_t dataset_sha256[32]);

#ifdef __cplusplus
}
#endif

#endif
