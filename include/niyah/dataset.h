#ifndef NIYAH_DATASET_H
#define NIYAH_DATASET_H

#include "niyah/common.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct niyah_dataset {
    uint32_t *tokens;
    uint64_t token_count;
    uint8_t content_sha256[32];
} niyah_dataset;

void niyah_dataset_init(niyah_dataset *dataset);
void niyah_dataset_free(niyah_dataset *dataset);
niyah_status niyah_dataset_from_tokens(
    const uint32_t *tokens,
    uint64_t token_count,
    niyah_dataset *out);
niyah_status niyah_dataset_save(const niyah_dataset *dataset, const char *path);
niyah_status niyah_dataset_load(const char *path, niyah_dataset *out);

#ifdef __cplusplus
}
#endif

#endif
