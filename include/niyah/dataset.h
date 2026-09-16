#ifndef NIYAH_DATASET_H
#define NIYAH_DATASET_H

#include "niyah/niyah.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NiyahDatasetCursor {
    size_t sample_count;
    size_t position;
    uint64_t seed;
    uint64_t epoch;
    size_t *order;
} NiyahDatasetCursor;

NiyahStatus niyah_dataset_cursor_init(NiyahDatasetCursor *cursor,
                                      size_t sample_count,
                                      uint64_t seed);
void niyah_dataset_cursor_destroy(NiyahDatasetCursor *cursor);
NiyahStatus niyah_dataset_cursor_next(NiyahDatasetCursor *cursor,
                                      size_t *out_sample_index);
NiyahStatus niyah_dataset_cursor_save(const NiyahDatasetCursor *cursor,
                                      const char *path);
NiyahStatus niyah_dataset_cursor_load(const char *path,
                                      NiyahDatasetCursor *out_cursor);

#ifdef __cplusplus
}
#endif

#endif
