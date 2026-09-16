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

/* Rebuild deterministic order for an explicit epoch and position.
 * Higher-level training code uses this to roll a cursor back when a
 * selected sample fails before an optimizer step commits.
 */
NiyahStatus niyah_dataset_cursor_seek(NiyahDatasetCursor *cursor,
                                      uint64_t epoch,
                                      size_t position);

NiyahStatus niyah_dataset_cursor_save(const NiyahDatasetCursor *cursor,
                                      const char *path);
NiyahStatus niyah_dataset_cursor_load(const char *path,
                                      NiyahDatasetCursor *out_cursor);

#ifdef __cplusplus
}
#endif

#endif
