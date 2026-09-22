#ifndef NIYAH_TRAIN_H
#define NIYAH_TRAIN_H

#include "niyah/checkpoint.h"
#include "niyah/common.h"
#include "niyah/dataset.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct niyah_train_result {
    uint64_t updates_requested;
    uint64_t updates_applied;
    uint64_t cursor_before;
    uint64_t cursor_after;
} niyah_train_result;

niyah_status niyah_train_new(
    const niyah_dataset *dataset,
    uint64_t updates,
    niyah_checkpoint *out,
    niyah_train_result *result);
niyah_status niyah_train_resume(
    const niyah_dataset *dataset,
    niyah_checkpoint *checkpoint,
    uint64_t updates,
    niyah_train_result *result);

#ifdef __cplusplus
}
#endif

#endif
