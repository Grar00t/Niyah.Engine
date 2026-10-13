#ifndef NIYAH_PAIR_INTERNAL_H
#define NIYAH_PAIR_INTERNAL_H
#include "niyah/checkpoint.h"
#include "niyah/dataset.h"
NiyahStatus niyah_pair_save(const char *checkpoint, const char *cursor_path,
    const NiyahModel *model, const NiyahAdamWState *state,
    const NiyahAdamWConfig *config, const NiyahTokenizer *tokenizer,
    const NiyahDatasetCursor *cursor);
/* legacy=1 only when neither protocol sidecar exists. Resume still checks binding. */
NiyahStatus niyah_pair_validate(const char *checkpoint, const char *cursor_path, int *legacy);
#endif
