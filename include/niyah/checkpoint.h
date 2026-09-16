#ifndef NIYAH_CHECKPOINT_H
#define NIYAH_CHECKPOINT_H

#include "niyah/optimizer.h"

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

#ifdef __cplusplus
}
#endif

#endif
