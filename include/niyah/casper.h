#ifndef NIYAH_CASPER_H
#define NIYAH_CASPER_H

#include "niyah/niyah.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Map a Casper/NIYAH v0x0005 raw model read-only without copying
 * or reordering tensors. The mapped model is inference-only.
 */
NiyahStatus niyah_casper_mmap_load(const char *path, NiyahModel *out_model);

#ifdef __cplusplus
}
#endif
#endif
