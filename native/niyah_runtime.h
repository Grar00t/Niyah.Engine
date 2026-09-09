#ifndef NIYAH_RUNTIME_H
#define NIYAH_RUNTIME_H

#include "niyah.h"

NiyahRuntime* niyah_runtime_create(const NiyahRuntimeConfig* config);
void niyah_runtime_destroy(NiyahRuntime* runtime);

NiyahStatus niyah_runtime_init_inplace(NiyahRuntime* runtime,
                                       const NiyahRuntimeConfig* config);
void niyah_runtime_deinit_inplace(NiyahRuntime* runtime);
NiyahStatus niyah_runtime_rewind(NiyahRuntime* runtime, size_t used);

#endif // NIYAH_RUNTIME_H
