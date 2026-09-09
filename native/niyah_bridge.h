#ifndef NIYAH_BRIDGE_H
#define NIYAH_BRIDGE_H

#include "niyah.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NiyahBridgeContext NiyahBridgeContext;

NIYAH_API const char *niyah_bridge_version(void);
NIYAH_API int32_t niyah_bridge_open(
    const char *conninfo,
    NiyahBridgeContext **out_context);
NIYAH_API void niyah_bridge_close(NiyahBridgeContext *context);
NIYAH_API char *niyah_bridge_search_json(
    NiyahBridgeContext *context,
    const char *query,
    int32_t max_hits);
NIYAH_API void niyah_bridge_free_string(char *text);

#ifdef __cplusplus
}
#endif
#endif
