#ifndef NIYAH_NATIVE_EXECUTE_H
#define NIYAH_NATIVE_EXECUTE_H

#include "niyah/router.h"
#include "niyah/niyah.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NiyahNativeExecutionResult {
    NiyahRouteKind route_kind;

    /*
     * Valid only when:
     * route_kind == NIYAH_ROUTE_NETWORK_IP_IN_CIDR
     *
     * 0 => address is outside network
     * 1 => address is inside network
     */
    int network_match;
} NiyahNativeExecutionResult;

/*
 * Execute a supported deterministic native route directly
 * from natural-language text.
 *
 * NIYAH_OK does not imply that a native route existed.
 *
 * route_kind == NIYAH_ROUTE_NONE:
 *   text had no supported deterministic native route.
 *
 * route_kind == NIYAH_ROUTE_NETWORK_IP_IN_CIDR:
 *   network_match contains the deterministic result.
 */
NiyahStatus niyah_native_execute_text(
    const char *text,
    NiyahNativeExecutionResult *out_result);

#ifdef __cplusplus
}
#endif

#endif
