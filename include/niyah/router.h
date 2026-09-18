#ifndef NIYAH_ROUTER_H
#define NIYAH_ROUTER_H

#include "niyah/network_ir.h"
#include "niyah/niyah.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum NiyahRouteKind {
    NIYAH_ROUTE_NONE = 0,
    NIYAH_ROUTE_NETWORK_IP_IN_CIDR = 1
} NiyahRouteKind;

typedef struct NiyahRoute {
    NiyahRouteKind kind;
    NiyahNetworkIr network_ir;
} NiyahRoute;

/*
 * Route natural-language text into a supported deterministic
 * native execution domain.
 *
 * NIYAH_OK does not imply that a route exists.
 *
 * NIYAH_ROUTE_NONE:
 *   text does not qualify for any currently supported
 *   deterministic native route.
 *
 * NIYAH_ROUTE_NETWORK_IP_IN_CIDR:
 *   network_ir contains validated typed network IR.
 */
NiyahStatus niyah_route_text(
    const char *text,
    NiyahRoute *out_route);

#ifdef __cplusplus
}
#endif

#endif
