#include "niyah/native_execute.h"
#include "niyah/network_ir.h"

#include <string.h>

NiyahStatus niyah_native_execute_text(
    const char *text,
    NiyahNativeExecutionResult *out_result)
{
    NiyahRoute route;
    NiyahStatus status;

    if (text == NULL || out_result == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    memset(out_result, 0, sizeof(*out_result));

    status =
        niyah_route_text(
            text,
            &route);

    if (status != NIYAH_OK) {
        return status;
    }

    out_result->route_kind =
        route.kind;

    switch (route.kind) {
        case NIYAH_ROUTE_NONE:
            return NIYAH_OK;

        case NIYAH_ROUTE_NETWORK_IP_IN_CIDR:
            out_result->network_ir =
                route.network_ir;

            return niyah_network_ir_execute(
                &out_result->network_ir,
                &out_result->network_match);

        default:
            return NIYAH_ERR_INVALID_ARGUMENT;
    }
}
