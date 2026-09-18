#include "niyah/router.h"
#include "niyah/network_admissibility.h"

#include <string.h>

NiyahStatus niyah_route_text(
    const char *text,
    NiyahRoute *out_route)
{
    NiyahNetworkIr network_ir;
    NiyahStatus status;

    if (text == NULL || out_route == NULL) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    memset(out_route, 0, sizeof(*out_route));

    status =
        niyah_network_admit_ip_in_cidr(
            text,
            &network_ir);

    if (status == NIYAH_OK) {
        out_route->kind =
            NIYAH_ROUTE_NETWORK_IP_IN_CIDR;

        out_route->network_ir =
            network_ir;

        return NIYAH_OK;
    }

    if (status == NIYAH_ERR_INVALID_ARGUMENT) {
        out_route->kind =
            NIYAH_ROUTE_NONE;

        return NIYAH_OK;
    }

    return status;
}
