#ifndef NIYAH_NETWORK_SLOTS_H
#define NIYAH_NETWORK_SLOTS_H

#include "niyah/niyah.h"
#include "niyah/network_ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Extract exactly:
 *   first standalone IPv4 address
 *   first IPv4/CIDR network
 *
 * from natural-language text.
 *
 * Output is validated through the existing strict network IR rules.
 */
NiyahStatus niyah_network_slots_extract_ip_in_cidr(
    const char *text,
    NiyahNetworkIr *out_ir);

#ifdef __cplusplus
}
#endif

#endif
