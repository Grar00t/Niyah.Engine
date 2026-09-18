#ifndef NIYAH_NETWORK_SLOTS_H
#define NIYAH_NETWORK_SLOTS_H

#include "niyah/niyah.h"
#include "niyah/network_ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Extract exactly one standalone valid IPv4 address and exactly
 * one valid IPv4/CIDR network from the complete natural-language
 * text.
 *
 * Additional valid IPv4 or IPv4/CIDR operands are ambiguous and
 * fail closed with NIYAH_ERR_INVALID_ARGUMENT.
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
