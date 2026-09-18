#ifndef NIYAH_NETWORK_ADMISSIBILITY_H
#define NIYAH_NETWORK_ADMISSIBILITY_H

#include "niyah/network_ir.h"
#include "niyah/niyah.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Fail-closed V1 admission gate for the IP_IN_CIDR path.
 *
 * This layer is intentionally separate from network_slots:
 *
 *   admissibility = whether the user's text expresses
 *                   an IP-membership request
 *
 *   slots         = structural extraction of IPv4/CIDR values
 *
 * On success, out_ir contains validated typed network IR.
 *
 * NIYAH_OK:
 *   text was admitted and a valid IP_IN_CIDR IR was produced.
 *
 * NIYAH_ERR_INVALID_ARGUMENT:
 *   invalid API arguments, non-membership text, or text that
 *   does not contain structurally valid IP/CIDR operands.
 *
 * V1 recognized membership cues are deliberately narrow.
 */
NiyahStatus niyah_network_admit_ip_in_cidr(
    const char *text,
    NiyahNetworkIr *out_ir);

#ifdef __cplusplus
}
#endif

#endif
