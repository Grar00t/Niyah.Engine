#ifndef NIYAH_NETWORK_IR_H
#define NIYAH_NETWORK_IR_H

#include "niyah/niyah.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum NiyahNetworkIrOp {
    NIYAH_NETWORK_IR_OP_INVALID = 0,
    NIYAH_NETWORK_IR_OP_IP_IN_CIDR = 1
} NiyahNetworkIrOp;

typedef struct NiyahNetworkIr {
    NiyahNetworkIrOp op;

    /*
     * IPv4 values are stored in canonical numeric form:
     * A.B.C.D => A<<24 | B<<16 | C<<8 | D.
     */
    uint32_t address;
    uint32_t network;
    uint8_t prefix_length;
} NiyahNetworkIr;

/*
 * Strict V1 syntax:
 *
 * IP_IN_CIDR|192.168.1.42|192.168.1.0|24
 *
 * Rules:
 * - ASCII only
 * - no whitespace
 * - no leading zeroes in octets/prefix except literal 0
 * - IPv4 octets 0..255
 * - prefix 0..32
 * - network operand must be canonical for its prefix
 */
NiyahStatus niyah_network_ir_parse(
    const char *text,
    NiyahNetworkIr *out_ir);

/*
 * out_match:
 *   0 => address is outside network
 *   1 => address is inside network
 */
NiyahStatus niyah_network_ir_execute(
    const NiyahNetworkIr *ir,
    int *out_match);

#ifdef __cplusplus
}
#endif

#endif
