#ifndef NIYAH_IR_H
#define NIYAH_IR_H

#include "niyah/niyah.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum NiyahIrOp {
    NIYAH_IR_OP_INVALID = 0,
    NIYAH_IR_OP_ADD = 1,
    NIYAH_IR_OP_SUB = 2,
    NIYAH_IR_OP_MUL = 3
} NiyahIrOp;

typedef struct NiyahIr {
    NiyahIrOp op;
    int64_t lhs;
    int64_t rhs;
} NiyahIr;

NiyahStatus niyah_ir_parse(
    const char *text,
    NiyahIr *out_ir);

NiyahStatus niyah_ir_execute(
    const NiyahIr *ir,
    int64_t *out_result);

#ifdef __cplusplus
}
#endif

#endif
