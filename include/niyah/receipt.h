#ifndef NIYAH_RECEIPT_H
#define NIYAH_RECEIPT_H

#include "niyah/ir.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NiyahExecutionReceipt {
    NiyahIr ir;
    int64_t result;
} NiyahExecutionReceipt;

NiyahStatus niyah_receipt_format(
    const NiyahExecutionReceipt *receipt,
    char *out,
    size_t out_size,
    size_t *out_length);

#ifdef __cplusplus
}
#endif

#endif
