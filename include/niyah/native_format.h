#ifndef NIYAH_NATIVE_FORMAT_H
#define NIYAH_NATIVE_FORMAT_H

#include "niyah/native_execute.h"
#include "niyah/niyah.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Canonically format a completed native execution result.
 *
 * out_length receives the number of bytes required/written,
 * excluding the terminating NUL.
 *
 * Query mode:
 *   out_text == NULL
 *   out_capacity == 0
 *
 * returns NIYAH_OK and reports the required length.
 *
 * NIYAH_ERR_BUFFER_TOO_SMALL:
 *   output buffer cannot hold formatted text plus NUL.
 */
NiyahStatus niyah_native_execution_format(
    const NiyahNativeExecutionResult *result,
    char *out_text,
    size_t out_capacity,
    size_t *out_length);

#ifdef __cplusplus
}
#endif

#endif
