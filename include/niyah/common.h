#ifndef NIYAH_COMMON_H
#define NIYAH_COMMON_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum niyah_status {
    NIYAH_OK = 0,
    NIYAH_ERR_INVALID = 1,
    NIYAH_ERR_IO = 2,
    NIYAH_ERR_FORMAT = 3,
    NIYAH_ERR_NOMEM = 4,
    NIYAH_ERR_MISMATCH = 5,
    NIYAH_ERR_STATE = 6
} niyah_status;

const char *niyah_status_string(niyah_status status);

#ifdef __cplusplus
}
#endif

#endif
