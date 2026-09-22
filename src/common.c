#include "niyah/common.h"

const char *niyah_status_string(niyah_status status) {
    switch (status) {
        case NIYAH_OK: return "ok";
        case NIYAH_ERR_INVALID: return "invalid_argument";
        case NIYAH_ERR_IO: return "io_error";
        case NIYAH_ERR_FORMAT: return "format_error";
        case NIYAH_ERR_NOMEM: return "out_of_memory";
        case NIYAH_ERR_MISMATCH: return "identity_mismatch";
        case NIYAH_ERR_STATE: return "invalid_state";
        default: return "unknown_error";
    }
}
