#include "niyah_io.h"
#include <stdio.h>
#include <stdlib.h>

niyah_status niyah_read_file(const char *path, uint8_t **out_data, size_t *out_size) {
    FILE *f;
    long end;
    uint8_t *buf;
    size_t got;

    if (!path || !out_data || !out_size) return NIYAH_ERR_INVALID_ARGUMENT;
    *out_data = NULL;
    *out_size = 0;

    f = fopen(path, "rb");
    if (!f) return NIYAH_ERR_IO;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NIYAH_ERR_IO; }
    end = ftell(f);
    if (end < 0) { fclose(f); return NIYAH_ERR_IO; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NIYAH_ERR_IO; }

    if (end == 0) { fclose(f); return NIYAH_OK; }
    buf = (uint8_t *)malloc((size_t)end);
    if (!buf) { fclose(f); return NIYAH_ERR_NOMEM; }
    got = fread(buf, 1, (size_t)end, f);
    fclose(f);
    if (got != (size_t)end) { free(buf); return NIYAH_ERR_IO; }

    *out_data = buf;
    *out_size = got;
    return NIYAH_OK;
}

niyah_status niyah_write_file(const char *path, const void *data, size_t size) {
    FILE *f;
    size_t wrote;
    if (!path || (!data && size != 0)) return NIYAH_ERR_INVALID_ARGUMENT;
    f = fopen(path, "wb");
    if (!f) return NIYAH_ERR_IO;
    wrote = size ? fwrite(data, 1, size, f) : 0;
    if (fclose(f) != 0) return NIYAH_ERR_IO;
    return (wrote == size) ? NIYAH_OK : NIYAH_ERR_IO;
}
