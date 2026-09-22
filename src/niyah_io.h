#ifndef NIYAH_IO_H
#define NIYAH_IO_H

#include <stddef.h>
#include <stdint.h>
#include "niyah/niyah.h"

niyah_status niyah_read_file(const char *path, uint8_t **out_data, size_t *out_size);
niyah_status niyah_write_file(const char *path, const void *data, size_t size);

#endif
