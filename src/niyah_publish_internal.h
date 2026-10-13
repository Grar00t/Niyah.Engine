#ifndef NIYAH_PUBLISH_INTERNAL_H
#define NIYAH_PUBLISH_INTERNAL_H
#include "niyah/niyah.h"
#include <stddef.h>
#include <stdio.h>
typedef struct NiyahPublication {
    char *directory;
    char *path;
} NiyahPublication;
/* Private staging directory, exclusive publication. No final-path cleanup. */
NiyahStatus niyah_publish_begin(const char *destination, NiyahPublication *stage);
NiyahStatus niyah_publish_commit(NiyahPublication *stage, const char *destination, int replace);
NiyahStatus niyah_publish_cleanup(NiyahPublication *stage);
/* 0 absent, 1 present, -1 access/error. Never used to authorize a write. */
int niyah_publish_path_state(const char *path);
/* Native path contract: UTF-8 on Windows; filesystem bytes on POSIX. */
FILE *niyah_file_open(const char *path, const char *mode);
FILE *niyah_atomic_open(const char *destination, NiyahPublication *stage);
NiyahStatus niyah_atomic_finish(FILE *file, NiyahPublication *stage, const char *destination, NiyahStatus status);
/* Internal deterministic test seam; production never sets this callback. */
typedef int (*NiyahPublishFault)(const char *point);
void niyah_publish_set_fault(NiyahPublishFault callback);
int niyah_publish_fault(const char *point);
#endif
