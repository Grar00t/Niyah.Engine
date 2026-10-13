#define _CRT_SECURE_NO_WARNINGS
#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200809L
#endif
#include "niyah_publish_internal.h"
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <direct.h>
#include <wchar.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#include <stdatomic.h>
#endif
#if defined(_WIN32)
static wchar_t *niyah_wide_path(const char *path)
{
    int count;
    wchar_t *wide;
    if (!path) { errno = EINVAL; return NULL; }
    count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, NULL, 0);
    if (count <= 0) { errno = EINVAL; return NULL; }
    wide = (wchar_t *)malloc((size_t)count * sizeof(*wide));
    if (!wide) { errno = ENOMEM; return NULL; }
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, wide, count) != count) {
        free(wide); errno = EINVAL; return NULL;
    }
    return wide;
}
#endif
FILE *niyah_file_open(const char *path, const char *mode)
{
#if defined(_WIN32)
    wchar_t *wide_path = niyah_wide_path(path);
    wchar_t *wide_mode = niyah_wide_path(mode);
    FILE *file = NULL;
    int saved_errno;
    if (wide_path && wide_mode) file = _wfopen(wide_path, wide_mode);
    saved_errno = errno;
    free(wide_path); free(wide_mode);
    errno = saved_errno;
    return file;
#else
    return fopen(path, mode);
#endif
}
static NiyahPublishFault fault_callback;
void niyah_publish_set_fault(NiyahPublishFault callback) { fault_callback = callback; }
int niyah_publish_fault(const char *point) { return fault_callback != NULL && fault_callback(point); }
int niyah_publish_path_state(const char *path)
{
#if defined(_WIN32)
    DWORD attr, error;
    wchar_t *wide;
    if (!path || !*path) return -1;
    wide = niyah_wide_path(path);
    if (!wide) return -1;
    attr = GetFileAttributesW(wide);
    error = GetLastError();
    free(wide);
    if (attr != INVALID_FILE_ATTRIBUTES) return 1;
    return error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND ? 0 : -1;
#else
    struct stat st;
    if (!path || !*path) return -1;
    if (lstat(path, &st) == 0) return 1;
    return errno == ENOENT ? 0 : -1;
#endif
}
NiyahStatus niyah_publish_begin(const char *destination, NiyahPublication *stage)
{
    size_t size;
    unsigned int attempt;
#if defined(_WIN32)
    static volatile LONG sequence;
    const unsigned long process = GetCurrentProcessId();
#else
    static atomic_ulong sequence;
    const unsigned long process = (unsigned long)getpid();
#endif
    if (!destination || !*destination || !stage || stage->directory || stage->path)
        return NIYAH_ERR_INVALID_ARGUMENT;
    if (strlen(destination) > SIZE_MAX - 112U) return NIYAH_ERR_OVERFLOW;
    size = strlen(destination) + 96U;
    stage->directory = (char *)malloc(size);
    stage->path = (char *)malloc(size + 16U);
    if (!stage->directory || !stage->path) {
        free(stage->directory); free(stage->path); memset(stage, 0, sizeof(*stage));
        return NIYAH_ERR_OUT_OF_MEMORY;
    }
    for (attempt = 0U; attempt < 128U; ++attempt) {
#if defined(_WIN32)
        unsigned long serial = (unsigned long)InterlockedIncrement(&sequence);
#else
        unsigned long serial = atomic_fetch_add(&sequence, 1UL) + 1UL;
#endif
        int n = snprintf(stage->directory, size, "%s.niyah-stage-%lu-%lu", destination, process, serial);
        if (n < 0 || (size_t)n >= size) break;
#if defined(_WIN32)
        wchar_t *wide = niyah_wide_path(stage->directory);
        DWORD create_error;
        BOOL created;
        if (!wide) break;
        created = CreateDirectoryW(wide, NULL);
        create_error = GetLastError();
        free(wide);
        if (created) {
#else
        if (mkdir(stage->directory, 0700) == 0) {
#endif
            (void)snprintf(stage->path, size + 16U, "%s/payload", stage->directory);
            return NIYAH_OK;
        }
#if defined(_WIN32)
        if (create_error != ERROR_ALREADY_EXISTS) break;
#else
        if (errno != EEXIST) break;
#endif
    }
    free(stage->directory); free(stage->path); memset(stage, 0, sizeof(*stage));
    return NIYAH_ERR_IO;
}
NiyahStatus niyah_publish_cleanup(NiyahPublication *stage)
{
    int failed = 0;
    if (!stage) return NIYAH_ERR_INVALID_ARGUMENT;
    if (stage->directory) {
        if (niyah_publish_fault("cleanup")) failed = 1;
        else {
#if defined(_WIN32)
            wchar_t *path = niyah_wide_path(stage->path);
            wchar_t *directory = niyah_wide_path(stage->directory);
            if (!path || !directory) failed = 1;
            if (path && _wremove(path) != 0 && errno != ENOENT) failed = 1;
            if (directory && _wrmdir(directory) != 0) failed = 1;
            free(path); free(directory);
#else
            if (stage->path && remove(stage->path) != 0 && errno != ENOENT) failed = 1;
            if (rmdir(stage->directory) != 0) failed = 1;
#endif
        }
        if (failed) fprintf(stderr, "publication_cleanup_failed=%s\n", stage->directory);
    }
    free(stage->directory); free(stage->path); memset(stage, 0, sizeof(*stage));
    return failed ? NIYAH_ERR_IO : NIYAH_OK;
}
NiyahStatus niyah_publish_commit(NiyahPublication *stage, const char *destination, int replace)
{
    if (!stage || !stage->path || !destination) return NIYAH_ERR_INVALID_ARGUMENT;
    if (niyah_publish_fault("publish")) return NIYAH_ERR_IO;
#if defined(_WIN32)
    {
        wchar_t *source = niyah_wide_path(stage->path);
        wchar_t *target = niyah_wide_path(destination);
        BOOL published = FALSE;
        if (source && target) published = replace
            ? MoveFileExW(source, target, MOVEFILE_REPLACE_EXISTING)
            : CreateHardLinkW(target, source, NULL);
        free(source); free(target);
        if (!published) return NIYAH_ERR_IO;
    }
#else
    if (replace) {
        if (rename(stage->path, destination) != 0) return NIYAH_ERR_IO;
    } else if (link(stage->path, destination) != 0) return NIYAH_ERR_IO;
#endif
    return NIYAH_OK;
}
FILE *niyah_atomic_open(const char *destination, NiyahPublication *stage)
{
    FILE *file;
    if (niyah_publish_begin(destination, stage) != NIYAH_OK) return NULL;
    file = niyah_file_open(stage->path, "wb");
    if (!file) (void)niyah_publish_cleanup(stage);
    return file;
}
NiyahStatus niyah_atomic_finish(FILE *file, NiyahPublication *stage, const char *destination, NiyahStatus status)
{
    if (fflush(file) != 0) status = NIYAH_ERR_IO;
    if (fclose(file) != 0) status = NIYAH_ERR_IO;
    if (niyah_publish_fault("partial_write")) status = NIYAH_ERR_IO;
    if (status == NIYAH_OK) status = niyah_publish_commit(stage, destination, 1);
    if (niyah_publish_cleanup(stage) != NIYAH_OK) status = NIYAH_ERR_IO;
    return status;
}
