#ifndef NIYAH_WINDOW_FORMAT_H
#define NIYAH_WINDOW_FORMAT_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define NIYAH_WINDOW_HEADER_SIZE 64U
#define NIYAH_WINDOW_VERSION 1U

typedef struct {
    unsigned char magic[8];
    uint32_t version;
    uint32_t sequence_length;
    uint32_t vocabulary_size;
    uint32_t flags;
    uint64_t window_count;
    unsigned char corpus_sha256[32];
} NiyahWindowHeader;

static uint32_t niyah_window_le32(const unsigned char *p)
{
    return ((uint32_t)p[0]) |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint64_t niyah_window_le64(const unsigned char *p)
{
    return ((uint64_t)p[0]) |
           ((uint64_t)p[1] << 8) |
           ((uint64_t)p[2] << 16) |
           ((uint64_t)p[3] << 24) |
           ((uint64_t)p[4] << 32) |
           ((uint64_t)p[5] << 40) |
           ((uint64_t)p[6] << 48) |
           ((uint64_t)p[7] << 56);
}

static int niyah_window_header_read(FILE *stream, NiyahWindowHeader *h)
{
    unsigned char raw[NIYAH_WINDOW_HEADER_SIZE];

    if (!stream || !h)
        return 0;

    if (fread(raw, 1U, sizeof(raw), stream) != sizeof(raw))
        return 0;

    if (memcmp(raw, "NIYAHW1", 7U) != 0 || raw[7] != '\0')
        return 0;

    memset(h, 0, sizeof(*h));
    memcpy(h->magic, raw, 8U);
    h->version = niyah_window_le32(raw + 8U);
    h->sequence_length = niyah_window_le32(raw + 12U);
    h->vocabulary_size = niyah_window_le32(raw + 16U);
    h->flags = niyah_window_le32(raw + 20U);
    h->window_count = niyah_window_le64(raw + 24U);
    memcpy(h->corpus_sha256, raw + 32U, 32U);
    return 1;
}

#endif
