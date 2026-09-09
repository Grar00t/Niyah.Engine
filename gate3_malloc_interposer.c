/* malloc_interposer.c — LD_PRELOAD shim to count and measure heap calls */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

static volatile long g_malloc_calls  = 0;
static volatile long g_calloc_calls  = 0;
static volatile long g_realloc_calls = 0;
static volatile long g_free_calls    = 0;
static volatile size_t g_peak_bytes  = 0;
static volatile size_t g_cur_bytes   = 0;

/* simple header-tag allocator */
typedef struct { size_t sz; unsigned magic; } Hdr;
#define MAGIC 0xDEAD1337u
#define HDR_SZ (sizeof(Hdr))

static void *(*real_malloc)(size_t)    = NULL;
static void *(*real_calloc)(size_t,size_t) = NULL;
static void *(*real_realloc)(void*,size_t) = NULL;
static void  (*real_free)(void*)       = NULL;

static void init_real(void) {
    if (!real_malloc) {
        real_malloc  = dlsym(RTLD_NEXT, "malloc");
        real_calloc  = dlsym(RTLD_NEXT, "calloc");
        real_realloc = dlsym(RTLD_NEXT, "realloc");
        real_free    = dlsym(RTLD_NEXT, "free");
    }
}

static void update_peak(void) {
    if (g_cur_bytes > g_peak_bytes) g_peak_bytes = g_cur_bytes;
}

void *malloc(size_t sz) {
    init_real();
    Hdr *p = (Hdr*)real_malloc(sz + HDR_SZ);
    if (!p) return NULL;
    p->sz = sz; p->magic = MAGIC;
    g_malloc_calls++;
    g_cur_bytes += sz;
    update_peak();
    return (char*)p + HDR_SZ;
}

void *calloc(size_t n, size_t sz) {
    init_real();
    size_t total = n * sz;
    Hdr *p = (Hdr*)real_malloc(total + HDR_SZ);
    if (!p) return NULL;
    memset(p, 0, total + HDR_SZ);
    p->sz = total; p->magic = MAGIC;
    g_calloc_calls++;
    g_cur_bytes += total;
    update_peak();
    return (char*)p + HDR_SZ;
}

void *realloc(void *ptr, size_t sz) {
    init_real();
    if (!ptr) { g_realloc_calls++; return malloc(sz); }
    Hdr *h = (Hdr*)((char*)ptr - HDR_SZ);
    if (h->magic == MAGIC) {
        size_t old = h->sz;
        Hdr *p = (Hdr*)real_realloc(h, sz + HDR_SZ);
        if (!p) return NULL;
        p->sz = sz; p->magic = MAGIC;
        g_realloc_calls++;
        g_cur_bytes = g_cur_bytes - old + sz;
        update_peak();
        return (char*)p + HDR_SZ;
    }
    return real_realloc(ptr, sz);
}

void free(void *ptr) {
    init_real();
    if (!ptr) return;
    Hdr *h = (Hdr*)((char*)ptr - HDR_SZ);
    if (h->magic == MAGIC) {
        g_free_calls++;
        g_cur_bytes -= h->sz;
        h->magic = 0;
        real_free(h);
    } else {
        real_free(ptr);
    }
}

__attribute__((destructor))
static void print_report(void) {
    fprintf(stderr,
        "\n=== HEAP REPORT ===\n"
        "MALLOC_CALLS=%ld\n"
        "CALLOC_CALLS=%ld\n"
        "REALLOC_CALLS=%ld\n"
        "FREE_CALLS=%ld\n"
        "PEAK_HEAP_BYTES=%zu\n"
        "LEAKED_BYTES=%zu\n"
        "===================\n",
        g_malloc_calls, g_calloc_calls, g_realloc_calls, g_free_calls,
        g_peak_bytes, g_cur_bytes);
}
