#include "niyah_bridge.h"
#include "store.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct NiyahBridgeContext {
    NiyahStore *store;
};

typedef struct {
    char *data;
    size_t length;
    size_t capacity;
    size_t row_count;
    int failed;
} JsonBuffer;

static int json_reserve(JsonBuffer *b, size_t extra)
{
    size_t required, next;
    char *grown;

    if (!b || b->failed) return 0;
    if (extra > SIZE_MAX - b->length - 1U) {
        b->failed = 1;
        return 0;
    }

    required = b->length + extra + 1U;
    if (required <= b->capacity) return 1;

    next = b->capacity ? b->capacity : 256U;
    while (next < required) {
        if (next > SIZE_MAX / 2U) {
            next = required;
            break;
        }
        next *= 2U;
    }

    grown = (char *)realloc(b->data, next);
    if (!grown) {
        b->failed = 1;
        return 0;
    }
    b->data = grown;
    b->capacity = next;
    return 1;
}

static int json_bytes(JsonBuffer *b, const char *s, size_t n)
{
    if (!b || (!s && n != 0U) || !json_reserve(b, n)) return 0;
    if (n) memcpy(b->data + b->length, s, n);
    b->length += n;
    b->data[b->length] = '\0';
    return 1;
}

static int json_literal(JsonBuffer *b, const char *s)
{
    return s ? json_bytes(b, s, strlen(s)) : 0;
}

static int json_char(JsonBuffer *b, char c)
{
    return json_bytes(b, &c, 1U);
}

static int json_string(JsonBuffer *b, const char *s)
{
    static const char hex[] = "0123456789abcdef";
    const unsigned char *p;

    if (!s) return json_literal(b, "null");
    if (!json_char(b, '"')) return 0;

    p = (const unsigned char *)s;
    while (*p) {
        const unsigned char c = *p++;
        char escaped[6];

        switch (c) {
            case '"':  if (!json_literal(b, "\\\"")) return 0; break;
            case '\\': if (!json_literal(b, "\\\\")) return 0; break;
            case '\b': if (!json_literal(b, "\\b")) return 0; break;
            case '\f': if (!json_literal(b, "\\f")) return 0; break;
            case '\n': if (!json_literal(b, "\\n")) return 0; break;
            case '\r': if (!json_literal(b, "\\r")) return 0; break;
            case '\t': if (!json_literal(b, "\\t")) return 0; break;
            default:
                if (c < 0x20U) {
                    escaped[0] = '\\'; escaped[1] = 'u';
                    escaped[2] = '0'; escaped[3] = '0';
                    escaped[4] = hex[c >> 4]; escaped[5] = hex[c & 0x0fU];
                    if (!json_bytes(b, escaped, sizeof(escaped))) return 0;
                } else {
                    const char raw = (char)c;
                    if (!json_bytes(b, &raw, 1U)) return 0;
                }
                break;
        }
    }

    return json_char(b, '"');
}

static int json_float(JsonBuffer *b, float value)
{
    char tmp[64];
    int n, i;

    if (!isfinite(value)) return 0;
    n = snprintf(tmp, sizeof(tmp), "%.9g", (double)value);
    if (n <= 0 || (size_t)n >= sizeof(tmp)) return 0;

    for (i = 0; i < n; ++i)
        if (tmp[i] == ',') tmp[i] = '.';

    return json_bytes(b, tmp, (size_t)n);
}

static int search_visitor(
    void *context,
    const char *chunk_id,
    const char *document_id,
    const char *heading,
    const char *chunk_text,
    float rank)
{
    JsonBuffer *b = (JsonBuffer *)context;

    if (!b || !chunk_id || !document_id || !chunk_text || !isfinite(rank))
        return 1;

    if (b->row_count && !json_char(b, ',')) return 1;

    if (!json_literal(b, "{\"chunk_id\":") ||
        !json_string(b, chunk_id) ||
        !json_literal(b, ",\"document_id\":") ||
        !json_string(b, document_id) ||
        !json_literal(b, ",\"heading\":") ||
        !json_string(b, heading) ||
        !json_literal(b, ",\"snippet\":") ||
        !json_string(b, chunk_text) ||
        !json_literal(b, ",\"score\":") ||
        !json_float(b, rank) ||
        !json_char(b, '}')) {
        return 1;
    }

    ++b->row_count;
    return 0;
}

const char *niyah_bridge_version(void)
{
    return NIYAH_VERSION;
}

int32_t niyah_bridge_open(const char *conninfo, NiyahBridgeContext **out_context)
{
    NiyahBridgeContext *ctx;
    NiyahStoreStatus status;

    if (!conninfo || conninfo[0] == '\0' || !out_context)
        return NIYAH_ERR_INVALID_ARG;

    *out_context = NULL;
    ctx = (NiyahBridgeContext *)calloc(1U, sizeof(*ctx));
    if (!ctx) return NIYAH_ERR_OUT_OF_MEMORY;

    status = niyah_store_open(conninfo, &ctx->store);
    if (status != NIYAH_STORE_OK) {
        free(ctx);
        return status == NIYAH_STORE_INVALID ? NIYAH_ERR_INVALID_ARG : NIYAH_ERR_IO;
    }

    status = niyah_store_init_schema(ctx->store);
    if (status != NIYAH_STORE_OK) {
        niyah_store_close(ctx->store);
        free(ctx);
        return NIYAH_ERR_IO;
    }

    *out_context = ctx;
    return NIYAH_OK;
}

void niyah_bridge_close(NiyahBridgeContext *context)
{
    if (!context) return;
    niyah_store_close(context->store);
    context->store = NULL;
    free(context);
}

char *niyah_bridge_search_json(
    NiyahBridgeContext *context,
    const char *query,
    int32_t max_hits)
{
    JsonBuffer b;
    NiyahStoreStatus status;
    size_t rows = 0U;

    if (!context || !context->store || !query ||
        max_hits < 1 || max_hits > 200) {
        return NULL;
    }

    memset(&b, 0, sizeof(b));
    if (!json_char(&b, '[')) {
        free(b.data);
        return NULL;
    }

    status = niyah_store_search_chunks(
        context->store, query, max_hits, search_visitor, &b, &rows);

    if (status != NIYAH_STORE_OK || rows != b.row_count) {
        free(b.data);
        return NULL;
    }

    if (!json_char(&b, ']')) {
        free(b.data);
        return NULL;
    }

    return b.data;
}

void niyah_bridge_free_string(char *text)
{
    free(text);
}
