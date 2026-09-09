#ifndef NIYAH_STORE_H
#define NIYAH_STORE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NiyahStore NiyahStore;

typedef enum {
    NIYAH_STORE_OK = 0,
    NIYAH_STORE_INVALID = 1,
    NIYAH_STORE_IO = 2,
    NIYAH_STORE_SCHEMA = 3,
    NIYAH_STORE_BUSY = 4,
    NIYAH_STORE_ABORTED = 5
} NiyahStoreStatus;

/* Strings are libpq-owned and valid only for the callback duration. */
typedef int (*NiyahStoreChunkVisitor)(
    void *context,
    const char *chunk_id,
    const char *document_id,
    const char *heading,
    const char *chunk_text,
    float rank);

NiyahStoreStatus niyah_store_open(const char *location, NiyahStore **out_store);
void niyah_store_close(NiyahStore *store);
NiyahStoreStatus niyah_store_init_schema(NiyahStore *store);

NiyahStoreStatus niyah_store_insert_source(
    NiyahStore *store,
    const char *id,
    const char *canonical_uri,
    const char *title,
    const char *media_type,
    const char *language,
    const char *content_sha256,
    const char *source_kind);

NiyahStoreStatus niyah_store_insert_document(
    NiyahStore *store,
    const char *id,
    const char *source_id,
    const char *canonical_uri,
    const char *title,
    const char *media_type,
    const char *language,
    const char *content_sha256,
    long long content_bytes,
    const char *retrieved_at,
    const char *parser_version,
    const char *status);

NiyahStoreStatus niyah_store_insert_chunk(
    NiyahStore *store,
    const char *id,
    const char *document_id,
    long long ordinal,
    long long start_offset,
    long long end_offset,
    const char *heading,
    const char *text,
    const char *text_sha256,
    long long token_count);

NiyahStoreStatus niyah_store_insert_claim(
    NiyahStore *store,
    const char *id,
    const char *chunk_id,
    const char *claim_text,
    const char *claim_sha256,
    const char *classification,
    const char *extractor_version,
    const char *created_at);

NiyahStoreStatus niyah_store_search_chunks(
    NiyahStore *store,
    const char *query,
    int limit,
    NiyahStoreChunkVisitor visitor,
    void *visitor_context,
    size_t *out_count);

#ifdef __cplusplus
}
#endif
#endif
