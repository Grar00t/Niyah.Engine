#undef NDEBUG
#include <assert.h>

#include "local_store.h"

int main(void)
{
    NiyahStore *store = NULL;
    assert(niyah_store_open(":memory:", &store) == NIYAH_STORE_OK);
    assert(store != NULL);
    assert(niyah_store_init_schema(store) == NIYAH_STORE_OK);
    assert(niyah_store_init_schema(store) == NIYAH_STORE_OK);

    assert(niyah_store_insert_source(
               store,
               "src_1",
               "local://fixture",
               "Fixture",
               "text/plain",
               "en",
               "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
               "local") == NIYAH_STORE_OK);

    assert(niyah_store_insert_source(
               store,
               "src_1",
               "local://duplicate",
               NULL,
               NULL,
               NULL,
               "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
               "local") == NIYAH_STORE_SCHEMA);

    assert(niyah_store_insert_document(
               store,
               "doc_1",
               "src_1",
               "local://fixture/doc",
               "Document",
               "text/plain",
               "en",
               "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
               12,
               "2026-09-07T00:00:00Z",
               "parser-1",
               "indexed") == NIYAH_STORE_OK);

    assert(niyah_store_insert_document(
               store,
               "doc_orphan",
               "missing_source",
               "local://orphan",
               NULL,
               "text/plain",
               NULL,
               "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
               1,
               "2026-09-07T00:00:00Z",
               "parser-1",
               "indexed") == NIYAH_STORE_SCHEMA);

    assert(niyah_store_insert_chunk(
               store,
               "chunk_1",
               "doc_1",
               0,
               0,
               12,
               "Heading",
               "hello world!",
               "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
               3) == NIYAH_STORE_OK);

    assert(niyah_store_insert_chunk(
               store,
               "chunk_bad",
               "doc_1",
               1,
               15,
               14,
               NULL,
               "bad",
               "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
               1) == NIYAH_STORE_INVALID);

    assert(niyah_store_insert_claim(
               store,
               "claim_1",
               "chunk_1",
               "Tokyo is the capital of Japan.",
               "1111111111111111111111111111111111111111111111111111111111111111",
               "FACT",
               "extractor-1",
               "2026-09-07T00:00:00Z") == NIYAH_STORE_OK);

    assert(niyah_store_insert_claim(
               store,
               "claim_2",
               "chunk_1",
               "invalid classification",
               "2222222222222222222222222222222222222222222222222222222222222222",
               "VERIFIED",
               "extractor-1",
               "2026-09-07T00:00:00Z") == NIYAH_STORE_SCHEMA);

    niyah_store_close(store);

    assert(niyah_store_open(NULL, &store) == NIYAH_STORE_INVALID);
    assert(niyah_store_open("", &store) == NIYAH_STORE_INVALID);
    assert(niyah_store_open(":memory:", NULL) == NIYAH_STORE_INVALID);
    return 0;
}
