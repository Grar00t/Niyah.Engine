#undef NDEBUG
#include <assert.h>
#include <libpq-fe.h>
#include <stdio.h>
#include <stdlib.h>

#include "store.h"

static int exec_ok(PGconn *db, const char *sql)
{
    PGresult *result = PQexec(db, sql);
    if (!result)
        return 0;
    const int ok = PQresultStatus(result) == PGRES_COMMAND_OK;
    PQclear(result);
    return ok;
}

int main(void)
{
    const char *conninfo = getenv("NIYAH_TEST_PG_CONNINFO");
    if (!conninfo || conninfo[0] == '\0') {
        fprintf(stderr, "SKIP: NIYAH_TEST_PG_CONNINFO is not set\n");
        return 77;
    }

    NiyahStore *store = NULL;
    assert(niyah_store_open(conninfo, &store) == NIYAH_STORE_OK);
    assert(store != NULL);
    assert(niyah_store_init_schema(store) == NIYAH_STORE_OK);

    PGconn *cleanup = PQconnectdb(conninfo);
    assert(cleanup != NULL);
    assert(PQstatus(cleanup) == CONNECTION_OK);
    assert(exec_ok(cleanup, "DELETE FROM niyah.sources WHERE id='src_store_test';"));

    assert(niyah_store_insert_source(
               store,
               "src_store_test",
               "local://fixture",
               "Fixture",
               "text/plain",
               "en",
               "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1",
               "local") == NIYAH_STORE_OK);

    assert(niyah_store_insert_source(
               store,
               "src_store_test",
               "local://duplicate",
               NULL,
               NULL,
               NULL,
               "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb1",
               "local") == NIYAH_STORE_SCHEMA);

    assert(niyah_store_insert_document(
               store,
               "doc_store_test",
               "src_store_test",
               "local://fixture/doc",
               "Document",
               "text/plain",
               "en",
               "ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc1",
               12,
               "2026-09-07T00:00:00Z",
               "parser-1",
               "indexed") == NIYAH_STORE_OK);

    assert(niyah_store_insert_document(
               store,
               "doc_orphan_store_test",
               "missing_source",
               "local://orphan",
               NULL,
               "text/plain",
               NULL,
               "ddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd1",
               1,
               "2026-09-07T00:00:00Z",
               "parser-1",
               "indexed") == NIYAH_STORE_SCHEMA);

    assert(niyah_store_insert_chunk(
               store,
               "chunk_store_test",
               "doc_store_test",
               0,
               0,
               12,
               "Heading",
               "hello world!",
               "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee1",
               3) == NIYAH_STORE_OK);

    assert(niyah_store_insert_chunk(
               store,
               "chunk_bad_store_test",
               "doc_store_test",
               1,
               15,
               14,
               NULL,
               "bad",
               "fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff1",
               1) == NIYAH_STORE_INVALID);

    assert(niyah_store_insert_claim(
               store,
               "claim_store_test",
               "chunk_store_test",
               "Tokyo is the capital of Japan.",
               "1111111111111111111111111111111111111111111111111111111111111112",
               "FACT",
               "extractor-1",
               "2026-09-07T00:00:00Z") == NIYAH_STORE_OK);

    assert(niyah_store_insert_claim(
               store,
               "claim_bad_store_test",
               "chunk_store_test",
               "invalid classification",
               "2222222222222222222222222222222222222222222222222222222222222223",
               "VERIFIED",
               "extractor-1",
               "2026-09-07T00:00:00Z") == NIYAH_STORE_SCHEMA);

    niyah_store_close(store);

    assert(exec_ok(cleanup, "DELETE FROM niyah.sources WHERE id='src_store_test';"));
    PQfinish(cleanup);

    assert(niyah_store_open(NULL, &store) == NIYAH_STORE_INVALID);
    assert(niyah_store_open("", &store) == NIYAH_STORE_INVALID);
    assert(niyah_store_open(conninfo, NULL) == NIYAH_STORE_INVALID);
    return 0;
}
