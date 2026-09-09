#include "store.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <libpq-fe.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct NiyahStore {
    PGconn *db;
};

static NiyahStoreStatus map_pg_error(PGconn *db, PGresult *result)
{
    if (result) {
        const ExecStatusType status = PQresultStatus(result);
        if (status == PGRES_COMMAND_OK || status == PGRES_TUPLES_OK)
            return NIYAH_STORE_OK;

        const char *state = PQresultErrorField(result, PG_DIAG_SQLSTATE);
        if (state) {
            if (strncmp(state, "22", 2) == 0)
                return NIYAH_STORE_INVALID;

            if (strncmp(state, "23", 2) == 0 ||
                strncmp(state, "42", 2) == 0 ||
                strcmp(state, "3F000") == 0) {
                return NIYAH_STORE_SCHEMA;
            }

            if (strcmp(state, "40001") == 0 ||
                strcmp(state, "40P01") == 0 ||
                strcmp(state, "55P03") == 0) {
                return NIYAH_STORE_BUSY;
            }
        }
    }

    if (!db || PQstatus(db) != CONNECTION_OK)
        return NIYAH_STORE_IO;

    return NIYAH_STORE_IO;
}

static NiyahStoreStatus exec_params(
    NiyahStore *store,
    const char *sql,
    int count,
    const char *const *values)
{
    if (!store || !store->db || !sql || count < 0)
        return NIYAH_STORE_INVALID;

    PGresult *result = PQexecParams(
        store->db,
        sql,
        count,
        NULL,
        values,
        NULL,
        NULL,
        0);

    if (!result)
        return NIYAH_STORE_IO;

    const NiyahStoreStatus status = map_pg_error(store->db, result);
    PQclear(result);
    return status;
}

NiyahStoreStatus niyah_store_open(const char *location, NiyahStore **out_store)
{
    if (!location || location[0] == '\0' || !out_store)
        return NIYAH_STORE_INVALID;

    *out_store = NULL;

    NiyahStore *store = (NiyahStore *)calloc(1u, sizeof(*store));
    if (!store)
        return NIYAH_STORE_IO;

    store->db = PQconnectdb(location);
    if (!store->db || PQstatus(store->db) != CONNECTION_OK) {
        niyah_store_close(store);
        return NIYAH_STORE_IO;
    }

    if (PQsetClientEncoding(store->db, "UTF8") != 0) {
        niyah_store_close(store);
        return NIYAH_STORE_IO;
    }

    PGresult *result = PQexec(
        store->db,
        "SET application_name = 'niyah-engine';"
        "SET lock_timeout = '2500ms';");

    if (!result || PQresultStatus(result) != PGRES_COMMAND_OK) {
        if (result)
            PQclear(result);
        niyah_store_close(store);
        return NIYAH_STORE_IO;
    }

    PQclear(result);
    *out_store = store;
    return NIYAH_STORE_OK;
}

void niyah_store_close(NiyahStore *store)
{
    if (!store)
        return;
    if (store->db)
        PQfinish(store->db);
    free(store);
}

NiyahStoreStatus niyah_store_init_schema(NiyahStore *store)
{
    if (!store || !store->db)
        return NIYAH_STORE_INVALID;

    static const char sql[] =
        "SELECT "
        "  EXISTS (SELECT 1 FROM public.niyah_schema_migrations "
        "          WHERE version = '001_skg_core') "
        "  AND EXISTS (SELECT 1 FROM public.niyah_schema_migrations "
        "              WHERE version = '002_niyah_runtime') "
        "  AND EXISTS (SELECT 1 FROM public.niyah_schema_migrations "
        "              WHERE version = '003_skg_updated_at') "
        "  AND to_regclass('niyah.sources') IS NOT NULL "
        "  AND to_regclass('niyah.documents') IS NOT NULL "
        "  AND to_regclass('niyah.document_chunks') IS NOT NULL "
        "  AND to_regprocedure('niyah.search_chunks(text,integer)') IS NOT NULL;";

    PGresult *result = PQexec(store->db, sql);
    if (!result)
        return NIYAH_STORE_IO;

    NiyahStoreStatus status = map_pg_error(store->db, result);
    if (status == NIYAH_STORE_OK) {
        if (PQntuples(result) != 1 ||
            PQnfields(result) != 1 ||
            strcmp(PQgetvalue(result, 0, 0), "t") != 0) {
            status = NIYAH_STORE_SCHEMA;
        }
    }

    PQclear(result);
    return status;
}

NiyahStoreStatus niyah_store_insert_source(
    NiyahStore *store,
    const char *id,
    const char *canonical_uri,
    const char *title,
    const char *media_type,
    const char *language,
    const char *content_sha256,
    const char *source_kind)
{
    if (!store || !store->db || !id || !canonical_uri ||
        !content_sha256 || !source_kind) {
        return NIYAH_STORE_INVALID;
    }

    static const char sql[] =
        "INSERT INTO niyah.sources("
        "id,canonical_uri,title,media_type,language,content_sha256,source_kind"
        ") VALUES($1,$2,$3,$4,$5,$6,$7);";

    const char *values[] = {
        id,
        canonical_uri,
        title,
        media_type,
        language,
        content_sha256,
        source_kind
    };

    return exec_params(store, sql, 7, values);
}

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
    const char *status_value)
{
    if (!store || !store->db || !id || !source_id || !canonical_uri ||
        !content_sha256 || content_bytes < 0 || !retrieved_at ||
        !parser_version || !status_value) {
        return NIYAH_STORE_INVALID;
    }

    char content_bytes_text[32];
    if (snprintf(
            content_bytes_text,
            sizeof(content_bytes_text),
            "%lld",
            content_bytes) >= (int)sizeof(content_bytes_text)) {
        return NIYAH_STORE_INVALID;
    }

    static const char sql[] =
        "INSERT INTO niyah.documents("
        "id,source_id,canonical_uri,title,media_type,language,content_sha256,"
        "content_bytes,retrieved_at,parser_version,status"
        ") VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11);";

    const char *values[] = {
        id,
        source_id,
        canonical_uri,
        title,
        media_type,
        language,
        content_sha256,
        content_bytes_text,
        retrieved_at,
        parser_version,
        status_value
    };

    return exec_params(store, sql, 11, values);
}

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
    long long token_count)
{
    if (!store || !store->db || !id || !document_id || ordinal < 0 ||
        start_offset < 0 || end_offset < start_offset || !text ||
        !text_sha256 || token_count < 0) {
        return NIYAH_STORE_INVALID;
    }

    char ordinal_text[32];
    char start_text[32];
    char end_text[32];
    char token_count_text[32];

    if (snprintf(ordinal_text, sizeof(ordinal_text), "%lld", ordinal) >= (int)sizeof(ordinal_text) ||
        snprintf(start_text, sizeof(start_text), "%lld", start_offset) >= (int)sizeof(start_text) ||
        snprintf(end_text, sizeof(end_text), "%lld", end_offset) >= (int)sizeof(end_text) ||
        snprintf(token_count_text, sizeof(token_count_text), "%lld", token_count) >= (int)sizeof(token_count_text)) {
        return NIYAH_STORE_INVALID;
    }

    static const char sql[] =
        "INSERT INTO niyah.document_chunks("
        "id,document_id,ordinal,start_offset,end_offset,heading,text,text_sha256,token_count"
        ") VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9);";

    const char *values[] = {
        id,
        document_id,
        ordinal_text,
        start_text,
        end_text,
        heading,
        text,
        text_sha256,
        token_count_text
    };

    return exec_params(store, sql, 9, values);
}

NiyahStoreStatus niyah_store_insert_claim(
    NiyahStore *store,
    const char *id,
    const char *chunk_id,
    const char *claim_text,
    const char *claim_sha256,
    const char *classification,
    const char *extractor_version,
    const char *created_at)
{
    if (!store || !store->db || !id || !chunk_id || !claim_text ||
        !claim_sha256 || !classification || !extractor_version || !created_at) {
        return NIYAH_STORE_INVALID;
    }

    static const char sql[] =
        "INSERT INTO niyah.claims("
        "id,chunk_id,claim_text,claim_sha256,classification,extractor_version,created_at"
        ") VALUES($1,$2,$3,$4,$5,$6,$7);";

    const char *values[] = {
        id,
        chunk_id,
        claim_text,
        claim_sha256,
        classification,
        extractor_version,
        created_at
    };

    return exec_params(store, sql, 7, values);
}

static int has_non_space(const char *text)
{
    const unsigned char *p;
    if (!text) return 0;
    p = (const unsigned char *)text;
    while (*p) {
        if (!isspace(*p)) return 1;
        ++p;
    }
    return 0;
}

NiyahStoreStatus niyah_store_search_chunks(
    NiyahStore *store,
    const char *query,
    int limit,
    NiyahStoreChunkVisitor visitor,
    void *visitor_context,
    size_t *out_count)
{
    static const char sql[] =
        "SELECT chunk_id, document_id, heading, chunk_text, rank "
        "FROM niyah.search_chunks($1, $2::integer);";

    char limit_text[16];
    const char *values[2];
    PGresult *result;
    NiyahStoreStatus status;
    size_t count = 0U;
    int rows;
    int row;

    if (out_count) *out_count = 0U;

    if (!store || !store->db || !query || !has_non_space(query) ||
        limit < 1 || limit > 200 || !visitor) {
        return NIYAH_STORE_INVALID;
    }

    if (snprintf(limit_text, sizeof(limit_text), "%d", limit) < 0)
        return NIYAH_STORE_INVALID;

    values[0] = query;
    values[1] = limit_text;

    result = PQexecParams(store->db, sql, 2, NULL, values, NULL, NULL, 0);
    if (!result)
        return NIYAH_STORE_IO;

    status = map_pg_error(store->db, result);
    if (status != NIYAH_STORE_OK) {
        PQclear(result);
        return status;
    }

    if (PQnfields(result) != 5) {
        PQclear(result);
        return NIYAH_STORE_SCHEMA;
    }

    rows = PQntuples(result);

    for (row = 0; row < rows; ++row) {
        const char *chunk_id;
        const char *document_id;
        const char *heading;
        const char *chunk_text;
        const char *rank_text;
        char *rank_end = NULL;
        float rank;

        if (PQgetisnull(result, row, 0) ||
            PQgetisnull(result, row, 1) ||
            PQgetisnull(result, row, 3) ||
            PQgetisnull(result, row, 4)) {
            PQclear(result);
            return NIYAH_STORE_SCHEMA;
        }

        chunk_id = PQgetvalue(result, row, 0);
        document_id = PQgetvalue(result, row, 1);
        heading = PQgetisnull(result, row, 2) ? NULL : PQgetvalue(result, row, 2);
        chunk_text = PQgetvalue(result, row, 3);
        rank_text = PQgetvalue(result, row, 4);

        errno = 0;
        rank = strtof(rank_text, &rank_end);
        if (errno == ERANGE || rank_end == rank_text || !rank_end ||
            *rank_end != '\0' || !isfinite(rank)) {
            PQclear(result);
            return NIYAH_STORE_SCHEMA;
        }

        if (visitor(visitor_context, chunk_id, document_id, heading, chunk_text, rank) != 0) {
            PQclear(result);
            return NIYAH_STORE_ABORTED;
        }
        ++count;
    }

    PQclear(result);
    if (out_count) *out_count = count;
    return NIYAH_STORE_OK;
}
