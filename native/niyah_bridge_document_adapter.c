#include "niyah_bridge.h"

static NiyahStatus legacy_get(
    void* implementation,
    const NiyahDocumentContext* context,
    const char* doc_id,
    char** out_content)
{
    char* content;

    (void)implementation;
    (void)context;

    if (!doc_id || !doc_id[0] || !out_content) {
        return NIYAH_ERR_INVALID_ARG;
    }

    *out_content = NULL;
    content = niyah_bridge_get_document(doc_id);
    if (!content) {
        return NIYAH_ERR_NOT_FOUND;
    }

    *out_content = content;
    return NIYAH_OK;
}

static NiyahStatus legacy_search(
    void* implementation,
    const NiyahDocumentContext* context,
    const char* query,
    NiyahDocumentSearchResults** out_results)
{
    void* raw = NULL;
    int count = 0;
    int32_t status;

    (void)implementation;
    (void)context;

    if (!query || !out_results) {
        return NIYAH_ERR_INVALID_ARG;
    }

    *out_results = NULL;
    status = niyah_bridge_search(query, &raw, &count);
    if (status != NIYAH_OK) {
        return (NiyahStatus)status;
    }

    *out_results = (NiyahDocumentSearchResults*)raw;
    return NIYAH_OK;
}

static NiyahStatus legacy_count(
    void* implementation,
    const NiyahDocumentContext* context,
    int32_t* out_count)
{
    (void)implementation;
    (void)context;

    if (!out_count) {
        return NIYAH_ERR_INVALID_ARG;
    }

    *out_count = niyah_bridge_document_count();
    return NIYAH_OK;
}

static void legacy_free_string(void* implementation, char* text)
{
    (void)implementation;
    niyah_bridge_free_string(text);
}

static void legacy_free_results(
    void* implementation,
    NiyahDocumentSearchResults* results)
{
    (void)implementation;
    niyah_bridge_free_results(results);
}

static NiyahStatus legacy_add(
    void* implementation,
    const NiyahDocumentContext* context,
    const char* content,
    char** out_doc_id)
{
    const char* legacy_id = NULL;
    int32_t status;

    (void)implementation;
    (void)context;

    if (!content || !out_doc_id) {
        return NIYAH_ERR_INVALID_ARG;
    }

    *out_doc_id = NULL;
    status = niyah_bridge_add_document(content, &legacy_id);
    if (status != NIYAH_OK) {
        return (NiyahStatus)status;
    }

    *out_doc_id = (char*)legacy_id;
    return NIYAH_OK;
}

static NiyahStatus legacy_delete(
    void* implementation,
    const NiyahDocumentContext* context,
    const char* doc_id)
{
    (void)implementation;
    (void)context;
    return (NiyahStatus)niyah_bridge_delete_document(doc_id);
}

static const NiyahDocumentServiceOps g_legacy_document_ops = {
    {
        legacy_get,
        legacy_search,
        legacy_count,
        legacy_free_string,
        legacy_free_results
    },
    legacy_add,
    legacy_delete
};

static NiyahDocumentService g_legacy_document_service = {
    NULL,
    &g_legacy_document_ops
};

const NiyahDocumentService* niyah_bridge_document_service(void)
{
    return &g_legacy_document_service;
}
