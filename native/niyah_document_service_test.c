#undef NDEBUG
#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "niyah_bridge.h"
#include "niyah_document_service.h"

static NiyahDocumentContext host_context(void)
{
    NiyahDocumentContext context;
    context.caller = NIYAH_DOCUMENT_CALLER_HOST;
    context.caller_context = NULL;
    return context;
}

static NiyahDocumentContext model_context(void)
{
    NiyahDocumentContext context;
    context.caller = NIYAH_DOCUMENT_CALLER_MODEL;
    context.caller_context = NULL;
    return context;
}

int main(void)
{
    const NiyahDocumentService* service = niyah_bridge_document_service();
    NiyahDocumentContext host = host_context();
    NiyahDocumentContext model = model_context();
    NiyahDocumentContext invalid_context = model_context();
    NiyahDocumentReader reader;
    NiyahDocumentSearchResults* results = NULL;
    char* doc_id = NULL;
    char* content = NULL;
    int32_t count = -1;

    assert(service != NULL);
    niyah_bridge_clear();

    assert(niyah_document_service_count(service, &host, &count) == NIYAH_OK);
    assert(count == 0);

    assert(niyah_document_service_add(service, &host,
                                      "contract document alpha", &doc_id) == NIYAH_OK);
    assert(doc_id != NULL);
    assert(niyah_document_service_count(service, &host, &count) == NIYAH_OK);
    assert(count == 1);

    assert(niyah_document_service_get(service, &host, doc_id, &content) == NIYAH_OK);
    assert(content != NULL);
    assert(strcmp(content, "contract document alpha") == 0);
    niyah_document_service_free_string(service, content);
    content = NULL;

    assert(niyah_document_service_get(service, &model, doc_id, &content) == NIYAH_OK);
    assert(content != NULL);
    assert(strcmp(content, "contract document alpha") == 0);
    niyah_document_service_free_string(service, content);
    content = NULL;

    assert(niyah_document_service_search(service, &model, "alpha", &results) == NIYAH_OK);
    assert(results != NULL);
    assert(results->count == 1);
    assert(results->hits != NULL);
    assert(strcmp(results->hits[0].doc_id, doc_id) == 0);
    niyah_document_service_free_results(service, results);
    results = NULL;

    assert(niyah_document_service_model_reader(service, &model, &reader) == NIYAH_OK);
    assert(reader.ops != NULL);
    assert(reader.ops->get != NULL);
    assert(reader.ops->search != NULL);
    assert(reader.ops->count != NULL);

    assert(niyah_document_reader_count(&reader, &count) == NIYAH_OK);
    assert(count == 1);
    assert(niyah_document_reader_get(&reader, doc_id, &content) == NIYAH_OK);
    assert(content != NULL);
    assert(strcmp(content, "contract document alpha") == 0);
    niyah_document_reader_free_string(&reader, content);
    content = NULL;

    assert(niyah_document_reader_search(&reader, "contract", &results) == NIYAH_OK);
    assert(results != NULL);
    assert(results->count == 1);
    niyah_document_reader_free_results(&reader, results);
    results = NULL;

    {
        char* blocked_id = (char*)(uintptr_t)1;
        assert(niyah_document_service_add(service, &model,
                                          "blocked model write",
                                          &blocked_id) == NIYAH_DOCUMENT_ERR_FORBIDDEN);
        assert(blocked_id == NULL);
        assert(niyah_document_service_count(service, &host, &count) == NIYAH_OK);
        assert(count == 1);
    }

    assert(niyah_document_service_delete(service, &model,
                                         doc_id) == NIYAH_DOCUMENT_ERR_FORBIDDEN);
    assert(niyah_document_service_get(service, &host, doc_id, &content) == NIYAH_OK);
    assert(content != NULL);
    niyah_document_service_free_string(service, content);
    content = NULL;

    assert(niyah_document_service_model_reader(service, &host,
                                               &reader) == NIYAH_DOCUMENT_ERR_FORBIDDEN);

    invalid_context.caller = (NiyahDocumentCaller)99;
    assert(niyah_document_service_count(service, &invalid_context,
                                        &count) == NIYAH_ERR_INVALID_ARG);

    assert(niyah_document_service_add(NULL, &host, "x", &content) == NIYAH_ERR_INVALID_ARG);
    assert(niyah_document_service_add(service, NULL, "x", &content) == NIYAH_ERR_INVALID_ARG);
    assert(niyah_document_service_add(service, &host, NULL, &content) == NIYAH_ERR_INVALID_ARG);
    assert(niyah_document_service_add(service, &host, "x", NULL) == NIYAH_ERR_INVALID_ARG);
    assert(niyah_document_service_get(service, &host, NULL, &content) == NIYAH_ERR_INVALID_ARG);
    assert(niyah_document_service_get(service, &host, "", &content) == NIYAH_ERR_INVALID_ARG);
    assert(niyah_document_service_get(service, &host, doc_id, NULL) == NIYAH_ERR_INVALID_ARG);
    assert(niyah_document_service_delete(service, &host, NULL) == NIYAH_ERR_INVALID_ARG);
    assert(niyah_document_service_delete(service, &host, "") == NIYAH_ERR_INVALID_ARG);
    assert(niyah_document_service_search(service, &host, NULL, &results) == NIYAH_ERR_INVALID_ARG);
    assert(niyah_document_service_search(service, &host, "alpha", NULL) == NIYAH_ERR_INVALID_ARG);
    assert(niyah_document_service_count(service, &host, NULL) == NIYAH_ERR_INVALID_ARG);

    assert(niyah_document_service_delete(service, &host, doc_id) == NIYAH_OK);
    assert(niyah_document_service_count(service, &host, &count) == NIYAH_OK);
    assert(count == 0);
    assert(niyah_document_service_get(service, &host,
                                      doc_id, &content) == NIYAH_ERR_NOT_FOUND);
    assert(content == NULL);

    niyah_document_service_free_string(service, doc_id);
    doc_id = NULL;

    {
        const char* legacy_id = NULL;
        assert(niyah_bridge_add_document("legacy adapter beta",
                                         &legacy_id) == NIYAH_OK);
        assert(legacy_id != NULL);
        assert(niyah_document_service_get(service, &model,
                                          legacy_id, &content) == NIYAH_OK);
        assert(content != NULL);
        assert(strcmp(content, "legacy adapter beta") == 0);
        niyah_document_service_free_string(service, content);
        content = NULL;
        assert(niyah_document_service_delete(service, &host,
                                             legacy_id) == NIYAH_OK);
        niyah_bridge_free_string((char*)legacy_id);
    }

    assert(niyah_bridge_document_count() == 0);
    niyah_bridge_clear();
    return 0;
}
