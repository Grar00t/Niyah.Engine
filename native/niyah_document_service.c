#include "niyah_document_service.h"

static int document_context_valid(const NiyahDocumentContext* context)
{
    return context &&
           (context->caller == NIYAH_DOCUMENT_CALLER_HOST ||
            context->caller == NIYAH_DOCUMENT_CALLER_MODEL);
}

static NiyahStatus document_service_validate(
    const NiyahDocumentService* service)
{
    if (!service || !service->ops) {
        return NIYAH_ERR_INVALID_ARG;
    }

    if (!service->ops->read.get ||
        !service->ops->read.search ||
        !service->ops->read.count ||
        !service->ops->read.free_string ||
        !service->ops->read.free_results ||
        !service->ops->add ||
        !service->ops->delete_document) {
        return NIYAH_ERR_INVALID_ARG;
    }

    return NIYAH_OK;
}

static NiyahStatus document_reader_validate(
    const NiyahDocumentReader* reader)
{
    if (!reader || !reader->ops ||
        !reader->ops->get ||
        !reader->ops->search ||
        !reader->ops->count ||
        !reader->ops->free_string ||
        !reader->ops->free_results ||
        reader->context.caller != NIYAH_DOCUMENT_CALLER_MODEL) {
        return NIYAH_ERR_INVALID_ARG;
    }

    return NIYAH_OK;
}

NiyahStatus niyah_document_service_bind(
    NiyahDocumentService* service,
    void* implementation,
    const NiyahDocumentServiceOps* ops)
{
    NiyahDocumentService candidate;

    if (!service || !ops) {
        return NIYAH_ERR_INVALID_ARG;
    }

    candidate.implementation = implementation;
    candidate.ops = ops;

    if (document_service_validate(&candidate) != NIYAH_OK) {
        return NIYAH_ERR_INVALID_ARG;
    }

    *service = candidate;
    return NIYAH_OK;
}

NiyahStatus niyah_document_service_add(
    const NiyahDocumentService* service,
    const NiyahDocumentContext* context,
    const char* content,
    char** out_doc_id)
{
    if (out_doc_id) {
        *out_doc_id = NULL;
    }

    if (document_service_validate(service) != NIYAH_OK ||
        !document_context_valid(context) ||
        !content ||
        !out_doc_id) {
        return NIYAH_ERR_INVALID_ARG;
    }

    if (context->caller != NIYAH_DOCUMENT_CALLER_HOST) {
        return NIYAH_DOCUMENT_ERR_FORBIDDEN;
    }

    return service->ops->add(
        service->implementation,
        context,
        content,
        out_doc_id);
}

NiyahStatus niyah_document_service_get(
    const NiyahDocumentService* service,
    const NiyahDocumentContext* context,
    const char* doc_id,
    char** out_content)
{
    if (out_content) {
        *out_content = NULL;
    }

    if (document_service_validate(service) != NIYAH_OK ||
        !document_context_valid(context) ||
        !doc_id ||
        !doc_id[0] ||
        !out_content) {
        return NIYAH_ERR_INVALID_ARG;
    }

    return service->ops->read.get(
        service->implementation,
        context,
        doc_id,
        out_content);
}

NiyahStatus niyah_document_service_delete(
    const NiyahDocumentService* service,
    const NiyahDocumentContext* context,
    const char* doc_id)
{
    if (document_service_validate(service) != NIYAH_OK ||
        !document_context_valid(context) ||
        !doc_id ||
        !doc_id[0]) {
        return NIYAH_ERR_INVALID_ARG;
    }

    if (context->caller != NIYAH_DOCUMENT_CALLER_HOST) {
        return NIYAH_DOCUMENT_ERR_FORBIDDEN;
    }

    return service->ops->delete_document(
        service->implementation,
        context,
        doc_id);
}

NiyahStatus niyah_document_service_search(
    const NiyahDocumentService* service,
    const NiyahDocumentContext* context,
    const char* query,
    NiyahDocumentSearchResults** out_results)
{
    if (out_results) {
        *out_results = NULL;
    }

    if (document_service_validate(service) != NIYAH_OK ||
        !document_context_valid(context) ||
        !query ||
        !out_results) {
        return NIYAH_ERR_INVALID_ARG;
    }

    return service->ops->read.search(
        service->implementation,
        context,
        query,
        out_results);
}

NiyahStatus niyah_document_service_count(
    const NiyahDocumentService* service,
    const NiyahDocumentContext* context,
    int32_t* out_count)
{
    if (out_count) {
        *out_count = 0;
    }

    if (document_service_validate(service) != NIYAH_OK ||
        !document_context_valid(context) ||
        !out_count) {
        return NIYAH_ERR_INVALID_ARG;
    }

    return service->ops->read.count(
        service->implementation,
        context,
        out_count);
}

void niyah_document_service_free_string(
    const NiyahDocumentService* service,
    char* text)
{
    if (!text || document_service_validate(service) != NIYAH_OK) {
        return;
    }

    service->ops->read.free_string(service->implementation, text);
}

void niyah_document_service_free_results(
    const NiyahDocumentService* service,
    NiyahDocumentSearchResults* results)
{
    if (!results || document_service_validate(service) != NIYAH_OK) {
        return;
    }

    service->ops->read.free_results(service->implementation, results);
}

NiyahStatus niyah_document_service_model_reader(
    const NiyahDocumentService* service,
    const NiyahDocumentContext* context,
    NiyahDocumentReader* out_reader)
{
    if (out_reader) {
        out_reader->implementation = NULL;
        out_reader->ops = NULL;
        out_reader->context.caller = NIYAH_DOCUMENT_CALLER_MODEL;
        out_reader->context.caller_context = NULL;
    }

    if (document_service_validate(service) != NIYAH_OK ||
        !document_context_valid(context) ||
        !out_reader) {
        return NIYAH_ERR_INVALID_ARG;
    }

    if (context->caller != NIYAH_DOCUMENT_CALLER_MODEL) {
        return NIYAH_DOCUMENT_ERR_FORBIDDEN;
    }

    out_reader->implementation = service->implementation;
    out_reader->ops = &service->ops->read;
    out_reader->context = *context;
    return NIYAH_OK;
}

NiyahStatus niyah_document_reader_get(
    const NiyahDocumentReader* reader,
    const char* doc_id,
    char** out_content)
{
    if (out_content) {
        *out_content = NULL;
    }

    if (document_reader_validate(reader) != NIYAH_OK ||
        !doc_id ||
        !doc_id[0] ||
        !out_content) {
        return NIYAH_ERR_INVALID_ARG;
    }

    return reader->ops->get(
        reader->implementation,
        &reader->context,
        doc_id,
        out_content);
}

NiyahStatus niyah_document_reader_search(
    const NiyahDocumentReader* reader,
    const char* query,
    NiyahDocumentSearchResults** out_results)
{
    if (out_results) {
        *out_results = NULL;
    }

    if (document_reader_validate(reader) != NIYAH_OK ||
        !query ||
        !out_results) {
        return NIYAH_ERR_INVALID_ARG;
    }

    return reader->ops->search(
        reader->implementation,
        &reader->context,
        query,
        out_results);
}

NiyahStatus niyah_document_reader_count(
    const NiyahDocumentReader* reader,
    int32_t* out_count)
{
    if (out_count) {
        *out_count = 0;
    }

    if (document_reader_validate(reader) != NIYAH_OK || !out_count) {
        return NIYAH_ERR_INVALID_ARG;
    }

    return reader->ops->count(
        reader->implementation,
        &reader->context,
        out_count);
}

void niyah_document_reader_free_string(
    const NiyahDocumentReader* reader,
    char* text)
{
    if (!text || document_reader_validate(reader) != NIYAH_OK) {
        return;
    }

    reader->ops->free_string(reader->implementation, text);
}

void niyah_document_reader_free_results(
    const NiyahDocumentReader* reader,
    NiyahDocumentSearchResults* results)
{
    if (!results || document_reader_validate(reader) != NIYAH_OK) {
        return;
    }

    reader->ops->free_results(reader->implementation, results);
}
