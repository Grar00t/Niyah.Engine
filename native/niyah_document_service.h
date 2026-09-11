#ifndef NIYAH_DOCUMENT_SERVICE_H
#define NIYAH_DOCUMENT_SERVICE_H

#include "niyah.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NIYAH_DOCUMENT_ERR_FORBIDDEN ((NiyahStatus)-9)

typedef enum {
    NIYAH_DOCUMENT_CALLER_HOST  = 0,
    NIYAH_DOCUMENT_CALLER_MODEL = 1
} NiyahDocumentCaller;

typedef struct {
    NiyahDocumentCaller caller;
    const void*         caller_context;
} NiyahDocumentContext;

typedef struct {
    char* doc_id;
    char* snippet;
    float score;
} NiyahDocumentSearchHit;

typedef struct {
    NiyahDocumentSearchHit* hits;
    int32_t                 count;
} NiyahDocumentSearchResults;

typedef struct {
    NiyahStatus (*get)(void* implementation,
                       const NiyahDocumentContext* context,
                       const char* doc_id,
                       char** out_content);
    NiyahStatus (*search)(void* implementation,
                          const NiyahDocumentContext* context,
                          const char* query,
                          NiyahDocumentSearchResults** out_results);
    NiyahStatus (*count)(void* implementation,
                         const NiyahDocumentContext* context,
                         int32_t* out_count);
    void (*free_string)(void* implementation, char* text);
    void (*free_results)(void* implementation,
                         NiyahDocumentSearchResults* results);
} NiyahDocumentReadOps;

typedef struct {
    NiyahDocumentReadOps read;
    NiyahStatus (*add)(void* implementation,
                       const NiyahDocumentContext* context,
                       const char* content,
                       char** out_doc_id);
    NiyahStatus (*delete_document)(void* implementation,
                                   const NiyahDocumentContext* context,
                                   const char* doc_id);
} NiyahDocumentServiceOps;

typedef struct {
    void*                          implementation;
    const NiyahDocumentServiceOps* ops;
} NiyahDocumentService;

/* Model-facing capability. This type intentionally contains read operations
 * only; there is no add/delete function pointer on the model surface. */
typedef struct {
    void*                       implementation;
    const NiyahDocumentReadOps* ops;
    NiyahDocumentContext        context;
} NiyahDocumentReader;

NIYAH_API NiyahStatus niyah_document_service_bind(
    NiyahDocumentService* service,
    void* implementation,
    const NiyahDocumentServiceOps* ops);

NIYAH_API NiyahStatus niyah_document_service_add(
    const NiyahDocumentService* service,
    const NiyahDocumentContext* context,
    const char* content,
    char** out_doc_id);

NIYAH_API NiyahStatus niyah_document_service_get(
    const NiyahDocumentService* service,
    const NiyahDocumentContext* context,
    const char* doc_id,
    char** out_content);

NIYAH_API NiyahStatus niyah_document_service_delete(
    const NiyahDocumentService* service,
    const NiyahDocumentContext* context,
    const char* doc_id);

NIYAH_API NiyahStatus niyah_document_service_search(
    const NiyahDocumentService* service,
    const NiyahDocumentContext* context,
    const char* query,
    NiyahDocumentSearchResults** out_results);

NIYAH_API NiyahStatus niyah_document_service_count(
    const NiyahDocumentService* service,
    const NiyahDocumentContext* context,
    int32_t* out_count);

NIYAH_API void niyah_document_service_free_string(
    const NiyahDocumentService* service,
    char* text);

NIYAH_API void niyah_document_service_free_results(
    const NiyahDocumentService* service,
    NiyahDocumentSearchResults* results);

NIYAH_API NiyahStatus niyah_document_service_model_reader(
    const NiyahDocumentService* service,
    const NiyahDocumentContext* context,
    NiyahDocumentReader* out_reader);

NIYAH_API NiyahStatus niyah_document_reader_get(
    const NiyahDocumentReader* reader,
    const char* doc_id,
    char** out_content);

NIYAH_API NiyahStatus niyah_document_reader_search(
    const NiyahDocumentReader* reader,
    const char* query,
    NiyahDocumentSearchResults** out_results);

NIYAH_API NiyahStatus niyah_document_reader_count(
    const NiyahDocumentReader* reader,
    int32_t* out_count);

NIYAH_API void niyah_document_reader_free_string(
    const NiyahDocumentReader* reader,
    char* text);

NIYAH_API void niyah_document_reader_free_results(
    const NiyahDocumentReader* reader,
    NiyahDocumentSearchResults* results);

#ifdef __cplusplus
}
#endif

#endif /* NIYAH_DOCUMENT_SERVICE_H */
