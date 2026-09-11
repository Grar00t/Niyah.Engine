#ifndef NIYAH_BRIDGE_H
#define NIYAH_BRIDGE_H

#include "niyah.h"
#include "niyah_document_service.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef NiyahDocumentSearchHit NiyahBridgeHit;
typedef NiyahDocumentSearchResults NiyahBridgeResults;

NIYAH_API const char* niyah_bridge_version(void);
NIYAH_API const char* niyah_get_version(void);
NIYAH_API const char* niyah_get_truth_string(NiyahTruth truth);

NIYAH_API int32_t niyah_bridge_add_document(const char* content,
                                            const char** doc_id);

/* Returns a heap copy of the document text, or NULL when the id is absent.
 * Release a non-NULL result with niyah_bridge_free_string. */
NIYAH_API char* niyah_bridge_get_document(const char* doc_id);

/* Deletes one document by exact id. */
NIYAH_API int32_t niyah_bridge_delete_document(const char* doc_id);

/* Legacy structured search entry point. *results receives a heap
 * NiyahBridgeResults*; release it with niyah_bridge_free_results. */
NIYAH_API int32_t niyah_bridge_search(const char* query,
                                      void** results,
                                      int* count);

/* Managed-friendly search entry point. Returns a heap JSON array string. */
NIYAH_API char* niyah_bridge_search_json(const char* query, int32_t max_hits);

NIYAH_API int32_t niyah_bridge_document_count(void);
NIYAH_API void    niyah_bridge_clear(void);
NIYAH_API void    niyah_bridge_free_results(void* results);
NIYAH_API void    niyah_bridge_free_string(char* text);

/* Contract-first adapter over the current BridgeStore. The legacy bridge
 * remains the storage implementation for P0-A.1; callers can bind to this
 * service now without taking a dependency on BridgeStore itself. */
NIYAH_API const NiyahDocumentService* niyah_bridge_document_service(void);

#ifdef __cplusplus
}
#endif

#endif /* NIYAH_BRIDGE_H */
