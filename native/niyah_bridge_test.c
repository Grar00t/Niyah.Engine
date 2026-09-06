#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "niyah_bridge.h"

int main(void)
{
    assert(niyah_bridge_version() != NULL);
    assert(niyah_get_version() != NULL);
    assert(strcmp(niyah_get_version(), "0.2.0") == 0);
    assert(strcmp(niyah_get_truth_string(NIYAH_UNKNOWN), "unknown") == 0);

    niyah_bridge_clear();
    assert(niyah_bridge_document_count() == 0);

    char* empty = niyah_bridge_search_json("anything", 10);
    assert(empty != NULL);
    assert(strcmp(empty, "[]") == 0);
    niyah_bridge_free_string(empty);

    void* results = NULL;
    int count = -1;
    assert(niyah_bridge_search("anything", &results, &count) == NIYAH_OK);
    assert(count == 0);
    niyah_bridge_free_results(results);

    const char* id_a = NULL;
    const char* id_b = NULL;
    const char* text_a = "the capital of japan is tokyo";
    const char* text_b = "rust is a systems programming language";

    assert(niyah_bridge_add_document(text_a, &id_a) == NIYAH_OK);
    assert(id_a != NULL);
    assert(niyah_bridge_document_count() == 1);

    assert(niyah_bridge_add_document(text_b, &id_b) == NIYAH_OK);
    assert(id_b != NULL);
    assert(niyah_bridge_document_count() == 2);
    assert(strcmp(id_a, "doc_new") != 0);
    assert(strcmp(id_a, id_b) != 0);

    char* fetched = niyah_bridge_get_document(id_a);
    assert(fetched != NULL);
    assert(strcmp(fetched, text_a) == 0);
    assert(fetched != text_a);
    niyah_bridge_free_string(fetched);
    assert(niyah_bridge_get_document("missing") == NULL);
    assert(niyah_bridge_get_document(NULL) == NULL);

    char* hit = niyah_bridge_search_json("tokyo", 10);
    assert(hit != NULL);
    assert(strstr(hit, id_a) != NULL);
    assert(strstr(hit, id_b) == NULL);
    niyah_bridge_free_string(hit);

    char* other = niyah_bridge_search_json("rust", 10);
    assert(other != NULL);
    assert(strstr(other, id_b) != NULL);
    assert(strstr(other, id_a) == NULL);
    niyah_bridge_free_string(other);

    char* miss = niyah_bridge_search_json("zebra", 10);
    assert(miss != NULL);
    assert(strcmp(miss, "[]") == 0);
    niyah_bridge_free_string(miss);

    char* capped = niyah_bridge_search_json("is", 1);
    assert(capped != NULL);
    niyah_bridge_free_string(capped);

    results = NULL;
    count = -1;
    assert(niyah_bridge_search("tokyo", &results, &count) == NIYAH_OK);
    assert(count == 1);
    assert(results != NULL);

    const NiyahBridgeResults* typed = (const NiyahBridgeResults*)results;
    assert(typed->count == 1);
    assert(typed->hits != NULL);
    assert(typed->hits[0].doc_id != NULL);
    assert(strcmp(typed->hits[0].doc_id, id_a) == 0);
    assert(typed->hits[0].score > 0.0f);
    niyah_bridge_free_results(results);

    assert(niyah_bridge_delete_document(id_b) == NIYAH_OK);
    assert(niyah_bridge_document_count() == 1);
    assert(niyah_bridge_get_document(id_b) == NULL);
    assert(niyah_bridge_delete_document(id_b) == NIYAH_ERR_NOT_FOUND);
    assert(niyah_bridge_delete_document(NULL) == NIYAH_ERR_INVALID_ARG);
    assert(niyah_bridge_delete_document("") == NIYAH_ERR_INVALID_ARG);

    char* after_delete = niyah_bridge_search_json("rust", 10);
    assert(after_delete != NULL);
    assert(strcmp(after_delete, "[]") == 0);
    niyah_bridge_free_string(after_delete);

    fetched = niyah_bridge_get_document(id_a);
    assert(fetched != NULL);
    assert(strcmp(fetched, text_a) == 0);
    niyah_bridge_free_string(fetched);

    niyah_bridge_free_string((char*)id_a);
    niyah_bridge_free_string((char*)id_b);

    niyah_bridge_clear();
    assert(niyah_bridge_document_count() == 0);

    char* after_clear = niyah_bridge_search_json("tokyo", 10);
    assert(after_clear != NULL);
    assert(strcmp(after_clear, "[]") == 0);
    niyah_bridge_free_string(after_clear);

    assert(niyah_bridge_add_document(NULL, &id_a) != NIYAH_OK);
    assert(niyah_bridge_add_document("content", NULL) != NIYAH_OK);
    assert(niyah_bridge_search(NULL, &results, &count) != NIYAH_OK);
    assert(niyah_bridge_search_json(NULL, 10) == NULL);
    niyah_bridge_free_string(NULL);
    niyah_bridge_free_results(NULL);

    NiyahBridgeContext* ctx = niyah_bridge_create(NULL);
    assert(ctx != NULL);
    assert(niyah_bridge_graph(ctx) != NULL);
    niyah_bridge_destroy(ctx);
    niyah_bridge_destroy(NULL);

    return 0;
}
