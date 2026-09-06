#ifndef NIYAH_INDEX_H
#define NIYAH_INDEX_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef NIYAH_DOCUMENT_H
#error "search/niyah_index.h and native/niyah_document.h both define NiyahDocument. They are different concepts; do not include both in the same translation unit."
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifndef NIYAH_TERM_MAX
#define NIYAH_TERM_MAX 64u
#endif

typedef struct {
    uint64_t    document_id;
    const char* text;
    uint32_t    term_count;
} NiyahDocument;

typedef struct {
    uint64_t document_id;
    uint32_t term_frequency;
} NiyahPosting;

typedef struct {
    char          term[NIYAH_TERM_MAX];
    NiyahPosting* postings;
    size_t        posting_count;
    size_t        posting_capacity;
    uint32_t      document_frequency;
} NiyahTermEntry;

typedef struct {
    uint64_t document_id;
    double   score;
} NiyahSearchHit;

typedef struct {
    NiyahTermEntry* terms;
    size_t          term_count;
    size_t          term_capacity;

    NiyahDocument*  documents;
    size_t          document_count;
    size_t          document_capacity;

    double          average_document_length;
    double          k1;
    double          b;
} NiyahInvertedIndex;

void niyah_index_init(NiyahInvertedIndex* index, double k1, double b);
void niyah_index_free(NiyahInvertedIndex* index);

/*
 * Indexes a document atomically. The index copies document->text and owns the
 * copy until niyah_index_free(), so callers may release or reuse their input
 * buffer immediately after this function returns.
 */
bool niyah_index_add_document(NiyahInvertedIndex* index,
                              const NiyahDocument* document);

size_t niyah_index_search(const NiyahInvertedIndex* index,
                          const char* query,
                          NiyahSearchHit* hits,
                          size_t hit_capacity);

const NiyahDocument* niyah_index_document(const NiyahInvertedIndex* index,
                                          uint64_t document_id);

#ifdef __cplusplus
}
#endif

#endif
