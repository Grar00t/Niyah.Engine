#include "niyah_index.h"

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define NIYAH_QUERY_TOKEN_LIMIT 128u
#define NIYAH_INITIAL_TERM_CAPACITY 128u
#define NIYAH_INITIAL_DOCUMENT_CAPACITY 64u
#define NIYAH_INITIAL_POSTING_CAPACITY 16u
#define NIYAH_INITIAL_DOC_TERM_CAPACITY 64u

static int hit_compare(const void *a, const void *b) {
    const NiyahSearchHit *ha = (const NiyahSearchHit *)a;
    const NiyahSearchHit *hb = (const NiyahSearchHit *)b;
    if (ha->score < hb->score) return 1;
    if (ha->score > hb->score) return -1;
    if (ha->document_id < hb->document_id) return -1;
    if (ha->document_id > hb->document_id) return 1;
    return 0;
}

static bool token_byte(unsigned char c) {
    return isalnum(c) != 0 || c >= 0x80u || c == '_' || c == '-';
}

/*
 * Streaming single-token extractor.  Advances *cursor past one token,
 * writes the lowercased token into buf, returns the token length.
 * Returns 0 when no more tokens remain.
 */
static size_t tokenize_one(const char **cursor, char *buf, size_t max_len) {
    const unsigned char *p = (const unsigned char *)*cursor;
    while (*p && !token_byte(*p)) ++p;
    if (!*p) { *cursor = (const char *)p; return 0; }
    size_t length = 0;
    while (*p && token_byte(*p)) {
        if (length + 1u < max_len) {
            const unsigned char byte = *p;
            buf[length++] = (char)(byte < 0x80u ? tolower((int)byte) : byte);
        }
        ++p;
    }
    buf[length] = '\0';
    *cursor = (const char *)p;
    return length;
}

static size_t tokenize(const char *text,
                       char tokens[][NIYAH_TERM_MAX],
                       size_t max_tokens) {
    if (!text || !tokens || max_tokens == 0) return 0;
    size_t count = 0;
    const unsigned char *cursor = (const unsigned char *)text;
    while (*cursor && count < max_tokens) {
        while (*cursor && !token_byte(*cursor)) ++cursor;
        if (!*cursor) break;
        size_t length = 0;
        while (*cursor && token_byte(*cursor)) {
            if (length + 1u < NIYAH_TERM_MAX) {
                const unsigned char byte = *cursor;
                tokens[count][length++] =
                    (char)(byte < 0x80u ? tolower((int)byte) : byte);
            }
            ++cursor;
        }
        tokens[count][length] = '\0';
        if (length > 0) ++count;
    }
    return count;
}

static NiyahTermEntry *find_term(NiyahInvertedIndex *index, const char *term) {
    if (!index || !term) return NULL;
    for (size_t i = 0; i < index->term_count; ++i) {
        if (strcmp(index->terms[i].term, term) == 0) return &index->terms[i];
    }
    return NULL;
}

static const NiyahTermEntry *find_term_const(const NiyahInvertedIndex *index,
                                              const char *term) {
    if (!index || !term) return NULL;
    for (size_t i = 0; i < index->term_count; ++i) {
        if (strcmp(index->terms[i].term, term) == 0) return &index->terms[i];
    }
    return NULL;
}

static bool checked_capacity_growth(size_t current,
                                    size_t initial,
                                    size_t element_size,
                                    size_t *next_out) {
    if (!next_out || element_size == 0) return false;
    const size_t next = current == 0 ? initial : current * 2u;
    if (next < current || next > SIZE_MAX / element_size) return false;
    *next_out = next;
    return true;
}

static bool grow_terms(NiyahInvertedIndex *index) {
    size_t next = 0;
    if (!index || !checked_capacity_growth(index->term_capacity,
                                            NIYAH_INITIAL_TERM_CAPACITY,
                                            sizeof(*index->terms), &next)) {
        return false;
    }
    NiyahTermEntry *terms = realloc(index->terms, next * sizeof(*terms));
    if (!terms) return false;
    memset(terms + index->term_capacity, 0,
           (next - index->term_capacity) * sizeof(*terms));
    index->terms = terms;
    index->term_capacity = next;
    return true;
}

static bool grow_documents(NiyahInvertedIndex *index) {
    size_t next = 0;
    if (!index || !checked_capacity_growth(index->document_capacity,
                                            NIYAH_INITIAL_DOCUMENT_CAPACITY,
                                            sizeof(*index->documents), &next)) {
        return false;
    }
    NiyahDocument *documents = realloc(index->documents,
                                       next * sizeof(*documents));
    if (!documents) return false;
    index->documents = documents;
    index->document_capacity = next;
    return true;
}

static bool grow_postings(NiyahTermEntry *entry) {
    size_t next = 0;
    if (!entry || !checked_capacity_growth(entry->posting_capacity,
                                            NIYAH_INITIAL_POSTING_CAPACITY,
                                            sizeof(*entry->postings), &next)) {
        return false;
    }
    NiyahPosting *postings = realloc(entry->postings,
                                     next * sizeof(*postings));
    if (!postings) return false;
    entry->postings = postings;
    entry->posting_capacity = next;
    return true;
}

static bool ensure_term_capacity(NiyahInvertedIndex *index, size_t additional) {
    if (!index || additional > SIZE_MAX - index->term_count) return false;
    const size_t required = index->term_count + additional;
    while (index->term_capacity < required) {
        if (!grow_terms(index)) return false;
    }
    return true;
}

static bool token_seen_before(char tokens[][NIYAH_TERM_MAX],
                              size_t index,
                              const char *token) {
    if (!tokens || !token) return false;
    for (size_t i = 0; i < index; ++i) {
        if (strcmp(tokens[i], token) == 0) return true;
    }
    return false;
}

static size_t document_position(const NiyahInvertedIndex *index,
                                uint64_t document_id) {
    if (!index || document_id == 0) return SIZE_MAX;
    for (size_t i = 0; i < index->document_count; ++i) {
        if (index->documents[i].document_id == document_id) return i;
    }
    return SIZE_MAX;
}

static char *copy_text(const char *text) {
    if (!text) return NULL;
    const size_t length = strlen(text);
    if (length == SIZE_MAX) return NULL;
    char *copy = malloc(length + 1u);
    if (!copy) return NULL;
    memcpy(copy, text, length + 1u);
    return copy;
}

static void free_pending_terms(NiyahTermEntry *terms, size_t count) {
    if (!terms) return;
    for (size_t i = 0; i < count; ++i) free(terms[i].postings);
    free(terms);
}

/* ---- streaming document tokenizer + per-document term accumulator ---- */

typedef struct {
    char     term[NIYAH_TERM_MAX];
    uint32_t frequency;
} NiyahDocTerm;

static NiyahDocTerm *find_doc_term(NiyahDocTerm *terms,
                                   size_t count,
                                   const char *term) {
    for (size_t i = 0; i < count; ++i) {
        if (strcmp(terms[i].term, term) == 0) return &terms[i];
    }
    return NULL;
}

static bool grow_doc_terms(NiyahDocTerm **terms, size_t *capacity) {
    size_t next = 0;
    if (!checked_capacity_growth(*capacity, NIYAH_INITIAL_DOC_TERM_CAPACITY,
                                 sizeof(NiyahDocTerm), &next))
        return false;
    NiyahDocTerm *grown = realloc(*terms, next * sizeof(NiyahDocTerm));
    if (!grown) return false;
    *terms = grown;
    *capacity = next;
    return true;
}

/*
 * Accumulate one token into the per-document term table.
 * Returns 0 on success, -1 on uint32 overflow, -2 on allocation failure.
 */
static int accumulate_one_token(NiyahDocTerm **terms, size_t *count,
                                size_t *capacity, const char *token,
                                size_t len, uint32_t *total) {
    if (*total == UINT32_MAX) return -1;
    ++*total;

    NiyahDocTerm *existing = find_doc_term(*terms, *count, token);
    if (existing) {
        if (existing->frequency == UINT32_MAX) return -1;
        ++existing->frequency;
        return 0;
    }

    if (*count == *capacity && !grow_doc_terms(terms, capacity)) return -2;
    memcpy((*terms)[*count].term, token, len + 1u);
    (*terms)[*count].frequency = 1u;
    ++*count;
    return 0;
}

/*
 * Stream the entire document text, building a unique-term table with
 * true per-term frequencies and the true total token count.  No cap.
 */
static bool accumulate_doc_terms(const char *text,
                                 NiyahDocTerm **out_terms,
                                 size_t *out_unique,
                                 uint32_t *out_total) {
    NiyahDocTerm *terms = NULL;
    size_t count = 0, capacity = 0;
    uint32_t total = 0;
    const char *cursor = text;
    char token[NIYAH_TERM_MAX];

    if (!text) {
        *out_terms = NULL; *out_unique = 0; *out_total = 0;
        return true;
    }

    for (;;) {
        size_t len = tokenize_one(&cursor, token, NIYAH_TERM_MAX);
        if (len == 0) break;
        int rc = accumulate_one_token(&terms, &count, &capacity,
                                      token, len, &total);
        if (rc != 0) { free(terms); return false; }
    }

    *out_terms = terms;
    *out_unique = count;
    *out_total = total;
    return true;
}

/* ---- public API ---- */

void niyah_index_init(NiyahInvertedIndex *index, double k1, double b) {
    if (!index) return;
    memset(index, 0, sizeof(*index));
    index->k1 = k1 > 0.0 ? k1 : 1.2;
    index->b = b >= 0.0 && b <= 1.0 ? b : 0.75;
}

void niyah_index_free(NiyahInvertedIndex *index) {
    if (!index) return;
    for (size_t i = 0; i < index->term_count; ++i)
        free(index->terms[i].postings);
    for (size_t i = 0; i < index->document_count; ++i)
        free((void *)index->documents[i].text);
    free(index->terms);
    free(index->documents);
    memset(index, 0, sizeof(*index));
}

/*
 * 62 lines: atomic multi-phase document insertion with 7 cleanup paths.
 * Splitting would pass heap pointers across function boundaries, making
 * leak-free error recovery strictly harder.
 */
bool niyah_index_add_document(NiyahInvertedIndex *index,
                              const NiyahDocument *document) {
    if (!index || !document || document->document_id == 0) return false;
    if (niyah_index_document(index, document->document_id)) return false;

    NiyahDocTerm *doc_terms = NULL;
    size_t doc_term_count = 0;
    uint32_t true_tokens = 0;

    if (!accumulate_doc_terms(document->text, &doc_terms,
                             &doc_term_count, &true_tokens))
        return false;

    size_t missing_count = 0;
    for (size_t i = 0; i < doc_term_count; ++i) {
        if (!find_term(index, doc_terms[i].term)) ++missing_count;
    }

    char *text_copy = copy_text(document->text);
    if (document->text && !text_copy) { free(doc_terms); return false; }

    if (!ensure_term_capacity(index, missing_count)) {
        free(text_copy); free(doc_terms); return false;
    }

    if (index->document_count == index->document_capacity &&
        !grow_documents(index)) {
        free(text_copy); free(doc_terms); return false;
    }

    NiyahTermEntry *pending = NULL;
    if (missing_count > 0) {
        pending = calloc(missing_count, sizeof(*pending));
        if (!pending) { free(text_copy); free(doc_terms); return false; }
    }

    size_t pending_count = 0;
    for (size_t i = 0; i < doc_term_count; ++i) {
        NiyahTermEntry *existing = find_term(index, doc_terms[i].term);
        if (existing) {
            if (existing->posting_count == existing->posting_capacity &&
                !grow_postings(existing)) {
                free_pending_terms(pending, pending_count);
                free(text_copy); free(doc_terms); return false;
            }
            continue;
        }
        NiyahTermEntry *entry = &pending[pending_count];
        memcpy(entry->term, doc_terms[i].term,
               strlen(doc_terms[i].term) + 1u);
        if (!grow_postings(entry)) {
            free_pending_terms(pending, pending_count + 1u);
            free(text_copy); free(doc_terms); return false;
        }
        ++pending_count;
    }

    const size_t original_term_count = index->term_count;
    for (size_t i = 0; i < pending_count; ++i) {
        index->terms[index->term_count++] = pending[i];
        pending[i].postings = NULL;
        pending[i].posting_capacity = 0;
    }
    free_pending_terms(pending, pending_count);

    for (size_t i = 0; i < doc_term_count; ++i) {
        NiyahTermEntry *entry = find_term(index, doc_terms[i].term);
        if (!entry || entry->posting_count >= entry->posting_capacity) {
            for (size_t j = original_term_count; j < index->term_count; ++j) {
                free(index->terms[j].postings);
                memset(&index->terms[j], 0, sizeof(index->terms[j]));
            }
            index->term_count = original_term_count;
            free(text_copy); free(doc_terms); return false;
        }
    }

    NiyahDocument *destination = &index->documents[index->document_count];
    destination->document_id = document->document_id;
    destination->text = text_copy;
    destination->term_count = true_tokens;

    for (size_t i = 0; i < doc_term_count; ++i) {
        NiyahTermEntry *entry = find_term(index, doc_terms[i].term);
        NiyahPosting *posting = &entry->postings[entry->posting_count++];
        posting->document_id = document->document_id;
        posting->term_frequency = doc_terms[i].frequency;
        ++entry->document_frequency;
    }

    const size_t previous_count = index->document_count;
    ++index->document_count;
    index->average_document_length = previous_count == 0
        ? (double)true_tokens
        : (index->average_document_length * (double)previous_count +
           (double)true_tokens) / (double)index->document_count;

    free(doc_terms);
    return true;
}

static double bm25_score(const NiyahInvertedIndex *index,
                         const NiyahTermEntry *entry,
                         uint32_t term_frequency,
                         uint32_t document_length) {
    if (!index || !entry || term_frequency == 0 || document_length == 0 ||
        index->document_count == 0) return 0.0;

    const double documents = (double)index->document_count;
    const double document_frequency = (double)entry->document_frequency;
    const double denominator = document_frequency + 0.5;
    if (denominator <= 0.0) return 0.0;

    const double idf = log1p((documents - document_frequency + 0.5) /
                             denominator);
    const double average_length = index->average_document_length > 0.0
        ? index->average_document_length : 1.0;
    const double normalization = index->k1 *
        (1.0 - index->b + index->b * ((double)document_length / average_length));
    const double tf = (double)term_frequency;
    const double score_denominator = tf + normalization;
    if (score_denominator <= 0.0) return 0.0;
    return idf * (tf * (index->k1 + 1.0) / score_denominator);
}

size_t niyah_index_search(const NiyahInvertedIndex *index,
                          const char *query,
                          NiyahSearchHit *hits,
                          size_t hit_capacity) {
    if (!index || !query || !hits || hit_capacity == 0 ||
        index->document_count == 0) return 0;

    double *scores = calloc(index->document_count, sizeof(*scores));
    if (!scores) return 0;

    char query_tokens[NIYAH_QUERY_TOKEN_LIMIT][NIYAH_TERM_MAX];
    const size_t query_token_count = tokenize(query, query_tokens,
                                               NIYAH_QUERY_TOKEN_LIMIT);

    for (size_t qi = 0; qi < query_token_count; ++qi) {
        if (token_seen_before(query_tokens, qi, query_tokens[qi])) continue;
        const NiyahTermEntry *entry = find_term_const(index, query_tokens[qi]);
        if (!entry) continue;

        for (size_t pi = 0; pi < entry->posting_count; ++pi) {
            const NiyahPosting *posting = &entry->postings[pi];
            const size_t di = document_position(index, posting->document_id);
            if (di == SIZE_MAX) continue;
            scores[di] += bm25_score(index, entry, posting->term_frequency,
                                     index->documents[di].term_count);
        }
    }

    size_t hit_count = 0;
    for (size_t di = 0; di < index->document_count; ++di) {
        const double score = scores[di];
        if (!(score > 0.0) || !isfinite(score)) continue;

        if (hit_count < hit_capacity) {
            hits[hit_count].document_id = index->documents[di].document_id;
            hits[hit_count].score = score;
            ++hit_count;
            continue;
        }

        size_t worst = 0;
        for (size_t hi = 1; hi < hit_capacity; ++hi) {
            if (hits[hi].score < hits[worst].score ||
                (hits[hi].score == hits[worst].score &&
                 hits[hi].document_id > hits[worst].document_id)) {
                worst = hi;
            }
        }
        if (score > hits[worst].score ||
            (score == hits[worst].score &&
             index->documents[di].document_id < hits[worst].document_id)) {
            hits[worst].document_id = index->documents[di].document_id;
            hits[worst].score = score;
        }
    }

    qsort(hits, hit_count, sizeof(*hits), hit_compare);
    free(scores);
    return hit_count;
}

const NiyahDocument *niyah_index_document(const NiyahInvertedIndex *index,
                                           uint64_t document_id) {
    if (!index || document_id == 0) return NULL;
    const size_t position = document_position(index, document_id);
    return position == SIZE_MAX ? NULL : &index->documents[position];
}
