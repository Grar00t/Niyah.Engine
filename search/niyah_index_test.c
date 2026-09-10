#ifdef NDEBUG
#undef NDEBUG
#endif

#include "niyah_index.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

static NiyahDocument make_document(uint64_t id, const char *text) {
    NiyahDocument document;
    memset(&document, 0, sizeof(document));
    document.document_id = id;
    document.text = text;
    return document;
}

static const NiyahPosting *find_posting(const NiyahInvertedIndex *index,
                                        const char *term,
                                        uint64_t document_id) {
    for (size_t i = 0; i < index->term_count; ++i) {
        if (strcmp(index->terms[i].term, term) != 0) continue;
        for (size_t j = 0; j < index->terms[i].posting_count; ++j) {
            if (index->terms[i].postings[j].document_id == document_id)
                return &index->terms[i].postings[j];
        }
    }
    return NULL;
}

/* ---- existing tests (unchanged from base) ---- */

static void test_empty_index(void) {
    NiyahInvertedIndex index;
    niyah_index_init(&index, 1.2, 0.75);
    assert(index.term_count == 0);
    assert(index.document_count == 0);
    assert(index.average_document_length == 0.0);
    niyah_index_free(&index);
}

static void test_add_and_find_document(void) {
    NiyahInvertedIndex index;
    niyah_index_init(&index, 1.2, 0.75);
    NiyahDocument document = make_document(1, "C programming systems programming");
    assert(niyah_index_add_document(&index, &document));
    assert(index.document_count == 1);
    assert(index.term_count == 3);
    assert(index.documents[0].term_count == 4);
    const NiyahDocument *found = niyah_index_document(&index, 1);
    assert(found && found->document_id == 1);
    assert(strcmp(found->text, "C programming systems programming") == 0);
    assert(niyah_index_document(&index, 999) == NULL);
    niyah_index_free(&index);
}

static void test_document_text_is_owned(void) {
    NiyahInvertedIndex index;
    niyah_index_init(&index, 1.2, 0.75);
    char buffer[] = "alpha beta";
    NiyahDocument document = make_document(7, buffer);
    assert(niyah_index_add_document(&index, &document));
    buffer[0] = 'X';
    memset(buffer + 1, 'z', sizeof(buffer) - 2u);
    const NiyahDocument *found = niyah_index_document(&index, 7);
    assert(found != NULL);
    assert(strcmp(found->text, "alpha beta") == 0);
    niyah_index_free(&index);
}

static void test_term_frequency_is_counted(void) {
    NiyahInvertedIndex index;
    niyah_index_init(&index, 1.2, 0.75);
    NiyahDocument document = make_document(1, "cache cache cache miss");
    assert(niyah_index_add_document(&index, &document));
    const NiyahPosting *repeated = find_posting(&index, "cache", 1);
    const NiyahPosting *single = find_posting(&index, "miss", 1);
    assert(repeated && single);
    assert(repeated->term_frequency == 3);
    assert(single->term_frequency == 1);
    for (size_t i = 0; i < index.term_count; ++i) {
        assert(index.terms[i].posting_count == 1);
        assert(index.terms[i].document_frequency == 1);
    }
    niyah_index_free(&index);
}

static void test_term_frequency_affects_ranking(void) {
    NiyahInvertedIndex index;
    niyah_index_init(&index, 1.2, 0.75);
    NiyahDocument dense = make_document(1, "cache cache cache miss");
    NiyahDocument sparse = make_document(2, "cache miss miss miss");
    assert(niyah_index_add_document(&index, &dense));
    assert(niyah_index_add_document(&index, &sparse));
    NiyahSearchHit hits[2] = {{0}};
    const size_t count = niyah_index_search(&index, "cache", hits, 2);
    assert(count == 2);
    assert(hits[0].document_id == 1);
    assert(hits[1].document_id == 2);
    assert(hits[0].score > hits[1].score);
    niyah_index_free(&index);
}

static void test_duplicate_document_rejected(void) {
    NiyahInvertedIndex index;
    niyah_index_init(&index, 1.2, 0.75);
    NiyahDocument first = make_document(1, "alpha beta");
    NiyahDocument duplicate = make_document(1, "different text");
    assert(niyah_index_add_document(&index, &first));
    assert(!niyah_index_add_document(&index, &duplicate));
    assert(index.document_count == 1);
    assert(index.term_count == 2);
    assert(strcmp(index.documents[0].text, "alpha beta") == 0);
    niyah_index_free(&index);
}

static void test_search_is_deterministic(void) {
    NiyahInvertedIndex index;
    niyah_index_init(&index, 1.2, 0.75);
    NiyahDocument first = make_document(1, "systems programming");
    NiyahDocument second = make_document(2, "systems programming");
    NiyahDocument third = make_document(3, "unrelated topic");
    assert(niyah_index_add_document(&index, &first));
    assert(niyah_index_add_document(&index, &second));
    assert(niyah_index_add_document(&index, &third));
    NiyahSearchHit hits[2] = {{0}};
    const size_t count = niyah_index_search(&index, "systems", hits, 2);
    assert(count == 2);
    assert(hits[0].document_id == 1);
    assert(hits[1].document_id == 2);
    niyah_index_free(&index);
}

static void test_empty_query(void) {
    NiyahInvertedIndex index;
    niyah_index_init(&index, 1.2, 0.75);
    NiyahDocument document = make_document(1, "alpha beta");
    assert(niyah_index_add_document(&index, &document));
    NiyahSearchHit hit = {0};
    assert(niyah_index_search(&index, "", &hit, 1) == 0);
    niyah_index_free(&index);
}

/* ---- regression gate helpers ---- */

/*
 * Build "filler filler ... filler <tail>" with filler_count filler tokens
 * followed by the tail string (which may itself contain multiple tokens).
 */
static char *build_doc_with_tail(size_t filler_count, const char *tail) {
    const char *filler = "filler";
    const size_t fl = strlen(filler);
    const size_t tl = tail ? strlen(tail) : 0;

    size_t needed = filler_count > 0
        ? fl + (filler_count - 1u) * (fl + 1u)
        : 0;
    if (tail) needed += (filler_count > 0 ? 1u : 0u) + tl;
    needed += 1u;

    char *buf = (char *)malloc(needed);
    assert(buf != NULL);

    char *cursor = buf;
    for (size_t i = 0; i < filler_count; ++i) {
        if (i > 0) *cursor++ = ' ';
        memcpy(cursor, filler, fl);
        cursor += fl;
    }
    if (tail) {
        if (filler_count > 0) *cursor++ = ' ';
        memcpy(cursor, tail, tl);
        cursor += tl;
    }
    *cursor = '\0';
    return buf;
}

static char *make_length_test_document(size_t token_count) {
    const char *first = "target";
    const char *filler = " filler";
    const size_t first_len = strlen(first);
    const size_t filler_len = strlen(filler);
    const size_t payload_len = first_len + (token_count - 1u) * filler_len;

    assert(token_count > 0u);

    char *text = (char *)malloc(payload_len + 1u);
    assert(text != NULL);

    char *cursor = text;
    memcpy(cursor, first, first_len);
    cursor += first_len;
    for (size_t i = 1u; i < token_count; ++i) {
        memcpy(cursor, filler, filler_len);
        cursor += filler_len;
    }
    *cursor = '\0';
    return text;
}

/* ---- GATE 1: term at token 1025 must be searchable ---- */

static void test_tail_term_is_indexed(void) {
    NiyahInvertedIndex index;
    niyah_index_init(&index, 1.2, 0.75);

    char *text = build_doc_with_tail(1024u, "sovereign_cuda");
    NiyahDocument doc = make_document(1u, text);
    assert(niyah_index_add_document(&index, &doc));

    NiyahSearchHit hit = {0};
    const size_t count = niyah_index_search(&index, "sovereign_cuda",
                                             &hit, 1u);
    assert(count == 1u);
    assert(hit.document_id == 1u);

    niyah_index_free(&index);
    free(text);
}

/* ---- GATE 2: term frequency after token 1024 must count ---- */

static void test_tail_term_frequency_counts(void) {
    NiyahInvertedIndex index;
    niyah_index_init(&index, 1.2, 0.75);

    /* 1024 fillers + 10 "cuda" = 1034 tokens, TF(cuda) must be 10 */
    char *text = build_doc_with_tail(1024u,
        "cuda cuda cuda cuda cuda cuda cuda cuda cuda cuda");
    NiyahDocument doc = make_document(1u, text);
    assert(niyah_index_add_document(&index, &doc));

    const NiyahPosting *posting = find_posting(&index, "cuda", 1u);
    assert(posting != NULL);
    assert(posting->term_frequency == 10u);
    assert(index.documents[0].term_count == 1034u);

    niyah_index_free(&index);
    free(text);
}

/* ---- GATE 3: 1024 vs 1025 length normalization must remain correct ---- */

static void test_bm25_length_normalisation_uses_true_document_length(void) {
    NiyahInvertedIndex index;
    NiyahSearchHit hits[2] = {{0}};
    char *at_limit;
    char *over_limit;
    NiyahDocument short_document;
    NiyahDocument long_document;
    size_t found;

    at_limit = make_length_test_document(1024u);
    over_limit = make_length_test_document(1025u);

    niyah_index_init(&index, 1.2, 0.75);

    short_document = make_document(1u, at_limit);
    long_document = make_document(2u, over_limit);

    assert(niyah_index_add_document(&index, &short_document));
    assert(niyah_index_add_document(&index, &long_document));

    assert(index.documents[0].term_count == 1024u);
    assert(index.documents[1].term_count == 1025u);
    assert(index.average_document_length == 1024.5);

    found = niyah_index_search(&index, "target", hits, 2u);

    assert(found == 2u);
    assert(hits[0].document_id == 1u);
    assert(hits[1].document_id == 2u);
    assert(hits[0].score > hits[1].score);

    niyah_index_free(&index);
    free(at_limit);
    free(over_limit);
}

int main(void) {
    test_empty_index();
    test_add_and_find_document();
    test_document_text_is_owned();
    test_term_frequency_is_counted();
    test_term_frequency_affects_ranking();
    test_duplicate_document_rejected();
    test_search_is_deterministic();
    test_empty_query();
    test_tail_term_is_indexed();
    test_tail_term_frequency_counts();
    test_bm25_length_normalisation_uses_true_document_length();
    return 0;
}
