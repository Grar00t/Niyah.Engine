#ifdef NDEBUG
#undef NDEBUG
#endif

#include "niyah_index.h"

#include <assert.h>
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

int main(void) {
    test_empty_index();
    test_add_and_find_document();
    test_document_text_is_owned();
    test_term_frequency_is_counted();
    test_term_frequency_affects_ranking();
    test_duplicate_document_rejected();
    test_search_is_deterministic();
    test_empty_query();
    return 0;
}
