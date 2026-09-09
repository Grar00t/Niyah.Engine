#include "niyah_mini_vocab.h"
#include "../niyah.h"

#include <stdio.h>
#include <string.h>

int main(void) {
    fprintf(stderr, "Testing NiyahMini vocabulary...\n");

    /* Test vocabulary initialization */
    NiyahMiniVocab vocab;
    memset(&vocab, 0, sizeof(vocab));

    /* Test loading from non-existent file */
    NiyahStatus status = niyah_mini_vocab_load(&vocab, "nonexistent_vocab.txt", "nonexistent_merges.txt");
    if (status == NIYAH_OK) {
        fprintf(stderr, "ERROR: Should fail to load non-existent file\n");
        return 1;
    }

    /* Create a small test vocabulary manually */
    niyah_mini_vocab_config_init(NULL);  /* Just to test it doesn't crash */

    /* Test token lookup */
    {
        NiyahMiniVocab small_vocab;
        memset(&small_vocab, 0, sizeof(small_vocab));

        /* Add some tokens manually */
        niyah_mini_vocab_add(&small_vocab, "<pad>", 0.0f);
        niyah_mini_vocab_add(&small_vocab, "hello", 0.0f);
        niyah_mini_vocab_add(&small_vocab, "world", 0.0f);
        niyah_mini_vocab_add(&small_vocab, "test", 0.0f);

        /* Test lookup */
        int32_t id = niyah_mini_vocab_lookup(&small_vocab, "nonexistent");
        if (id != -1) {
            fprintf(stderr, "ERROR: Should return -1 for non-existent token\n");
            niyah_mini_vocab_free(&small_vocab);
            return 1;
        }

        id = niyah_mini_vocab_lookup(&small_vocab, "hello");
        if (id != 1) {  /* hello should be at index 1 */
            fprintf(stderr, "ERROR: Expected hello at index 1, got %d\n", id);
            niyah_mini_vocab_free(&small_vocab);
            return 1;
        }

        /* Test id_to_token */
        const char* token = niyah_mini_vocab_id_to_token(&small_vocab, 1);
        if (!token || strcmp(token, "hello") != 0) {
            fprintf(stderr, "ERROR: Expected 'hello' at index 1\n");
            niyah_mini_vocab_free(&small_vocab);
            return 1;
        }

        niyah_mini_vocab_free(&small_vocab);
    }

    /* Test BPE tokenizer */
    {
        NiyahMiniBPE bpe;
        memset(&bpe, 0, sizeof(bpe));

        NiyahMiniVocab vocab;
        memset(&vocab, 0, sizeof(vocab));

        /* Add some tokens */
        niyah_mini_vocab_add(&vocab, "<pad>", 0.0f);
        niyah_mini_vocab_add(&vocab, "hello", 0.0f);
        niyah_mini_vocab_add(&vocab, "world", 0.0f);
        niyah_mini_vocab_add(&vocab, " ", 0.0f);

        /* Initialize BPE without merges */
        status = niyah_mini_bpe_init(&bpe, &vocab, NULL, 0);
        if (status != NIYAH_OK) {
            fprintf(stderr, "ERROR: Failed to initialize BPE\n");
            niyah_mini_vocab_free(&vocab);
            return 1;
        }

        /* Test tokenization */
        int32_t tokens[256];
        const char* text = "hello world";
        int32_t count = niyah_mini_bpe_tokenize(&bpe, text, tokens, 256);
        /* Vocabulary contains an explicit space token:
         * "hello world" -> "hello", " ", "world". */
        if (count != 3) {
            fprintf(stderr, "ERROR: Expected 3 tokens for 'hello world', got %d\n", count);
            niyah_mini_bpe_free(&bpe);
            niyah_mini_vocab_free(&vocab);
            return 1;
        }

        if (tokens[0] != 1 || tokens[1] != 3 || tokens[2] != 2) {
            fprintf(stderr,
                    "ERROR: Unexpected token IDs for 'hello world': %d %d %d\n",
                    tokens[0], tokens[1], tokens[2]);
            niyah_mini_bpe_free(&bpe);
            niyah_mini_vocab_free(&vocab);
            return 1;
        }

        niyah_mini_bpe_free(&bpe);
        niyah_mini_vocab_free(&vocab);
    }

    /* A vocabulary file defines the id space: line N must load as id N,
     * because those ids index embedding and LM-head rows. A skipped blank
     * line or a silently dropped repeat shifts every later id. */
    {
        const char *ok_path = "niyah_mini_vocab_test_ok.txt";
        const char *blank_path = "niyah_mini_vocab_test_blank.txt";
        const char *dup_path = "niyah_mini_vocab_test_dup.txt";
        NiyahMiniVocab loaded;
        FILE *f;

        f = fopen(ok_path, "wb");
        if (!f) {
            fprintf(stderr, "ERROR: Cannot create %s\n", ok_path);
            return 1;
        }
        fprintf(f, "<pad>\nhello\nworld\n");
        fclose(f);

        f = fopen(blank_path, "wb");
        if (!f) {
            fprintf(stderr, "ERROR: Cannot create %s\n", blank_path);
            remove(ok_path);
            return 1;
        }
        fprintf(f, "<pad>\n\nworld\n");
        fclose(f);

        f = fopen(dup_path, "wb");
        if (!f) {
            fprintf(stderr, "ERROR: Cannot create %s\n", dup_path);
            remove(ok_path);
            remove(blank_path);
            return 1;
        }
        fprintf(f, "<pad>\nhello\nhello\n");
        fclose(f);

        memset(&loaded, 0, sizeof(loaded));
        status = niyah_mini_vocab_load(&loaded, ok_path, NULL);
        if (status != NIYAH_OK ||
            loaded.n_vocab != 3 ||
            niyah_mini_vocab_lookup(&loaded, "<pad>") != 0 ||
            niyah_mini_vocab_lookup(&loaded, "hello") != 1 ||
            niyah_mini_vocab_lookup(&loaded, "world") != 2) {
            fprintf(stderr,
                    "ERROR: Well-formed vocabulary did not load line N as id N\n");
            niyah_mini_vocab_free(&loaded);
            remove(ok_path);
            remove(blank_path);
            remove(dup_path);
            return 1;
        }
        niyah_mini_vocab_free(&loaded);

        memset(&loaded, 0, sizeof(loaded));
        status = niyah_mini_vocab_load(&loaded, blank_path, NULL);
        if (status == NIYAH_OK || loaded.n_vocab != 0) {
            fprintf(stderr,
                    "ERROR: A blank vocabulary line must be refused, not skipped\n");
            niyah_mini_vocab_free(&loaded);
            remove(ok_path);
            remove(blank_path);
            remove(dup_path);
            return 1;
        }

        memset(&loaded, 0, sizeof(loaded));
        status = niyah_mini_vocab_load(&loaded, dup_path, NULL);
        if (status == NIYAH_OK || loaded.n_vocab != 0) {
            fprintf(stderr,
                    "ERROR: A repeated vocabulary line must be refused, not dropped\n");
            niyah_mini_vocab_free(&loaded);
            remove(ok_path);
            remove(blank_path);
            remove(dup_path);
            return 1;
        }

        remove(ok_path);
        remove(blank_path);
        remove(dup_path);
    }

    fprintf(stderr, "\nAll vocabulary tests passed!\n");
    return 0;
}
