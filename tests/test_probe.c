#define NIYAH_PROBE_TEST 1
#include "../tools/niyah_probe.c"
#include <float.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "probe CHECK line=%d: %s\n", __LINE__, #x); return 1; } } while (0)

static int invalid_options(void)
{
    char *cases[][10] = {
        {"probe", "--tokenizer", "x", "--unknown", "1", NULL},
        {"probe", "--tokenizer", NULL},
        {"probe", "--tokenizer", "x", NULL},
        {"probe", "--tokenizer", "x", "--checkpoint", "x", NULL},
        {"probe", "--tokenizer", "x", "--prompt", "", NULL},
        {"probe", "--tokenizer", "x", "--shard", "x", "--topk", "2", NULL},
        {"probe", "--tokenizer", "x", "--shard", "x", "--tokenizer", "x", NULL},
        {"probe", "--tokenizer", "--shard", "x", NULL},
        {"probe", "--tokenizer", "x", "--checkpoint", "x", "--prompt", "", "--topk", "0", NULL},
        {"probe", "--tokenizer", "x", "--checkpoint", "x", "--prompt", "", "--topk", "-1", NULL},
        {"probe", "--tokenizer", "x", "--checkpoint", "x", "--prompt", "", "--topk", "1x", NULL},
        {"probe", "--tokenizer", "x", "--checkpoint", "x", "--prompt", "", "--trace-steps", "9999999999999999999999999", NULL}
    };
    size_t i;
    for (i = 0U; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        int argc = 0;
        while (cases[i][argc] != NULL) ++argc;
        CHECK(niyah_cli_main(argc, cases[i]) == 2);
    }
    return 0;
}

int main(void)
{
    float logits[] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    uint8_t taken[3] = {0U, 0U, 0U};
    double maximum, total;
    size_t value = 99U, i;
    NiyahTokenizer *tok = NULL;
    NiyahTokenizerTrainConfig config = {270U, 2U};
    uint8_t corpus[1024];
    uint32_t id = 0U;
    size_t longest = 0U;
    CHECK(invalid_options() == 0);
    CHECK(parse_size("0", &value) && value == 0U);
    CHECK(!parse_size("+2", &value));
    CHECK(!parse_size(" 2", &value));
    CHECK(!parse_size("2.0", &value));
    CHECK(distribution(logits, 3U, &maximum, &total));
    CHECK(total == 3.0 && maximum == -(double)FLT_MAX);
    CHECK(best_untaken(logits, taken, 3U) == 0U);
    taken[0] = 1U;
    CHECK(best_untaken(logits, taken, 3U) == 1U);
    logits[0] = NAN;
    CHECK(!distribution(logits, 3U, &maximum, &total));
    logits[0] = INFINITY;
    CHECK(!distribution(logits, 3U, &maximum, &total));
    logits[0] = -INFINITY;
    CHECK(!distribution(logits, 3U, &maximum, &total));
    logits[0] = 0.0f; logits[1] = 1.0f; logits[2] = 2.0f;
    CHECK(distribution(logits, 3U, &maximum, &total));
    CHECK(fabs(total - (1.0 + exp(-1.0) + exp(-2.0))) < 1e-15);
    memset(corpus, 'a', sizeof(corpus));
    CHECK(niyah_tokenizer_train(corpus, sizeof(corpus), &config, &tok) == NIYAH_OK);
    for (i = 0U; i < niyah_tokenizer_vocab_size(tok); ++i) {
        uint32_t candidate = (uint32_t)i;
        size_t bytes = 0U;
        CHECK(niyah_tokenizer_decode(tok, &candidate, 1U, NULL, 0U, &bytes) == NIYAH_OK);
        if (bytes > longest) { longest = bytes; id = candidate; }
    }
    CHECK(longest > 64U);
    CHECK(print_token_bytes(tok, id));
    niyah_tokenizer_destroy(tok);
    CHECK(fflush(stdout) == 0);
    puts("PROBE_CONTRACT=PASS");
    return 0;
}
