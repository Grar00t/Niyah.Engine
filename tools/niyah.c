#include "niyah/niyah.h"
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(FILE *f) {
    fprintf(f,
        "Niyah.Engine v0.1.0\n"
        "usage:\n"
        "  niyah train --input FILE --model FILE\n"
        "  niyah eval --input FILE --model FILE\n"
        "  niyah generate --model FILE [--prompt TEXT] [--tokens N] [--seed N]\n"
        "  niyah inspect --model FILE\n");
}

static const char *arg_value(int argc, char **argv, const char *name) {
    int i;
    for (i = 2; i + 1 < argc; ++i) if (strcmp(argv[i], name) == 0) return argv[i + 1];
    return NULL;
}

static int parse_u64(const char *s, uint64_t *out) {
    char *end = NULL;
    unsigned long long v;
    if (!s || !*s) return 0;
    errno = 0;
    v = strtoull(s, &end, 10);
    if (errno || !end || *end != '\0') return 0;
    *out = (uint64_t)v;
    return 1;
}

static int fail_status(const char *op, niyah_status s) {
    fprintf(stderr, "error=%s status=%s\n", op, niyah_status_string(s));
    return (int)s;
}

int main(int argc, char **argv) {
    niyah_model model;
    niyah_status s;
    const char *input;
    const char *model_path;

    if (argc < 2) { usage(stderr); return 2; }

    if (strcmp(argv[1], "train") == 0) {
        input = arg_value(argc, argv, "--input");
        model_path = arg_value(argc, argv, "--model");
        if (!input || !model_path) { usage(stderr); return 2; }
        niyah_model_init(&model);
        s = niyah_model_train_file(&model, input);
        if (s != NIYAH_OK) return fail_status("train", s);
        s = niyah_model_save(&model, model_path);
        if (s != NIYAH_OK) return fail_status("save", s);
        printf("status=ok\nbytes_seen=%" PRIu64 "\nmodel=%s\n", model.bytes_seen, model_path);
        return 0;
    }

    if (strcmp(argv[1], "eval") == 0) {
        double bpb, ppl;
        input = arg_value(argc, argv, "--input");
        model_path = arg_value(argc, argv, "--model");
        if (!input || !model_path) { usage(stderr); return 2; }
        s = niyah_model_load(&model, model_path);
        if (s != NIYAH_OK) return fail_status("load", s);
        s = niyah_model_eval_file(&model, input, &bpb, &ppl);
        if (s != NIYAH_OK) return fail_status("eval", s);
        printf("status=ok\nbits_per_byte=%.9f\nperplexity=%.9f\n", bpb, ppl);
        return 0;
    }

    if (strcmp(argv[1], "generate") == 0) {
        const char *prompt = arg_value(argc, argv, "--prompt");
        const char *tokens_s = arg_value(argc, argv, "--tokens");
        const char *seed_s = arg_value(argc, argv, "--seed");
        uint64_t tokens64 = 128, seed = 42;
        uint8_t *out;
        size_t out_size = 0;
        if (!prompt) prompt = "";
        model_path = arg_value(argc, argv, "--model");
        if (!model_path) { usage(stderr); return 2; }
        if (tokens_s && !parse_u64(tokens_s, &tokens64)) { fprintf(stderr, "invalid --tokens\n"); return 2; }
        if (seed_s && !parse_u64(seed_s, &seed)) { fprintf(stderr, "invalid --seed\n"); return 2; }
        if (tokens64 > 1024 * 1024) { fprintf(stderr, "--tokens too large\n"); return 2; }
        s = niyah_model_load(&model, model_path);
        if (s != NIYAH_OK) return fail_status("load", s);
        out = (uint8_t *)malloc((size_t)tokens64);
        if (!out && tokens64) return fail_status("alloc", NIYAH_ERR_NOMEM);
        s = niyah_model_generate(&model, (const uint8_t *)prompt, strlen(prompt),
                                 (size_t)tokens64, seed, out, (size_t)tokens64, &out_size);
        if (s != NIYAH_OK) { free(out); return fail_status("generate", s); }
        if (*prompt) fwrite(prompt, 1, strlen(prompt), stdout);
        fwrite(out, 1, out_size, stdout);
        fputc('\n', stdout);
        free(out);
        return 0;
    }

    if (strcmp(argv[1], "inspect") == 0) {
        model_path = arg_value(argc, argv, "--model");
        if (!model_path) { usage(stderr); return 2; }
        s = niyah_model_load(&model, model_path);
        if (s != NIYAH_OK) return fail_status("load", s);
        printf("status=ok\nformat=NIYAHBG1\nversion=%u\nalphabet=%u\nbytes_seen=%" PRIu64 "\n",
               NIYAH_MODEL_VERSION, NIYAH_ALPHABET_SIZE, model.bytes_seen);
        return 0;
    }

    usage(stderr);
    return 2;
}
