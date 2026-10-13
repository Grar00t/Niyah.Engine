/* tools/niyah_probe.c — standalone, does not touch niyah_generate.c / niyah_sampler.c
 *
 * Usage:
 *   niyah_probe --tokenizer TOK
 *               [--shard SHARD]
 *               [--checkpoint CKPT --prompt TEXT [--topk N] [--trace-steps N]]
 */
#include "niyah/niyah.h"
#include "niyah/tokenizer.h"
#include "niyah/dataset.h"
#include "niyah/checkpoint.h"
#include "niyah/decode.h"
#include "niyah/optimizer.h"
#include "niyah/sampler.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <stdint.h>

static int fail(const char *stage, NiyahStatus status)
{
    fprintf(stderr, "stage=%s status=%d\n", stage, (int)status);
    return 1;
}

static int print_token_bytes(const NiyahTokenizer *tok, uint32_t id)
{
    uint8_t *buf = NULL;
    size_t out = 0U;
    size_t i;
    NiyahStatus st = niyah_tokenizer_decode(tok, &id, 1U, NULL, 0U, &out);
    if (st != NIYAH_OK) { fail("token_decode_size", st); return 0; }
    buf = (uint8_t *)malloc(out == 0U ? 1U : out);
    if (buf == NULL) { fail("token_decode_allocation", NIYAH_ERR_OUT_OF_MEMORY); return 0; }
    st = niyah_tokenizer_decode(tok, &id, 1U, buf, out, &out);
    if (st != NIYAH_OK) { free(buf); fail("token_decode", st); return 0; }
    putchar('"');
    for (i = 0U; i < out; ++i) {
        unsigned char c = buf[i];
        if (c == '\n') printf("\\n");
        else if (c == '\r') printf("\\r");
        else if (c == ' ') printf("\\x20");
        else if (c == '"' || c == '\\') printf("\\%c", (int)c);
        else if (c >= 0x20 && c < 0x7F) putchar((int)c);
        else printf("\\x%02X", (unsigned int)c);
    }
    putchar('"');
    free(buf);
    return !ferror(stdout);
}

static int distribution(const float *logits, size_t vocab, double *out_max,
                        double *out_total)
{
    double max_logit;
    double total = 0.0;
    size_t i;

    if (logits == NULL || vocab == 0U || out_max == NULL || out_total == NULL) return 0;
    max_logit = (double)logits[0];
    for (i = 0U; i < vocab; ++i) {
        if (!isfinite(logits[i])) return 0;
        if ((double)logits[i] > max_logit) max_logit = (double)logits[i];
    }
    for (i = 0U; i < vocab; ++i) {
        total += exp((double)logits[i] - max_logit);
    }
    if (!(total > 0.0) || !isfinite(total)) return 0;
    *out_max = max_logit;
    *out_total = total;
    return 1;
}

static size_t best_untaken(const float *logits, const uint8_t *taken, size_t vocab)
{
    size_t best = vocab;
    size_t i;
    for (i = 0U; i < vocab; ++i)
        if (!taken[i] && (best == vocab || logits[i] > logits[best])) best = i;
    return best;
}

static int shard_report(const char *tokenizer_path, const char *shard_path)
{
    NiyahTokenizer *tok = NULL;
    NiyahDatasetShard shard;
    uint64_t *counts = NULL;
    uint8_t *taken = NULL;
    size_t vocab, i;
    double entropy = 0.0;
    NiyahStatus st;
    int rc = 1;

    memset(&shard, 0, sizeof(shard));

    st = niyah_tokenizer_load(tokenizer_path, &tok);
    if (st != NIYAH_OK) return fail("tokenizer_load", st);

    st = niyah_dataset_shard_load(shard_path, tok, &shard);
    if (st != NIYAH_OK) { niyah_tokenizer_destroy(tok); return fail("shard_load", st); }

    vocab = niyah_tokenizer_vocab_size(tok);
    if (vocab <= 32U || vocab > SIZE_MAX / sizeof(*counts) || shard.token_count == 0U) {
        fail("shard_geometry", NIYAH_ERR_INVALID_CONFIG); goto cleanup;
    }
    counts = (uint64_t *)calloc(vocab, sizeof(uint64_t));
    if (counts == NULL) { fprintf(stderr, "oom\n"); goto cleanup; }

    for (i = 0U; i < shard.token_count; ++i) {
        uint32_t id = shard.tokens[i];
        if ((size_t)id < vocab) counts[id] += 1U;
    }
    for (i = 0U; i < vocab; ++i) {
        if (counts[i] == 0U) continue;
        {
            double p = (double)counts[i] / (double)shard.token_count;
            entropy -= p * log(p);
        }
    }

    printf("shard_tokens=%zu vocab=%zu unigram_entropy_nats=%.6f\n",
           shard.token_count, vocab, entropy);
    printf("standalone_space_token_id=32 standalone_space_token_count=%llu standalone_space_token_frac=%.6f merged_token_spaces=not_counted\n",
           (unsigned long long)counts[32],
           (double)counts[32] / (double)shard.token_count);

    taken = (uint8_t *)calloc(vocab, 1U);
    if (taken == NULL) { fprintf(stderr, "oom\n"); goto cleanup; }
    printf("top10_by_count:\n");
    {
        int printed = 0;
        while (printed < 10) {
            size_t best = vocab;
            uint64_t best_count = 0U;
            for (i = 0U; i < vocab; ++i) {
                if (!taken[i] && counts[i] > best_count) { best_count = counts[i]; best = i; }
            }
            if (best == vocab || best_count == 0U) break;
            taken[best] = 1U;
            printf("  rank=%d id=%zu count=%llu frac=%.6f bytes=",
                   printed + 1, best, (unsigned long long)best_count,
                   (double)best_count / (double)shard.token_count);
            if (!print_token_bytes(tok, (uint32_t)best)) goto cleanup;
            putchar('\n');
            ++printed;
        }
    }
    rc = 0;

cleanup:
    free(taken);
    free(counts);
    niyah_dataset_shard_destroy(&shard);
    niyah_tokenizer_destroy(tok);
    return rc;
}

static int topk_logit_report(const char *tokenizer_path,
                             const char *checkpoint_path,
                             const char *prompt,
                             size_t topk,
                             size_t trace_steps)
{
    NiyahTokenizer *tok = NULL;
    NiyahModel model;
    NiyahAdamWState opt_state;
    NiyahAdamWConfig opt_config;
    NiyahKVCache cache;
    uint32_t *prompt_tokens = NULL;
    float *decode_ws = NULL;
    float *logits = NULL;
    uint8_t *taken = NULL;
    size_t encoded = 0U, prompt_count, decode_ws_count = 0U, vocab, i;
    NiyahStatus st;
    int rc = 1;

    memset(&model, 0, sizeof(model));
    memset(&opt_state, 0, sizeof(opt_state));
    memset(&opt_config, 0, sizeof(opt_config));
    memset(&cache, 0, sizeof(cache));

    st = niyah_tokenizer_load(tokenizer_path, &tok);
    if (st != NIYAH_OK) return fail("tokenizer_load", st);

    st = niyah_checkpoint_load_with_tokenizer(checkpoint_path, tok, &model, &opt_state, &opt_config);
    if (st != NIYAH_OK) { niyah_tokenizer_destroy(tok); return fail("checkpoint_load", st); }

    vocab = niyah_tokenizer_vocab_size(tok);
    if (vocab == 0U || vocab != (size_t)model.config.vocab_size) {
        fprintf(stderr, "vocab_mismatch\n"); goto cleanup;
    }
    if (model.config.n_segments != 0U) {
        fprintf(stderr, "segment_conditioning_unsupported=probe\n"); goto cleanup;
    }
    if (topk > vocab) topk = vocab;

    st = niyah_tokenizer_encode(tok, (const uint8_t *)prompt, strlen(prompt), NULL, 0U, &encoded);
    if (st != NIYAH_OK) { fail("prompt_encode_query", st); goto cleanup; }

    if (encoded == SIZE_MAX || encoded + 1U > SIZE_MAX / sizeof(*prompt_tokens)) {
        fail("prompt_size", NIYAH_ERR_OVERFLOW); goto cleanup;
    }
    prompt_count = encoded + 1U; /* + BOS — matches tools/niyah.c run_command */
    if (prompt_count > (size_t)model.config.context_length ||
        trace_steps > (size_t)model.config.context_length - prompt_count) {
        fprintf(stderr, "trace_context_overflow prompt_tokens=%zu trace_steps=%zu context=%u\n",
                prompt_count, trace_steps, model.config.context_length);
        goto cleanup;
    }

    prompt_tokens = (uint32_t *)malloc(prompt_count * sizeof(*prompt_tokens));
    if (prompt_tokens == NULL) { fprintf(stderr, "oom\n"); goto cleanup; }
    prompt_tokens[0] = NIYAH_TOKEN_BOS;

    st = niyah_tokenizer_encode(tok, (const uint8_t *)prompt, strlen(prompt),
                                prompt_tokens + 1U, encoded, &encoded);
    if (st != NIYAH_OK) { fail("prompt_encode", st); goto cleanup; }

    printf("prompt_token_ids:");
    for (i = 0U; i < prompt_count; ++i) printf(" %u", prompt_tokens[i]);
    putchar('\n');

    st = niyah_kv_cache_create(&cache, &model.config);
    if (st != NIYAH_OK) { fail("kv_cache_create", st); goto cleanup; }

    st = niyah_decode_workspace_floats(&model.config, &decode_ws_count);
    if (st != NIYAH_OK) { fail("workspace_size", st); goto cleanup; }
    if (decode_ws_count > SIZE_MAX / sizeof(float) || vocab > SIZE_MAX / sizeof(float)) {
        fail("workspace_size", NIYAH_ERR_OVERFLOW); goto cleanup;
    }

    decode_ws = (float *)malloc(decode_ws_count * sizeof(float));
    logits = (float *)malloc(vocab * sizeof(float));
    if (decode_ws == NULL || logits == NULL) { fprintf(stderr, "oom\n"); goto cleanup; }

    for (i = 0U; i < prompt_count; ++i) {
        st = niyah_transformer_decode_token(&model, &cache, prompt_tokens[i],
                                            logits, vocab, decode_ws, decode_ws_count);
        if (st != NIYAH_OK) { fail("decode_token", st); goto cleanup; }
    }
    /* logits[] now holds the raw next-token distribution right after the
     * prompt — the exact buffer niyah_sampler_sample() sees before the first
     * generated token. */

    taken = (uint8_t *)calloc(vocab, 1U);
    if (taken == NULL) { fprintf(stderr, "oom\n"); goto cleanup; }
    {
        double max_logit, total;
        size_t printed = 0U;
        if (!distribution(logits, vocab, &max_logit, &total)) {
            fail("logit_distribution", NIYAH_ERR_INVALID_CONFIG); goto cleanup;
        }

        printf("prompt_tokens=%zu vocab=%zu\n", prompt_count, vocab);
        printf("top%zu_next_token_logits:\n", topk);
        while (printed < topk) {
            size_t best = best_untaken(logits, taken, vocab);
            if (best == vocab) break;
            taken[best] = 1U;
            {
                double prob = exp((double)logits[best] - max_logit) / total;
                printf("  rank=%zu id=%zu logit=%.6f prob=%.6f bytes=",
                       printed + 1, best, (double)logits[best], prob);
                if (!print_token_bytes(tok, (uint32_t)best)) goto cleanup;
                putchar('\n');
            }
            ++printed;
        }
    }

    if (trace_steps > 0U) {
        NiyahSampler sampler;
        NiyahSamplerConfig sampler_config;
        size_t step;

        memset(&sampler, 0, sizeof(sampler));
        memset(&sampler_config, 0, sizeof(sampler_config));
        sampler_config.temperature = 0.0f;
        sampler_config.seed = 0U;

        st = niyah_sampler_init(&sampler, &sampler_config);
        if (st != NIYAH_OK) { fail("trace_sampler_init", st); goto cleanup; }

        printf("greedy_trace_steps=%zu trace_stop_on_eos=0 eos_token=%u\n", trace_steps, NIYAH_TOKEN_EOS);
        for (step = 0U; step < trace_steps; ++step) {
            uint32_t token = 0U;
            size_t cache_before = niyah_kv_cache_position(&cache);
            size_t cache_after;
            float selected_logit;
            double prob;
            double eos_prob;
            double max_logit, total;

            if (!distribution(logits, vocab, &max_logit, &total)) {
                fail("trace_distribution", NIYAH_ERR_INVALID_CONFIG); goto cleanup;
            }

            st = niyah_sampler_sample(&sampler, logits, vocab, &token);
            if (st != NIYAH_OK) { fail("trace_sampler_sample", st); goto cleanup; }

            selected_logit = logits[token];
            prob = exp((double)logits[token] - max_logit) / total;
            eos_prob = exp((double)logits[NIYAH_TOKEN_EOS] - max_logit) / total;

            st = niyah_transformer_decode_token(&model, &cache, token,
                                                logits, vocab,
                                                decode_ws, decode_ws_count);
            if (st != NIYAH_OK) { fail("trace_decode_token", st); goto cleanup; }
            cache_after = niyah_kv_cache_position(&cache);

            printf("trace_step=%zu cache=%zu->%zu id=%u logit=%.6f prob=%.6f eos=%d eos_prob=%.9g finite_distribution=1 bytes=",
                   step, cache_before, cache_after, token,
                   (double)selected_logit, prob, token == NIYAH_TOKEN_EOS, eos_prob);
            if (!print_token_bytes(tok, token)) goto cleanup;
            putchar('\n');
        }
    }

    rc = 0;

cleanup:
    free(taken);
    free(prompt_tokens);
    free(decode_ws);
    free(logits);
    niyah_kv_cache_destroy(&cache);
    niyah_adamw_state_destroy(&opt_state);
    niyah_model_destroy(&model);
    niyah_tokenizer_destroy(tok);
    return rc;
}

static int arg_eq(const char *a, const char *b) { return strcmp(a, b) == 0; }

static int parse_size(const char *text, size_t *out)
{
    size_t value = 0U;
    const unsigned char *p = (const unsigned char *)text;
    if (text == NULL || *p == 0U) return 0;
    for (; *p != 0U; ++p) {
        size_t digit;
        if (*p < '0' || *p > '9') return 0;
        digit = (size_t)(*p - '0');
        if (value > (SIZE_MAX - digit) / 10U) return 0;
        value = value * 10U + digit;
    }
    *out = value;
    return 1;
}

static int niyah_cli_main(int argc, char **argv)
{
    const char *tokenizer_path = NULL, *shard_path = NULL;
    const char *checkpoint_path = NULL, *prompt = NULL;
    size_t topk = 10U;
    size_t trace_steps = 0U;
    unsigned int seen = 0U;
    int i, rc = 0;

    for (i = 1; i < argc; ++i) {
        unsigned int bit;
        const char *key = argv[i];
        const char *value;
        if (arg_eq(key, "--tokenizer")) bit = 1U;
        else if (arg_eq(key, "--shard")) bit = 2U;
        else if (arg_eq(key, "--checkpoint")) bit = 4U;
        else if (arg_eq(key, "--prompt")) bit = 8U;
        else if (arg_eq(key, "--topk")) bit = 16U;
        else if (arg_eq(key, "--trace-steps")) bit = 32U;
        else { fprintf(stderr, "unknown_option=%s\n", key); return 2; }
        if ((seen & bit) != 0U || i + 1 >= argc) {
            fprintf(stderr, "duplicate_or_incomplete_option=%s\n", key); return 2;
        }
        seen |= bit;
        value = argv[++i];
        if (strncmp(value, "--", 2U) == 0 || (bit != 8U && value[0] == '\0')) {
            fprintf(stderr, "invalid_value=%s\n", key); return 2;
        }
        if (bit == 1U) tokenizer_path = value;
        else if (bit == 2U) shard_path = value;
        else if (bit == 4U) checkpoint_path = value;
        else if (bit == 8U) prompt = value;
        else if (bit == 16U && (!parse_size(value, &topk) || topk == 0U)) {
            fprintf(stderr, "invalid_topk\n"); return 2;
        } else if (bit == 32U && !parse_size(value, &trace_steps)) {
            fprintf(stderr, "invalid_trace_steps\n"); return 2;
        }
    }

    if (tokenizer_path == NULL || (shard_path == NULL && checkpoint_path == NULL) ||
        ((checkpoint_path == NULL) != (prompt == NULL)) ||
        ((seen & (16U | 32U)) != 0U && checkpoint_path == NULL)) {
        fprintf(stderr, "usage: niyah_probe --tokenizer TOK [--shard SHARD] "
                        "[--checkpoint CKPT --prompt TEXT [--topk N] [--trace-steps N]]\n");
        return 2;
    }

    if (shard_path != NULL) {
        rc |= shard_report(tokenizer_path, shard_path);
    }
    if (checkpoint_path != NULL && prompt != NULL) {
        rc |= topk_logit_report(tokenizer_path, checkpoint_path, prompt, topk, trace_steps);
    }
    if (fflush(stdout) == EOF || ferror(stdout)) return fail("stdout_write", NIYAH_ERR_IO);
    return rc;
}


/* NIYAH_UTF8_ARGV_WRAPPER_V1 */
#if !defined(NIYAH_PROBE_TEST)
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

static void niyah_utf8_free_args(int argc, char **argv)
{
    int i;
    if (argv == NULL) return;
    for (i = 0; i < argc; ++i) free(argv[i]);
    free(argv);
}

int wmain(int argc, wchar_t **wargv)
{
    char **argv;
    int i;
    int rc;

    if (argc < 0) return 1;
    argv = (char **)calloc((size_t)argc + 1U, sizeof(*argv));
    if (argv == NULL) return 1;

    for (i = 0; i < argc; ++i) {
        int required = WideCharToMultiByte(
            CP_UTF8, WC_ERR_INVALID_CHARS,
            wargv[i], -1, NULL, 0, NULL, NULL);

        if (required <= 0) {
            fprintf(stderr, "argv_utf8_size_failed index=%d\n", i);
            niyah_utf8_free_args(argc, argv);
            return 1;
        }

        argv[i] = (char *)malloc((size_t)required);
        if (argv[i] == NULL) {
            niyah_utf8_free_args(argc, argv);
            return 1;
        }

        if (WideCharToMultiByte(
                CP_UTF8, WC_ERR_INVALID_CHARS,
                wargv[i], -1, argv[i], required, NULL, NULL
            ) != required) {
            fprintf(stderr, "argv_utf8_conversion_failed index=%d\n", i);
            niyah_utf8_free_args(argc, argv);
            return 1;
        }

        if (required > 1 &&
            MultiByteToWideChar(
                CP_UTF8, MB_ERR_INVALID_CHARS,
                argv[i], required - 1, NULL, 0
            ) <= 0) {
            fprintf(stderr, "argv_utf8_validation_failed index=%d\n", i);
            niyah_utf8_free_args(argc, argv);
            return 1;
        }
    }

    if (GetEnvironmentVariableW(L"NIYAH_ARGV_UTF8_DIAG", NULL, 0) > 0U) {
        for (i = 1; i + 1 < argc; ++i) {
            if (strcmp(argv[i], "--prompt") == 0) {
                const unsigned char *p =
                    (const unsigned char *)argv[i + 1];
                fprintf(stderr, "prompt_utf8_hex=");
                while (*p != 0U) {
                    fprintf(stderr, "%02x", (unsigned int)*p++);
                }
                fprintf(stderr, "\n");
            }
        }
    }

    rc = niyah_cli_main(argc, argv);
    niyah_utf8_free_args(argc, argv);
    return rc;
}
#else
int main(int argc, char **argv)
{
    return niyah_cli_main(argc, argv);
}
#endif
#endif /* NIYAH_PROBE_TEST */
