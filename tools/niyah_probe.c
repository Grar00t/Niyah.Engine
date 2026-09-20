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

static int fail(const char *stage, NiyahStatus status)
{
    fprintf(stderr, "stage=%s status=%d\n", stage, (int)status);
    return 1;
}

static void print_token_bytes(const NiyahTokenizer *tok, uint32_t id)
{
    uint8_t buf[64];
    size_t out = 0U;
    size_t i;
    NiyahStatus st = niyah_tokenizer_decode(tok, &id, 1U, buf, sizeof(buf), &out);
    if (st != NIYAH_OK) { printf("<decode_err=%d>", (int)st); return; }
    putchar('"');
    for (i = 0U; i < out; ++i) {
        unsigned char c = buf[i];
        if (c == '\n') printf("\\n");
        else if (c == '\r') printf("\\r");
        else if (c == ' ') printf("\\x20");
        else if (c >= 0x20 && c < 0x7F) putchar((int)c);
        else printf("\\x%02X", (unsigned int)c);
    }
    putchar('"');
}

static double token_probability(const float *logits, size_t vocab, size_t token_index)
{
    double max_logit = -1e300;
    double total = 0.0;
    size_t i;

    for (i = 0U; i < vocab; ++i) {
        if ((double)logits[i] > max_logit) max_logit = (double)logits[i];
    }
    for (i = 0U; i < vocab; ++i) {
        total += exp((double)logits[i] - max_logit);
    }
    if (!(total > 0.0) || !isfinite(total)) return 0.0;
    return exp((double)logits[token_index] - max_logit) / total;
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
    printf("space_byte_id=32 space_byte_count=%llu space_byte_frac=%.6f\n",
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
            print_token_bytes(tok, (uint32_t)best);
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

    st = niyah_tokenizer_encode(tok, (const uint8_t *)prompt, strlen(prompt), NULL, 0U, &encoded);
    if (st != NIYAH_OK) { fail("prompt_encode_query", st); goto cleanup; }

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
        double max_logit = -1e300, total = 0.0;
        int printed = 0;
        for (i = 0U; i < vocab; ++i) if ((double)logits[i] > max_logit) max_logit = (double)logits[i];
        for (i = 0U; i < vocab; ++i) total += exp((double)logits[i] - max_logit);

        printf("prompt_tokens=%zu vocab=%zu\n", prompt_count, vocab);
        printf("top%zu_next_token_logits:\n", topk);
        while ((size_t)printed < topk) {
            size_t best = vocab;
            float best_val = -1e30f;
            for (i = 0U; i < vocab; ++i)
                if (!taken[i] && logits[i] > best_val) { best_val = logits[i]; best = i; }
            if (best == vocab) break;
            taken[best] = 1U;
            {
                double prob = exp((double)logits[best] - max_logit) / total;
                printf("  rank=%d id=%zu logit=%.6f prob=%.6f bytes=",
                       printed + 1, best, (double)logits[best], prob);
                print_token_bytes(tok, (uint32_t)best);
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

        printf("greedy_trace_steps=%zu\n", trace_steps);
        for (step = 0U; step < trace_steps; ++step) {
            uint32_t token = 0U;
            size_t cache_before = niyah_kv_cache_position(&cache);
            size_t cache_after;
            float selected_logit;
            double prob;

            st = niyah_sampler_sample(&sampler, logits, vocab, &token);
            if (st != NIYAH_OK) { fail("trace_sampler_sample", st); goto cleanup; }

            selected_logit = logits[token];
            prob = token_probability(logits, vocab, (size_t)token);

            st = niyah_transformer_decode_token(&model, &cache, token,
                                                logits, vocab,
                                                decode_ws, decode_ws_count);
            if (st != NIYAH_OK) { fail("trace_decode_token", st); goto cleanup; }
            cache_after = niyah_kv_cache_position(&cache);

            printf("trace_step=%zu cache=%zu->%zu id=%u logit=%.6f prob=%.6f bytes=",
                   step, cache_before, cache_after, token,
                   (double)selected_logit, prob);
            print_token_bytes(tok, token);
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

int main(int argc, char **argv)
{
    const char *tokenizer_path = NULL, *shard_path = NULL;
    const char *checkpoint_path = NULL, *prompt = NULL;
    size_t topk = 10U;
    size_t trace_steps = 0U;
    int i, rc = 0;

    for (i = 1; i + 1 <= argc; ++i) {
        if (i + 1 < argc && arg_eq(argv[i], "--tokenizer")) { tokenizer_path = argv[++i]; }
        else if (i + 1 < argc && arg_eq(argv[i], "--shard")) { shard_path = argv[++i]; }
        else if (i + 1 < argc && arg_eq(argv[i], "--checkpoint")) { checkpoint_path = argv[++i]; }
        else if (i + 1 < argc && arg_eq(argv[i], "--prompt")) { prompt = argv[++i]; }
        else if (i + 1 < argc && arg_eq(argv[i], "--topk")) { topk = (size_t)strtoul(argv[++i], NULL, 10); }
        else if (i + 1 < argc && arg_eq(argv[i], "--trace-steps")) { trace_steps = (size_t)strtoul(argv[++i], NULL, 10); }
    }

    if (tokenizer_path == NULL) {
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
    return rc;
}
