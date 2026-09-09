#include "niyah_sampler_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/*
 * Was a stub, so niyah_sample() was declared in niyah.h but never defined and
 * every executable that touched it failed at link time.
 *
 * xoshiro256** seeded through splitmix64. Deterministic for a given seed,
 * which matters for reproducible evaluation runs.
 */

static uint64_t g_state[4] = {
    0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL,
    0x2545f4914f6cdd1dULL, 0x9e3779b97f4a7c15ULL
};

static uint64_t splitmix64(uint64_t* x)
{
    uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

void niyah_sampler_seed(uint64_t seed)
{
    uint64_t x = seed ? seed : 0x9e3779b97f4a7c15ULL;
    for (int i = 0; i < 4; ++i) {
        g_state[i] = splitmix64(&x);
    }
}

static uint64_t rotl(uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}

static uint64_t next_u64(void)
{
    const uint64_t result = rotl(g_state[1] * 5ULL, 7) * 9ULL;
    const uint64_t t = g_state[1] << 17;

    g_state[2] ^= g_state[0];
    g_state[3] ^= g_state[1];
    g_state[1] ^= g_state[2];
    g_state[0] ^= g_state[3];
    g_state[2] ^= t;
    g_state[3] = rotl(g_state[3], 45);

    return result;
}

static float next_float(void)
{
    return (float)((next_u64() >> 11) * (1.0 / 9007199254740992.0));
}

static NiyahSamplerConfig sampler_config_resolve(const NiyahSamplerConfig* config)
{
    NiyahSamplerConfig cfg;
    if (config) {
        cfg = *config;
    } else {
        cfg.strategy = NIYAH_SAMPLE_GREEDY;
        cfg.temperature = 1.0f;
        cfg.top_k = 0;
        cfg.top_p = 1.0f;
    }
    return cfg;
}

static int32_t sampler_pool_required(const NiyahSamplerConfig* config,
                                     int32_t n_vocab)
{
    const NiyahSamplerConfig cfg = sampler_config_resolve(config);
    if (cfg.strategy == NIYAH_SAMPLE_GREEDY || !(cfg.temperature > 0.0f)) {
        return 0;
    }
    if (cfg.strategy == NIYAH_SAMPLE_TOP_K) {
        int32_t k = cfg.top_k > 0 ? cfg.top_k : 40;
        if (k > n_vocab) {
            k = n_vocab;
        }
        return k;
    }
    return n_vocab;
}

static bool candidate_is_worse(const NiyahSamplerCandidate* a,
                               const NiyahSamplerCandidate* b)
{
    if (a->prob < b->prob) return true;
    if (a->prob > b->prob) return false;
    return a->index > b->index;
}

static void candidate_swap(NiyahSamplerCandidate* a,
                           NiyahSamplerCandidate* b)
{
    const NiyahSamplerCandidate tmp = *a;
    *a = *b;
    *b = tmp;
}

static void candidate_sift_down(NiyahSamplerCandidate* pool,
                                int32_t root,
                                int32_t count)
{
    for (;;) {
        const int32_t child = root * 2 + 1;
        if (child >= count) {
            return;
        }

        int32_t worst = child;
        if (child + 1 < count &&
            candidate_is_worse(&pool[child + 1], &pool[child])) {
            worst = child + 1;
        }

        if (!candidate_is_worse(&pool[worst], &pool[root])) {
            return;
        }

        candidate_swap(&pool[root], &pool[worst]);
        root = worst;
    }
}

static void candidate_sort_desc(NiyahSamplerCandidate* pool, int32_t count)
{
    if (!pool || count <= 1) {
        return;
    }

    for (int32_t start = count / 2; start > 0; --start) {
        candidate_sift_down(pool, start - 1, count);
    }

    for (int32_t end = count - 1; end > 0; --end) {
        candidate_swap(&pool[0], &pool[end]);
        candidate_sift_down(pool, 0, end);
    }
}

static int32_t sample_from(const NiyahSamplerCandidate* pool,
                           int32_t count,
                           float total)
{
    if (count <= 0) {
        return -1;
    }
    if (!(total > 0.0f)) {
        return pool[0].index;
    }

    const float target = next_float() * total;
    float running = 0.0f;

    for (int32_t i = 0; i < count; ++i) {
        running += pool[i].prob;
        if (running >= target) {
            return pool[i].index;
        }
    }

    return pool[count - 1].index;
}

void niyah_sampler_apply_repetition_penalty(float* logits,
                                            int32_t n_vocab,
                                            const int32_t* history,
                                            int32_t history_len,
                                            float penalty)
{
    if (!logits || n_vocab <= 0 || !history || history_len <= 0) {
        return;
    }
    if (!(penalty > 0.0f) || penalty == 1.0f) {
        return;
    }

    for (int32_t i = 0; i < history_len; ++i) {
        const int32_t token = history[i];
        if (token < 0 || token >= n_vocab) {
            continue;
        }
        logits[token] = logits[token] > 0.0f
            ? logits[token] / penalty
            : logits[token] * penalty;
    }
}

int32_t niyah_sample_with_scratch(const float* logits,
                                  int32_t n_vocab,
                                  const NiyahSamplerConfig* config,
                                  float* probs,
                                  NiyahSamplerCandidate* pool,
                                  int32_t pool_capacity)
{
    if (!logits || n_vocab <= 0) {
        return -1;
    }

    const NiyahSamplerConfig cfg = sampler_config_resolve(config);
    if (cfg.strategy == NIYAH_SAMPLE_GREEDY || !(cfg.temperature > 0.0f)) {
        return niyah_argmax(logits, n_vocab);
    }

    const int32_t required = sampler_pool_required(&cfg, n_vocab);
    if (!probs || !pool || pool_capacity < required) {
        return -1;
    }

    memcpy(probs, logits, (size_t)n_vocab * sizeof(float));
    niyah_softmax_temperature(probs, n_vocab, cfg.temperature);

    if (cfg.strategy == NIYAH_SAMPLE_TEMPERATURE) {
        float total = 0.0f;
        for (int32_t i = 0; i < n_vocab; ++i) {
            pool[i].prob = probs[i];
            pool[i].index = i;
            total += probs[i];
        }
        return sample_from(pool, n_vocab, total);
    }

    if (cfg.strategy == NIYAH_SAMPLE_TOP_K) {
        int32_t k = required;
        for (int32_t slot = 0; slot < k; ++slot) {
            int32_t best = -1;
            float best_p = -1.0f;
            for (int32_t i = 0; i < n_vocab; ++i) {
                if (probs[i] < 0.0f) {
                    continue;
                }
                if (probs[i] > best_p) {
                    best_p = probs[i];
                    best = i;
                }
            }
            if (best < 0) {
                k = slot;
                break;
            }
            pool[slot].prob = best_p;
            pool[slot].index = best;
            probs[best] = -1.0f;
        }

        float total = 0.0f;
        for (int32_t i = 0; i < k; ++i) {
            total += pool[i].prob;
        }
        return sample_from(pool, k, total);
    }

    const float top_p = (cfg.top_p > 0.0f && cfg.top_p <= 1.0f)
        ? cfg.top_p : 0.9f;

    for (int32_t i = 0; i < n_vocab; ++i) {
        pool[i].prob = probs[i];
        pool[i].index = i;
    }
    candidate_sort_desc(pool, n_vocab);

    float cumulative = 0.0f;
    int32_t cutoff = 0;
    while (cutoff < n_vocab) {
        cumulative += pool[cutoff].prob;
        ++cutoff;
        if (cumulative >= top_p) {
            break;
        }
    }
    return sample_from(pool, cutoff, cumulative);
}

int32_t niyah_sample(const float* logits,
                     int32_t n_vocab,
                     const NiyahSamplerConfig* config)
{
    if (!logits || n_vocab <= 0) {
        return -1;
    }

    const NiyahSamplerConfig cfg = sampler_config_resolve(config);
    if (cfg.strategy == NIYAH_SAMPLE_GREEDY || !(cfg.temperature > 0.0f)) {
        return niyah_argmax(logits, n_vocab);
    }

    const int32_t required = sampler_pool_required(&cfg, n_vocab);
    float* probs = (float*)malloc((size_t)n_vocab * sizeof(float));
    NiyahSamplerCandidate* pool = required > 0
        ? (NiyahSamplerCandidate*)malloc((size_t)required * sizeof(NiyahSamplerCandidate))
        : NULL;

    if (!probs || (required > 0 && !pool)) {
        free(probs);
        free(pool);
        return niyah_argmax(logits, n_vocab);
    }

    int32_t chosen = niyah_sample_with_scratch(logits, n_vocab, &cfg,
                                               probs, pool, required);
    free(probs);
    free(pool);

    if (chosen < 0) {
        chosen = niyah_argmax(logits, n_vocab);
    }
    return chosen;
}
