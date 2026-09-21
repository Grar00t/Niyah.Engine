#include "niyah/baseline.h"
#include "niyah/tokenizer.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

static int niyah_u64_compare(const void *left, const void *right)
{
    const uint64_t a = *(const uint64_t *)left;
    const uint64_t b = *(const uint64_t *)right;
    return (a > b) - (a < b);
}

static int niyah_is_record_boundary(uint32_t previous, uint32_t next)
{
    return previous == NIYAH_TOKEN_EOS && next == NIYAH_TOKEN_BOS;
}

static NiyahStatus niyah_add1_bigram_pairs_mean_nll(
    uint64_t *pairs,
    size_t transition_count,
    size_t vocab_size,
    double *out_mean_nll_nats)
{
    size_t i;
    double total_nll = 0.0;
    double mean_nll;

    if (pairs == NULL || out_mean_nll_nats == NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;
    if (transition_count == 0U || vocab_size == 0U)
        return NIYAH_ERR_INVALID_CONFIG;

    qsort(pairs, transition_count, sizeof(*pairs), niyah_u64_compare);

    i = 0U;
    while (i < transition_count) {
        const uint32_t previous = (uint32_t)(pairs[i] >> 32U);
        size_t previous_end = i;
        size_t pair_index;
        double denominator;

        while (previous_end < transition_count &&
               (uint32_t)(pairs[previous_end] >> 32U) == previous) {
            previous_end += 1U;
        }

        denominator = (double)(previous_end - i) + (double)vocab_size;
        if (!isfinite(denominator) || denominator <= 0.0)
            return NIYAH_ERR_OVERFLOW;

        pair_index = i;
        while (pair_index < previous_end) {
            size_t pair_end = pair_index + 1U;
            size_t pair_count;
            double probability;
            double contribution;

            while (pair_end < previous_end &&
                   pairs[pair_end] == pairs[pair_index]) {
                pair_end += 1U;
            }

            pair_count = pair_end - pair_index;
            probability = ((double)pair_count + 1.0) / denominator;
            if (!isfinite(probability) ||
                probability <= 0.0 ||
                probability > 1.0)
                return NIYAH_ERR_OVERFLOW;

            contribution = -(double)pair_count * log(probability);
            if (!isfinite(contribution) ||
                !isfinite(total_nll + contribution))
                return NIYAH_ERR_OVERFLOW;

            total_nll += contribution;
            pair_index = pair_end;
        }

        i = previous_end;
    }

    mean_nll = total_nll / (double)transition_count;
    if (!isfinite(mean_nll))
        return NIYAH_ERR_OVERFLOW;

    *out_mean_nll_nats = mean_nll;
    return NIYAH_OK;
}

NiyahStatus niyah_add1_bigram_mean_nll(
    const uint32_t *tokens,
    size_t token_count,
    size_t vocab_size,
    double *out_mean_nll_nats)
{
    uint64_t *pairs = NULL;
    size_t transition_count = 0U;
    size_t write_index = 0U;
    size_t i;
    NiyahStatus status;

    if (tokens == NULL || out_mean_nll_nats == NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;
    if (token_count < 2U || vocab_size == 0U)
        return NIYAH_ERR_INVALID_CONFIG;

    for (i = 0U; i < token_count; ++i) {
        if ((size_t)tokens[i] >= vocab_size)
            return NIYAH_ERR_INVALID_ARGUMENT;
    }

    for (i = 0U; i + 1U < token_count; ++i) {
        if (!niyah_is_record_boundary(tokens[i], tokens[i + 1U]))
            transition_count += 1U;
    }

    if (transition_count == 0U)
        return NIYAH_ERR_INVALID_CONFIG;
    if (transition_count > SIZE_MAX / sizeof(*pairs))
        return NIYAH_ERR_OVERFLOW;

    pairs = (uint64_t *)malloc(transition_count * sizeof(*pairs));
    if (pairs == NULL)
        return NIYAH_ERR_OUT_OF_MEMORY;

    for (i = 0U; i + 1U < token_count; ++i) {
        if (niyah_is_record_boundary(tokens[i], tokens[i + 1U]))
            continue;

        pairs[write_index++] =
            ((uint64_t)tokens[i] << 32U) |
            (uint64_t)tokens[i + 1U];
    }

    if (write_index != transition_count) {
        free(pairs);
        return NIYAH_ERR_INVALID_CONFIG;
    }

    status = niyah_add1_bigram_pairs_mean_nll(
        pairs, transition_count, vocab_size, out_mean_nll_nats);
    free(pairs);
    return status;
}

NiyahStatus niyah_add1_bigram_samples_mean_nll(
    const NiyahEvaluationSample *samples,
    size_t sample_count,
    size_t vocab_size,
    double *out_mean_nll_nats)
{
    uint64_t *pairs = NULL;
    size_t transition_count = 0U;
    size_t write_index = 0U;
    size_t i;
    NiyahStatus status;

    if (samples == NULL || out_mean_nll_nats == NULL)
        return NIYAH_ERR_INVALID_ARGUMENT;
    if (sample_count == 0U || vocab_size == 0U)
        return NIYAH_ERR_INVALID_CONFIG;

    for (i = 0U; i < sample_count; ++i) {
        size_t t;
        size_t scored_count;

        if (samples[i].tokens == NULL ||
            samples[i].targets == NULL ||
            samples[i].token_count == 0U)
            return NIYAH_ERR_INVALID_ARGUMENT;
        if (samples[i].loss_start >= samples[i].token_count)
            return NIYAH_ERR_INVALID_CONFIG;

        for (t = 0U; t < samples[i].token_count; ++t) {
            if ((size_t)samples[i].tokens[t] >= vocab_size)
                return NIYAH_ERR_INVALID_ARGUMENT;
        }
        for (t = samples[i].loss_start;
             t < samples[i].token_count;
             ++t) {
            if ((size_t)samples[i].targets[t] >= vocab_size)
                return NIYAH_ERR_INVALID_ARGUMENT;
        }

        scored_count =
            samples[i].token_count - samples[i].loss_start;
        if (transition_count > SIZE_MAX - scored_count)
            return NIYAH_ERR_OVERFLOW;
        transition_count += scored_count;
    }

    if (transition_count == 0U)
        return NIYAH_ERR_INVALID_CONFIG;
    if (transition_count > SIZE_MAX / sizeof(*pairs))
        return NIYAH_ERR_OVERFLOW;

    pairs = (uint64_t *)malloc(transition_count * sizeof(*pairs));
    if (pairs == NULL)
        return NIYAH_ERR_OUT_OF_MEMORY;

    for (i = 0U; i < sample_count; ++i) {
        size_t t;
        for (t = samples[i].loss_start;
             t < samples[i].token_count;
             ++t) {
            pairs[write_index++] =
                ((uint64_t)samples[i].tokens[t] << 32U) |
                (uint64_t)samples[i].targets[t];
        }
    }

    if (write_index != transition_count) {
        free(pairs);
        return NIYAH_ERR_INVALID_CONFIG;
    }

    status = niyah_add1_bigram_pairs_mean_nll(
        pairs, transition_count, vocab_size, out_mean_nll_nats);
    free(pairs);
    return status;
}
