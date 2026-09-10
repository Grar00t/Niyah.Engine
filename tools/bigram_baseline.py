#!/usr/bin/env python3
# SPDX-License-Identifier: UNKNOWN
"""Producer for the three baseline numbers this project measures against.

Every evidence file in this repository compares the model to three constants:

    UNIFORM_VOCAB269_BPB = 8.071462
    UNIGRAM_ADD1_BPB     = 4.796900
    BIGRAM_ADD1_BPB      = 3.768400   <- the gate

Until this file, none of them had a committed producer. A number whose
producer is not committed cannot be re-derived, only quoted. This script
re-derives all three from the same window streams the model is evaluated on,
so the gate can be checked instead of trusted.

Self-check: run it on the same train/eval pair used for the release. If
BIGRAM_ADD1_BPB does not land within ~1e-4 of 3.768400, either this script or
the frozen constant is wrong, and the disagreement must be resolved before
any PASS is claimed.

Model convention, matched exactly to tools/niyah_eval.c:
  - a window is WINDOW_TOKENS int32 little-endian ids
  - the first id is context only; the remaining WINDOW_TOKENS-1 are predicted
  - loss is mean negative log-likelihood in nats over predicted positions
  - BPB = mean_nll_nats / ln(2), which assumes one symbol is one byte

usage: bigram_baseline.py TRAIN_BIN EVAL_BIN [WINDOW_TOKENS]

exit 0  all three baselines printed
exit 2  malformed input or bad arguments
"""

import math
import struct
import sys

DEFAULT_WINDOW_TOKENS = 65
TOKEN_BYTES = 4
VOCAB = 269


def load_tokens(path, window_tokens):
    """Read a window stream into a flat tuple of ids, or exit 2 with a reason."""
    record_bytes = window_tokens * TOKEN_BYTES
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError as exc:
        print("FATAL=cannot read %s: %s" % (path, exc))
        sys.exit(2)
    if len(data) == 0:
        print("FATAL=%s is empty" % path)
        sys.exit(2)
    if len(data) % record_bytes != 0:
        print(
            "FATAL=%s is %d bytes, not a multiple of RECORD_BYTES=%d"
            % (path, len(data), record_bytes)
        )
        sys.exit(2)
    tokens = struct.unpack("<%di" % (len(data) // TOKEN_BYTES), data)
    for token in tokens:
        if token < 0 or token >= VOCAB:
            print("FATAL=%s contains id %d outside [0,%d)" % (path, token, VOCAB))
            sys.exit(2)
    return tokens


def pairs(tokens, window_tokens):
    """Yield (context, target) for every predicted position."""
    for start in range(0, len(tokens), window_tokens):
        window = tokens[start:start + window_tokens]
        for index in range(window_tokens - 1):
            yield window[index], window[index + 1]


def main(argv):
    if len(argv) not in (3, 4):
        print("usage: %s TRAIN_BIN EVAL_BIN [WINDOW_TOKENS]" % argv[0])
        return 2

    train_path, eval_path = argv[1], argv[2]

    if len(argv) == 4:
        try:
            window_tokens = int(argv[3])
        except ValueError:
            print("FATAL=WINDOW_TOKENS must be an integer, got %r" % argv[3])
            return 2
    else:
        window_tokens = DEFAULT_WINDOW_TOKENS

    if window_tokens <= 1:
        print("FATAL=WINDOW_TOKENS must be greater than 1, got %d" % window_tokens)
        return 2

    print("VOCAB=%d" % VOCAB)
    print("WINDOW_TOKENS=%d" % window_tokens)

    train_tokens = load_tokens(train_path, window_tokens)
    eval_tokens = load_tokens(eval_path, window_tokens)

    train_windows = len(train_tokens) // window_tokens
    eval_windows = len(eval_tokens) // window_tokens
    per_window = window_tokens - 1

    # Fit on the training stream only.
    unigram = [0] * VOCAB
    bigram = [0] * (VOCAB * VOCAB)
    context_total = [0] * VOCAB
    train_predictions = 0
    for context, target in pairs(train_tokens, window_tokens):
        unigram[target] += 1
        bigram[context * VOCAB + target] += 1
        context_total[context] += 1
        train_predictions += 1

    print("TRAIN_WINDOWS=%d TRAIN_PREDICTIONS=%d" % (train_windows, train_predictions))

    unigram_total = train_predictions
    contexts_seen = sum(1 for count in context_total if count > 0)
    print("TRAIN_CONTEXTS_SEEN=%d of %d" % (contexts_seen, VOCAB))

    # Score the evaluation stream.
    unigram_nll = 0.0
    bigram_nll = 0.0
    eval_predictions = 0
    unseen_contexts = 0
    unseen_pairs = 0
    for context, target in pairs(eval_tokens, window_tokens):
        eval_predictions += 1
        p_unigram = (unigram[target] + 1.0) / (unigram_total + VOCAB)
        unigram_nll -= math.log(p_unigram)
        denominator = context_total[context] + VOCAB
        numerator = bigram[context * VOCAB + target] + 1.0
        if context_total[context] == 0:
            unseen_contexts += 1
        if numerator == 1.0:
            unseen_pairs += 1
        bigram_nll -= math.log(numerator / denominator)

    if eval_predictions == 0:
        print("FATAL=no predictions in %s" % eval_path)
        return 2

    print("EVAL_WINDOWS=%d EVAL_PREDICTIONS=%d" % (eval_windows, eval_predictions))
    print("EVAL_UNSEEN_CONTEXTS=%d" % unseen_contexts)
    print("EVAL_UNSEEN_PAIRS=%d" % unseen_pairs)

    if eval_predictions != eval_windows * per_window:
        print("FATAL=prediction accounting mismatch")
        return 2

    ln2 = math.log(2.0)

    uniform_nats = math.log(VOCAB)
    unigram_nats = unigram_nll / eval_predictions
    bigram_nats = bigram_nll / eval_predictions

    print("UNIFORM_NLL_NATS=%.6f" % uniform_nats)
    print("UNIFORM_VOCAB%d_BPB=%.6f" % (VOCAB, uniform_nats / ln2))
    print("UNIGRAM_ADD1_NLL_NATS=%.6f" % unigram_nats)
    print("UNIGRAM_ADD1_BPB=%.6f" % (unigram_nats / ln2))
    print("BIGRAM_ADD1_NLL_NATS=%.6f" % bigram_nats)
    print("BIGRAM_ADD1_BPB=%.9f" % (bigram_nats / ln2))
    print("BASELINE_COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
