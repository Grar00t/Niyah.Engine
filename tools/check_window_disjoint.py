#!/usr/bin/env python3
"""Window-level train/eval disjointness gate.

A held-out split that shares records with the training split is not held out,
and every number measured on it is suspect. This tool hashes each fixed-size
window record in both streams and reports how many evaluation records also
appear in the training stream.

The exit code is derived from the measured overlap, not from reaching the end
of the script:

    0  OVERLAP_WINDOWS == 0        HELDOUT_DISJOINT=PASS
    1  OVERLAP_WINDOWS  > 0        HELDOUT_DISJOINT=FAIL
    2  malformed input or usage    FAIL=<reason>

Usage:
    python3 tools/check_window_disjoint.py TRAIN.bin EVAL.bin [--window-tokens N]

Default window is 65 int32 tokens (64 shifted next-token predictions),
matching the v5-clean c64 window format.

No network. No temporary files. Deterministic.

Author: Suliman Nazal Alshammari
"""

import hashlib
import sys

TOKEN_BYTES = 4
DEFAULT_WINDOW_TOKENS = 65


def read_digests(path, record_bytes):
    """Return (list_of_record_digests, total_bytes) or exit 2 if malformed."""
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError as exc:
        print("FAIL=UNREADABLE path=%s errno=%s" % (path, exc.errno))
        sys.exit(2)
    if not data:
        print("FAIL=EMPTY_STREAM path=%s" % path)
        sys.exit(2)
    if len(data) % record_bytes != 0:
        print("FAIL=RECORD_SIZE path=%s bytes=%d record_bytes=%d remainder=%d"
              % (path, len(data), record_bytes, len(data) % record_bytes))
        sys.exit(2)
    count = len(data) // record_bytes
    digests = [
        hashlib.sha256(data[i * record_bytes:(i + 1) * record_bytes]).digest()
        for i in range(count)
    ]
    return digests, len(data)


def parse_window_tokens(argv):
    """Return the window size in tokens, or exit 2 on a bad value."""
    if "--window-tokens" not in argv:
        return DEFAULT_WINDOW_TOKENS
    index = argv.index("--window-tokens") + 1
    if index >= len(argv):
        print("FAIL=USAGE missing value for --window-tokens")
        sys.exit(2)
    try:
        tokens = int(argv[index])
    except ValueError:
        print("FAIL=USAGE non-integer --window-tokens")
        sys.exit(2)
    if tokens <= 0:
        print("FAIL=USAGE non-positive --window-tokens")
        sys.exit(2)
    return tokens


def report(record_bytes, train, train_bytes, evaluation, eval_bytes):
    """Print every measured value and return the overlap count."""
    train_set = set(train)
    overlap = sum(1 for digest in evaluation if digest in train_set)
    print("RECORD_BYTES=%d" % record_bytes)
    print("TRAIN_BYTES=%d TRAIN_WINDOWS=%d TRAIN_UNIQUE=%d"
          % (train_bytes, len(train), len(train_set)))
    print("EVAL_BYTES=%d EVAL_WINDOWS=%d EVAL_UNIQUE=%d"
          % (eval_bytes, len(evaluation), len(set(evaluation))))
    print("OVERLAP_WINDOWS=%d" % overlap)
    print("LEAKAGE_PCT=%.6f" % (100.0 * overlap / len(evaluation)))
    print("HELDOUT_DISJOINT=%s" % ("PASS" if overlap == 0 else "FAIL"))
    return overlap


def main(argv):
    positional = [arg for arg in argv[1:] if not arg.startswith("--")]
    if "--window-tokens" in argv:
        value_index = argv.index("--window-tokens") + 1
        if value_index < len(argv) and argv[value_index] in positional:
            positional.remove(argv[value_index])
    if len(positional) != 2:
        print("FAIL=USAGE expected TRAIN.bin EVAL.bin [--window-tokens N]")
        return 2
    window_tokens = parse_window_tokens(argv)
    record_bytes = window_tokens * TOKEN_BYTES
    print("WINDOW_TOKENS=%d" % window_tokens)
    train, train_bytes = read_digests(positional[0], record_bytes)
    evaluation, eval_bytes = read_digests(positional[1], record_bytes)
    overlap = report(record_bytes, train, train_bytes, evaluation, eval_bytes)
    return 0 if overlap == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
