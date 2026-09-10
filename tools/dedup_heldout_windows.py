#!/usr/bin/env python3
# SPDX-License-Identifier: UNKNOWN
"""Repair a held-out window stream by removing every window whose bytes also
occur in the training stream.

tools/check_window_disjoint.py measures the leak. This one removes it, and
prints exactly what it removed and why.

Why window-level dedup is required at all: splitting a corpus by document is
not sufficient when the corpus contains templated text. Two different source
documents can contain a byte-identical span of WINDOW_TOKENS symbols -- a
standard header, a citation line, a licence block. No document-level split
can prevent that collision. Only comparing the windows themselves can.

Stream format: a flat array of little-endian int32 token ids, WINDOW_TOKENS
ids per record, no header, no padding.

usage: dedup_heldout_windows.py TRAIN_BIN VAL_BIN OUT_BIN [WINDOW_TOKENS]

exit 0  OUT_BIN written (whether or not anything was dropped)
exit 2  malformed input or bad arguments; nothing written
exit 3  refused: every held-out window is a duplicate
"""

import hashlib
import os
import sys

DEFAULT_WINDOW_TOKENS = 65
TOKEN_BYTES = 4

# Vocabulary layout of the niyah-mini byte tokenizer: ids [0, 13) are control
# symbols, ids [13, 269) are the 256 raw bytes. Used only to render a preview
# of a dropped window. It never affects which windows are dropped.
FIRST_BYTE_ID = 13
LAST_BYTE_ID = 269


def read_stream(path, record_bytes):
    """Read a window stream, or exit 2 with a reason."""
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
    return data


def split_windows(data, record_bytes):
    return [
        data[offset:offset + record_bytes]
        for offset in range(0, len(data), record_bytes)
    ]


def preview(window):
    """Render a window as text under the byte-tokenizer assumption.

    Control ids render as '.', so a preview never claims a byte that is not
    in the stream.
    """
    ids = [
        int.from_bytes(window[i:i + TOKEN_BYTES], "little", signed=True)
        for i in range(0, len(window), TOKEN_BYTES)
    ]
    raw = bytes(
        (token - FIRST_BYTE_ID) if FIRST_BYTE_ID <= token < LAST_BYTE_ID else 0x2E
        for token in ids
    )
    return ids, raw.decode("utf-8", "replace")


def main(argv):
    if len(argv) not in (4, 5):
        print("usage: %s TRAIN_BIN VAL_BIN OUT_BIN [WINDOW_TOKENS]" % argv[0])
        return 2

    train_path, val_path, out_path = argv[1], argv[2], argv[3]

    if len(argv) == 5:
        try:
            window_tokens = int(argv[4])
        except ValueError:
            print("FATAL=WINDOW_TOKENS must be an integer, got %r" % argv[4])
            return 2
    else:
        window_tokens = DEFAULT_WINDOW_TOKENS

    if window_tokens <= 1:
        print("FATAL=WINDOW_TOKENS must be greater than 1, got %d" % window_tokens)
        return 2

    if os.path.exists(out_path):
        print("FATAL=OUT_BIN already exists, refusing to overwrite: %s" % out_path)
        return 2

    record_bytes = window_tokens * TOKEN_BYTES
    print("WINDOW_TOKENS=%d" % window_tokens)
    print("RECORD_BYTES=%d" % record_bytes)

    train_data = read_stream(train_path, record_bytes)
    val_data = read_stream(val_path, record_bytes)

    train_windows = split_windows(train_data, record_bytes)
    val_windows = split_windows(val_data, record_bytes)

    train_index = {}
    for ordinal, window in enumerate(train_windows):
        train_index.setdefault(window, ordinal)

    print(
        "TRAIN_WINDOWS=%d TRAIN_UNIQUE=%d"
        % (len(train_windows), len(train_index))
    )
    print(
        "VAL_WINDOWS=%d VAL_UNIQUE=%d"
        % (len(val_windows), len(set(val_windows)))
    )

    kept = []
    dropped = []
    seen = set()
    for ordinal, window in enumerate(val_windows):
        if window in train_index:
            dropped.append((ordinal, "TRAIN_COLLISION", train_index[window], window))
        elif window in seen:
            dropped.append((ordinal, "VAL_DUPLICATE", -1, window))
        else:
            seen.add(window)
            kept.append(window)

    for ordinal, reason, train_ordinal, window in dropped:
        ids, text = preview(window)
        specials = [token for token in ids if token < FIRST_BYTE_ID]
        print(
            "DROP VAL_IDX=%d REASON=%s TRAIN_IDX=%d SPECIALS=%s"
            % (ordinal, reason, train_ordinal, specials)
        )
        print("     PREVIEW=%r" % text)

    print("DROPPED_WINDOWS=%d" % len(dropped))
    print("KEPT_WINDOWS=%d" % len(kept))
    if len(val_windows) > 0:
        print(
            "DROPPED_PCT=%.6f"
            % (100.0 * len(dropped) / len(val_windows))
        )
    print("PREDICTIONS_BEFORE=%d" % (len(val_windows) * (window_tokens - 1)))
    print("PREDICTIONS_AFTER=%d" % (len(kept) * (window_tokens - 1)))

    if not kept:
        print("FATAL=every held-out window collides; nothing written")
        return 3

    payload = b"".join(kept)
    try:
        with open(out_path, "wb") as handle:
            handle.write(payload)
    except OSError as exc:
        print("FATAL=cannot write %s: %s" % (out_path, exc))
        return 2

    print("OUT_PATH=%s" % out_path)
    print("OUT_BYTES=%d" % len(payload))
    print("OUT_SHA256=%s" % hashlib.sha256(payload).hexdigest())
    print("HELDOUT_DISJOINT=PASS")
    print("DEDUP_COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
