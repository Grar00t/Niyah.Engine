#!/usr/bin/env python3
"""Build line-oriented Arabic text from explicitly pinned dataset sources.

NFKC normalization, HTML/URL removal, tatweel removal and optional diacritic
removal precede line extraction. Each retained line is one record; original
multi-line document boundaries are not preserved. --max-mb is a strict UTF-8
MiB cap including newline bytes (no overshoot); it is not a token budget.
License labels/evidence are recorded source claims, not permission proof.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import unicodedata

from corpus_common import publish_files
from prep_kaggle_ar import arabic_letter_ratio

TASHKEEL = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
TATWEEL = re.compile(r"\u0640")
URL = re.compile(r"https?://\S+|www\.\S+")
HTML = re.compile(r"<[^>]+>")
SPACES = re.compile(r"[ \t\u00A0\u200B-\u200F\u202A-\u202E\uFEFF]+")


def clean(text, keep_tashkeel):
    text = unicodedata.normalize("NFKC", text)
    text = HTML.sub(" ", URL.sub(" ", text))
    text = TATWEEL.sub("", text)
    if not keep_tashkeel:
        text = TASHKEEL.sub("", text)
    text = SPACES.sub(" ", text)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def good(line, min_chars, min_ar):
    return len(line) >= min_chars and any(c.isalpha() for c in line) and arabic_letter_ratio(line) >= min_ar


def validate_arguments(max_mb, min_chars, ratio):
    if not math.isfinite(max_mb) or not math.isfinite(max_mb * 1048576) or max_mb <= 0 or max_mb * 1048576 < 1:
        raise ValueError("max-mb must be finite and at least one byte")
    if min_chars < 1 or not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError("min-chars must be positive and min-arabic-ratio in [0,1]")


def prepare_rows(rows, *, max_bytes, min_chars, ratio, keep_tashkeel):
    seen, lines, written = set(), [], 0
    for raw in rows:
        if not isinstance(raw, str):
            raise ValueError("source text must be a string")
        for line in clean(raw, keep_tashkeel).split("\n"):
            if not good(line, min_chars, ratio) or line in seen or "<|doc|>" in line:
                continue
            encoded_size = len(line.encode("utf-8")) + 1
            if written + encoded_size > max_bytes:
                return lines
            seen.add(line)
            lines.append(line)
            written += encoded_size
            if written == max_bytes:
                return lines
    return lines


def source_rows(sources, loader=None):
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources must be a nonempty JSON list")
    for source in sources:
        keys = ("repo", "split", "revision", "text_key", "license_claim", "license_evidence")
        if not isinstance(source, dict) or any(not isinstance(source.get(k), str) or not source[k].strip() for k in keys):
            raise ValueError("source requires repo/split/revision/text_key/license_claim/license_evidence")
        if not re.fullmatch(r"[0-9a-fA-F]{40}", source["revision"]):
            raise ValueError("source revision must be a pinned 40-hex commit")
    if loader is None:
        from datasets import load_dataset
        loader = load_dataset
    for source in sources:
        kwargs = dict(split=source["split"], revision=source["revision"], streaming=True)
        if source.get("config") is not None:
            if not isinstance(source["config"], str):
                raise ValueError("source config must be a string")
            kwargs["name"] = source["config"]
        for row in loader(source["repo"], **kwargs):
            if not isinstance(row, dict) or not isinstance(row.get(source["text_key"]), str):
                raise ValueError(f"invalid source schema: {source['repo']}:{source['text_key']}")
            yield row[source["text_key"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", type=Path, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-mb", type=float, default=300)
    ap.add_argument("--min-chars", type=int, default=30)
    ap.add_argument("--min-arabic-ratio", type=float, default=0.6)
    ap.add_argument("--keep-tashkeel", action="store_true")
    args = ap.parse_args()
    try:
        validate_arguments(args.max_mb, args.min_chars, args.min_arabic_ratio)
        sources = json.loads(args.sources.read_text(encoding="utf-8"))
        lines = prepare_rows(source_rows(sources), max_bytes=int(args.max_mb * 1048576),
                             min_chars=args.min_chars, ratio=args.min_arabic_ratio, keep_tashkeel=args.keep_tashkeel)
        if not lines:
            raise ValueError("no retained lines; no output published")
        payload = ("\n".join(lines) + "\n").encode("utf-8")
        metadata = {"schema_version": 1, "sources": sources, "bytes": len(payload), "lines": len(lines),
                    "sha256": hashlib.sha256(payload).hexdigest(), "normalization": "NFKC; one retained line per record",
                    "keep_tashkeel": args.keep_tashkeel, "min_chars": args.min_chars,
                    "min_arabic_letter_ratio": args.min_arabic_ratio, "max_bytes": int(args.max_mb * 1048576),
                    "license_note": "recorded source claims, not independent permission verification",
                    "privacy_note": "URL removal is not comprehensive personal-data removal"}
        publish_files([(args.out, payload), (args.out + ".provenance.json", (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))])
    except (OSError, UnicodeError, ValueError, ImportError) as exc:
        ap.exit(1, f"PREPARE_FAILED: {exc}\n")
    print(f"lines={len(lines)} utf8_bytes={len(payload)}", file=sys.stderr)


if __name__ == "__main__":
    main()
