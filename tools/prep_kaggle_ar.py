"""Prepare local CSV/JSON/JSONL with an explicit text column and strict parsing.

URL/email removal is a limited text filter, not comprehensive PII removal.
JSON inputs must be arrays of records; JSONL inputs must have one object/line.
"""
import argparse
import csv
import json
from pathlib import Path
import re
import unicodedata

from corpus_common import publish_corpus, sha256_file, unique_texts

CTRL = re.compile(r"[\u0000-\u0008\u000b-\u001f\u200b-\u200f\u202a-\u202e]")
URL = re.compile(r"https?://\S+|www\.\S+")
MAIL = re.compile(r"\S+@\S+\.\S+")


def arabic_letter_ratio(text):
    letters = [c for c in text if c.isalpha()]
    return sum("ARABIC" in unicodedata.name(c, "") for c in letters) / len(letters) if letters else 0.0


def records(path):
    with path.open(encoding="utf-8", newline="") as stream:
        if path.suffix.lower() == ".csv":
            reader = csv.DictReader(stream, strict=True)
            if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError(f"SCHEMA_FAIL path={path}: missing or duplicate header")
            for row in reader:
                if None in row or None in row.values():
                    raise ValueError(f"PARSE_FAIL path={path} row={reader.line_num}: field count mismatch")
                yield row
        elif path.suffix.lower() == ".jsonl":
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except ValueError as exc:
                    raise ValueError(f"PARSE_FAIL path={path} line={line_no}: {exc}") from exc
        else:
            rows = json.load(stream)
            if not isinstance(rows, list):
                raise ValueError(f"SCHEMA_FAIL path={path}: JSON must be an array of records")
            yield from rows


def prepare(source, output, text_column, min_chars=200, min_arabic_ratio=0.5):
    if min_chars < 1 or not 0 <= min_arabic_ratio <= 1:
        raise ValueError("min_chars must be positive and min_arabic_ratio in [0,1]")
    paths = sorted((p for p in Path(source).rglob("*") if p.suffix.lower() in {".csv", ".json", ".jsonl"}),
                   key=lambda p: p.as_posix())
    if not paths:
        raise ValueError("no supported input files")
    identities = [{"path": str(p.resolve()), "raw_sha256": sha256_file(p)} for p in paths]
    documents = []
    for path in paths:
        try:
            for number, row in enumerate(records(path), 1):
                if not isinstance(row, dict) or not isinstance(row.get(text_column), str):
                    raise ValueError(f"SCHEMA_FAIL path={path} record={number}: text column {text_column!r} must be a string")
                text = URL.sub("", MAIL.sub("", CTRL.sub("", row[text_column].replace("\r", "")))).strip()
                if len(text) >= min_chars and "<|doc|>" not in text and arabic_letter_ratio(text) >= min_arabic_ratio:
                    documents.append(text)
        except (OSError, UnicodeError, ValueError, csv.Error) as exc:
            raise ValueError(f"INPUT_FAIL path={path}: {exc}") from exc
    documents = unique_texts(documents)
    if any(sha256_file(path) != identity["raw_sha256"] for path, identity in zip(paths, identities)):
        raise ValueError("input changed during preparation; no output published")
    publish_corpus([(output, documents)], {"operation": "prep_kaggle_ar", "text_column": text_column,
                   "min_characters": min_chars, "min_arabic_letter_ratio": min_arabic_ratio,
                   "privacy_note": "URL/email/control filtering only; not a comprehensive PII review",
                   "license_note": "not verified by this tool",
                   "inputs": identities})
    return len(documents)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--text-column", required=True)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--min-arabic-ratio", type=float, default=0.5)
    args = ap.parse_args()
    try:
        count = prepare(args.source, args.out, args.text_column, args.min_chars, args.min_arabic_ratio)
    except (OSError, ValueError) as exc:
        ap.exit(1, f"PREPARE_FAILED: {exc}\n")
    print(f"kept_documents={count}")


if __name__ == "__main__":
    main()
