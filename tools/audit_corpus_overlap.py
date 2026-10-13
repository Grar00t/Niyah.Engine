"""Read-only exact and exhaustive word-5-gram Jaccard overlap for JSONL text."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import unicodedata

from corpus_common import publish_files, sha256_file


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def shingles(text):
    words = re.findall(r"\w+", normalized(text))
    return {tuple(words[i:i + 5]) for i in range(len(words) - 4)}


def rows(path):
    with path.open(encoding="utf-8", errors="strict") as stream:
        for index, line in enumerate(stream, 1):
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise ValueError(f"invalid text record at {path}:{index}")
            yield index, row["text"]


def measure(train, validation, threshold=0.8):
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0,1]")
    before = {str(path): sha256_file(path) for path in (train, validation)}
    val_rows = list(rows(validation))
    exact = {text for _, text in val_rows}
    nfc = {unicodedata.normalize("NFC", text) for _, text in val_rows}
    norm = {normalized(text) for _, text in val_rows}
    sizes, postings = {}, defaultdict(list)
    for index, text in val_rows:
        grams = shingles(text)
        sizes[index] = len(grams)
        for gram in grams:
            postings[gram].append(index)
    counts = {"train_documents": 0, "validation_documents": len(val_rows),
              "exact_train_documents_matching_validation": 0,
              "nfc_train_documents_matching_validation": 0,
              "nfkc_casefold_whitespace_train_documents_matching_validation": 0,
              "train_documents_without_five_word_shingles": 0,
              "validation_documents_without_five_word_shingles": sum(size == 0 for size in sizes.values())}
    near = []
    maximum = {"jaccard": 0.0, "train_line": None, "validation_line": None}
    for index, text in rows(train):
        counts["train_documents"] += 1
        counts["exact_train_documents_matching_validation"] += text in exact
        counts["nfc_train_documents_matching_validation"] += unicodedata.normalize("NFC", text) in nfc
        counts["nfkc_casefold_whitespace_train_documents_matching_validation"] += normalized(text) in norm
        grams = shingles(text)
        if not grams:
            counts["train_documents_without_five_word_shingles"] += 1
            continue
        intersections = defaultdict(int)
        for gram in grams:
            for val_index in postings.get(gram, ()):
                intersections[val_index] += 1
        for val_index, intersection in intersections.items():
            union = len(grams) + sizes[val_index] - intersection
            score = intersection / union
            pair = {"train_line": index, "validation_line": val_index, "jaccard": score,
                    "intersection": intersection, "union": union}
            if score > maximum["jaccard"]:
                maximum = pair
            if score >= threshold:
                near.append(pair)
    after = {str(path): sha256_file(path) for path in (train, validation)}
    if before != after:
        raise ValueError("input changed while measuring")
    return {"schema_version": 1, "inputs_raw_sha256": before, "inputs_preserved": True,
            "counts": counts, "near_duplicate_definition": "NFKC + casefold + whitespace normalization; regex word tokens; set Jaccard of contiguous word 5-grams",
            "near_duplicate_threshold": threshold, "pairs_at_or_above_threshold": near,
            "maximum_jaccard_pair": maximum,
            "coverage": "All train/validation document pairs; empty 5-gram sets excluded from Jaccard. No semantic, substring or historical-source independence claim."}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--validation", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.8)
    args = ap.parse_args()
    result = measure(args.train, args.validation, args.threshold)
    publish_files([(args.out, (json.dumps(result, indent=2) + "\n").encode())])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
