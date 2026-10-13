"""Exact document deduplication and deterministic document-count split."""
import argparse
import hashlib
from pathlib import Path

from corpus_common import SEP, publish_corpus, split_documents, unique_texts


def merge(inputs, output, validation, seed=42, validation_percent=1):
    paths = sorted(map(Path, inputs), key=lambda p: str(p.resolve()))
    documents = []
    identities = []
    for path in paths:
        raw = path.read_bytes()
        identities.append({"path": str(path.resolve()), "raw_sha256": hashlib.sha256(raw).hexdigest()})
        text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        documents.extend(t for t in text.split(SEP) if t.strip())
    documents = unique_texts(documents)
    train, heldout = split_documents(documents, seed, validation_percent)
    provenance = {"operation": "merge_mix", "seed": seed,
                  "validation_percent_documents": validation_percent,
                  "inputs": identities}
    publish_corpus([(output, train), (validation, heldout)], provenance)
    return len(train), len(heldout)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--validation-out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--validation-percent", type=int, default=1)
    args = ap.parse_args()
    try:
        train, heldout = merge(args.input, args.out, args.validation_out, args.seed, args.validation_percent)
    except (OSError, UnicodeError, ValueError) as exc:
        ap.exit(1, f"MERGE_FAILED: {exc}\n")
    print(f"train_documents={train} validation_documents={heldout}")


if __name__ == "__main__":
    main()
