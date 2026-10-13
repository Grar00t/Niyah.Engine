"""Import a hash-pinned Tanzil numbered text without altering verse text.

The output manifest is written last. Its presence marks a complete import.
Source authentication and overlap with earlier training corpora are not proved.
No network access, tokenizer training, model training, or source overwrite occurs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

SOURCE = "https://tanzil.net"
NOTICE_MARKER = "# PLEASE DO NOT REMOVE OR CHANGE THIS COPYRIGHT BLOCK"


def parse_source(raw, expected_chapters=114, expected_verses=6236):
    text = raw.decode("utf-8-sig", errors="strict")
    if "\ufffd" in text or "\x00" in text:
        raise ValueError("replacement character or NUL in source")
    position = text.find(NOTICE_MARKER)
    if position < 0:
        raise ValueError("Tanzil notice is missing")
    body, notice = text[:position], text[position:]
    if "Tanzil Quran Text (Simple Clean, Version 1.1)" not in notice:
        raise ValueError("unexpected source edition")
    if "Creative Commons Attribution 3.0" not in notice:
        raise ValueError("unexpected license notice")
    if any(line and not line.startswith("#") for line in notice.splitlines()):
        raise ValueError("unexpected content after notice")
    rows, counts, chapter, ayah = [], {}, 0, 0
    for line in body.splitlines():
        if not line:
            continue
        match = re.fullmatch(r"([1-9][0-9]*)\|([1-9][0-9]*)\|(.+)", line)
        if not match:
            raise ValueError("invalid numbered row")
        s, a, verse = int(match[1]), int(match[2]), match[3]
        if any(ord(c) < 32 for c in verse) or "\u2028" in verse or "\u2029" in verse:
            raise ValueError("unexpected control in verse")
        if s == chapter + 1 and a == 1:
            chapter, ayah = s, 0
        if s != chapter or a != ayah + 1:
            raise ValueError("non-contiguous or duplicate verse")
        ayah = a
        counts[s] = a
        rows.append({"surah": s, "ayah": a, "text": verse})
    if len(counts) != expected_chapters or len(rows) != expected_verses:
        raise ValueError("unexpected chapter or verse count")
    return rows, notice, counts


def prepare(source, expected_sha256, destination, expected_chapters=114, expected_verses=6236):
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("expected SHA256 must be 64 hex digits")
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256.lower():
        raise ValueError("source SHA256 mismatch")
    rows, notice, counts = parse_source(raw, expected_chapters, expected_verses)
    # Whole chapters remain together; the split is not an overlap audit.
    train = [r for r in rows if r["surah"] % 20 != 0]
    validation = [r for r in rows if r["surah"] % 20 == 0]
    payloads = {
        "source.txt": raw,
        "TANZIL-NOTICE.txt": notice.encode("utf-8"),
        "quran.jsonl": "".join(json.dumps({**r, "source": SOURCE,
            "edition": "Tanzil Simple Clean 1.1", "notice": "TANZIL-NOTICE.txt"},
            ensure_ascii=False) + "\n" for r in rows).encode("utf-8"),
        "train.txt": ("\n\n".join(r["text"] for r in train) + "\n").encode("utf-8"),
        "validation.txt": ("\n\n".join(r["text"] for r in validation) + "\n").encode("utf-8"),
    }
    payloads["README.md"] = ("# Tanzil import\n\nSource: " + SOURCE +
        "/download/\nLicense evidence: https://tanzil.net/docs/Text_License\n\n"
        "Keep TANZIL-NOTICE.txt with this entire dataset and derived shards. "
        "Verse strings are unchanged. No normalization or diacritic removal occurred.\n\n"
        "Whole surahs divisible by 20 form validation; all others form training. "
        "This is a proposed new experiment, not the old held-out evaluation. "
        "Overlap with prior corpora and canonical-text equivalence are NOT VERIFIED.\n"
        "The manifest contains a distribution hash, not proof of correctness or a legal opinion.\n").encode("utf-8")
    manifest = {"result": "PREPARED_NOT_TRAINED", "source": SOURCE,
        "source_sha256": digest, "edition": "Tanzil Simple Clean 1.1",
        "chapter_count": len(counts), "verse_count": len(rows),
        "chapter_verse_counts": counts, "train_verses": len(train),
        "validation_verses": len(validation), "validation_rule": "surah % 20 == 0",
        "verse_text_changed": False, "overlap_with_prior_corpora": "NOT_VERIFIED",
        "canonical_text_equivalence": "NOT_INDEPENDENTLY_VERIFIED",
        "exact_text_overlap_train_validation": len(
            {r["text"] for r in train} & {r["text"] for r in validation}),
        "files": {k: {"bytes": len(v), "sha256": hashlib.sha256(v).hexdigest()}
                  for k, v in payloads.items()}}
    # Validate fully before creating the output; never replace an existing directory.
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in payloads.items():
        with (destination / name).open("xb") as output:
            output.write(data)
    for name, data in payloads.items():
        if (destination / name).read_bytes() != data:
            raise OSError("output readback mismatch: " + name)
    with (destination / "manifest.json").open("x", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--out-directory", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = prepare(args.source, args.sha256, args.out_directory)
    except (OSError, ValueError, UnicodeError) as exc:
        parser.exit(1, "TANZIL_IMPORT_FAILED: " + str(exc) + "\n")
    print("TANZIL_IMPORT=PASS")
    print("VERSES=" + str(manifest["verse_count"]))
    print("CHAPTERS=" + str(manifest["chapter_count"]))
    print("MANIFEST=" + str(args.out_directory / "manifest.json"))


if __name__ == "__main__":
    main()
