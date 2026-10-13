"""Strict UTF-8 character diagnostics; Arabic block and Arabic letters differ."""
import argparse
from collections import Counter
from pathlib import Path
import sys
import unicodedata


def classify(character):
    if character.isalpha() and "ARABIC" in unicodedata.name(character, ""):
        return "arabic_letter"
    if character.isascii() and character.isalpha():
        return "latin_letter"
    if character.isdigit():
        return "digit"
    if character.isspace():
        return "space"
    return "other"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    args = ap.parse_args()
    try:
        raw = args.input.read_bytes()
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        print(f"UTF8=INVALID byte_offset={exc.start} reason={exc.reason}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"READ_FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"UTF8=VALID bytes={len(raw)} characters={len(text)}")
    counts = Counter(classify(c) for c in text)
    for key, value in sorted(counts.items()):
        print(f"{key} count={value} character_fraction={value / len(text):.8f}")
    letters = sum(c.isalpha() for c in text)
    print(f"arabic_letter_ratio_among_letters={counts['arabic_letter'] / letters if letters else 0:.8f}")
    print(f"arabic_block_codepoints={sum(0x0600 <= ord(c) <= 0x06ff for c in text)}")
    print(f"literal_replacement_characters={text.count(chr(0xfffd))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
