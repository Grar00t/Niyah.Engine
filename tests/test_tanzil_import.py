"""Offline importer contracts; synthetic fixtures are not Quran quotations."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from import_tanzil import parse_source, prepare

NOTICE = ("# PLEASE DO NOT REMOVE OR CHANGE THIS COPYRIGHT BLOCK\n"
          "# Tanzil Quran Text (Simple Clean, Version 1.1)\n"
          "# License: Creative Commons Attribution 3.0\n")
RAW = ("1|1|  first text  \n1|2|second text\n2|1|third text\n\n" + NOTICE).encode()


class ImportTests(unittest.TestCase):
    def test_exact_text(self):
        rows, _, counts = parse_source(RAW, 2, 3)
        self.assertEqual(rows[0]["text"], "  first text  ")
        self.assertEqual(counts, {1: 2, 2: 1})

    def test_utf8(self):
        with self.assertRaises(UnicodeError): parse_source(b"\xff" + RAW, 2, 3)

    def test_replacement(self):
        with self.assertRaises(ValueError): parse_source(RAW.replace(b"first", "\ufffd".encode()), 2, 3)

    def test_duplicate(self):
        with self.assertRaises(ValueError): parse_source(RAW.replace(b"1|2", b"1|1"), 2, 3)

    def test_gap(self):
        with self.assertRaises(ValueError): parse_source(RAW.replace(b"2|1", b"3|1"), 2, 3)

    def test_count(self):
        with self.assertRaises(ValueError): parse_source(RAW, 2, 4)

    def test_notice_required(self):
        with self.assertRaises(ValueError): parse_source(b"1|1|text\n", 1, 1)

    def test_trailing_content(self):
        with self.assertRaises(ValueError): parse_source(RAW + b"3|1|bad\n", 2, 3)

    def test_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / "source.txt"; source.write_bytes(RAW)
            dest = root / "prepared"; digest = hashlib.sha256(RAW).hexdigest()
            manifest = prepare(source, digest, dest, 2, 3)
            self.assertEqual((dest / "source.txt").read_bytes(), RAW)
            rows = [json.loads(x) for x in (dest / "quran.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["text"], "  first text  ")
            self.assertEqual(manifest["verse_count"], 3)
            self.assertTrue((dest / "manifest.json").is_file())
            with self.assertRaises(FileExistsError): prepare(source, digest, dest, 2, 3)
            self.assertEqual((dest / "source.txt").read_bytes(), RAW)

    def test_unicode_preserved(self):
        value = "\u0646\u0635 \u0639\u0631\u0628\u064a \u062a\u062c\u0631\u064a\u0628\u064a"
        raw = RAW.replace(b"second text", value.encode("utf-8"))
        rows, _, _ = parse_source(raw, 2, 3)
        self.assertEqual(rows[1]["text"], value)

    def test_edition_required(self):
        with self.assertRaises(ValueError):
            parse_source(RAW.replace(b"Version 1.1", b"Version 9.9"), 2, 3)

    def test_split_reports_identical_text(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / "source.txt"
            raw = ("".join(f"{i}|1|identical synthetic text\n" for i in range(1, 21)) + NOTICE).encode()
            source.write_bytes(raw)
            m = prepare(source, hashlib.sha256(raw).hexdigest(), root / "out", 20, 20)
            self.assertEqual(m["train_verses"], 19)
            self.assertEqual(m["validation_verses"], 1)
            self.assertEqual(m["exact_text_overlap_train_validation"], 1)
            self.assertEqual(m["overlap_with_prior_corpora"], "NOT_VERIFIED")

    def test_bad_hash_no_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / "source.txt"; source.write_bytes(RAW)
            dest = root / "prepared"
            with self.assertRaises(ValueError): prepare(source, "0" * 64, dest, 2, 3)
            self.assertFalse(dest.exists())


if __name__ == "__main__":
    unittest.main()
