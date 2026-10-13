"""Offline behavior regressions; no corpus download or existing artifact writes."""
import concurrent.futures
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_mix
import corpus_common as common
import merge_mix
import prep_kaggle_ar as kaggle
import prepare_arabic_corpus as arabic
import validate_onion_manifest as validator
import audit_corpus_overlap as overlap


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.fixture = tempfile.TemporaryDirectory(prefix="niyah-corpus-test-")
        self.root = Path(self.fixture.name)

    def tearDown(self):
        self.fixture.cleanup()

    def test_exact_dedup_and_digest_collision(self):
        self.assertEqual(common.unique_texts(["alpha", "beta", "alpha"], lambda _: b"collision"), ["alpha", "beta"])

    def test_publication_existing_destination_unchanged(self):
        path = self.root / "existing"
        path.write_bytes(b"preserve")
        with self.assertRaises(FileExistsError):
            common.publish_files([(path, b"replace")])
        self.assertEqual(path.read_bytes(), b"preserve")
        self.assertEqual(list(self.root.glob(".niyah-corpus-*")), [])

    def test_partial_staging_is_not_published(self):
        path = self.root / "output"
        with patch.object(common.os, "fsync", side_effect=OSError("injected disk failure")):
            with self.assertRaisesRegex(OSError, "injected disk failure"):
                common.publish_files([(path, b"staged")])
        self.assertFalse(path.exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_publication_failure_preserves_racing_destination(self):
        path = self.root / "output"
        def fail_link(temporary, destination):
            Path(destination).write_bytes(b"other writer")
            raise FileExistsError(destination)
        with patch.object(common.os, "link", side_effect=fail_link):
            with self.assertRaises(FileExistsError):
                common.publish_files([(path, b"ours")])
        self.assertEqual(path.read_bytes(), b"other writer")
        self.assertEqual(list(self.root.glob(".niyah-corpus-*")), [])

    def test_access_failure_fails_closed(self):
        path = self.root / "output"
        with patch.object(common.os, "link", side_effect=PermissionError("injected ACL refusal")):
            with self.assertRaises(PermissionError):
                common.publish_files([(path, b"ours")])
        self.assertFalse(path.exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_cleanup_failure_is_visible(self):
        with patch.object(Path, "unlink", side_effect=PermissionError("injected cleanup refusal")):
            with self.assertRaisesRegex(OSError, "temporary cleanup failed"):
                common.publish_files([(self.root / "output", b"complete")])
        self.assertEqual((self.root / "output").read_bytes(), b"complete")
        for path in self.root.glob(".niyah-corpus-*"):
            path.unlink()

    def test_invalid_parent(self):
        path = self.root / "absent" / "output"
        with self.assertRaises(FileNotFoundError):
            common.publish_files([(path, b"ours")])
        self.assertFalse(path.exists())

    def test_hardlink_alias_does_not_overwrite(self):
        path, alias = self.root / "original", self.root / "alias"
        path.write_bytes(b"preserve")
        os.link(path, alias)
        with self.assertRaises(FileExistsError):
            common.publish_files([(alias, b"replace")])
        self.assertEqual(path.read_bytes(), b"preserve")

    def test_concurrent_contenders_one_winner(self):
        path = self.root / "destination"
        barrier = threading.Barrier(8)
        def contender(number):
            barrier.wait()
            try:
                common.publish_files([(path, str(number).encode())])
                return number
            except FileExistsError:
                return None
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            winners = [n for n in pool.map(contender, range(8)) if n is not None]
        self.assertEqual(len(winners), 1)
        self.assertEqual(path.read_bytes(), str(winners[0]).encode())
        self.assertEqual(list(self.root.glob(".niyah-corpus-*")), [])

    def test_merge_deterministic_sorted_inputs_and_no_overlap(self):
        a, b = self.root / "a", self.root / "b"
        a.write_text(common.SEP.join(["one", "two", "one"]), encoding="utf-8")
        b.write_text(common.SEP.join(["three", "two"]), encoding="utf-8")
        for name, inputs in (("first", [b, a]), ("second", [a, b])):
            merge_mix.merge(inputs, self.root / name, self.root / (name + "val"))
        self.assertEqual((self.root / "first").read_bytes(), (self.root / "second").read_bytes())
        train = set((self.root / "first").read_text().split(common.SEP)) - {""}
        val = set((self.root / "firstval").read_text().split(common.SEP)) - {""}
        self.assertEqual(train | val, {"one", "two", "three"})
        self.assertFalse(train & val)
        metadata = json.loads((self.root / "first.provenance.json").read_text())
        self.assertEqual(metadata["sha256"], common.sha256_file(self.root / "first"))

    def test_empty_or_one_document_split_rejected(self):
        for docs in ([], ["one"]):
            with self.assertRaises(ValueError):
                common.split_documents(docs)
        with self.assertRaises(ValueError):
            common.publish_corpus([(self.root / "out", [])], {})
        self.assertEqual(list(self.root.iterdir()), [])

    def source(self):
        return dict(name="fixture", repo="local/fixture", revision="a" * 40,
                    config="ar", split="train", text_key="text", character_budget=300,
                    license_claim="fixture", license_evidence="generated locally")

    def test_cache_hit_verifies_contract_content_and_pin_forwarding(self):
        source = self.source()
        calls = []
        def loader(*args, **kwargs):
            calls.append((args, kwargs))
            return [{"text": "a" * 200}, {"text": "b" * 200}]
        first = build_mix.fetch(source, self.root, loader)
        second = build_mix.fetch(source, self.root, lambda *a, **k: self.fail("cache fetched remote"))
        self.assertEqual(first, second)
        self.assertEqual(calls[0][0], (source["repo"], source["config"]))
        self.assertEqual(calls[0][1]["revision"], source["revision"])
        cache = next(self.root.glob("*.json"))
        data = json.loads(cache.read_text())
        data["documents"][0] = "c" * 200
        cache.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "invalid cache"):
            build_mix.fetch(source, self.root, loader)

    def test_cache_revision_budget_selection_and_legacy(self):
        source = self.source()
        legacy = self.root / "_cache_fixture.jsonl"
        legacy.write_text('"legacy"\n')
        for changes in ({}, {"revision": "b" * 40}, {"character_budget": 200}, {"text_key": "content"}):
            current = dict(source, **changes)
            build_mix.fetch(current, self.root, lambda *a, **k: [{"text": "a" * 400, "content": "b" * 400}])
        self.assertEqual(len(list(self.root.glob("*.json"))), 4)
        self.assertEqual(legacy.read_text(), '"legacy"\n')
        with self.assertRaises(ValueError):
            build_mix.source_contract(dict(source, revision="main"))
        with self.assertRaises(ValueError):
            build_mix.source_contract(dict(source, character_budget=True))

    def test_cache_short_or_bad_schema_does_not_publish(self):
        for rows in ([], [{"text": "short"}], [{"text": None}]):
            with self.assertRaises(ValueError):
                build_mix.fetch(self.source(), self.root, lambda *a, **k: rows)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_kaggle_explicit_schema_sorted_deterministic_inputs(self):
        source = self.root / "source"
        source.mkdir()
        (source / "b.json").write_text(json.dumps([{"body": "مرحبا بكم", "irrelevant": "x" * 1000}]), encoding="utf-8")
        (source / "a.csv").write_text('body,other\nأهلا بالعالم,x\n', encoding="utf-8")
        out = self.root / "out"
        self.assertEqual(kaggle.prepare(source, out, "body", min_chars=2), 2)
        self.assertTrue(out.read_text(encoding="utf-8").startswith("أهلا بالعالم"))
        self.assertNotIn("irrelevant", out.read_text(encoding="utf-8"))

    def test_kaggle_parse_schema_and_empty_fail_before_publication(self):
        source = self.root / "source"
        source.mkdir()
        file = source / "bad.jsonl"
        for content in ('{bad}\n', '{"other":"hello"}\n', '{"body":false}\n', '{"body":"x"}\n'):
            file.write_text(content)
            with self.assertRaises(ValueError):
                kaggle.prepare(source, self.root / "out", "body")
            self.assertFalse((self.root / "out").exists())
        file.unlink()
        (source / "bad.csv").write_text('body,other\n"unterminated,x\n')
        with self.assertRaisesRegex(ValueError, "INPUT_FAIL"):
            kaggle.prepare(source, self.root / "out", "body")

    def test_kaggle_input_change_rejects_provenance_mismatch(self):
        source = self.root / "source"
        source.mkdir()
        path = source / "input.jsonl"
        path.write_text('{"body":"original"}\n')
        def changing_records(_):
            yield {"body": "مرحبا بالعالم"}
            path.write_text('{"body":"changed"}\n')
        with patch.object(kaggle, "records", side_effect=changing_records):
            with self.assertRaisesRegex(ValueError, "input changed"):
                kaggle.prepare(source, self.root / "out", "body", min_chars=1)
        self.assertFalse((self.root / "out").exists())

    def test_arabic_ratio_compatible_letter_categories(self):
        self.assertFalse(arabic.good("١٢٣٤٥٦٧٨٩٠!!!abc", 1, 0.6))
        self.assertAlmostEqual(kaggle.arabic_letter_ratio("اa١٢٣!?"), 0.5)
        self.assertEqual(kaggle.arabic_letter_ratio("١٢٣!?"), 0)
        self.assertTrue(arabic.good("مرحبا بالعالم", 1, 1))

    def test_arabic_argument_ranges(self):
        for maximum, minimum, ratio in ((0, 1, .5), (-1, 1, .5), (math.nan, 1, .5),
                                         (math.inf, 1, .5), (1e308, 1, .5), (1, 0, .5),
                                         (1, 1, math.nan), (1, 1, -0.1), (1, 1, 1.1)):
            with self.assertRaises(ValueError):
                arabic.validate_arguments(maximum, minimum, ratio)

    def test_arabic_normalization_lines_exact_dedup_and_byte_cap(self):
        lines = arabic.prepare_rows(["<b>مَرْحَبًا</b>\nمَرْحَبًا\nأهلا"], max_bytes=100,
                                    min_chars=1, ratio=.5, keep_tashkeel=False)
        self.assertEqual(lines, ["مرحبا", "أهلا"])
        limited = arabic.prepare_rows(["مرحبا\nأهلا"], max_bytes=12,
                                      min_chars=1, ratio=.5, keep_tashkeel=False)
        self.assertEqual(limited, ["مرحبا"])
        self.assertLessEqual(len(("\n".join(limited) + "\n").encode()), 12)
        self.assertIn("َ", arabic.clean("مَرْحَبًا", True))

    def test_arabic_source_revision_forwarded_and_schema_rejected(self):
        source = self.source()
        calls = []
        def loader(*args, **kwargs):
            calls.append(kwargs)
            return [{"text": "مرحبا"}]
        self.assertEqual(list(arabic.source_rows([source], loader)), ["مرحبا"])
        self.assertEqual(calls[0]["revision"], source["revision"])
        with self.assertRaises(ValueError):
            list(arabic.source_rows([dict(source, revision="main")], loader))
        with self.assertRaises(ValueError):
            list(arabic.source_rows([source], lambda *a, **k: [{"text": []}]))

    def test_imports_do_not_load_optional_remote_dependencies(self):
        result = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, 'tools'); import build_mix, prep_kaggle_ar, prepare_arabic_corpus; assert 'datasets' not in sys.modules; assert 'pandas' not in sys.modules"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_overlap_independent_small_fixture(self):
        train, val = self.root / "train.jsonl", self.root / "validation.jsonl"
        text = "one two three four five six seven eight nine ten eleven twelve"
        train.write_text(json.dumps({"text": text}) + "\n" + json.dumps({"text": "tiny"}) + "\n")
        val.write_text(json.dumps({"text": text + " thirteen"}) + "\n" + json.dumps({"text": "tiny"}) + "\n")
        result = overlap.measure(train, val)
        self.assertEqual(result["counts"]["exact_train_documents_matching_validation"], 1)
        self.assertEqual(len(result["pairs_at_or_above_threshold"]), 1)
        self.assertAlmostEqual(result["maximum_jaccard_pair"]["jaccard"], 8 / 9)
        self.assertTrue(result["inputs_preserved"])

    def test_character_diagnostics_strict_utf8_and_letter_ratio(self):
        path = self.root / "bytes"
        path.write_bytes(b"abc\xff")
        command = [sys.executable, str(ROOT / "tools" / "char_dist.py"), str(path)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("byte_offset=3", result.stderr)
        path.write_text("اa١٢٣!?", encoding="utf-8")
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("arabic_letter_ratio_among_letters=0.50000000", result.stdout)
        self.assertIn("arabic_block_codepoints=4", result.stdout)
        path.write_bytes(b"")
        self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_corpus_cli_positive_fixtures_and_existing_output(self):
        # Offline replacement for the optional datasets module. This does not
        # exercise the live Hugging Face service or licensing provenance.
        (self.root / "datasets.py").write_text("def load_dataset(*args, **kwargs):\n    return [{'text': '\\u0627' * 200}, {'text': '\\u0628' * 200}]\n")
        sources = self.root / "sources.json"
        sources.write_text(json.dumps([self.source()]))
        env = dict(os.environ, PYTHONPATH=str(self.root))
        commands = [
            ["build_mix.py", "--sources", str(sources), "--cache-directory", str(self.root / "cache"),
             "--out", str(self.root / "train"), "--validation-out", str(self.root / "val")],
            ["prepare_arabic_corpus.py", "--sources", str(sources), "--out", str(self.root / "arabic"), "--max-mb", "0.01"],
        ]
        for args in commands:
            command = [sys.executable, str(ROOT / "tools" / args[0]), *args[1:]]
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            out = Path(args[args.index("--out") + 1])
            before = out.read_bytes()
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(out.read_bytes(), before)
            self.assertEqual(list(self.root.glob(".niyah-corpus-*")), [])


class ManifestTests(unittest.TestCase):
    def fixture(self):
        return dict(source_url="https://" + "a" * 56 + ".onion/path", source_name="Fixture",
                    retrieved_at="2026-10-12T12:00:00Z", content_type="text/plain", language="ar",
                    category="language", license="CC0", license_evidence="generated fixture",
                    sha256="a" * 64, bytes=12, pii_review="PASS", malware_review="NOT_APPLICABLE",
                    redistribution_allowed=True, training_allowed=True, rag_allowed=True, decision="TRAIN_ALLOWED")

    def run_manifest(self, raw):
        with tempfile.TemporaryDirectory(prefix="niyah-manifest-test-") as name:
            path = Path(name) / "manifest.jsonl"
            path.write_bytes(raw)
            return subprocess.run([sys.executable, str(ROOT / "tools" / "validate_onion_manifest.py"), str(path)], capture_output=True, text=True)

    def test_actual_valid_manifest_cli(self):
        result = self.run_manifest((json.dumps(self.fixture()) + "\n").encode())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("VALID=YES", result.stdout)
        self.assertIn("ONION_VALIDATION=hostname_pattern_only", result.stdout)

    def test_actual_invalid_manifest_fixtures_never_traceback(self):
        base = self.fixture()
        cases = [b"", b" \n\t\n", b"\xff", b"[]\n", b"{broken}\n"]
        missing = dict(base)
        missing.pop("source_name")
        cases.append(json.dumps(missing).encode())
        for field, value in (("category", []), ("category", {}), ("decision", []), ("decision", {}),
                             ("bytes", True), ("bytes", -1), ("license", ""), ("license_evidence", " "),
                             ("sha256", "x" * 64), ("source_url", "https://example.com"),
                             ("source_url", "http://" + "a" * 56 + ".onion/\nevil"),
                             ("source_name", []), ("retrieved_at", "2026-10-12"),
                             ("pii_review", {}), ("training_allowed", "yes"), ("training_allowed", False),
                             ("redistribution_allowed", False), ("decision", "REFERENCE_ONLY"),
                             ("decision", "QUARANTINE")):
            cases.append(json.dumps(dict(base, **{field: value})).encode())
        cases.append(json.dumps(dict(base, decision="RAG_ALLOWED", rag_allowed=False)).encode())
        for raw in cases:
            with self.subTest(raw=raw):
                result = self.run_manifest(raw)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("VALID=NO", result.stdout)
                self.assertIn("FAIL=", result.stderr)
                self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
