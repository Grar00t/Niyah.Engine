# Pinned Tanzil import

This offline data-preparation tool does not train or replace a model or tokenizer.
It accepts Tanzil Simple Clean 1.1 text with numbered `surah|ayah|text` rows and
requires the original copyright notice and an independently retained SHA-256.
Obtain the source from https://tanzil.net/download/ and retain the terms at
https://tanzil.net/docs/Text_License. Preserve attribution and the complete notice
with the output collection and any derived shards. No Quran text is distributed
in this source-code change.

## Command

```text
python tools/import_tanzil.py --source tanzil-simple-clean.txt --sha256 <SHA256> --out-directory new-import
```

Replace `<SHA256>` with the digest from the reviewed download, not a digest
chosen merely to silence a mismatch. The destination must not already exist.
The importer verifies UTF-8, numbering, the declared edition, 114 chapters and
6,236 verses before creating output. It does not strip, normalize or rewrite
verse strings. It retains exact input bytes as `source.txt` and writes JSONL,
blank-line-separated training and validation records, the original notice and
a content manifest. The manifest is written last after output readback.
A partial output directory without a valid manifest is not a completed import.

Whole chapters with `surah % 20 == 0` form a proposed validation partition.
Repeated text can occur across chapters; the manifest reports exact shared text
count without removing or changing verses. This is not a contamination audit of
existing corpora, a validated benchmark split, or proof of canonical accuracy.
Source authentication and independent canonical-text comparison remain separate
obligations. The expected hash pins bytes, not their semantic correctness.

Use an existing tokenizer when preparing shards for an existing checkpoint.
Do not use tokenizer-training commands to reproduce a tokenizer by vocabulary
size alone. Check the CLI of the actual build before invoking shard preparation.
Do not silently add these records to an ongoing resume experiment.

## Regression tests

```text
python -B tests/test_tanzil_import.py -v
```

Tests use synthetic numbered text, not Quran quotations. They cover exact strings,
Unicode, invalid encodings, edition/notice/count/numbering checks, refusal to
overwrite, hash mismatch, and explicit reporting of cross-partition identical
text. They do not claim model-quality improvement or completed model training.
