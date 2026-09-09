#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import pathlib
import struct
import sys
import unicodedata
from dataclasses import dataclass
from typing import Iterable

MAGIC = b"NIYAHW1\x00"
VERSION = 1
BYTE_BASE = 13
BOS = 1
EOS = 2
VOCAB_SIZE = 269
BLOCK_BYTES = 256


@dataclass(frozen=True)
class Source:
    source_id: str
    title: str
    publisher: str
    canonical_uri: str
    retrieved_at: str
    language: str
    media_type: str
    path: pathlib.Path
    raw: bytes
    normalized: bytes
    raw_sha256: str
    normalized_sha256: str


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_text(raw: bytes) -> bytes:
    text = raw.decode("utf-8", errors="strict")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = unicodedata.normalize("NFC", text)
    return (text.rstrip("\n") + "\n").encode("utf-8")


def read_tsv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise RuntimeError(f"empty manifest: {path}")
    return rows


def resolve_path(manifest: pathlib.Path, row: dict[str, str]) -> pathlib.Path:
    value = row.get("path", "").strip()
    if not value:
        raise RuntimeError(f"{manifest}: missing path for {row.get('source_id', '?')}")
    p = pathlib.Path(value)
    if not p.is_absolute():
        p = (manifest.parent / p).resolve()
    if not p.is_file():
        raise RuntimeError(f"source missing: {p}")
    return p


def make_source(manifest: pathlib.Path, row: dict[str, str]) -> Source:
    sid = row.get("source_id", "").strip()
    if not sid:
        raise RuntimeError(f"{manifest}: empty source_id")

    path = resolve_path(manifest, row)
    raw = path.read_bytes()
    normalized = normalize_text(raw)
    raw_hash = sha256_hex(raw)
    norm_hash = sha256_hex(normalized)

    declared_raw = row.get("raw_sha256", "").strip()
    declared_norm = row.get("normalized_sha256", "").strip()
    if declared_raw and declared_raw != raw_hash:
        raise RuntimeError(f"{sid}: declared raw SHA mismatch")
    if declared_norm and declared_norm != norm_hash:
        raise RuntimeError(f"{sid}: declared normalized SHA mismatch")

    return Source(
        source_id=sid,
        title=row.get("title", ""),
        publisher=row.get("publisher", ""),
        canonical_uri=row.get("canonical_uri", "").strip(),
        retrieved_at=row.get("retrieved_at", ""),
        language=row.get("language", ""),
        media_type=row.get("media_type", "text/plain"),
        path=path,
        raw=raw,
        normalized=normalized,
        raw_sha256=raw_hash,
        normalized_sha256=norm_hash,
    )


def load_training(manifests: Iterable[pathlib.Path]) -> list[Source]:
    result: list[Source] = []
    for manifest in manifests:
        for row in read_tsv(manifest):
            result.append(make_source(manifest, row))
    if not result:
        raise RuntimeError("no training sources loaded")
    return result


def load_candidates(manifest: pathlib.Path) -> list[Source]:
    result: list[Source] = []
    seen_ids: set[str] = set()
    seen_uris: set[str] = set()

    for row in read_tsv(manifest):
        sid = row.get("source_id", "").strip()
        uri = row.get("canonical_uri", "").strip()

        if row.get("author_type", "").strip().lower() != "human":
            raise RuntimeError(f"{sid}: author_type must be human")
        if row.get("source_kind", "").strip().lower() != "primary":
            raise RuntimeError(f"{sid}: source_kind must be primary")
        if row.get("synthetic", "").strip().lower() not in {"no", "false", "0"}:
            raise RuntimeError(f"{sid}: synthetic must be no")

        if sid in seen_ids:
            raise RuntimeError(f"duplicate candidate source_id: {sid}")
        if uri and uri in seen_uris:
            raise RuntimeError(f"duplicate candidate canonical_uri: {uri}")

        src = make_source(manifest, row)
        result.append(src)
        seen_ids.add(sid)
        if uri:
            seen_uris.add(uri)

    if not result:
        raise RuntimeError("no candidate sources loaded")
    return result


def block_digest(block: memoryview) -> bytes:
    return hashlib.blake2b(block, digest_size=16).digest()


def build_block_index(training: list[Source]) -> set[bytes]:
    index: set[bytes] = set()
    for source in training:
        data = memoryview(source.normalized)
        if len(data) < BLOCK_BYTES:
            continue
        for i in range(len(data) - BLOCK_BYTES + 1):
            index.add(block_digest(data[i:i + BLOCK_BYTES]))
    return index


def exact_block_exists(block: bytes, training: list[Source]) -> bool:
    return any(block in source.normalized for source in training)


def contamination_reason(
    candidate: Source,
    training: list[Source],
    training_ids: set[str],
    training_uris: set[str],
    raw_hashes: set[str],
    normalized_hashes: set[str],
    block_index: set[bytes],
) -> str | None:
    if candidate.source_id in training_ids:
        return "SOURCE_ID_MATCH"
    if candidate.canonical_uri and candidate.canonical_uri in training_uris:
        return "CANONICAL_URI_MATCH"
    if candidate.raw_sha256 in raw_hashes:
        return "RAW_SHA256_MATCH"
    if candidate.normalized_sha256 in normalized_hashes:
        return "NORMALIZED_SHA256_MATCH"

    data = memoryview(candidate.normalized)
    if len(data) < BLOCK_BYTES:
        return None

    for i in range(len(data) - BLOCK_BYTES + 1):
        view = data[i:i + BLOCK_BYTES]
        digest = block_digest(view)
        if digest not in block_index:
            continue
        block = bytes(view)
        if exact_block_exists(block, training):
            return f"EXACT_256_BYTE_BLOCK_OVERLAP:offset={i}"

    return None


def tokenize_bytes(body: bytes) -> list[int]:
    return [BOS, *(BYTE_BASE + b for b in body), EOS]


def make_windows(source: Source, seq: int) -> list[list[int]]:
    tokens = tokenize_bytes(source.normalized)
    width = seq + 1
    if len(tokens) < width:
        return []

    windows: list[list[int]] = []
    for start in range(0, len(tokens) - width + 1, seq):
        record = tokens[start:start + width]
        if len(record) != width:
            raise AssertionError("window width invariant")
        windows.append(record)
    return windows


def write_niyahw1(
    path: pathlib.Path,
    windows: list[list[int]],
    seq: int,
    corpus_sha256: bytes,
) -> None:
    header = struct.pack(
        "<8sIIIIQ32s",
        MAGIC,
        VERSION,
        seq,
        VOCAB_SIZE,
        0,
        len(windows),
        corpus_sha256,
    )
    if len(header) != 64:
        raise AssertionError("NIYAHW1 header must be 64 bytes")

    fmt = "<" + "i" * (seq + 1)
    with path.open("wb") as f:
        f.write(header)
        for record in windows:
            f.write(struct.pack(fmt, *record))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-manifest", required=True, type=pathlib.Path)
    ap.add_argument("--train-manifest", required=True, action="append", type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--seq", type=int, required=True)
    args = ap.parse_args()

    if args.seq <= 0:
        raise RuntimeError("seq must be positive")

    training = load_training(args.train_manifest)
    candidates = load_candidates(args.candidate_manifest)

    training_ids = {s.source_id for s in training}
    training_uris = {s.canonical_uri for s in training if s.canonical_uri}
    raw_hashes = {s.raw_sha256 for s in training}
    norm_hashes = {s.normalized_sha256 for s in training}
    block_index = build_block_index(training)

    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / "contamination-report.tsv"

    accepted: list[Source] = []
    with report_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["source_id", "status", "reason"])

        for candidate in candidates:
            reason = contamination_reason(
                candidate,
                training,
                training_ids,
                training_uris,
                raw_hashes,
                norm_hashes,
                block_index,
            )
            if reason:
                writer.writerow([candidate.source_id, "REJECTED", reason])
            else:
                writer.writerow([candidate.source_id, "ACCEPTED", "NONE"])
                accepted.append(candidate)

    if not accepted:
        raise RuntimeError("no clean candidate source survived contamination gates")

    corpus = b"".join(s.normalized for s in accepted)
    text_path = args.out / "heldout-v3.txt"
    text_path.write_bytes(corpus)

    all_windows: list[list[int]] = []
    counts: dict[str, int] = {}
    for source in accepted:
        windows = make_windows(source, args.seq)
        counts[source.source_id] = len(windows)
        all_windows.extend(windows)

    if not all_windows:
        raise RuntimeError("no evaluation windows produced")

    bin_path = args.out / "heldout-v3.niyahw1"
    write_niyahw1(bin_path, all_windows, args.seq, hashlib.sha256(corpus).digest())

    manifest_path = args.out / "manifest.tsv"
    fields = [
        "source_id", "title", "publisher", "canonical_uri", "retrieved_at",
        "language", "media_type", "raw_sha256", "normalized_sha256",
        "normalized_bytes", "window_count", "provenance", "training_overlap",
        "synthetic", "accepted",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for source in accepted:
            writer.writerow({
                "source_id": source.source_id,
                "title": source.title,
                "publisher": source.publisher,
                "canonical_uri": source.canonical_uri,
                "retrieved_at": source.retrieved_at,
                "language": source.language,
                "media_type": source.media_type,
                "raw_sha256": source.raw_sha256,
                "normalized_sha256": source.normalized_sha256,
                "normalized_bytes": len(source.normalized),
                "window_count": counts[source.source_id],
                "provenance": "HUMAN_PRIMARY",
                "training_overlap": "NONE",
                "synthetic": "NO",
                "accepted": "YES",
            })

    checksum_targets = [text_path, bin_path, manifest_path, report_path]
    sums = args.out / "SHA256SUMS"
    with sums.open("w", encoding="utf-8") as f:
        for p in checksum_targets:
            f.write(f"{sha256_hex(p.read_bytes())}  {p.name}\n")

    print(f"TRAINING_SOURCES={len(training)}")
    print(f"CANDIDATE_SOURCES={len(candidates)}")
    print(f"ACCEPTED_SOURCES={len(accepted)}")
    print(f"REJECTED_SOURCES={len(candidates) - len(accepted)}")
    print(f"SEQ={args.seq}")
    print(f"WINDOWS={len(all_windows)}")
    print(f"HELDOUT_SHA256={sha256_hex(corpus)}")
    print("HELDOUT_V3_CONTAMINATION_GATE=PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL={exc}", file=sys.stderr)
        raise SystemExit(1)
