#!/usr/bin/env python3
"""Deterministically split JSONL records by ID hash.

Record-level splitting avoids cutting a single instruction/response record across
train and validation. The split is stable across machines and input ordering.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def bucket(record_id: str, seed: str) -> int:
    h = hashlib.sha256((seed + "\0" + record_id).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % 10000


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--validation", type=Path, required=True)
    ap.add_argument("--validation-percent", type=float, default=5.0)
    ap.add_argument("--seed", default="1448")
    args = ap.parse_args()
    if not (0.1 <= args.validation_percent <= 50.0):
        raise SystemExit("validation percent must be between 0.1 and 50")

    threshold = int(round(args.validation_percent * 100))
    seen: set[str] = set()
    train: list[tuple[str, str]] = []
    val: list[tuple[str, str]] = []

    for line_no, raw in enumerate(args.input.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        obj = json.loads(raw)
        rid = obj.get("id")
        if not isinstance(rid, str) or not rid:
            raise SystemExit(f"line {line_no}: missing string id")
        if rid in seen:
            raise SystemExit(f"line {line_no}: duplicate id {rid}")
        seen.add(rid)
        canonical = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        target = val if bucket(rid, args.seed) < threshold else train
        target.append((rid, canonical))

    train.sort(key=lambda x: x[0])
    val.sort(key=lambda x: x[0])
    if not train or not val:
        raise SystemExit(f"empty split: train={len(train)} validation={len(val)}")

    write_lines(args.train, [x[1] for x in train])
    write_lines(args.validation, [x[1] for x in val])

    train_ids = {x[0] for x in train}
    val_ids = {x[0] for x in val}
    overlap = train_ids & val_ids
    if overlap:
        raise SystemExit(f"split overlap detected: {sorted(overlap)[:5]}")

    print(f"INPUT_RECORDS={len(seen)}")
    print(f"TRAIN_RECORDS={len(train)}")
    print(f"VALIDATION_RECORDS={len(val)}")
    print("OVERLAP=0")
    print(f"TRAIN_SHA256={hashlib.sha256(args.train.read_bytes()).hexdigest()}")
    print(f"VALIDATION_SHA256={hashlib.sha256(args.validation.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
