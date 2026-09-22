#!/usr/bin/env python3
"""Validate mechanically-verifiable NIYAH math JSONL records."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

EXPECTED_SCHEMA = "niyah.verified_math.v1"


def fail(msg: str) -> None:
    raise ValueError(msg)


def verify(v: dict) -> None:
    t = v.get("type")
    if t == "integer_exact":
        a, b, op, ans = v["a"], v["b"], v["op"], v["answer"]
        got = a + b if op == "+" else a - b if op == "-" else a * b if op == "*" else None
        if got is None or got != ans:
            fail("integer_exact mismatch")
    elif t == "fraction_exact":
        a = Fraction(*v["a"])
        b = Fraction(*v["b"])
        got = a + b if v["op"] == "+" else a - b if v["op"] == "-" else None
        if got is None or [got.numerator, got.denominator] != v["answer"]:
            fail("fraction_exact mismatch")
    elif t == "linear_exact":
        a, b, c, x = v["a"], v["b"], v["c"], v["answer"]
        if a == 0 or a * x + b != c:
            fail("linear_exact mismatch")
    elif t == "gcd_exact":
        if math.gcd(v["a"], v["b"]) != v["answer"]:
            fail("gcd_exact mismatch")
    elif t == "mod_inverse_exact":
        a, m, ans = v["a"], v["modulus"], v["answer"]
        if math.gcd(a, m) != 1 or (a * ans) % m != 1:
            fail("mod_inverse_exact mismatch")
    elif t == "comb_exact":
        if math.comb(v["n"], v["k"]) != v["answer"]:
            fail("comb_exact mismatch")
    else:
        fail(f"unsupported verification type: {t!r}")


def canonical_content(rec: dict) -> str:
    content = {
        "instruction": rec["instruction"],
        "response": rec["response"],
        "verification": rec["verification"],
    }
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--expect-records", type=int)
    args = ap.parse_args()

    ids: set[str] = set()
    count = 0
    bad = 0
    for line_no, raw in enumerate(args.path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            rec = json.loads(raw)
            if rec.get("schema") != EXPECTED_SCHEMA:
                fail("schema mismatch")
            rid = rec.get("id")
            if not isinstance(rid, str) or not rid:
                fail("invalid id")
            if rid in ids:
                fail("duplicate id")
            ids.add(rid)
            if rec.get("domain") != "mathematics" or rec.get("language") != "ar":
                fail("domain/language mismatch")
            p = rec.get("provenance") or {}
            if p.get("source") != "deterministic-generator":
                fail("unexpected provenance source")
            if p.get("status") != "MECHANICALLY_VERIFIABLE":
                fail("unexpected provenance status")
            expected_hash = hashlib.sha256(canonical_content(rec).encode("utf-8")).hexdigest()
            if p.get("content_sha256") != expected_hash:
                fail("content_sha256 mismatch")
            verify(rec["verification"])
            count += 1
        except Exception as exc:
            bad += 1
            print(f"BAD line={line_no}: {exc}")

    if args.expect_records is not None and count != args.expect_records:
        print(f"COUNT_MISMATCH valid={count} expected={args.expect_records}")
        return 2
    if bad:
        print(f"VALID={count} BAD={bad}")
        return 1

    print(f"VALID={count}")
    print("BAD=0")
    print(f"UNIQUE_IDS={len(ids)}")
    print(f"SHA256={hashlib.sha256(args.path.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
