#!/usr/bin/env python3
"""Offline validator for benign onion-source manifests.

This tool does not crawl Tor and does not make network requests. It validates
newline-delimited JSON metadata before a source is admitted to an external
collection/RAG/training pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ALLOWED_DECISIONS = {"REFERENCE_ONLY", "RAG_ALLOWED", "TRAIN_ALLOWED", "QUARANTINE"}
ALLOWED_CATEGORIES = {
    "academic",
    "archive",
    "books",
    "culture",
    "journalism",
    "language",
    "open_source_docs",
    "public_interest",
    "privacy_rights",
    "technical",
    "forum",
}
REQUIRED = {
    "source_url",
    "source_name",
    "retrieved_at",
    "content_type",
    "language",
    "category",
    "license",
    "license_evidence",
    "sha256",
    "bytes",
    "pii_review",
    "malware_review",
    "redistribution_allowed",
    "training_allowed",
    "rag_allowed",
    "decision",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ONION_RE = re.compile(r"^https?://[a-z2-7]{56}\.onion(?:/|$)", re.I)


def fail(msg: str) -> None:
    print(f"FAIL={msg}", file=sys.stderr)


def validate_record(obj: dict, line_no: int) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED - obj.keys())
    if missing:
        errors.append(f"line_{line_no}:missing:{','.join(missing)}")
        return errors

    if not isinstance(obj["source_url"], str) or not ONION_RE.match(obj["source_url"]):
        errors.append(f"line_{line_no}:invalid_onion_url")
    if obj["category"] not in ALLOWED_CATEGORIES:
        errors.append(f"line_{line_no}:invalid_category:{obj['category']}")
    if obj["decision"] not in ALLOWED_DECISIONS:
        errors.append(f"line_{line_no}:invalid_decision:{obj['decision']}")
    if not isinstance(obj["sha256"], str) or not SHA256_RE.fullmatch(obj["sha256"]):
        errors.append(f"line_{line_no}:invalid_sha256")
    if not isinstance(obj["bytes"], int) or obj["bytes"] < 0:
        errors.append(f"line_{line_no}:invalid_bytes")

    for key in ("redistribution_allowed", "training_allowed", "rag_allowed"):
        if not isinstance(obj[key], bool):
            errors.append(f"line_{line_no}:{key}_must_be_boolean")

    if obj["decision"] == "TRAIN_ALLOWED":
        if not obj["training_allowed"]:
            errors.append(f"line_{line_no}:train_decision_without_training_permission")
        if not obj["redistribution_allowed"]:
            errors.append(f"line_{line_no}:train_decision_without_redistribution_permission")
        if not str(obj["license"]).strip() or str(obj["license"]).lower() in {"unknown", "none", "n/a"}:
            errors.append(f"line_{line_no}:train_decision_without_license")
        if not str(obj["license_evidence"]).strip():
            errors.append(f"line_{line_no}:train_decision_without_license_evidence")

    if obj["decision"] == "RAG_ALLOWED" and not obj["rag_allowed"]:
        errors.append(f"line_{line_no}:rag_decision_without_rag_permission")

    if obj["decision"] != "QUARANTINE":
        for key in ("pii_review", "malware_review"):
            if str(obj[key]).upper() not in {"PASS", "NOT_APPLICABLE"}:
                errors.append(f"line_{line_no}:{key}_not_cleared")

    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    args = ap.parse_args()

    if not args.manifest.is_file():
        fail("manifest_not_found")
        return 2

    errors: list[str] = []
    count = 0
    decisions: dict[str, int] = {x: 0 for x in sorted(ALLOWED_DECISIONS)}

    with args.manifest.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            count += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"line_{line_no}:invalid_json:{e.msg}")
                continue
            if not isinstance(obj, dict):
                errors.append(f"line_{line_no}:record_must_be_object")
                continue
            errors.extend(validate_record(obj, line_no))
            if obj.get("decision") in decisions:
                decisions[obj["decision"]] += 1

    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    print(f"MANIFEST={args.manifest}")
    print(f"MANIFEST_SHA256={manifest_hash}")
    print(f"RECORDS={count}")
    for key in sorted(decisions):
        print(f"{key}={decisions[key]}")

    if errors:
        print("VALID=NO")
        for e in errors[:100]:
            fail(e)
        return 1

    print("VALID=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
