#!/usr/bin/env python3
"""Offline onion manifest metadata validation; no network or content inspection.

URL validation checks hostname syntax only, NOT onion v3 checksum/version.
Hashes, license evidence and review labels are declarations; this tool does
not verify content hashes, legal permission, malware absence or actual PII.
REFERENCE_ONLY and QUARANTINE do not grant training/RAG permissions.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit

ALLOWED_DECISIONS = {"REFERENCE_ONLY", "RAG_ALLOWED", "TRAIN_ALLOWED", "QUARANTINE"}
ALLOWED_CATEGORIES = {"academic", "archive", "books", "culture", "journalism", "language",
                      "open_source_docs", "public_interest", "privacy_rights", "technical", "forum"}
STRING_FIELDS = {"source_url", "source_name", "retrieved_at", "content_type", "language", "category",
                 "license", "license_evidence", "sha256", "pii_review", "malware_review", "decision"}
PERMISSIONS = {"redistribution_allowed", "training_allowed", "rag_allowed"}
REQUIRED = STRING_FIELDS | PERMISSIONS | {"bytes"}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
ONION_HOST_RE = re.compile(r"[a-z2-7]{56}\.onion", re.I)


def valid_url(value):
    if not isinstance(value, str) or any(c.isspace() or ord(c) < 32 for c in value):
        return False
    try:
        parts = urlsplit(value)
        return (parts.scheme in {"http", "https"} and parts.username is None and parts.password is None
                and parts.port is None and ONION_HOST_RE.fullmatch(parts.hostname or "") is not None)
    except ValueError:
        return False


def validate_record(obj, line_no):
    prefix = f"line_{line_no}:"
    if not isinstance(obj, dict):
        return [prefix + "record_must_be_object"]
    missing = sorted(REQUIRED - obj.keys())
    if missing:
        return [prefix + "missing:" + ",".join(missing)]
    errors = []
    for key in sorted(STRING_FIELDS):
        if not isinstance(obj[key], str) or not obj[key].strip():
            errors.append(prefix + key + "_must_be_nonempty_string")
    for key in sorted(PERMISSIONS):
        if type(obj[key]) is not bool:
            errors.append(prefix + key + "_must_be_boolean")
    if type(obj["bytes"]) is not int or obj["bytes"] < 0:
        errors.append(prefix + "invalid_bytes")
    if not valid_url(obj["source_url"]):
        errors.append(prefix + "invalid_onion_url")
    for field, choices in (("category", ALLOWED_CATEGORIES), ("decision", ALLOWED_DECISIONS)):
        if not isinstance(obj[field], str) or obj[field] not in choices:
            errors.append(prefix + "invalid_" + field)
    if not isinstance(obj["sha256"], str) or not SHA256_RE.fullmatch(obj["sha256"]):
        errors.append(prefix + "invalid_sha256")
    try:
        value = obj["retrieved_at"]
        if not isinstance(value, str):
            raise ValueError("not a timestamp")
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if stamp.tzinfo is None or "T" not in value:
            raise ValueError("timestamp requires date, time and timezone")
    except (ValueError, TypeError, OverflowError):
        errors.append(prefix + "invalid_retrieved_at")
    decision = obj["decision"]
    if decision == "TRAIN_ALLOWED":
        for key in ("training_allowed", "redistribution_allowed"):
            if obj[key] is not True:
                errors.append(prefix + "train_decision_without_" + key)
        if isinstance(obj["license"], str) and obj["license"].strip().lower() in {"unknown", "none", "n/a"}:
            errors.append(prefix + "train_decision_without_license")
    if decision == "RAG_ALLOWED" and obj["rag_allowed"] is not True:
        errors.append(prefix + "rag_decision_without_rag_permission")
    if decision == "REFERENCE_ONLY" or decision == "QUARANTINE":
        if obj["training_allowed"] is True or obj["rag_allowed"] is True:
            errors.append(prefix + "decision_conflicts_with_permissions")
    for key in ("pii_review", "malware_review"):
        if not isinstance(obj[key], str) or obj[key] not in {"PASS", "NOT_APPLICABLE", "FAIL", "PENDING"}:
            errors.append(prefix + key + "_invalid_review")
        elif decision != "QUARANTINE" and obj[key] not in {"PASS", "NOT_APPLICABLE"}:
            errors.append(prefix + key + "_not_cleared")
    return errors


def validate_bytes(raw):
    errors, count = [], 0
    decisions = {x: 0 for x in sorted(ALLOWED_DECISIONS)}
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        return [f"invalid_utf8:{exc.start}"], count, decisions
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        count += 1
        try:
            obj = json.loads(line)
        except ValueError as exc:
            errors.append(f"line_{line_no}:invalid_json:{exc}")
            continue
        errors.extend(validate_record(obj, line_no))
        decision = obj.get("decision") if isinstance(obj, dict) else None
        if isinstance(decision, str) and decision in decisions:
            decisions[decision] += 1
    if count == 0:
        errors.append("empty_manifest")
    return errors, count, decisions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", type=Path)
    args = ap.parse_args()
    try:
        raw = args.manifest.read_bytes()
    except OSError as exc:
        print(f"VALID=NO\nFAIL=manifest_read:{exc}", file=sys.stderr)
        return 2
    errors, count, decisions = validate_bytes(raw)
    print(f"MANIFEST={args.manifest}\nMANIFEST_SHA256={hashlib.sha256(raw).hexdigest()}\nRECORDS={count}")
    print("VALIDATION_SCOPE=metadata_only\nONION_VALIDATION=hostname_pattern_only")
    for key, value in decisions.items():
        print(f"{key}={value}")
    if errors:
        print("VALID=NO")
        for error in errors[:100]:
            print(f"FAIL={error}", file=sys.stderr)
        return 1
    print("VALID=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
