"""Build a pinned source mixture with validated content-addressed caches.

The --sources JSON list specifies name, repo, revision (40 hex commit), config
(string or null), split, text_key, character_budget, license_claim and
license_evidence. License metadata records claims, not independent permission
proof. No remote download occurs on import or when every cache validates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from corpus_common import publish_corpus, publish_files, split_documents, unique_texts

SELECTION = {"version": 1, "min_characters": 200, "remove_cr": True,
             "strip": True, "reject_document_marker": True,
             "budget": "Unicode characters; final document may overshoot"}


def source_contract(source):
    keys = ("name", "repo", "revision", "split", "text_key", "license_claim", "license_evidence")
    if not isinstance(source, dict) or any(not isinstance(source.get(k), str) or not source[k].strip() for k in keys):
        raise ValueError("source requires nonempty name/repo/revision/split/text_key/license_claim/license_evidence")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", source["name"]):
        raise ValueError("invalid source name")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", source["revision"]):
        raise ValueError("source revision must be a pinned 40-hex commit")
    if source.get("config") is not None and not isinstance(source["config"], str):
        raise ValueError("config must be a string or null")
    budget = source.get("character_budget")
    if type(budget) is not int or budget <= 0:
        raise ValueError("character_budget must be a positive integer")
    return {**{key: source[key] for key in keys}, "config": source.get("config"),
            "character_budget": budget, "selection": SELECTION}


def document_hash(documents):
    return hashlib.sha256(json.dumps(documents, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def fetch(source, cache_directory, loader=None):
    contract = source_contract(source)
    key = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    cache = Path(cache_directory) / f"{source['name']}-{key}.json"
    if cache.exists():
        content = json.loads(cache.read_text(encoding="utf-8"))
        docs = content.get("documents") if isinstance(content, dict) else None
        if (not isinstance(content, dict) or content.get("contract") != contract or not isinstance(docs, list) or not docs or
                any(not isinstance(t, str) or len(t) < 200 or "<|doc|>" in t or t != t.replace("\r", "").strip() for t in docs) or
                content.get("documents_sha256") != document_hash(docs) or
                sum(map(len, docs)) < source["character_budget"]):
            raise ValueError(f"invalid cache: {cache}")
        return docs
    # Legacy _cache_NAME.jsonl lacks source/revision/content binding. Never reuse it.
    if loader is None:
        from datasets import load_dataset
        loader = load_dataset
    positional = [source["repo"]]
    if source.get("config") is not None:
        positional.append(source["config"])
    rows = loader(*positional, revision=source["revision"], split=source["split"], streaming=True)
    docs, used = [], 0
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"invalid record schema: {source['name']}")
        text = row.get(source["text_key"])
        if not isinstance(text, str):
            raise ValueError(f"invalid text schema: {source['name']}:{source['text_key']}")
        text = text.replace("\r", "").strip()
        if len(text) < 200 or "<|doc|>" in text:
            continue
        docs.append(text)
        used += len(text)
        if used >= source["character_budget"]:
            break
    if not docs or used < source["character_budget"]:
        raise ValueError(f"{source['name']}: character budget not reached ({used})")
    payload = {"contract": contract, "documents_sha256": document_hash(docs), "documents": docs}
    publish_files([(cache, (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))])
    return docs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", type=Path, required=True)
    ap.add_argument("--cache-directory", type=Path, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--validation-out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    try:
        sources = json.loads(args.sources.read_text(encoding="utf-8"))
        if not isinstance(sources, list) or not sources:
            raise ValueError("sources must be a nonempty list")
        contracts = [source_contract(source) for source in sources]
        if len({s["name"] for s in contracts}) != len(contracts):
            raise ValueError("duplicate source name")
        args.cache_directory.mkdir(parents=True, exist_ok=True)
        documents = []
        for source in sorted(sources, key=lambda s: s["name"]):
            documents.extend(fetch(source, args.cache_directory))
        train, val = split_documents(unique_texts(documents), args.seed)
        publish_corpus([(args.out, train), (args.validation_out, val)],
                       {"operation": "build_mix", "sources": contracts, "seed": args.seed,
                        "license_note": "source claims; not independent permission verification"})
    except (OSError, UnicodeError, ValueError, ImportError) as exc:
        ap.exit(1, f"BUILD_MIX_FAILED: {exc}\n")


if __name__ == "__main__":
    main()
