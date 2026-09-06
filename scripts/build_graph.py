#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def canonical_key(value: Any) -> str:
    if isinstance(value, dict) and value.get("id"):
        return f"id:{value['id']}"
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "json:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def extend_unique(target: list, seen: set[str], values: Any) -> None:
    if values is None:
        return
    if not isinstance(values, list):
        raise ValueError("graph collections must be arrays")
    for value in values:
        key = canonical_key(value)
        if key in seen:
            continue
        seen.add(key)
        target.append(value)


def build_graph(chunks_dir: Path, output_path: Path, manifest_path: Path | None) -> dict:
    full = {"nodes": [], "edges": [], "evidence": [], "schemas": [], "constraints": []}
    seen = {name: set() for name in full}
    manifest = {"chunks": [], "errors": []}

    chunk_files = sorted(chunks_dir.glob("khawrizm_graph_chunk_*.json"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found in {chunks_dir}")

    for path in chunk_files:
        try:
            raw = path.read_bytes()
            if not raw.strip():
                raise ValueError("empty file")
            payload = json.loads(raw.decode("utf-8-sig"))
            graph = payload.get("graph", payload)
            if not isinstance(graph, dict):
                raise ValueError("graph payload must be an object")

            before = {name: len(full[name]) for name in full}
            for name in full:
                extend_unique(full[name], seen[name], graph.get(name, []))

            manifest["chunks"].append(
                {
                    "file": path.name,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "added": {name: len(full[name]) - before[name] for name in full},
                }
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            manifest["errors"].append({"file": path.name, "error": str(exc)})
            print(f"[SKIP] {path.name}: {exc}", file=sys.stderr)

    if not manifest["chunks"]:
        raise ValueError("No valid graph chunks were loaded")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(full, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if manifest_path:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(
        f"Built graph: {len(full['nodes'])} nodes, {len(full['edges'])} edges, "
        f"{len(full['evidence'])} evidence from {len(manifest['chunks'])} chunks"
    )
    if manifest["errors"]:
        print(f"Skipped {len(manifest['errors'])} invalid chunks", file=sys.stderr)
    return full


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge graph chunks without duplicating records")
    parser.add_argument("--chunks-dir", default=str(ROOT / "chunks"))
    parser.add_argument("--output", default=str(ROOT / "data" / "rebuilt_graph.json"))
    parser.add_argument("--manifest", default=str(ROOT / "data" / "build_manifest.json"))
    args = parser.parse_args()

    try:
        build_graph(
            chunks_dir=Path(args.chunks_dir),
            output_path=Path(args.output),
            manifest_path=Path(args.manifest) if args.manifest else None,
        )
        return 0
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
