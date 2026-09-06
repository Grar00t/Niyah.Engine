#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bind an embedding model and vector dimension to a local space"
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dimensions", required=True, type=int)
    parser.add_argument("--version", default="1")
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "data" / "embedding_space.json"),
    )
    args = parser.parse_args()

    if args.dimensions <= 0:
        raise SystemExit("dimensions must be > 0")

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": args.model_name,
        "dimensions": args.dimensions,
        "version": args.version,
        "distance": "cosine",
        "single_model_per_space": True,
    }

    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("model_name") != args.model_name
            or existing.get("dimensions") != args.dimensions
        ):
            raise SystemExit(
                "single_model_per_space violation: manifest already binds a different model or dimension"
            )

    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", **payload}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
