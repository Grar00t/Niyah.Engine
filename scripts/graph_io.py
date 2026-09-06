from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH = ROOT / "knowledge" / "canonical_knowledge_v2.json"


def load_graph(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"graph file must contain a JSON object: {source}")

    graph = value.get("graph", value)
    if not isinstance(graph, dict):
        raise ValueError(f"graph payload must be a JSON object: {source}")

    return graph
