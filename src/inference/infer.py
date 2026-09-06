#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from graph_io import DEFAULT_GRAPH, load_graph


def stable_edge_id(source: str, edge_type: str, target: str) -> str:
    raw = f"{source}|{edge_type}|{target}".encode("utf-8")
    return "e_" + hashlib.sha256(raw).hexdigest()


def edge_confidence(edge: dict) -> float:
    value = edge.get("confidence", 1.0)
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(confidence):
        return 0.0
    return max(0.0, min(1.0, confidence))


def infer_edges(graph: dict, confidence_threshold: float = 0.7) -> list[dict]:
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    node_ids = {
        node.get("id")
        for node in nodes
        if isinstance(node, dict) and node.get("id")
    }

    adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
    existing: set[tuple[str, str]] = set()
    for edge in edges:
        if not isinstance(edge, dict) or edge.get("type") != "part_of":
            continue
        source = edge.get("source")
        target = edge.get("target")
        if source not in node_ids or target not in node_ids or source == target:
            continue
        confidence = edge_confidence(edge)
        adjacency[source].append((target, confidence))
        existing.add((source, target))

    for neighbors in adjacency.values():
        neighbors.sort(key=lambda item: item[0])

    inferred: dict[tuple[str, str], dict] = {}
    for source in sorted(adjacency):
        queue: deque[tuple[str, float, tuple[str, ...]]] = deque()
        for target, confidence in adjacency[source]:
            queue.append((target, confidence, (source, target)))

        best_seen: dict[str, float] = {}
        while queue:
            current, path_confidence, path = queue.popleft()
            if path_confidence <= best_seen.get(current, -1.0):
                continue
            best_seen[current] = path_confidence

            if len(path) >= 3 and current != source:
                pair = (source, current)
                if pair not in existing and path_confidence >= confidence_threshold:
                    previous = inferred.get(pair)
                    if previous is None or path_confidence > previous["confidence"]:
                        inferred[pair] = {
                            "id": stable_edge_id(source, "part_of", current),
                            "source": source,
                            "target": current,
                            "type": "part_of",
                            "status": "candidate",
                            "origin": "deterministic_transitive_closure",
                            "confidence": path_confidence,
                            "reason": {
                                "rule": "transitive_part_of",
                                "path": list(path),
                            },
                        }

            for target, edge_conf in adjacency.get(current, []):
                if target in path:
                    continue
                next_confidence = min(path_confidence, edge_conf)
                if next_confidence < confidence_threshold:
                    continue
                queue.append((target, next_confidence, path + (target,)))

    return [inferred[key] for key in sorted(inferred)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Infer deterministic transitive part_of edges"
    )
    parser.add_argument("--graph", default=str(DEFAULT_GRAPH))
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "inferred_edges.json"),
    )
    parser.add_argument("--confidence", type=float, default=0.7)
    args = parser.parse_args()

    try:
        graph = load_graph(args.graph)
        inferred = infer_edges(graph, args.confidence)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(inferred, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Inferred {len(inferred)} edges")
        print(f"Wrote to {output_path}")
        return 0
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
