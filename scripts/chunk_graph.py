#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from graph_io import DEFAULT_GRAPH, load_graph

ROOT = Path(__file__).resolve().parents[1]


def chunk_graph(input_path: Path, output_dir: Path, chunk_size: int) -> int:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    graph = load_graph(input_path)
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    evidence = graph.get("evidence", [])
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("graph nodes and edges must be arrays")

    output_dir.mkdir(parents=True, exist_ok=True)
    total_chunks = (len(nodes) + chunk_size - 1) // chunk_size

    for i in range(0, len(nodes), chunk_size):
        chunk_num = (i // chunk_size) + 1
        chunk_nodes = nodes[i:i + chunk_size]
        node_ids = {n["id"] for n in chunk_nodes if isinstance(n, dict) and "id" in n}
        chunk_edges = [
            e
            for e in edges
            if e.get("source") in node_ids or e.get("target") in node_ids
        ]
        chunk = {
            "graph": {
                "nodes": chunk_nodes,
                "edges": chunk_edges,
                "evidence": evidence,
                "schemas": graph.get("schemas", []),
                "constraints": graph.get("constraints", []),
            },
            "metadata": {
                "chunk": chunk_num,
                "total_chunks": total_chunks,
                "chunk_size": chunk_size,
            },
        }
        output_file = output_dir / f"khawrizm_graph_chunk_{chunk_num:04d}.json"
        output_file.write_text(
            json.dumps(chunk, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"[{chunk_num}/{total_chunks}] {output_file.name}: "
            f"{len(chunk_nodes)} nodes, {len(chunk_edges)} edges"
        )

    print(f"Created {total_chunks} chunks in {output_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Split graph into chunks")
    parser.add_argument("--input", default=str(DEFAULT_GRAPH))
    parser.add_argument("--output-dir", default=str(ROOT / "chunks"))
    parser.add_argument("--chunk-size", type=int, default=50)
    args = parser.parse_args()

    try:
        return chunk_graph(Path(args.input), Path(args.output_dir), args.chunk_size)
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
