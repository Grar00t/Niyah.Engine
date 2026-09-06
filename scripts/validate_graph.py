#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from graph_io import DEFAULT_GRAPH, load_graph

ALLOWED_NODE_TYPES = {
    "domain", "topic", "concept", "practice", "technology", "language",
    "standard", "protocol", "algorithm", "architecture", "component",
    "constraint", "requirement", "risk", "decision", "evidence", "document",
}

ALLOWED_EDGE_TYPES = {
    "contains", "part_of", "related_to", "implements", "enables", "depends_on",
    "requires", "constrains", "satisfies", "supports", "derived_from", "causes",
    "mitigates", "conflicts_with", "contradicts", "supersedes", "validated_by",
    "evidenced_by", "uses", "validates",
}


def validate_graph(graph: dict) -> tuple[bool, list[str]]:
    errors: list[str] = []
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    if not isinstance(nodes, list):
        return False, ["nodes must be an array"]
    if not isinstance(edges, list):
        return False, ["edges must be an array"]
    if not nodes:
        errors.append("Graph has no nodes")

    node_ids = {n.get("id") for n in nodes if isinstance(n, dict)}

    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            errors.append(f"Node {i} is not an object")
            continue
        if not node.get("id"):
            errors.append(f"Node {i} missing id")
        if not node.get("type"):
            errors.append(f"Node {node.get('id', i)} missing type")
        elif node["type"] not in ALLOWED_NODE_TYPES:
            errors.append(f"Node {node.get('id')} has unknown type: {node['type']}")
        if not node.get("label"):
            errors.append(f"Node {node.get('id', i)} missing label")

    for i, edge in enumerate(edges):
        if not isinstance(edge, dict):
            errors.append(f"Edge {i} is not an object")
            continue
        if not edge.get("id"):
            errors.append(f"Edge {i} missing id")
        if not edge.get("source"):
            errors.append(f"Edge {edge.get('id', i)} missing source")
        elif edge["source"] not in node_ids:
            errors.append(f"Edge {edge.get('id')} source {edge['source']} not in nodes")
        if not edge.get("target"):
            errors.append(f"Edge {edge.get('id', i)} missing target")
        elif edge["target"] not in node_ids:
            errors.append(f"Edge {edge.get('id')} target {edge['target']} not in nodes")
        if not edge.get("type"):
            errors.append(f"Edge {edge.get('id', i)} missing type")
        elif edge["type"] not in ALLOWED_EDGE_TYPES:
            errors.append(f"Edge {edge.get('id')} has unknown type: {edge['type']}")

    return not errors, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate graph structure")
    parser.add_argument("--graph", default=str(DEFAULT_GRAPH))
    args = parser.parse_args()

    try:
        graph = load_graph(args.graph)
        is_valid, errors = validate_graph(graph)
        if is_valid:
            print("Graph is valid")
            return 0

        print(f"Graph has {len(errors)} errors:", file=sys.stderr)
        for error in errors[:20]:
            print(f"  - {error}", file=sys.stderr)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
