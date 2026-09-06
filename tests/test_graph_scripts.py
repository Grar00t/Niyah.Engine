from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_graph import build_graph
from chunk_graph import chunk_graph
from export_graph import export_graphml
from graph_io import load_graph


class GraphScriptTests(unittest.TestCase):
    def test_chunk_rebuild_deduplicates_cross_chunk_edges(self) -> None:
        graph = {
            "graph": {
                "nodes": [
                    {"id": "a", "type": "concept", "label": "A"},
                    {"id": "b", "type": "concept", "label": "B"},
                ],
                "edges": [
                    {
                        "id": "ab",
                        "source": "a",
                        "target": "b",
                        "type": "related_to",
                    }
                ],
                "evidence": [{"id": "ev1", "content": "source"}],
            }
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            chunks = root / "chunks"
            rebuilt = root / "rebuilt.json"
            source.write_text(json.dumps(graph), encoding="utf-8")

            self.assertEqual(chunk_graph(source, chunks, 1), 0)
            result = build_graph(chunks, rebuilt, None)

            self.assertEqual(len(result["nodes"]), 2)
            self.assertEqual(len(result["edges"]), 1)
            self.assertEqual(result["edges"][0]["id"], "ab")
            self.assertEqual(len(result["evidence"]), 1)

    def test_graph_loader_accepts_wrapped_and_unwrapped_graphs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrapped = root / "wrapped.json"
            plain = root / "plain.json"
            payload = {"nodes": [{"id": "a"}], "edges": []}
            wrapped.write_text(json.dumps({"graph": payload}), encoding="utf-8")
            plain.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(load_graph(wrapped), payload)
            self.assertEqual(load_graph(plain), payload)

    def test_graphml_escapes_identifiers_and_values(self) -> None:
        graph = {
            "nodes": [{"id": 'a&"', "label": "<A&B>", "type": "x&y"}],
            "edges": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "graph.graphml"
            export_graphml(graph, output)
            parsed = ET.parse(output)
            node = parsed.find("{http://graphml.graphdrawing.org/xmlns}graph/{http://graphml.graphdrawing.org/xmlns}node")
            self.assertIsNotNone(node)
            self.assertEqual(node.attrib["id"], 'a&"')


if __name__ == "__main__":
    unittest.main()
