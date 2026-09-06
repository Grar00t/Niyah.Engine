from __future__ import annotations

import unittest

from src.inference.infer import infer_edges, stable_edge_id


class GraphInferenceTests(unittest.TestCase):
    def test_stable_edge_id_is_deterministic(self) -> None:
        first = stable_edge_id("a", "part_of", "c")
        second = stable_edge_id("a", "part_of", "c")
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("e_"))
        self.assertEqual(len(first), 66)

    def test_transitive_part_of_is_inferred(self) -> None:
        graph = {
            "nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            "edges": [
                {
                    "id": "ab",
                    "source": "a",
                    "target": "b",
                    "type": "part_of",
                    "confidence": 0.9,
                },
                {
                    "id": "bc",
                    "source": "b",
                    "target": "c",
                    "type": "part_of",
                    "confidence": 0.8,
                },
            ],
        }
        inferred = infer_edges(graph, 0.7)
        self.assertEqual(len(inferred), 1)
        self.assertEqual(inferred[0]["source"], "a")
        self.assertEqual(inferred[0]["target"], "c")
        self.assertEqual(inferred[0]["confidence"], 0.8)
        self.assertEqual(inferred[0]["reason"]["path"], ["a", "b", "c"])

    def test_existing_edge_is_not_duplicated(self) -> None:
        graph = {
            "nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            "edges": [
                {"source": "a", "target": "b", "type": "part_of"},
                {"source": "b", "target": "c", "type": "part_of"},
                {"source": "a", "target": "c", "type": "part_of"},
            ],
        }
        self.assertEqual(infer_edges(graph), [])

    def test_confidence_threshold_is_enforced(self) -> None:
        graph = {
            "nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            "edges": [
                {
                    "source": "a",
                    "target": "b",
                    "type": "part_of",
                    "confidence": 0.95,
                },
                {
                    "source": "b",
                    "target": "c",
                    "type": "part_of",
                    "confidence": 0.6,
                },
            ],
        }
        self.assertEqual(infer_edges(graph, 0.7), [])

    def test_cycles_do_not_infer_self_edges(self) -> None:
        graph = {
            "nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            "edges": [
                {"source": "a", "target": "b", "type": "part_of"},
                {"source": "b", "target": "c", "type": "part_of"},
                {"source": "c", "target": "a", "type": "part_of"},
            ],
        }
        inferred = infer_edges(graph)
        self.assertTrue(all(edge["source"] != edge["target"] for edge in inferred))

    def test_invalid_threshold_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            infer_edges({"nodes": [], "edges": []}, 1.1)


if __name__ == "__main__":
    unittest.main()
