"""Saved embeddings and a persistent local Qdrant collection work together."""

import io
import json
import os
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from devui import server as dev_server
from src.ingestion_pipeline.embedding.build import build_embeddings
from src.ingestion_pipeline.indexing.store import build_index, inspect_index
from src.retrieval_pipeline.semantic import search_vector


class FakeEmbedder:
    model = "test-model"

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[1.0, 0.0] if "supply" in value.lower() else [0.0, 1.0] for value in texts]


class IngestionQdrantTest(unittest.TestCase):
    def test_build_search_cache_and_stale_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = root / "data/index"
            directory.mkdir(parents=True)
            chunks = directory / "structure_chunks.jsonl"
            embeddings = directory / "structure_embeddings.db"
            cache = directory / "cache.db"
            store = directory / "qdrant"
            records = [
                {"id": "apple", "strategy": "structure", "ticker": "AAPL", "fiscal_year": 2025,
                 "filename": "a.pdf", "pages": [8], "anchors": [{"page": 8, "bbox": [1, 2, 3, 4]}],
                 "text": "Supply chain disruptions can affect production."},
                {"id": "microsoft", "strategy": "structure", "ticker": "MSFT", "fiscal_year": 2024,
                 "filename": "m.pdf", "pages": [12], "anchors": [],
                 "text": "Cloud service revenue increased."},
            ]
            chunks.write_text("".join(json.dumps(row) + "\n" for row in records))
            fake = FakeEmbedder()
            self.assertEqual(build_embeddings(chunks, embeddings, cache, fake), (2, 2, False))
            self.assertEqual(build_embeddings(chunks, embeddings, cache, fake), (2, 2, True))
            self.assertEqual(fake.calls, 1)
            self.assertEqual(build_index(chunks, embeddings, store, "structure"), (2, 2))
            self.assertEqual(inspect_index(store, "structure")["count"], 2)
            hit = search_vector(store, "structure", "supply risk", fake)[0]
            self.assertEqual(hit["id"], "apple")
            from qdrant_client import QdrantClient
            def slow_client(*args, **kwargs):
                client = QdrantClient(*args, **kwargs)
                time.sleep(0.02)  # Expose overlapping local-client lifetimes.
                return client
            with patch("qdrant_client.QdrantClient", side_effect=slow_client), \
                 ThreadPoolExecutor(max_workers=3) as pool:
                hits = list(pool.map(lambda _: search_vector(store, "structure", "supply risk", fake),
                                     range(3)))
            self.assertTrue(all(rows[0]["id"] == "apple" for rows in hits))
            alternate = FakeEmbedder()
            alternate.model = "alternate-test-model"
            with self.assertRaisesRegex(ValueError, "Query model must match"):
                search_vector(store, "structure", "supply risk", alternate)
            self.assertEqual(alternate.calls, 0)
            self.assertEqual(hit["anchors"][0]["page"], 8)
            self.assertEqual(search_vector(store, "structure", "supply risk", fake, ticker="MSFT")[0]["id"], "microsoft")
            self.assertEqual(search_vector(store, "structure", "supply risk", fake, fiscal_year=2023), [])
            with patch.object(dev_server, "ROOT", root), patch.object(dev_server, "QDRANT_STORE", store):
                self.assertTrue(dev_server.vector_state("structure")["embedding"]["ready"])
                self.assertTrue(dev_server.vector_state("structure")["index"]["ready"])
                self.assertEqual(dev_server.embedding_preview("structure")["items"][0]["dimension"], 2)
                self.assertEqual(len(dev_server.qdrant_points("structure")["points"]), 2)
                first_page = dev_server.qdrant_points("structure", limit=1)
                second_page = dev_server.qdrant_points("structure", first_page["next_cursor"], limit=1)
                self.assertNotEqual(first_page["points"][0]["chunk_id"], second_page["points"][0]["chunk_id"])
                with patch.dict(os.environ, {"APP_MODE": "developer"}):
                    for route in ("/api/dev/embeddings?strategy=structure",
                                  "/api/dev/qdrant-points?strategy=structure"):
                        handler = dev_server.Handler.__new__(dev_server.Handler)
                        handler.path = route
                        handler.wfile = io.BytesIO()
                        handler.send_response = lambda status: self.assertEqual(status, 200)
                        handler.send_header = lambda *args: None
                        handler.end_headers = lambda: None
                        handler.do_GET()
                        self.assertIn("items" if "embeddings" in route else "points",
                                      json.loads(handler.wfile.getvalue()))

            records[0]["text"] = "Different disclosure"
            chunks.write_text("".join(json.dumps(row) + "\n" for row in records))
            with self.assertRaisesRegex(ValueError, "different chunk file"):
                build_index(chunks, embeddings, store, "structure")
            self.assertEqual(search_vector(store, "structure", "supply risk", fake)[0]["id"], "apple")
            self.assertEqual(build_embeddings(chunks, embeddings, cache, fake), (2, 2, False))
            self.assertEqual(build_index(chunks, embeddings, store, "structure"), (2, 2))
            self.assertEqual(inspect_index(store, "structure")["count"], 2)


if __name__ == "__main__":
    unittest.main()
