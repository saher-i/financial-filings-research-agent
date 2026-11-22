"""Hybrid retrieval keeps source identity and combines independent rankings."""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from src.ingestion_pipeline.embedding.build import build_embeddings
from src.ingestion_pipeline.indexing.store import build_index
from src.retrieval_pipeline.hybrid import reciprocal_rank_fusion, search_hybrid
from src.retrieval_pipeline.lexical import ensure_lexical_index, search_lexical
from src.retrieval_pipeline.ranking import local_rerank, diversify_pages


class FakeEmbedder:
    model = "test-model"

    def embed(self, texts):
        return [[1.0, 0.0] if "supply" in text.lower() else [0.0, 1.0] for text in texts]


class HybridRetrievalTest(unittest.TestCase):
    def test_lexical_semantic_and_fused_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            chunks = root / "chunks.jsonl"
            lexical = root / "lexical.db"
            embeddings = root / "embeddings.db"
            store = root / "qdrant"
            records = [
                {"id": "apple-supply", "strategy": "structure", "ticker": "AAPL", "fiscal_year": 2025,
                 "filename": "a.pdf", "pages": [8], "anchors": [],
                 "text": "Supply chain disruption and component sourcing risks."},
                {"id": "apple-revenue", "strategy": "structure", "ticker": "AAPL", "fiscal_year": 2025,
                 "filename": "a.pdf", "pages": [25], "anchors": [],
                 "text": "Services revenue increased from subscriptions."},
                {"id": "microsoft-supply", "strategy": "structure", "ticker": "MSFT", "fiscal_year": 2024,
                 "filename": "m.pdf", "pages": [12], "anchors": [],
                 "text": "Supply of cloud data center equipment."},
            ]
            chunks.write_text("".join(json.dumps(record) + "\n" for record in records))
            fake = FakeEmbedder()
            build_embeddings(chunks, embeddings, root / "cache.db", fake)
            build_index(chunks, embeddings, store, "structure")
            ensure_lexical_index(chunks, lexical)
            self.assertEqual(search_lexical(lexical, "supply chain", ticker="AAPL")[0]["id"], "apple-supply")
            result = search_hybrid("supply chain", chunks, lexical, store, "structure",
                                   fake, ticker="AAPL", fiscal_year=2025)
            self.assertEqual(result["results"][0]["id"], "apple-supply")
            self.assertEqual(result["results"][0]["lexical_rank"], 1)
            self.assertEqual(result["results"][0]["semantic_rank"], 1)
            self.assertTrue(all(hit["ticker"] == "AAPL" for hit in result["results"]))
            self.assertTrue(result["lexical"] and result["semantic"])
            self.assertIn("local_rerank", result)
            self.assertIn("page_diverse", result)
            self.assertEqual(result["results"][0]["id"], "apple-supply")

            records[0]["text"] = "Changed chunk text"
            chunks.write_text("".join(json.dumps(record) + "\n" for record in records))
            with self.assertRaisesRegex(ValueError, "does not match"):
                search_hybrid("supply chain", chunks, lexical, store, "structure", fake)

    def test_both_retrievals_start_before_either_finishes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            store = root / "qdrant"
            store.mkdir()
            (store / "active_structure.json").write_text(json.dumps({"source_hash": "same"}))
            barrier = threading.Barrier(2)
            hit = {"id": "one", "ticker": "AAPL", "fiscal_year": 2025,
                   "filename": "a.pdf", "pages": [1], "anchors": [], "text": "evidence", "score": 1.0}

            def branch(*args, **kwargs):
                barrier.wait(timeout=2)
                return [hit]

            with patch("src.retrieval_pipeline.hybrid.file_hash", return_value="same"), \
                 patch("src.retrieval_pipeline.hybrid.ensure_lexical_index"), \
                 patch("src.retrieval_pipeline.hybrid.search_lexical", side_effect=branch), \
                 patch("src.retrieval_pipeline.hybrid.search_vector", side_effect=branch):
                result = search_hybrid("evidence", root / "chunks.jsonl", root / "lexical.db",
                                       store, "structure", FakeEmbedder())
            self.assertEqual(len(result["results"]), 1)
            self.assertAlmostEqual(result["results"][0]["rrf_score"], 2 / 61)

    def test_rewrite_changes_only_keyword_query_and_records_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            store = root / "qdrant"
            store.mkdir()
            (store / "active_structure.json").write_text(json.dumps({"source_hash": "same"}))
            hit = {"id": "one", "ticker": "GOOGL", "fiscal_year": 2025,
                   "filename": "g.pdf", "pages": [1], "anchors": [], "text": "YouTube revenue"}
            seen = []
            def keyword(path, query, *args):
                seen.append(("keyword", query))
                return [hit]
            def vector(store, strategy, query, *args):
                seen.append(("vector", query))
                return [hit]
            with patch("src.retrieval_pipeline.hybrid.file_hash", return_value="same"), \
                 patch("src.retrieval_pipeline.hybrid.ensure_lexical_index"), \
                 patch("src.retrieval_pipeline.hybrid.search_lexical", side_effect=keyword), \
                 patch("src.retrieval_pipeline.hybrid.search_vector", side_effect=vector):
                result = search_hybrid("YouTu ads", root / "chunks", root / "lexical", store,
                                       "structure", FakeEmbedder(),
                                       rewriter=lambda _: {"query": "YouTube ads", "status": "rewritten"})
                self.assertIn(("keyword", "YouTube ads"), seen)
                self.assertIn(("vector", "YouTu ads"), seen)
                self.assertEqual(result["rewrite"]["lexical_query"], "YouTube ads")
                fallback = search_hybrid("YouTu ads", root / "chunks", root / "lexical", store,
                                         "structure", FakeEmbedder(),
                                         rewriter=lambda _: (_ for _ in ()).throw(RuntimeError("offline")))
                self.assertEqual(fallback["rewrite"]["status"], "fallback")
                self.assertIn(("keyword", "YouTu ads"), seen)

    def test_rrf_uses_ranks_not_raw_scores(self):
        base = {"ticker": "AAPL", "fiscal_year": 2025, "filename": "a.pdf",
                "pages": [1], "anchors": [], "text": "evidence"}
        lexical = [base | {"id": "a", "score": 0.001}, base | {"id": "b", "score": 1000}]
        semantic = [base | {"id": "b", "score": 0.99}, base | {"id": "a", "score": 0.01}]
        fused = reciprocal_rank_fusion(lexical, semantic)
        self.assertEqual({item["id"] for item in fused}, {"a", "b"})
        self.assertAlmostEqual(fused[0]["rrf_score"], 1 / 61 + 1 / 62)

    def test_post_rankings_use_same_candidates_without_mutating_rrf(self):
        hits = [
            {"id":"one", "filename":"a.pdf", "pages":[8], "text":"Apple supply chain risks", "rrf_score":.032},
            {"id":"two", "filename":"a.pdf", "pages":[8], "text":"Apple supplier components", "rrf_score":.031},
            {"id":"three", "filename":"a.pdf", "pages":[9], "text":"Apple supply component shortages", "rrf_score":.030},
        ]
        reranked = local_rerank(hits, "Apple component shortages", 3)
        self.assertEqual({hit["id"] for hit in reranked}, {hit["id"] for hit in hits})
        self.assertEqual(reranked[0]["id"], "three")
        diverse = diversify_pages(hits, 2)
        self.assertEqual([hit["id"] for hit in diverse], ["one", "three"])
        self.assertEqual([hit["id"] for hit in hits], ["one", "two", "three"])
        self.assertNotIn("local_rerank_score", hits[0])


if __name__ == "__main__":
    unittest.main()
