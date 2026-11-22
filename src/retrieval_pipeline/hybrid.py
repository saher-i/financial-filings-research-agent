"""Parallel lexical and semantic retrieval with reciprocal rank fusion."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.ingestion_pipeline.embedding.build import file_hash

from .lexical import ensure_lexical_index, search_lexical
from .semantic import search_vector
from .ranking import local_rerank, diversify_pages
from .rewrite import rewrite_lexical_query


def reciprocal_rank_fusion(lexical: list[dict], semantic: list[dict],
                           limit: int = 5, rank_constant: int = 60) -> list[dict]:
    """Merge by chunk ID; rank positions, not unrelated raw scores, set the weight."""
    if rank_constant < 1:
        raise ValueError("rank_constant must be positive")
    combined = {}
    for method, hits in (("lexical", lexical), ("semantic", semantic)):
        for rank, hit in enumerate(hits, 1):
            key = hit["id"]
            entry = combined.setdefault(key, {"id": key, "ticker": hit["ticker"],
                                              "fiscal_year": hit["fiscal_year"],
                                              "filename": hit["filename"], "pages": hit["pages"],
                                              "anchors": hit["anchors"], "text": hit["text"],
                                              "rrf_score": 0.0, "lexical_rank": None,
                                              "semantic_rank": None})
            entry["rrf_score"] += 1 / (rank_constant + rank)
            entry[f"{method}_rank"] = rank
    ordered = sorted(combined.values(), key=lambda hit: (-hit["rrf_score"],
                     min(rank for rank in (hit["lexical_rank"], hit["semantic_rank"])
                         if rank is not None), hit["id"]))
    return ordered[:limit]


def search_hybrid(question: str, chunks_path: Path, lexical_path: Path,
                  qdrant_store: Path, strategy: str, embedder,
                  limit: int = 5, candidate_limit: int = 20,
                  ticker: str | None = None, fiscal_year: int | None = None,
                  rewriter=None) -> dict:
    """Run both retrieval methods at once, then fuse their ordered candidates."""
    if strategy not in {"structure", "semantic"}:
        raise ValueError("Unknown chunking strategy")
    if not question.strip() or limit < 1 or candidate_limit < limit:
        raise ValueError("Provide a question and candidate_limit >= limit >= 1")
    source_hash = file_hash(chunks_path)
    manifest = json.loads((qdrant_store / f"active_{strategy}.json").read_text())
    if manifest["source_hash"] != source_hash:
        raise ValueError("Qdrant index does not match this chunk file; rebuild embeddings and index")
    ensure_lexical_index(chunks_path, lexical_path)
    rewriter = rewriter or rewrite_lexical_query
    with ThreadPoolExecutor(max_workers=2) as pool:
        semantic_task = pool.submit(search_vector, qdrant_store, strategy, question,
                                    embedder, candidate_limit, ticker, fiscal_year)
        try:
            rewrite = rewriter(question)
        except Exception:
            rewrite = {"query": question, "status": "fallback", "duration_ms": 0.0}
        lexical_query = rewrite.get("query") or question
        lexical_task = pool.submit(search_lexical, lexical_path, lexical_query,
                                   candidate_limit, ticker, fiscal_year)
        lexical_hits = lexical_task.result()
        semantic_hits = semantic_task.result()
    fused_candidates = reciprocal_rank_fusion(lexical_hits, semantic_hits, candidate_limit * 2)
    locally_reranked = local_rerank(fused_candidates, question, max(1, len(fused_candidates)))
    return {"results": fused_candidates[:limit],
            "lexical": lexical_hits, "semantic": semantic_hits,
            "local_rerank": locally_reranked[:limit],
            "page_diverse": diversify_pages(fused_candidates, limit),
            "rerank_diverse": diversify_pages(locally_reranked, limit),
            "strategy": strategy, "source_hash": source_hash,
            "rewrite": {**rewrite, "original_query": question, "lexical_query": lexical_query}}
