"""Embed a question and retrieve matching filing passages from Qdrant."""

import json
import threading
from pathlib import Path

import numpy as np

from src.ingestion_pipeline.embedding.client import query_input


# Embedded Qdrant takes an exclusive folder lock. Keep parallel embedding and
# lexical work, but serialize the short local-client lifetime within this process.
_LOCAL_QDRANT_LOCK = threading.Lock()


def search_vector(store: Path, strategy: str, question: str, embedder, limit: int = 5,
                  ticker: str | None = None, fiscal_year: int | None = None) -> list[dict]:
    """Return ranked passages and their source information."""
    from qdrant_client import QdrantClient, models

    if strategy not in {"structure", "semantic"}:
        raise ValueError("Unknown chunking strategy")
    if limit < 1:
        return []
    if not question.strip():
        raise ValueError("Question cannot be empty")
    manifest = json.loads((store / f"active_{strategy}.json").read_text())
    if embedder.model != manifest["model"]:
        raise ValueError("Query model must match the index model")
    query = np.asarray(embedder.embed([query_input(question, embedder.model)])[0], dtype="<f4")
    if query.size != manifest["dimension"] or not np.isfinite(query).all() or not np.linalg.norm(query):
        raise ValueError("Invalid query embedding")
    conditions = []
    if ticker:
        conditions.append(models.FieldCondition(key="ticker", match=models.MatchValue(value=ticker.upper())))
    if fiscal_year is not None:
        conditions.append(models.FieldCondition(key="fiscal_year", match=models.MatchValue(value=fiscal_year)))
    with _LOCAL_QDRANT_LOCK:
        client = QdrantClient(path=str(store))
        try:
            result = client.query_points(manifest["collection"], query=query.tolist(),
                                         query_filter=models.Filter(must=conditions) if conditions else None,
                                         limit=limit, with_payload=True)
        finally:
            client.close()
    return [{"id": point.payload["chunk_id"], "ticker": point.payload["ticker"],
             "fiscal_year": point.payload["fiscal_year"], "filename": point.payload["filename"],
             "pages": point.payload["pages"], "anchors": point.payload["anchors"],
             "text": point.payload["text"], "score": point.score} for point in result.points]
