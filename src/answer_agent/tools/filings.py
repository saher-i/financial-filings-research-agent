"""Hybrid search over the supplied annual filings."""

import json

from src.ingestion_pipeline import ROOT
from src.ingestion_pipeline.embedding.client import FireworksEmbedder
from src.retrieval_pipeline.hybrid import search_hybrid


def _pdf_search(query: str, strategy: str, ticker: str | None = None,
                fiscal_year: int | None = None) -> dict:
    directory = ROOT / "data/index"
    store = directory / "qdrant"
    manifest = json.loads((store / f"active_{strategy}.json").read_text())
    lexical_path = directory / f"{strategy}_lexical.db"
    return search_hybrid(query, directory / f"{strategy}_chunks.jsonl",
                         lexical_path, store, strategy,
                         FireworksEmbedder(manifest["model"]), 5, 20,
                         ticker, fiscal_year)


