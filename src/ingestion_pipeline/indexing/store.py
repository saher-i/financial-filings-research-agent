"""Persistent local Qdrant collections for completed filing chunks."""

import hashlib
import json
import sqlite3
import uuid
from pathlib import Path

import numpy as np

from ..embedding.build import FORMAT_VERSION, file_hash, read_chunks
from ..embedding.client import document_input


def _manifest_path(store: Path, strategy: str) -> Path:
    return store / f"active_{strategy}.json"


def _manifest(store: Path, strategy: str) -> dict:
    return json.loads(_manifest_path(store, strategy).read_text())


def _client(store: Path):
    from qdrant_client import QdrantClient

    store.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(store))


def build_index(chunks_path: Path, embeddings_path: Path, store: Path,
                strategy: str, batch_size: int = 64) -> tuple[int, int]:
    """Validate every input, then activate a complete Qdrant collection."""
    from qdrant_client import models

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    chunks = read_chunks(chunks_path)
    if any(chunk.get("strategy") != strategy for chunk in chunks):
        raise ValueError("Chunk strategy does not match the selected strategy")
    source_hash = file_hash(chunks_path)
    with sqlite3.connect(embeddings_path) as source:
        metadata = dict(source.execute("SELECT key, value FROM metadata"))
        if metadata.get("source_hash") != source_hash or metadata.get("format") != FORMAT_VERSION:
            raise ValueError("Embeddings were made from a different chunk file or input format; rerun embedding")
        dimension = int(metadata["dimension"])
        if int(metadata["count"]) != len(chunks):
            raise ValueError("Embedding count does not match chunk count")
        vectors = {row[0]: (row[1], row[2]) for row in source.execute("SELECT chunk_id, input_hash, vector FROM embeddings")}
    if set(vectors) != {chunk["id"] for chunk in chunks}:
        raise ValueError("Embedding chunk IDs do not match chunk file")

    points = []
    for chunk in chunks:
        digest, blob = vectors[chunk["id"]]
        if digest != hashlib.sha256(document_input(chunk, metadata["model"]).encode("utf-8")).hexdigest():
            raise ValueError(f"Embedding input changed for {chunk['id']}")
        vector = np.frombuffer(blob, dtype="<f4")
        if vector.size != dimension or not np.isfinite(vector).all():
            raise ValueError(f"Invalid vector for {chunk['id']}")
        points.append(models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["id"])), vector=vector.tolist(),
            payload={"chunk_id": chunk["id"], "strategy": strategy, "ticker": chunk["ticker"],
                     "fiscal_year": chunk["fiscal_year"], "filename": chunk["filename"],
                     "pages": chunk["pages"], "anchors": chunk["anchors"], "text": chunk["text"]},
        ))

    version = hashlib.sha256(f"{source_hash}:{metadata['model']}:{FORMAT_VERSION}".encode()).hexdigest()[:16]
    collection = f"filings_{strategy}_{version}"
    new_manifest = {"collection": collection, "strategy": strategy, "source_hash": source_hash,
                    "model": metadata["model"], "format": FORMAT_VERSION,
                    "dimension": dimension, "count": len(chunks)}
    client = _client(store)
    try:
        if not client.collection_exists(collection):
            client.create_collection(collection, vectors_config=models.VectorParams(
                size=dimension, distance=models.Distance.COSINE))
        for start in range(0, len(points), batch_size):
            client.upsert(collection, points[start:start + batch_size], wait=True)
        if client.count(collection, exact=True).count != len(chunks):
            raise RuntimeError("Qdrant collection count does not match the chunk count")
    finally:
        client.close()
    manifest_path = _manifest_path(store, strategy)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(new_manifest, indent=2) + "\n")
    temporary.replace(manifest_path)
    return len(chunks), dimension


def inspect_index(store: Path, strategy: str) -> dict:
    """Show the active collection and counts by filing."""
    manifest = _manifest(store, strategy)
    client = _client(store)
    groups = {}
    try:
        if not client.collection_exists(manifest["collection"]):
            raise ValueError("Active Qdrant collection is missing; rebuild the index")
        total = client.count(manifest["collection"], exact=True).count
        offset = None
        while True:
            points, offset = client.scroll(manifest["collection"], offset=offset, limit=256,
                                           with_payload=["ticker", "fiscal_year"], with_vectors=False)
            for point in points:
                key = (point.payload["ticker"], point.payload["fiscal_year"])
                groups[key] = groups.get(key, 0) + 1
            if offset is None:
                break
    finally:
        client.close()
    if total != manifest["count"]:
        raise ValueError("Active Qdrant collection is incomplete; rebuild the index")
    return {"store": str(store), "collection": manifest["collection"],
            "model": manifest["model"], "dimension": manifest["dimension"], "count": total,
            "filings": [{"ticker": ticker, "year": year, "chunks": count}
                        for (ticker, year), count in sorted(groups.items())]}
