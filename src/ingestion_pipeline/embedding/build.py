"""Cache vectors and write a complete, versioned embedding file."""

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np

from .client import document_input

FORMAT_VERSION = "document-v1"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def read_chunks(path: Path) -> list[dict]:
    chunks = []
    ids = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            chunk = json.loads(line)
            for field in ("id", "ticker", "fiscal_year", "filename", "pages", "text", "anchors"):
                if field not in chunk:
                    raise ValueError(f"{path}:{line_number}: missing {field}")
            if chunk["id"] in ids:
                raise ValueError(f"{path}:{line_number}: duplicate chunk id {chunk['id']}")
            ids.add(chunk["id"])
            chunks.append(chunk)
    if not chunks:
        raise ValueError(f"No chunks in {path}")
    return chunks


def vector_blob(values: list[float] | np.ndarray, dimension: int | None = None) -> bytes:
    vector = np.asarray(values, dtype="<f4")
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
        raise ValueError("Embedding must be a nonempty finite vector")
    if dimension is not None and vector.size != dimension:
        raise ValueError(f"Expected {dimension} dimensions, got {vector.size}")
    return vector.tobytes()


def build_embeddings(chunks_path: Path, output: Path, cache_path: Path,
                     embedder, batch_size: int = 32) -> tuple[int, int, bool]:
    """Return (chunk count, dimension, reused existing output)."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    source_hash = file_hash(chunks_path)
    chunks = read_chunks(chunks_path)
    expected = {"source_hash": source_hash, "model": embedder.model, "format": FORMAT_VERSION}
    if output.exists():
        try:
            with sqlite3.connect(output) as old:
                meta = dict(old.execute("SELECT key, value FROM metadata"))
                count = old.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
                if all(meta.get(k) == v for k, v in expected.items()) and count == len(chunks):
                    return count, int(meta["dimension"]), True
        except (sqlite3.DatabaseError, KeyError, ValueError):
            pass

    output.parent.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.unlink(missing_ok=True)
    dimension = None
    try:
        with sqlite3.connect(cache_path) as cache, sqlite3.connect(temporary) as target:
            cache.execute("CREATE TABLE IF NOT EXISTS vectors (model TEXT NOT NULL, text_hash TEXT NOT NULL, vector BLOB NOT NULL, PRIMARY KEY(model, text_hash))")
            target.execute("CREATE TABLE embeddings (chunk_id TEXT PRIMARY KEY, input_hash TEXT NOT NULL, vector BLOB NOT NULL)")
            target.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            inputs = [document_input(chunk, embedder.model) for chunk in chunks]
            hashes = [hashlib.sha256(value.encode("utf-8")).hexdigest() for value in inputs]
            cached = {}
            missing = {}
            for digest, value in zip(hashes, inputs, strict=True):
                row = cache.execute("SELECT vector FROM vectors WHERE model=? AND text_hash=?", (embedder.model, digest)).fetchone()
                if row:
                    cached[digest] = row[0]
                else:
                    missing[digest] = value
            items = list(missing.items())
            for offset in range(0, len(items), batch_size):
                batch = items[offset:offset + batch_size]
                vectors = embedder.embed([value for _, value in batch])
                if len(vectors) != len(batch):
                    raise ValueError("Embedding response count does not match input count")
                for (digest, _), vector in zip(batch, vectors, strict=True):
                    blob = vector_blob(vector, dimension)
                    dimension = len(blob) // 4
                    cached[digest] = blob
                    cache.execute("INSERT OR REPLACE INTO vectors VALUES (?, ?, ?)", (embedder.model, digest, blob))
                cache.commit()
            for chunk, digest in zip(chunks, hashes, strict=True):
                blob = cached[digest]
                vector_blob(np.frombuffer(blob, dtype="<f4"), dimension)
                dimension = len(blob) // 4
                target.execute("INSERT INTO embeddings VALUES (?, ?, ?)", (chunk["id"], digest, blob))
            metadata = expected | {"dimension": str(dimension), "count": str(len(chunks))}
            target.executemany("INSERT INTO metadata VALUES (?, ?)", metadata.items())
            target.commit()
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return len(chunks), dimension, False
