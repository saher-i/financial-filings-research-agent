"""Fireworks embeddings and a local cache for semantic boundaries."""

import hashlib
import os
import sqlite3
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"


class FireworksEmbedder:
    def __init__(self, model: str, cache_path: Path):
        from openai import OpenAI

        key = os.getenv("FIREWORKS_API_KEY")
        if not key:
            raise RuntimeError("Semantic chunking needs FIREWORKS_API_KEY")
        self.client = OpenAI(api_key=key, base_url="https://api.fireworks.ai/inference/v1")
        self.model = model
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = sqlite3.connect(cache_path)
        self.cache.execute("CREATE TABLE IF NOT EXISTS vectors (model TEXT, text_hash TEXT, vector BLOB, PRIMARY KEY(model, text_hash))")

    def embed(self, texts: list[str]) -> np.ndarray:
        result: list[np.ndarray | None] = [None] * len(texts)
        missing: dict[str, str] = {}
        hashes = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
        for i, digest in enumerate(hashes):
            row = self.cache.execute("SELECT vector FROM vectors WHERE model=? AND text_hash=?", (self.model, digest)).fetchone()
            if row:
                result[i] = np.frombuffer(row[0], dtype=np.float32)
            else:
                missing[digest] = texts[i]
        items = list(missing.items())
        for offset in range(0, len(items), 32):
            batch = items[offset:offset + 32]
            response = self.client.embeddings.create(model=self.model, input=[text for _, text in batch])
            vectors = sorted(response.data, key=lambda item: item.index)
            with self.cache:
                for (digest, _), item in zip(batch, vectors, strict=True):
                    vector = np.asarray(item.embedding, dtype=np.float32)
                    self.cache.execute("INSERT OR REPLACE INTO vectors VALUES (?,?,?)", (self.model, digest, vector.tobytes()))
        for i, digest in enumerate(hashes):
            if result[i] is None:
                row = self.cache.execute("SELECT vector FROM vectors WHERE model=? AND text_hash=?", (self.model, digest)).fetchone()
                result[i] = np.frombuffer(row[0], dtype=np.float32)
        return np.stack(result)
