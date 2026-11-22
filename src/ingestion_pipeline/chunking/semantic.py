"""Use neighboring paragraph similarity to suggest topic boundaries."""

from typing import Protocol

import numpy as np

from .models import Block


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


def semantic_boundaries(blocks: list[Block], embedder: EmbeddingProvider, percentile: float = 80) -> set[int]:
    narrative = [(i, block) for i, block in enumerate(blocks) if block.kind == "paragraph"]
    if len(narrative) < 3:
        return set()
    windows = ["\n".join(item.text for _, item in narrative[max(0, i - 1):i + 2]) for i in range(len(narrative))]
    vectors = embedder.embed(windows)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)
    distances = 1 - np.sum(vectors[:-1] * vectors[1:], axis=1)
    threshold = float(np.percentile(distances, percentile))
    return {narrative[i + 1][0] for i, distance in enumerate(distances) if distance > threshold}
