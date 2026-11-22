"""Embed the completed structure or semantic chunks."""

import argparse
from pathlib import Path

from .. import ROOT
from .build import build_embeddings
from .client import DEFAULT_MODEL, FireworksEmbedder


def embed_chunks(chunks: Path, output: Path, cache: Path,
                 model: str, batch_size: int) -> tuple[int, int, bool]:
    """Make reusable vectors for all finished chunks in one chunk set."""
    return build_embeddings(chunks, output, cache, FireworksEmbedder(model), batch_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=["structure", "semantic"], required=True)
    parser.add_argument("--chunks", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/index/chunk_embedding_cache.db")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    directory = ROOT / "data/index"
    chunks = args.chunks or directory / f"{args.strategy}_chunks.jsonl"
    output = args.output or directory / f"{args.strategy}_embeddings.db"
    count, dimension, reused = embed_chunks(chunks, output, args.cache,
                                            args.model, args.batch_size)
    print(f"{'Reused' if reused else 'Wrote'} {count} {dimension}-dimensional embeddings: {output}")


if __name__ == "__main__":
    main()
