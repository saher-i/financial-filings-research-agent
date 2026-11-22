"""Build or inspect the local Qdrant filing index."""

import argparse
import json
from pathlib import Path

from .. import ROOT
from .store import build_index, inspect_index


def create_index(chunks: Path, embeddings: Path, store: Path, strategy: str) -> tuple[int, int]:
    """Load one chunk set and its vectors into Qdrant."""
    return build_index(chunks, embeddings, store, strategy)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=["structure", "semantic"], required=True)
    parser.add_argument("--chunks", type=Path)
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--store", type=Path, default=ROOT / "data/index/qdrant")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build")
    commands.add_parser("inspect")
    args = parser.parse_args()
    directory = ROOT / "data/index"
    if args.command == "build":
        chunks = args.chunks or directory / f"{args.strategy}_chunks.jsonl"
        embeddings = args.embeddings or directory / f"{args.strategy}_embeddings.db"
        count, dimension = create_index(chunks, embeddings, args.store, args.strategy)
        print(f"Indexed {count} chunks with {dimension}-dimensional vectors in {args.store}")
    elif args.command == "inspect":
        print(json.dumps(inspect_index(args.store, args.strategy), indent=2))


if __name__ == "__main__":
    main()
