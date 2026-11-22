"""Command-line entry point for preparing the six filing PDFs."""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from .. import ROOT
from .build import build_chunks
from .semantic_embeddings import DEFAULT_MODEL, FireworksEmbedder
from .extract import extract_blocks


def create_chunks(strategy: str, target: int, maximum: int, overlap: int,
                  model: str, output: Path) -> int:
    """Read every filing and save one JSON record per completed chunk."""
    output.parent.mkdir(parents=True, exist_ok=True)
    embedder = FireworksEmbedder(model, ROOT / "data" / "index" / "embedding_cache.db") if strategy == "semantic" else None
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
    total = 0
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w") as stream:
        for item in manifest["pdfs"]:
            filename = Path(item).name
            ticker, fiscal = filename.split("_")[:2]
            blocks, pages = extract_blocks(ROOT / item)
            chunks = build_chunks(blocks, filename, ticker, int(fiscal.removeprefix("FY")),
                                  strategy, target, maximum, overlap, embedder)
            for chunk in chunks:
                stream.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
            total += len(chunks)
            print(f"{filename}: {pages} pages, {len(blocks)} blocks, {len(chunks)} chunks")
    temporary.replace(output)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=["structure", "semantic"], required=True)
    parser.add_argument("--target", type=int, default=3600)
    parser.add_argument("--maximum", type=int, default=4800)
    parser.add_argument("--overlap", type=int, default=400)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 <= args.overlap < args.target <= args.maximum:
        parser.error("Require 0 <= overlap < target <= maximum")
    output = args.output or ROOT / "data" / "index" / f"{args.strategy}_chunks.jsonl"
    if args.strategy == "semantic" and not os.getenv("FIREWORKS_API_KEY"):
        parser.error("semantic chunking requires FIREWORKS_API_KEY in the environment")
    total = create_chunks(args.strategy, args.target, args.maximum, args.overlap, args.model, output)
    print(f"Wrote {total} chunks to {output}")


if __name__ == "__main__":
    main()
