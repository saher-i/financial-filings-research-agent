"""Command-line entry points for asking the agent and inspecting retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_local_fireworks_key
from src.ingestion_pipeline import ROOT


def ask(args: argparse.Namespace) -> dict:
    from src.answer_agent.graph import answer_question

    load_local_fireworks_key()
    return answer_question(args.question, strategy=args.strategy, model=args.model,
                           verify=True)


def search(args: argparse.Namespace) -> dict:
    from src.ingestion_pipeline.embedding.client import FireworksEmbedder
    from src.retrieval_pipeline.hybrid import search_hybrid

    load_local_fireworks_key()
    manifest = json.loads((args.store / f"active_{args.strategy}.json").read_text())
    directory = ROOT / "data/index"
    return search_hybrid(
        args.question,
        args.chunks or directory / f"{args.strategy}_chunks.jsonl",
        args.lexical_index or directory / f"{args.strategy}_lexical.db",
        args.store, args.strategy, FireworksEmbedder(manifest["model"]),
        args.limit, args.candidates, args.ticker, args.year,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ask_parser = commands.add_parser("ask", help="Ask the grounded research agent")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--strategy", choices=("structure", "semantic"), default="structure")
    ask_parser.add_argument("--model")

    search_parser = commands.add_parser("search", help="Inspect fused PDF retrieval")
    search_parser.add_argument("question")
    search_parser.add_argument("--strategy", choices=("structure", "semantic"), default="structure")
    search_parser.add_argument("--store", type=Path, default=ROOT / "data/index/qdrant")
    search_parser.add_argument("--chunks", type=Path)
    search_parser.add_argument("--lexical-index", type=Path)
    search_parser.add_argument("--ticker")
    search_parser.add_argument("--year", type=int)
    search_parser.add_argument("--limit", type=int, default=5)
    search_parser.add_argument("--candidates", type=int, default=20)

    args = parser.parse_args()
    result = ask(args) if args.command == "ask" else search(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
