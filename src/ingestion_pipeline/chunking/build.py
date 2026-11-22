"""Group source blocks into citeable structure or semantic chunks."""

import re

from .models import Block, Chunk
from .semantic import EmbeddingProvider, semantic_boundaries


def _split_long_block(block: Block, maximum: int) -> list[Block]:
    if len(block.text) <= maximum:
        return [block]
    if block.kind == "table":
        rows = block.text.splitlines()
        header = rows[0]
        pieces: list[Block] = []
        pending = [header]
        for row in rows[1:]:
            if len("\n".join([*pending, row])) > maximum and len(pending) > 1:
                pieces.append(Block("\n".join(pending), "table", block.page, block.bbox))
                pending = [header]
            pending.append(row)
        if len(pending) > 1:
            pieces.append(Block("\n".join(pending), "table", block.page, block.bbox))
        return pieces or [block]
    words = re.findall(r"\S+\s*", block.text)
    pieces: list[Block] = []
    pending = ""
    for word in words:
        if pending and len(pending) + len(word) > maximum:
            pieces.append(Block(pending.strip(), block.kind, block.page, block.bbox))
            pending = ""
        pending += word
    if pending.strip():
        pieces.append(Block(pending.strip(), block.kind, block.page, block.bbox))
    return pieces


def _pack(blocks: list[Block], boundaries: set[int], target: int, maximum: int, overlap: int) -> list[list[Block]]:
    """Pack complete blocks; carry bounded overlap only across size boundaries."""
    expanded: list[Block] = []
    expanded_boundaries: set[int] = set()
    for index, block in enumerate(blocks):
        if index in boundaries:
            expanded_boundaries.add(len(expanded))
        expanded.extend(_split_long_block(block, maximum))
    chunks: list[list[Block]] = []
    pending: list[Block] = []

    def flush() -> None:
        if pending:
            chunks.append(pending.copy())

    for index, block in enumerate(expanded):
        size = sum(len(item.text) + 2 for item in pending)
        # Consecutive headings belong with the following content. Flushing on
        # every heading produces many useless two-word chunks in SEC filings.
        heading_boundary = block.kind == "heading" and any(item.kind == "paragraph" for item in pending)
        table_boundary = block.kind == "table" and any(item.kind != "heading" for item in pending)
        hard_boundary = index in expanded_boundaries or heading_boundary or table_boundary
        size_boundary = bool(pending) and size + len(block.text) > target
        if pending and (hard_boundary or size_boundary):
            flush()
            carry: list[Block] = []
            if size_boundary and not hard_boundary and pending[-1].kind != "table":
                for previous in reversed(pending):
                    if previous.kind == "table" or sum(len(item.text) + 2 for item in carry) + len(previous.text) > overlap:
                        break
                    carry.insert(0, previous)
            while carry and sum(len(item.text) + 2 for item in carry) + len(block.text) > maximum:
                carry.pop(0)
            pending = carry
        pending.append(block)
        if block.kind == "table":
            flush()
            pending = []
    flush()
    return chunks


def build_chunks(blocks: list[Block], filename: str, ticker: str, year: int, strategy: str,
                 target: int, maximum: int, overlap: int, embedder: EmbeddingProvider | None = None) -> list[Chunk]:
    if strategy not in {"structure", "semantic"}:
        raise ValueError("Choose structure or semantic chunking")
    if strategy == "semantic" and embedder is None:
        raise ValueError("Semantic chunking requires an embedder")
    boundaries = semantic_boundaries(blocks, embedder) if strategy == "semantic" else set()
    packed = _pack(blocks, boundaries, target, maximum, overlap)
    chunks = []
    for ordinal, group in enumerate(packed):
        pages = sorted({block.page for block in group})
        chunks.append(Chunk(
            id=f"{ticker}-FY{year}-{strategy}-t{target}-m{maximum}-o{overlap}-{ordinal:04d}", strategy=strategy,
            ticker=ticker, fiscal_year=year, filename=filename, pages=pages,
            text="\n\n".join(block.text for block in group),
            anchors=[{"page": block.page, "bbox": block.bbox, "kind": block.kind} for block in group],
        ))
    return chunks
