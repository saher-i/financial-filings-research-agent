"""Read PDF text blocks and detected tables with source positions."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pymupdf

from .. import ROOT
from .models import Block


def _table_text(rows: list[list[str | None]]) -> str:
    return "\n".join(" | ".join((cell or "").replace("\n", " ") for cell in row) for row in rows)


def extract_blocks(path: Path) -> tuple[list[Block], int]:
    """Return reading-order blocks with physical-page and bounding-box provenance."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    cache = ROOT / "data" / "index" / "extractions" / f"{digest}-blocks-v1.json"
    if cache.exists():
        saved = json.loads(cache.read_text())
        return [Block(**item) for item in saved["blocks"]], saved["pages"]
    blocks: list[Block] = []
    with pymupdf.open(path) as pdf:
        if pdf.is_encrypted:
            raise ValueError(f"Encrypted PDF: {path.name}")
        for page_number, page in enumerate(pdf, 1):
            tables = []
            try:
                for table in page.find_tables().tables:
                    rows = table.extract()
                    if table.row_count > 1 and table.col_count > 1 and rows:
                        tables.append((pymupdf.Rect(table.bbox), rows))
            except (ValueError, RuntimeError):
                # The page text remains searchable if a table cannot be parsed.
                tables = []

            page_blocks: list[Block] = []
            for raw in page.get_text("dict", sort=True)["blocks"]:
                if raw["type"] != 0:
                    continue
                box = pymupdf.Rect(raw["bbox"])
                if any((box & table_box).get_area() / max(box.get_area(), 1) > 0.6 for table_box, _ in tables):
                    continue
                lines = raw.get("lines", [])
                text = "\n".join("".join(span["text"] for span in line["spans"]) for line in lines).strip()
                if not text:
                    continue
                spans = [span for line in lines for span in line["spans"]]
                largest = max((span["size"] for span in spans), default=0)
                bold = any("bold" in span["font"].lower() for span in spans)
                kind = "heading" if len(text) <= 140 and (bold or largest >= 13) else "paragraph"
                page_blocks.append(Block(text, kind, page_number, list(box)))
            for box, rows in tables:
                page_blocks.append(Block(_table_text(rows), "table", page_number, list(box)))
            page_blocks.sort(key=lambda block: (block.bbox[1], block.bbox[0]))
            blocks.extend(page_blocks)
        pages = len(pdf)
    if sum(len(block.text) for block in blocks) < 1000:
        raise ValueError(f"No useful extractable text in {path.name}; OCR is required")
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".tmp")
    temporary.write_text(json.dumps({"pages": pages, "blocks": [asdict(block) for block in blocks]}, ensure_ascii=False))
    temporary.replace(cache)
    return blocks, pages
