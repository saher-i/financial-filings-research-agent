"""Read a small number of physical PDF pages when a chunk omits table context."""

from __future__ import annotations

import re

import pymupdf

from src.ingestion_pipeline import ROOT

FILING_NAME = re.compile(r"(?:AAPL|MSFT|GOOGL)_FY202[45]_10-K\.pdf\Z")


def expand_table_preview(hit: dict, page_reader=None) -> tuple[str, bool]:
    """Open a short single-page passage to recover nearby omitted context."""
    preview = hit.get("text", "")
    pages = hit.get("pages", [])
    if (not isinstance(preview, str) or len(preview) > 350
            or not isinstance(pages, list) or len(pages) != 1
            or any(isinstance(anchor, dict) and anchor.get("kind") == "table"
                   for anchor in (hit.get("anchors") or []))):
        return preview, False
    reader = page_reader or read_pages
    try:
        full_page = reader(hit["filename"], pages[0])
        text = full_page.get("text", "")
        if isinstance(text, str) and len(text) > len(preview):
            return text, True
    except (KeyError, TypeError, ValueError, FileNotFoundError, RuntimeError):
        pass
    return preview, False


def read_pages(filename: str, page: int, end_page: int | None = None) -> dict:
    if not isinstance(filename, str) or not FILING_NAME.fullmatch(filename):
        raise ValueError("Choose a supplied filing filename")
    if type(page) is not int or not 1 <= page <= 1000:
        raise ValueError("Page must be a physical PDF page number")
    if end_page is None:
        end_page = page
    if type(end_page) is not int or not page <= end_page <= page + 1:
        raise ValueError("Read at most two consecutive pages")
    path = ROOT / "data" / "pdfs" / filename
    if not path.is_file():
        raise FileNotFoundError("The selected filing is unavailable")
    with pymupdf.open(path) as document:
        if end_page > len(document):
            raise ValueError("Page is outside this filing")
        pages = list(range(page, end_page + 1))
        text = "\n\n".join(f"PDF page {number}\n{document[number - 1].get_text()}"
                           for number in pages)
    return {"filename": filename, "pages": pages, "text": text[:14_000],
            "truncated": len(text) > 14_000}
