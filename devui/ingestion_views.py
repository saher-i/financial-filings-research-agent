"""Read-only ingestion summaries, chunk browsing, and filing previews."""

from __future__ import annotations

import json
import re
import threading
from collections import defaultdict
from pathlib import Path
from statistics import median

import pymupdf

from src.ingestion_pipeline import ROOT

STRATEGIES = ("structure", "semantic")
VARIANTS_DIR = ROOT / "data/index/variants"
SUMMARY_CACHE: dict[str, tuple[tuple[int, int], dict]] = {}
SUMMARY_LOCK = threading.Lock()


def chunk_path(strategy: str) -> Path:
    if strategy not in STRATEGIES:
        raise ValueError("Unknown chunking strategy")
    return ROOT / "data" / "index" / f"{strategy}_chunks.jsonl"


def chunk_summary(path: Path) -> dict:
    if not path.is_file():
        return {"ready": False, "chunks": 0}
    stat = path.stat()
    identity = (stat.st_mtime_ns, stat.st_size)
    cache_key = str(path)
    with SUMMARY_LOCK:
        cached = SUMMARY_CACHE.get(cache_key)
    if cached and cached[0] == identity:
        return cached[1]
    sizes: list[int] = []
    documents: dict[str, dict] = defaultdict(lambda: {"chunks": 0, "pages": set(), "table_chunks": 0})
    cross_page = table_chunks = 0
    with path.open() as stream:
        for line in stream:
            chunk = json.loads(line)
            sizes.append(len(chunk["text"]))
            document = documents[chunk["filename"]]
            document["chunks"] += 1
            document["pages"].update(chunk["pages"])
            has_table = any(anchor["kind"] == "table" for anchor in chunk["anchors"])
            document["table_chunks"] += has_table
            table_chunks += has_table
            cross_page += len(chunk["pages"]) > 1
    summary = {
        "ready": True, "chunks": len(sizes), "median_characters": int(median(sizes)) if sizes else 0,
        "mean_characters": round(sum(sizes) / len(sizes)) if sizes else 0,
        "cross_page_chunks": cross_page, "table_chunks": table_chunks,
        "documents": [
            {"filename": name, "chunks": info["chunks"], "pages": len(info["pages"]),
             "table_chunks": info["table_chunks"]}
            for name, info in sorted(documents.items())
        ],
        "file": str(path.relative_to(ROOT)),
    }
    with SUMMARY_LOCK:
        SUMMARY_CACHE[cache_key] = (identity, summary)
    return summary


def chunk_state() -> dict:
    return {strategy: chunk_summary(chunk_path(strategy)) for strategy in STRATEGIES}


def variant_id(strategy: str, target: int, maximum: int, overlap: int) -> str:
    if strategy not in STRATEGIES or not 800 <= target <= 6000 or not target <= maximum <= 8000 or not 0 <= overlap < target:
        raise ValueError("Use a valid strategy, target (800–6000), maximum (target–8000), and overlap (less than target)")
    return f"{strategy}-t{target}-m{maximum}-o{overlap}"


def variant_path(identifier: str) -> Path:
    if identifier in {f"{strategy}-default" for strategy in STRATEGIES}:
        return chunk_path(identifier.removesuffix("-default"))
    if not re.fullmatch(r"(?:structure|semantic)-t\d+-m\d+-o\d+", identifier):
        raise ValueError("Unknown trial")
    return VARIANTS_DIR / f"{identifier}.jsonl"


def variant_catalog() -> list[dict]:
    result = []
    for strategy in STRATEGIES:
        identifier = f"{strategy}-default"
        result.append({
            "id": identifier, "strategy": strategy, "target": 3600, "maximum": 4800,
            "overlap": 400, "default": True, "summary": chunk_summary(variant_path(identifier)),
        })
    if VARIANTS_DIR.exists():
        for path in sorted(VARIANTS_DIR.glob("*.json")):
            if not re.fullmatch(r"(?:structure|semantic)-t\d+-m\d+-o\d+", path.stem):
                continue
            metadata = json.loads(path.read_text())
            identifier = path.stem
            result.append({
                "id": identifier, "strategy": metadata["strategy"],
                "target": metadata["target"], "maximum": metadata["maximum"],
                "overlap": metadata["overlap"], "default": False,
                "build_seconds": metadata.get("build_seconds"),
                "summary": chunk_summary(variant_path(identifier)),
            })
    return result


def resolve_chunk_path(identifier: str) -> Path:
    return chunk_path(identifier) if identifier in STRATEGIES else variant_path(identifier)


def browse_chunks(identifier: str, filename: str = "", page: int | None = None,
                  query: str = "", offset: int = 0, limit: int = 10) -> dict:
    path = resolve_chunk_path(identifier)
    if not path.is_file():
        raise FileNotFoundError("Build the chunk set first")
    matches = []
    with path.open() as stream:
        for line in stream:
            chunk = json.loads(line)
            if filename and chunk["filename"] != filename:
                continue
            if page is not None and page not in chunk["pages"]:
                continue
            if query and query.casefold() not in chunk["text"].casefold():
                continue
            matches.append(chunk)
    selected = matches[offset:offset + limit]
    return {"total": len(matches), "offset": offset, "limit": limit, "chunks": [
        {"id": chunk["id"], "filename": chunk["filename"], "pages": chunk["pages"],
         "characters": len(chunk["text"]), "kinds": sorted({a["kind"] for a in chunk["anchors"]}),
         "text": chunk["text"]}
        for chunk in selected
    ]}


def filing_path(filename: str) -> Path | None:
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
    allowed = {Path(item).name: ROOT / item for item in manifest["pdfs"]}
    return allowed.get(filename)


def cited_page_highlights(filename: str, page_number: int, chunk_id: str) -> dict:
    """Locate a cited chunk's page boxes in either active chunk index."""
    path = filing_path(filename)
    if path is None or not path.is_file():
        raise FileNotFoundError("Unknown filing")
    with pymupdf.open(path) as pdf:
        if not 1 <= page_number <= len(pdf):
            raise ValueError("Page is outside this filing")
        bounds = pdf[page_number - 1].rect
    boxes = []
    if chunk_id:
        for strategy in STRATEGIES:
            chunks = chunk_path(strategy)
            if not chunks.is_file():
                continue
            with chunks.open() as stream:
                for line in stream:
                    chunk = json.loads(line)
                    if chunk.get("id") != chunk_id or chunk.get("filename") != filename:
                        continue
                    boxes = [anchor["bbox"] for anchor in chunk.get("anchors", [])
                             if anchor.get("page") == page_number and "bbox" in anchor][:100]
                    break
            if boxes:
                break
    return {"width": bounds.width, "height": bounds.height, "boxes": boxes}


def document_page(identifier: str, filename: str, page_number: int) -> dict:
    path = filing_path(filename)
    if path is None:
        raise ValueError("Unknown filing")
    chunks_path = resolve_chunk_path(identifier)
    if not chunks_path.is_file():
        raise FileNotFoundError("Build the chunk set first")
    with pymupdf.open(path) as pdf:
        if not 1 <= page_number <= len(pdf):
            raise ValueError("Page is outside this filing")
        bounds = pdf[page_number - 1].rect
        page_count = len(pdf)
    marks = []
    with chunks_path.open() as stream:
        for line in stream:
            chunk = json.loads(line)
            if chunk["filename"] != filename or page_number not in chunk["pages"]:
                continue
            boxes = [anchor["bbox"] for anchor in chunk["anchors"] if anchor["page"] == page_number]
            if boxes:
                marks.append({"id": chunk["id"], "boxes": boxes,
                              "preview": chunk["text"][:140]})
    return {"filename": filename, "page": page_number, "page_count": page_count,
            "width": bounds.width, "height": bounds.height, "marks": marks}


