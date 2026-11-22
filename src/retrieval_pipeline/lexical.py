"""Keyword search over the same filing chunks stored in Qdrant."""

import json
import re
import sqlite3
from pathlib import Path

from src.ingestion_pipeline.embedding.build import file_hash, read_chunks


def build_lexical_index(chunks_path: Path, index_path: Path) -> int:
    """Atomically write an FTS5 index for one chunk file."""
    chunks = read_chunks(chunks_path)
    source_hash = file_hash(chunks_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = index_path.with_name(index_path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with sqlite3.connect(temporary) as db:
            db.executescript("""
                CREATE TABLE chunks (
                    id TEXT PRIMARY KEY, ticker TEXT NOT NULL, fiscal_year INTEGER NOT NULL,
                    filename TEXT NOT NULL, pages TEXT NOT NULL, anchors TEXT NOT NULL,
                    text TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE chunks_fts USING fts5(id UNINDEXED, text, tokenize='unicode61');
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE INDEX chunk_filters ON chunks(ticker, fiscal_year);
            """)
            for chunk in chunks:
                db.execute("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
                           (chunk["id"], chunk["ticker"], chunk["fiscal_year"],
                            chunk["filename"], json.dumps(chunk["pages"]),
                            json.dumps(chunk["anchors"]), chunk["text"]))
                db.execute("INSERT INTO chunks_fts(id, text) VALUES (?, ?)",
                           (chunk["id"], chunk["text"]))
            db.executemany("INSERT INTO metadata VALUES (?, ?)",
                           [("source_hash", source_hash), ("count", str(len(chunks)))])
            db.commit()
        temporary.replace(index_path)
    finally:
        temporary.unlink(missing_ok=True)
    return len(chunks)


def ensure_lexical_index(chunks_path: Path, index_path: Path) -> str:
    """Reuse the index only when it contains the current chunk file."""
    source_hash = file_hash(chunks_path)
    if index_path.is_file():
        try:
            with sqlite3.connect(index_path) as db:
                metadata = dict(db.execute("SELECT key, value FROM metadata"))
                count = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                if metadata.get("source_hash") == source_hash and count == int(metadata["count"]):
                    return source_hash
        except (sqlite3.DatabaseError, KeyError, ValueError):
            pass
    build_lexical_index(chunks_path, index_path)
    return source_hash


def search_lexical(index_path: Path, question: str, limit: int = 20,
                   ticker: str | None = None, fiscal_year: int | None = None) -> list[dict]:
    """Rank chunks by SQLite FTS5 BM25; return the same evidence fields as Qdrant."""
    if limit < 1:
        return []
    terms = re.findall(r"\w+", question, flags=re.UNICODE)
    if not terms:
        raise ValueError("Question must contain searchable words")
    expression = " OR ".join(f'"{term}"' for term in dict.fromkeys(terms))
    filters = []
    values = [expression]
    if ticker:
        filters.append("c.ticker = ?")
        values.append(ticker.upper())
    if fiscal_year is not None:
        filters.append("c.fiscal_year = ?")
        values.append(fiscal_year)
    where = " AND " + " AND ".join(filters) if filters else ""
    with sqlite3.connect(index_path) as db:
        rows = db.execute(f"""SELECT c.id, c.ticker, c.fiscal_year, c.filename,
                                  c.pages, c.anchors, c.text, bm25(chunks_fts)
                           FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.id
                           WHERE chunks_fts MATCH ?{where}
                           ORDER BY bm25(chunks_fts) LIMIT ?""",
                          [*values, limit]).fetchall()
    return [{"id": row[0], "ticker": row[1], "fiscal_year": row[2],
             "filename": row[3], "pages": json.loads(row[4]),
             "anchors": json.loads(row[5]), "text": row[6], "score": -float(row[7])}
            for row in rows]
