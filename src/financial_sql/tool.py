"""A bounded, read-only SQL evidence tool for the supplied financial tables."""

from __future__ import annotations

import re
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from src.ingestion_pipeline import ROOT

DEFAULT_DB = ROOT / "data" / "financials.db"
TABLES = frozenset({"companies", "income_statements", "balance_sheets",
                    "segment_revenue", "geographic_revenue"})
FUNCTIONS = frozenset({"abs", "avg", "coalesce", "count", "ifnull", "length", "like",
                       "lower", "max", "min", "nullif", "round", "sum", "total",
                       "upper", "rank", "dense_rank", "row_number", "lag", "lead"})
MAX_SQL_LENGTH = 20_000
TIMEOUT_SECONDS = 2.0


def _connect(db_path: Path) -> sqlite3.Connection:
    path = Path(db_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Financial database not found: {path}")
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=1)
    db.execute("PRAGMA query_only = ON")
    db.enable_load_extension(False)
    db.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, MAX_SQL_LENGTH)
    return db


def get_schema(db_path: Path | None = None) -> dict:
    """Return only the financial tables and columns that SQL questions may use."""
    path = Path(db_path or DEFAULT_DB).resolve()
    with closing(_connect(path)) as db:
        tables = {
            table: [{"name": row[1], "type": row[2]} for row in db.execute(f"PRAGMA table_info({table})")]
            for table in sorted(TABLES)
        }
        companies = [{"ticker": ticker, "name": name}
                     for ticker, name in db.execute("SELECT ticker, name FROM companies ORDER BY ticker")]
        years = [row[0] for row in db.execute(
            "SELECT DISTINCT fiscal_year FROM income_statements ORDER BY fiscal_year")]
    return {"database": path.name, "tables": tables, "companies": companies,
            "fiscal_years": years,
            "notes": ["Money columns are stored as whole U.S. dollars.",
                      "Filter period_type = 'FY' for annual values.",
                      "Use period_end or fiscal_year to select the intended reporting period."]}


def _unique_columns(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    result = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        result.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return result


def execute_readonly_sql(sql: str, db_path: Path | None = None, max_rows: int = 100) -> dict:
    """Execute a single SELECT/WITH statement against the five supplied tables.

    SQLite's read-only connection and authorizer both enforce the boundary.
    The progress handler interrupts long-running queries, and returned rows are capped.
    """
    if not isinstance(sql, str) or not re.match(r"\s*(?:SELECT|WITH)\b", sql, re.IGNORECASE):
        raise ValueError("Only a SELECT or WITH query is allowed")
    if not sql.strip() or len(sql) > MAX_SQL_LENGTH:
        raise ValueError("SQL query is empty or too long")
    if type(max_rows) is not int or not 1 <= max_rows <= 1_000:
        raise ValueError("max_rows must be between 1 and 1000")

    path = Path(db_path or DEFAULT_DB).resolve()
    read_tables: set[str] = set()
    started = time.monotonic()
    interrupted = False

    def authorize(action: int, arg1: str | None, arg2: str | None,
                  database: str | None, trigger: str | None) -> int:
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ and database in ("main", None) and arg1 in TABLES:
            read_tables.add(arg1)
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in FUNCTIONS:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    def deadline() -> int:
        nonlocal interrupted
        interrupted = time.monotonic() - started > TIMEOUT_SECONDS
        return int(interrupted)

    with closing(_connect(path)) as db:
        db.set_authorizer(authorize)
        db.set_progress_handler(deadline, 1_000)
        try:
            cursor = db.execute(sql)
            if cursor.description is None or not read_tables:
                raise ValueError("Query must read at least one financial table")
            columns = _unique_columns([item[0] for item in cursor.description])
            values = cursor.fetchmany(max_rows + 1)
        except sqlite3.DatabaseError as exc:
            if interrupted:
                raise TimeoutError("Financial query exceeded the time limit") from exc
            raise ValueError(f"Financial query rejected: {exc}") from exc
        finally:
            db.set_progress_handler(None, 0)
            db.set_authorizer(None)

    rows = [dict(zip(columns, row)) for row in values[:max_rows]]
    return {"columns": columns, "rows": rows, "row_count": len(rows),
            "truncated": len(values) > max_rows, "sql": sql.strip(),
            "source": {"type": "sqlite", "database": path.name,
                       "tables": sorted(read_tables)}}
