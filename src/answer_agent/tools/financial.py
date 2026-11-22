"""Read-only financial SQL lookup used by the answering agent."""

from pathlib import Path

from src.financial_sql import execute_readonly_sql


def query_financials(sql: str, db_path: Path | None = None) -> dict:
    return execute_readonly_sql(sql, db_path=db_path, max_rows=100)
