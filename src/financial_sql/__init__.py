"""Read-only access to the supplied financial database."""

from .tool import execute_readonly_sql, get_schema

__all__ = ["execute_readonly_sql", "get_schema"]
