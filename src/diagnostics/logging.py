"""One logging setup and bounded error details for local evaluation runs."""

from __future__ import annotations

import logging


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")


def error_detail(exc: Exception, limit: int = 240) -> str:
    """Keep one line of diagnostic text without printing a full response body."""
    return " ".join(str(exc).split())[:limit]
