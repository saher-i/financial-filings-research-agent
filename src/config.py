"""Local configuration shared by the HTTP server and command-line agent."""

from __future__ import annotations

import os
import shlex
from pathlib import Path

from src.ingestion_pipeline import ROOT


def load_local_fireworks_key(env_file: Path = ROOT / ".env") -> None:
    """Load the local key only when the process has not exported one."""
    if os.getenv("FIREWORKS_API_KEY") or not env_file.is_file():
        return
    for raw_line in env_file.read_text().splitlines():
        name, separator, raw_value = raw_line.strip().partition("=")
        if name.removeprefix("export ").strip() != "FIREWORKS_API_KEY" or not separator:
            continue
        try:
            values = shlex.split(raw_value, comments=True)
        except ValueError as exc:
            raise ValueError("Invalid FIREWORKS_API_KEY entry in .env") from exc
        if len(values) == 1 and values[0]:
            os.environ["FIREWORKS_API_KEY"] = values[0]
        return
