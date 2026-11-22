"""Small records shared by extraction and chunking."""

from dataclasses import dataclass


@dataclass
class Block:
    text: str
    kind: str
    page: int
    bbox: list[float]


@dataclass
class Chunk:
    id: str
    strategy: str
    ticker: str
    fiscal_year: int
    filename: str
    pages: list[int]
    text: str
    anchors: list[dict]
