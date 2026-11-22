"""Fast, bounded query reconstruction for the lexical branch only."""

from __future__ import annotations

import os
import re
import threading
import time
from functools import lru_cache

from src.financial_sql import get_schema

MODEL = "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b"
_CACHE: dict[str, str] = {}
_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _known_entities() -> tuple[str, ...]:
    """Use the available financial corpus, not a development-question name list."""
    try:
        companies = get_schema().get("companies", [])
    except (OSError, ValueError):
        return ()
    names = set()
    for company in companies:
        ticker, name = company.get("ticker"), company.get("name")
        if isinstance(ticker, str) and ticker:
            names.add(ticker)
        if isinstance(name, str) and name:
            names.add(name)
            short_name = name.split()[0]
            if len(short_name) >= 4:
                names.add(short_name)
    return tuple(sorted(names, key=lambda name: (-len(name), name)))


def _clean(candidate: str, original: str, entities: tuple[str, ...] | None = None) -> str:
    value = candidate.strip().strip('`"\' ').splitlines()[0].strip() if candidate.strip() else ""
    if (not value or len(value) > 280 or not re.search(r"[A-Za-z0-9]", value)
            or any(char in value for char in "{}<>")):
        return original
    # An abbreviated search may shorten the question, but it must keep its scope.
    original_years = set(re.findall(r"(?<!\d)20\d{2}\b", original))
    rewritten_years = set(re.findall(r"(?<!\d)20\d{2}\b", value))
    if original_years != rewritten_years:
        return original
    for entity in entities if entities is not None else _known_entities():
        pattern = rf"(?<!\w){re.escape(entity)}(?!\w)"
        if re.search(pattern, original, re.I) and not re.search(pattern, value, re.I):
            return original
    # Preserve an unfamiliar all-capitals identifier even before it is indexed.
    for identifier in re.findall(r"(?<!\w)[A-Z]{2,8}(?!\w)", original):
        if not re.search(rf"(?<!\w){re.escape(identifier)}(?!\w)", value):
            return original
    return value


def rewrite_lexical_query(question: str, *, client=None) -> dict:
    """Return a short spelling-aware query; fail open to the original text."""
    started = time.perf_counter()
    if not question.strip():
        raise ValueError("Search query cannot be empty")
    with _LOCK:
        cached = _CACHE.get(question)
    if cached is not None and client is None:
        return {"query": cached, "status": "cached", "model": MODEL, "duration_ms": 0.0}
    status = "rewritten"
    rewritten = question
    try:
        if client is None:
            key = os.getenv("FIREWORKS_API_KEY")
            if not key:
                raise RuntimeError("Fireworks key unavailable")
            from openai import OpenAI
            client = OpenAI(api_key=key, base_url="https://api.fireworks.ai/inference/v1",
                            timeout=6.0, max_retries=0)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Rewrite the user's filing search into one short keyword search query. "
                 "Correct obvious spelling errors or incomplete names using context. Preserve company names, "
                 "fiscal years, requested concepts, and meaning. Do not answer the question. "
                 "Return only the query, with no explanation or quotation marks."},
                {"role": "user", "content": question},
            ], max_tokens=80, temperature=0)
        candidate = response.choices[0].message.content or ""
        rewritten = _clean(candidate, question)
        if rewritten == question:
            status = "unchanged"
        elif client is not None:
            with _LOCK:
                if len(_CACHE) >= 256:
                    _CACHE.pop(next(iter(_CACHE)))
                _CACHE[question] = rewritten
    except Exception:
        status = "fallback"
    return {"query": rewritten, "status": status, "model": MODEL,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1)}
