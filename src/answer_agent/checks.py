"""Deterministic citation and calculation checks for draft answers."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


def _valid_source_ids(answer: str, sources: list[dict]) -> bool:
    cited = set(re.findall(r"\[S\d+\]", answer))
    known = {f"[{source['id']}]" for source in sources}
    return bool(cited) and cited <= known


def _normalize_citations(answer: str) -> str:
    """Use one visible citation style for common bracket variants from models."""
    def source_range(match: re.Match) -> str:
        start, end = int(match.group(1)), int(match.group(2))
        return ("".join(f"[S{number}]" for number in range(start, end + 1))
                if start <= end <= start + 10 else match.group(0))

    # Invisible formatting can occur anywhere inside an otherwise valid tag.
    answer = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", answer)
    answer = re.sub(r"\[\s*((?:[Ss]\d+\s*,\s*)*[Ss]\d+)\s*;\s*calculate\s*\]",
                    lambda match: "".join(f"[S{number}]" for number in
                                          re.findall(r"[Ss](\d+)", match.group(1))),
                    answer, flags=re.IGNORECASE)
    answer = re.sub(r"[\[【]\s*[Ss](\d+)\s*[-‑–—]\s*[Ss](\d+)\s*[\]】]",
                    source_range, answer)
    answer = re.sub(r"\[\s*[Ss](\d+)\s*,\s*p\.?\s*(\d+)\s*\]",
                    lambda match: f"[S{match.group(1)}] (p. {match.group(2)})", answer)
    answer = re.sub(r"\[\s*[Ss](\d+)\s*,\s*[Ss](\d+)\s*\]",
                    lambda match: f"[S{match.group(1)}][S{match.group(2)}]", answer)
    return re.sub(r"[\[（(【〔]\s*[Ss](\d+)\s*[\]）)】〕]",
                  lambda match: f"[S{match.group(1)}]", answer)


def _format_calculated_precision(answer: str, trace: list[dict]) -> str:
    """Replace verbose raw calculator decimals with that result's display rounding."""
    for entry in trace:
        if entry.get("tool") != "calculate" or "error" in entry:
            continue
        result = entry.get("result") or {}
        operation = result.get("operation")
        if operation not in {"ratio", "percentage", "percent_change"}:
            continue
        raw = result.get("result")
        rounded = result.get("rounded_2" if operation == "ratio" else "rounded_1")
        if not isinstance(raw, str) or not isinstance(rounded, str) or "." not in raw:
            continue
        if len(raw.partition(".")[2]) <= 4:
            continue
        answer = re.sub(r"(?<![\d.])" + re.escape(raw) + r"(?!\d)", rounded, answer)
    return answer


_NUMERIC_CLAIM = re.compile(
    r"(?<![A-Za-z0-9])(?P<sign>[-−])?\s*(?P<currency>\$)?\s*(?P<number>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<unit>%|trillion|billion|million|[BMTx]\b)?", re.IGNORECASE)
_CITATIONS = re.compile(r"(?:\s*\[S\d+\])+")
_SOURCE_NUMBER = re.compile(r"(?<![A-Za-z])\d[\d,]*(?:\.\d+)?")


def _has_calculation(state: dict, operation: str | None = None) -> bool:
    return any(entry.get("tool") == "calculate" and "error" not in entry
               and (operation is None or entry.get("arguments", {}).get("operation") == operation)
               for entry in state.get("trace", []))


def _required_calculation_operation(state: dict) -> str | None:
    """Retained for trace inspection; a model's plan is never an answer gate."""
    plan = state.get("research_plan") or {}
    return "grouped_change" if plan.get("calculation_type") == "grouped_change" else None


def _number(value: object) -> Decimal | None:
    try:
        result = Decimal(str(value).replace(",", ""))
    except (InvalidOperation, ValueError):
        return None
    return result if result.is_finite() else None


def _matches(claim: re.Match, raw: Decimal, kind: str) -> bool:
    """Compare a displayed amount at its displayed precision, allowing unit conversion."""
    value = _number(("-" if claim.group("sign") else "") + claim.group("number"))
    if value is None:
        return False
    places = len(claim.group("number").partition(".")[2])
    tolerance = Decimal("0.5").scaleb(-places)
    unit = (claim.group("unit") or "").casefold()
    if unit == "%" and kind not in {"percent", "ratio", "pdf_percent"}:
        return False
    if unit == "x" and kind not in {"ratio", "sql_ratio"}:
        return False
    if kind in {"percent", "ratio"}:
        if unit not in ({"%"} if kind == "percent" else {"x", "%"}):
            return False
        factors = (Decimal("0.01"),) if kind == "ratio" and unit == "%" else (Decimal(1),)
    elif kind == "billions":
        if unit not in {"b", "billion"}:
            return False
        factors = (Decimal(1),)
    elif kind == "millions":
        if unit not in {"m", "million"}:
            return False
        factors = (Decimal(1),)
    elif unit in {"b", "billion"}:
        factors = (Decimal(1_000_000_000),) if kind in {"sql", "calculated_raw"} else (
            Decimal(1), Decimal(1_000), Decimal(1_000_000_000))
    elif unit in {"m", "million"}:
        factors = (Decimal(1_000_000),) if kind in {"sql", "calculated_raw"} else (
            Decimal(1), Decimal(1_000_000))
    elif unit in {"t", "trillion"}:
        factors = (Decimal(1_000_000_000_000),) if kind in {"sql", "calculated_raw"} else (
            Decimal(1), Decimal(1_000), Decimal(1_000_000_000_000))
    else:
        factors = (Decimal(1),)
    return any(abs(raw / factor - value) <= tolerance for factor in factors)


def _source_values(source: dict) -> list[tuple[Decimal, str]]:
    if source.get("type") == "sql":
        values = ((value, "billions" if key.endswith(("_billion", "_billions")) else
                   "millions" if key.endswith(("_million", "_millions")) else
                   "percent" if key.endswith(("_percent", "_percentage", "_pct")) else
                   "sql_ratio" if key.endswith("_ratio") else "sql")
                  for row in source.get("rows", []) if isinstance(row, dict)
                  for key, value in row.items())
    else:
        body = str(source.get("text", ""))
        values = ((match.group(), "pdf_percent" if body[match.end():].lstrip().startswith("%")
                   else "pdf") for match in _SOURCE_NUMBER.finditer(body))
    return [(number, kind) for value, kind in values if (number := _number(value)) is not None]


def _calculated_values(entry: dict) -> list[tuple[Decimal, str]]:
    if entry.get("tool") != "calculate" or "error" in entry:
        return []
    result = entry.get("result") or {}
    operation = result.get("operation") or (entry.get("arguments") or {}).get("operation")
    result_kind = "percent" if operation in {"percent_change", "percentage"} else (
        "ratio" if operation == "ratio" else "calculated_raw")
    fields = [(result.get(key), result_kind) for key in ("result", "rounded_1", "rounded_2")]
    fields.extend([(result.get("result_billions"), "billions"),
                   (result.get("total_change"), "calculated_raw"),
                   (result.get("total_change_billions"), "billions")])
    for group in result.get("groups", []):
        fields.extend([(group.get("change"), "calculated_raw"),
                       (group.get("change_billions"), "billions")])
    return [(number, kind) for value, kind in fields if (number := _number(value)) is not None]


def _unverified_numeric_claims(draft: str, sources: list[dict], trace: list[dict]) -> list[str]:
    """Check numeric presence and units against cited evidence or a calculation.

    A number printed directly in its cited source needs no calculator call. This is
    a conservative numeric check; semantic claim and citation support require the
    model review rather than a keyword classifier.
    """
    by_id = {source.get("id"): source for source in sources}
    findings = []
    start = 0
    for citation_group in _CITATIONS.finditer(draft):
        text = draft[start:citation_group.start()]
        cited_ids = re.findall(r"\[S(\d+)\]", citation_group.group())
        cited_sources = [by_id[f"S{number}"] for number in cited_ids if f"S{number}" in by_id]
        direct = [item for source in cited_sources for item in _source_values(source)]
        calculated = [item for entry in trace
                      if entry.get("source_ids") and
                      set(entry["source_ids"]) <= {source.get("id") for source in cited_sources}
                      for item in _calculated_values(entry)]
        for claim in _NUMERIC_CLAIM.finditer(text):
            if not claim.group("currency") and not claim.group("unit"):
                continue
            if not any(_matches(claim, value, kind) for value, kind in direct + calculated):
                findings.append(claim.group().strip())
        start = citation_group.end()
    for claim in _NUMERIC_CLAIM.finditer(draft[start:]):
        if claim.group("currency") or claim.group("unit"):
            findings.append(claim.group().strip())
    return findings


def _has_unverified_numeric_claim(draft: str, sources: list[dict],
                                  trace: list[dict] | None = None) -> bool:
    return bool(_unverified_numeric_claims(draft, sources, trace or []))


