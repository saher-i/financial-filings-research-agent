"""Deterministic arithmetic using numbers verified in gathered SQL or PDF evidence."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import re


def _number(value: object) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError("Selected field is not numeric")
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("Selected field is not numeric") from exc
    if not number.is_finite():
        raise ValueError("Selected field is not finite")
    return number


def _text(value: Decimal) -> str:
    return format(value.normalize(), "f") if value else "0"


def _rounded(value: Decimal, places: int) -> str:
    return _text(value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def _sql_source(sources: list[dict], source_id: str) -> dict:
    source = next((item for item in sources if item.get("id") == source_id), None)
    if source is None or source.get("type") != "sql":
        raise ValueError("Calculator operands must reference a gathered SQL source")
    if source.get("truncated"):
        raise ValueError("Query returned truncated rows; narrow the SQL query first")
    return source


_PDF_SCALE = {"thousands": Decimal(1_000), "millions": Decimal(1_000_000),
              "billions": Decimal(1_000_000_000)}
_PDF_UNIT_HEADING = re.compile(r"\b(?:in|amounts?\s+in)\s+(thousands|millions|billions)\b", re.I)


def _operand(sources: list[dict], item: dict, value_unit: str = "raw") -> tuple[Decimal, dict]:
    if not isinstance(item, dict):
        raise ValueError("Each operand must select a gathered source value")
    source_id, row_index, field = item.get("source_id"), item.get("row"), item.get("field")
    if isinstance(source_id, str) and "value" in item:
        source = next((source for source in sources if source.get("id") == source_id), None)
        if source is None or source.get("type") != "pdf":
            raise ValueError("PDF value operands must reference a gathered PDF source")
        raw = item["value"]
        anchor = item.get("anchor")
        if (not isinstance(raw, str) or len(raw) > 30
                or not re.fullmatch(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", raw)
                or not isinstance(anchor, str) or not 2 <= len(anchor) <= 100):
            raise ValueError("PDF operand needs a written number and a nearby text anchor")
        passage = str(source.get("text", ""))
        if source.get("truncated"):
            raise ValueError("Read the complete PDF page before calculating")
        number_pattern = re.compile(r"(?<![\d,])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\d,])")
        matched_at = None
        for occurrence in re.finditer(re.escape(anchor), passage, re.IGNORECASE):
            window = passage[occurrence.end():occurrence.end() + 220]
            if any(token.group() == raw for token in number_pattern.finditer(window)):
                matched_at = occurrence.start()
                break
        if matched_at is None:
            raise ValueError("PDF number was not found near the text anchor in that source")
        value = _number(raw.replace(",", ""))
        selector = {"source_id": source_id, "anchor": anchor, "written_value": raw}
        if value_unit == "usd":
            headings = list(_PDF_UNIT_HEADING.finditer(passage[max(0, matched_at - 700):matched_at]))
            if headings:
                unit = headings[-1].group(1).lower()
                value *= _PDF_SCALE[unit]
                selector["written_unit"] = unit
        selector["value"] = _text(value)
        return value, selector
    if not isinstance(source_id, str) or type(row_index) is not int or not isinstance(field, str):
        raise ValueError("Operand needs source_id, row, and field")
    source = _sql_source(sources, source_id)
    rows = source.get("rows", [])
    if row_index < 0 or row_index >= len(rows) or field not in rows[row_index]:
        raise ValueError("Operand does not identify a returned SQL value")
    value = _number(rows[row_index][field])
    return value, {"source_id": source_id, "row": row_index, "field": field, "value": _text(value)}


def calculate(sources: list[dict], arguments: dict) -> dict:
    """Compute a basic operation or a full year-over-year category breakdown."""
    operation = arguments.get("operation")
    value_unit = arguments.get("value_unit")
    if value_unit is not None and value_unit not in {"usd", "raw"}:
        raise ValueError("value_unit must be usd or raw")
    if operation == "grouped_change":
        source_id = arguments.get("source_id")
        year_field = arguments.get("year_field", "fiscal_year")
        group_field = arguments.get("group_field", "segment_name")
        value_field = arguments.get("value_field", "revenue")
        # The supplied SQL schema defines revenue as whole USD. Other fields
        # need an explicit unit for dollar scaling; otherwise keep raw values.
        inferred_unit = value_unit is None
        value_unit = value_unit or ("usd" if value_field == "revenue" else "raw")
        base_year = arguments.get("base_year")
        new_year = arguments.get("new_year")
        if (not isinstance(source_id, str) or type(base_year) is not int
                or type(new_year) is not int or base_year == new_year
                or any(not isinstance(name, str) or not name.isidentifier()
                       for name in (year_field, group_field, value_field))):
            raise ValueError("Provide a SQL source, two distinct years, and valid field names")
        source = _sql_source(sources, source_id)
        by_group: dict[str, dict[int, Decimal]] = {}
        for row in source.get("rows", []):
            if not all(key in row for key in (year_field, group_field, value_field)):
                raise ValueError("The SQL result must include year, group, and value fields")
            year = row[year_field]
            if year not in (base_year, new_year):
                continue
            group = str(row[group_field])
            years = by_group.setdefault(group, {})
            if year in years:
                raise ValueError("The SQL result has duplicate rows for a group and year")
            years[year] = _number(row[value_field])
        if not by_group or any(set(years) != {base_year, new_year} for years in by_group.values()):
            raise ValueError("Both years are required for every group; query complete rows")
        groups = [{"group": group, "base": _text(years[base_year]),
                   "new": _text(years[new_year]),
                   "change": _text(years[new_year] - years[base_year]),
                   "change_billions": _rounded((years[new_year] - years[base_year]) / Decimal(1_000_000_000), 3)
                   if value_unit == "usd" else None}
                  for group, years in sorted(by_group.items())]
        total = sum((years[new_year] - years[base_year] for years in by_group.values()), Decimal(0))
        result = {"operation": operation, "base_year": base_year, "new_year": new_year,
                  "value_unit": value_unit, "unit_inferred": inferred_unit,
                  "source_ids": [source_id], "groups": groups, "total_change": _text(total),
                  "total_change_billions": _rounded(total / Decimal(1_000_000_000), 3)
                  if value_unit == "usd" else None}
        expected = arguments.get("expected_total")
        if expected is not None:
            value, selector = _operand(sources, expected, value_unit)
            result["expected_total"] = selector
            result["difference_from_expected"] = _text(total - value)
            result["reconciled"] = total == value
            result["source_ids"] = list(dict.fromkeys([source_id, selector["source_id"]]))
        return result

    value_unit = value_unit or "raw"
    items = arguments.get("operands")
    if not isinstance(items, list) or not 1 <= len(items) <= 20:
        raise ValueError("Provide 1–20 SQL row or PDF value operands")
    selected = [_operand(sources, item, value_unit) for item in items]
    values = [value for value, _ in selected]
    if operation == "percent_change" and len(items) == 2 and all("anchor" in item for item in items):
        if items[0]["anchor"].casefold() != items[1]["anchor"].casefold():
            raise ValueError("For percent_change, PDF operands must use the same metric anchor")
    if operation == "percent_change" and len(items) == 2 and all("row" in item for item in items):
        first_source = _sql_source(sources, items[0]["source_id"])
        second_source = _sql_source(sources, items[1]["source_id"])
        newer = first_source["rows"][items[0]["row"]]
        older = second_source["rows"][items[1]["row"]]
        if (isinstance(newer.get("fiscal_year"), int)
                and isinstance(older.get("fiscal_year"), int)):
            if newer["fiscal_year"] <= older["fiscal_year"]:
                raise ValueError("For percent_change, the first operand must be the newer fiscal year and the second the older year")
            for field in ("company_ticker", "segment_name"):
                if field in newer and field in older and newer[field] != older[field]:
                    raise ValueError(f"For percent_change, both operands must have the same {field}")
    if operation == "sum":
        result = sum(values, Decimal(0))
    elif operation in {"difference", "ratio", "percentage", "percent_change"}:
        if len(values) != 2:
            raise ValueError("This operation requires exactly two operands")
        if operation == "difference":
            result = values[0] - values[1]
        else:
            if values[1] == 0:
                raise ValueError("Cannot divide by zero")
            result = values[0] / values[1]
            if operation == "percentage":
                result *= 100
            if operation == "percent_change":
                result = (result - 1) * 100
    else:
        raise ValueError("Choose sum, difference, ratio, percentage, percent_change, or grouped_change")
    return {"operation": operation, "result": _text(result),
            "rounded_1": _rounded(result, 1), "rounded_2": _rounded(result, 2),
            "result_billions": _rounded(result / Decimal(1_000_000_000), 3)
            if operation in {"sum", "difference"} and value_unit == "usd"
            else None,
            "source_ids": list(dict.fromkeys(item["source_id"] for _, item in selected)),
            "operands": [item for _, item in selected]}
