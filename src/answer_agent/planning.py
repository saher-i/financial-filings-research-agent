"""Validate the model's research checklist before executing tools."""

from __future__ import annotations

import json
from typing import Literal, TypedDict

CalculationType = Literal["year_over_year_change", "grouped_change", "ratio", "other"]


class ResearchPlan(TypedDict):
    needs_sql: bool
    needs_pdf: bool
    needs_calculation: bool
    calculation_type: CalculationType | None
    required_facts: list[str]


PLAN_TOOL = {"type": "function", "function": {
    "name": "submit_research_plan",
    "description": ("Record a brief checklist of evidence and calculations needed to answer the current "
                    "question, considering recent conversation. Submit it alongside evidence tool calls "
                    "when possible. This checklist guides verification but never limits later tool choice."),
    "parameters": {"type": "object", "properties": {
        "needs_sql": {"type": "boolean"},
        "needs_pdf": {"type": "boolean"},
        "needs_calculation": {"type": "boolean"},
        "calculation_type": {"type": ["string", "null"],
                             "enum": ["year_over_year_change", "grouped_change", "ratio", "other", None]},
        "required_facts": {"type": "array", "items": {"type": "string"}, "maxItems": 12}},
        "required": ["needs_sql", "needs_pdf", "needs_calculation", "calculation_type",
                     "required_facts"], "additionalProperties": False}}}


def validate_research_plan(arguments: str | dict) -> ResearchPlan:
    """Reject malformed plans without limiting later tool selection."""
    try:
        data = json.loads(arguments) if isinstance(arguments, str) else arguments
    except (TypeError, ValueError) as exc:
        raise ValueError("Research plan is not valid JSON") from exc
    if not isinstance(data, dict) or set(data) != {
            "needs_sql", "needs_pdf", "needs_calculation", "calculation_type", "required_facts"}:
        raise ValueError("Research plan has missing or unknown fields")
    if any(type(data[key]) is not bool for key in ("needs_sql", "needs_pdf", "needs_calculation")):
        raise ValueError("Research plan evidence requirements must be booleans")
    calculation_type = data["calculation_type"]
    if calculation_type not in {None, "year_over_year_change", "grouped_change", "ratio", "other"}:
        raise ValueError("Research plan has unsupported calculation type")
    if bool(calculation_type) != data["needs_calculation"]:
        raise ValueError("Research plan calculation fields disagree")
    facts = data["required_facts"]
    if (not isinstance(facts, list) or not 1 <= len(facts) <= 12
            or any(not isinstance(fact, str) or not 1 <= len(fact.strip()) <= 180
                   for fact in facts)):
        raise ValueError("Research plan needs one to twelve short required facts")
    return {**data, "required_facts": [fact.strip() for fact in facts]}


