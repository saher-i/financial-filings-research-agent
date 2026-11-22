"""Apply the development question file's declared grading methods to saved answers."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from eval_pipeline.generation_metrics import judge_record

ROOT = Path(__file__).resolve().parents[1]
NUMBER = re.compile(r"(?<![\w])\$?\s*(\d[\d,]*(?:\.\d+)?)\s*"
                    r"(billion|million|trillion|%|x|B|M|T)?(?![A-Za-z0-9])", re.I)
ALIASES = {"AAPL": ("apple", "aapl"), "MSFT": ("microsoft", "msft"),
           "GOOGL": ("alphabet", "google", "googl")}
QUESTION_WORDS = frozenset({"which", "whose", "where", "what", "when", "from", "their",
                            "these", "those", "between", "would", "could", "about",
                            "fiscal", "years", "year", "most", "recent", "company"})


def numeric_matches(answer: str, expected: float, question: str = "",
                    entity_aliases: tuple[str, ...] = ()) -> tuple[bool, list[float]]:
    """Compare amounts and reject a match explicitly tied to another period."""
    numbers = []
    clean_answer = re.sub(r"\[S\d+\]", "", answer)
    requested_years = set(re.findall(r"(?<!\d)20\d{2}\b", question))
    question_terms = {word for word in re.findall(r"[a-z]{6,}", question.lower())
                      if word not in QUESTION_WORDS
                      and not any(word in names for names in ALIASES.values())}
    multiple_values = len(list(NUMBER.finditer(clean_answer))) > 1
    mentioned_entities = {key for key, names in ALIASES.items()
                          if any(re.search(rf"\b{re.escape(name)}\b", clean_answer, re.I)
                                 for name in names)}
    matched = False
    for match in NUMBER.finditer(clean_answer):
        value = float(match.group(1).replace(",", ""))
        unit = (match.group(2) or "").lower()
        if unit in {"b", "billion"}:
            value *= 1e9 if expected > 1e6 else 1
        elif unit in {"m", "million"}:
            value *= 1e6 if expected > 1e6 else 1e-3
        elif unit in {"t", "trillion"}:
            value *= 1e12 if expected > 1e6 else 1e3
        numbers.append(value)
        # A matching number in an explicitly different year is not an answer to
        # a single-year question. This remains a lightweight numeric check.
        boundaries = list(re.finditer(r"(?:[;\n]|[.!?](?=\s+[A-Z]))", clean_answer))
        clause_start = max((item.end() for item in boundaries if item.end() <= match.start()), default=0)
        ends = [item.start() for item in boundaries if item.start() >= match.end()]
        clause = clean_answer[clause_start:min(ends) if ends else len(clean_answer)]
        clause_years = set(re.findall(r"(?<!\d)20\d{2}\b", clause))
        year_conflict = len(requested_years) == 1 and bool(clause_years) and not (requested_years & clause_years)
        # A semicolon can introduce the calculated answer after its operands.
        # Use the whole sentence for topic context, while checking period conflicts
        # against the narrower clause containing the particular number.
        sentence_boundaries = list(re.finditer(r"[.!?](?=\s+[A-Z])", clean_answer))
        sentence_start = max((item.end() for item in sentence_boundaries
                              if item.end() <= match.start()), default=0)
        sentence_ends = [item.start() for item in sentence_boundaries
                         if item.start() >= match.end()]
        sentence = clean_answer[sentence_start:min(sentence_ends) if sentence_ends else len(clean_answer)]
        nearby_terms = set(re.findall(r"[a-z]{6,}", sentence.lower()))
        needed_overlap = min(2, len(question_terms))
        context_mismatch = multiple_values and needed_overlap > 0 and len(question_terms & nearby_terms) < needed_overlap
        entity_mismatch = (entity_aliases and len(mentioned_entities) > 1
                           and not any(re.search(rf"\b{re.escape(alias)}\b", clause, re.I)
                                       for alias in entity_aliases))
        tolerance = max(abs(expected) * 0.001, 0.051 if expected < 1000 else 5e7)
        matched |= (abs(value - expected) <= tolerance and not year_conflict
                    and not context_mismatch and not entity_mismatch)
    return matched, numbers


def grade(question: dict, record: dict, judge=judge_record) -> dict:
    method = question["evaluation"]
    answer = str(record.get("answer") or "")
    if not answer or record.get("error") or record.get("answer_status", "answered") not in {
            "answered", "needs_verification"}:
        return {"status": "unavailable", "reason": "No completed answer", "method": method}
    if method == "fuzzy_numeric":
        expected = question["gold_answer_numeric"]
        matched, values = numeric_matches(answer, expected, question["question"])
        return {"status": "correct" if matched else "incorrect", "method": method,
                "reason": "Expected numeric value found within rounding tolerance" if matched else "Expected numeric value not found",
                "expected_numeric": expected, "extracted_numbers": values}
    if method == "exact_match_entity":
        expected = question["gold_answer_numeric"]
        expected_entity = next((key for key, aliases in ALIASES.items()
                                if any(re.search(rf"\b{re.escape(alias)}\b", question["gold_answer"], re.I)
                                       for alias in aliases)), None)
        aliases = ALIASES.get(expected_entity, ())
        matched_number, values = numeric_matches(answer, expected, question.get("question", ""), aliases)
        matched_entity = any(re.search(rf"\b{re.escape(alias)}\b", answer, re.I) for alias in aliases)
        return {"status": "correct" if matched_entity and matched_number else "incorrect",
                "method": method, "reason": "Entity and numeric result both match" if matched_entity and matched_number
                else "Entity or numeric result missing", "expected_entity": expected_entity,
                "entity_match": matched_entity, "numeric_match": matched_number,
                "expected_numeric": expected, "extracted_numbers": values}
    if method == "llm_judge":
        review = judge(record, question["gold_answer"])
        return {"status": review["status"], "method": method,
                "reason": review.get("reason"), "findings": review.get("findings", []),
                "error_stage": review.get("error_stage"),
                "error_detail": review.get("error_detail"),
                "model": review.get("model"), "dimensions": {
                    key: review.get(key) for key in ("coverage", "faithfulness", "relevance", "citation_support")}}
    raise ValueError(f"Unknown evaluation method: {method}")


def evaluate(root: Path = ROOT, judge=judge_record, progress=None,
             answer_run_path: Path | None = None,
             result_path: Path | None = None) -> dict:
    questions = json.loads((root / "questions/dev_questions_with_answers.json").read_text())
    answer_run = json.loads((answer_run_path or root / "evaluation/dev_run.json").read_text())
    records = {row["id"]: row for row in answer_run["questions"]}
    rows = []
    if progress and questions:
        progress(0, len(questions), questions[0]["id"])
    for index, question in enumerate(questions, 1):
        record = records.get(question["id"], {})
        result = grade(question, record, judge)
        rows.append({"id": question["id"], "question": question["question"],
                     "expected_answer": question["gold_answer"], "answer": record.get("answer"), **result})
        if progress:
            next_id = questions[index]["id"] if index < len(questions) else None
            progress(index, len(questions), next_id)
    counts = _counts(rows)
    result = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "answer_run_id": answer_run.get("run_id"), "counts": counts, "rows": rows}
    _save(root, result, result_path)
    return result


def _counts(rows: list[dict]) -> dict[str, int]:
    return {status: sum(row["status"] == status for row in rows)
            for status in ("correct", "partial", "incorrect", "unavailable")}


def _save(root: Path, result: dict, result_path: Path | None = None) -> None:
    destination = result_path or root / "evaluation/benchmark_run.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(destination)


def retry_unavailable(question_id: str, root: Path = ROOT, judge=judge_record) -> dict:
    """Retry only one unavailable saved benchmark judgment."""
    path = root / "evaluation/benchmark_run.json"
    result = json.loads(path.read_text())
    answer_run = json.loads((root / "evaluation/dev_run.json").read_text())
    if result.get("answer_run_id") != answer_run.get("run_id"):
        raise ValueError("Benchmark scores belong to another answer run")
    row = next((item for item in result["rows"] if item["id"] == question_id), None)
    question = next((item for item in json.loads((root / "questions/dev_questions_with_answers.json").read_text())
                     if item["id"] == question_id), None)
    record = next((item for item in answer_run["questions"] if item["id"] == question_id), None)
    if row is None or question is None or record is None or row["status"] != "unavailable":
        raise ValueError("Choose an unavailable question in the current benchmark run")
    if not record.get("answer") or record.get("error"):
        raise ValueError("This question has no saved answer to regrade")
    verdict = grade(question, record, judge)
    row.update(verdict)
    result["counts"] = _counts(result["rows"])
    result["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _save(root, result)
    return verdict
