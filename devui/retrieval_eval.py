"""Score cited and agent-exposed evidence from saved answers with official RAGAS.

Cited sources follow first-citation order. All exposed sources follow tool-call
order, which is not a single global search ranking when the agent searches again.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from eval_pipeline.generation import RAGAS_JUDGE_MODEL, _sql_context
from eval_pipeline.retrieval import build_metrics, score_methods

ROOT = Path(__file__).resolve().parents[1]
CITATION = re.compile(r"\[S\d+\]")
PILOT_IDS = {"q_006", "q_012"}


def cases(root: Path = ROOT) -> list[dict]:
    questions = json.loads((root / "questions/dev_questions_with_answers.json").read_text())
    return [{"question_id": row["id"], "question": row["question"],
             "reference": row["gold_answer"], "modalities": row["required_modalities"]}
            for row in questions]


def cited_hits(record: dict) -> list[dict]:
    """Preserve first citation order, with one context per unique cited source."""
    sources = {f"[{source['id']}]": source for source in record.get("sources", [])}
    hits = []
    for citation in dict.fromkeys(CITATION.findall(record.get("answer") or "")):
        source = sources.get(citation)
        if source is None:
            continue
        kind = source.get("type") or source.get("kind")
        if kind == "sql":
            rows = source.get("rows", [])
            text = (f"Financial database query: {source.get('sql', '')}\nRows: "
                    + _sql_context(json.dumps(rows, ensure_ascii=False)))
            hit_id = source["id"]
        elif kind == "pdf":
            text = str(source.get("text") or "")
            hit_id = source.get("chunk_id") or source["id"]
        else:
            continue
        if not text.strip():
            continue
        hits.append({"id": str(hit_id), "source_id": source["id"], "type": kind,
                     "filename": source.get("filename"), "pages": source.get("pages") or [],
                     "text": text})
    return hits


def retrieved_hits(record: dict) -> list[dict]:
    """Return evidence in the order the agent's tool calls exposed it.

    This includes uncited passages. Multiple searches do not form one global
    ranking, so the resulting order is labeled as agent exposure order.
    """
    sources = {source.get("id"): source for source in record.get("sources", [])}
    ordered_ids = dict.fromkeys(
        source_id
        for call in record.get("trace", [])
        if call.get("tool") in {"query_financials", "search_filings", "read_pages"}
        and "error" not in call
        for source_id in call.get("source_ids", [])
    )
    return _hits_for_sources(sources, ordered_ids)


def _hits_for_sources(sources: dict, ordered_ids) -> list[dict]:
    hits = []
    for source_id in ordered_ids:
        source = sources.get(source_id)
        if not isinstance(source, dict):
            continue
        kind = source.get("type") or source.get("kind")
        if kind == "sql":
            text = (f"Financial database query: {source.get('sql', '')}\nRows: "
                    + _sql_context(json.dumps(source.get("rows", []), ensure_ascii=False)))
            hit_id = source_id
        elif kind == "pdf":
            text = str(source.get("text") or "")
            hit_id = source.get("chunk_id") or source_id
        else:
            continue
        if text.strip():
            hits.append({"id": str(hit_id), "source_id": source_id, "type": kind,
                         "filename": source.get("filename"), "pages": source.get("pages") or [],
                         "text": text})
    return hits


def latest_result(root: Path = ROOT, answer_run_id: str | None = None) -> dict | None:
    """Restore the latest result for the current saved answers and scoring scope."""
    folder = root / "evaluation/retrieval_runs"
    for path in sorted(folder.glob("retrieval-*.json"), reverse=True):
        try:
            result = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if (result.get("ranking_scope") == "saved_cited_and_agent_exposure_v2"
                and result.get("answer_run_id") == answer_run_id):
            return result
    return None


def _summary(rows: list[dict]) -> dict:
    summary = {}
    for method in ("cited", "retrieved"):
        scored = {name: [row["metrics"][method][name] for row in rows
                         if method in row["metrics"] and row["metrics"][method][name] is not None]
                  for name in ("context_precision", "context_recall")}
        if method == "cited" or any(scored.values()):
            summary[method] = {
                "precision_scored": len(scored["context_precision"]),
                "recall_scored": len(scored["context_recall"]),
                **{name: round(sum(values) / len(values), 3) if values else None
                   for name, values in scored.items()}}
    return summary


def evaluate(*, root: Path = ROOT, progress=None, metrics=None,
             question_count: int = 10) -> dict:
    if question_count not in {2, 10}:
        raise ValueError("Choose the two-question pilot or all ten questions")
    answer_run = json.loads((root / "evaluation/dev_run.json").read_text())
    records = {row["id"]: row for row in answer_run["questions"]}
    selected = [case for case in cases(root)
                if question_count == 10 or case["question_id"] in PILOT_IDS]
    if len(selected) != question_count:
        raise ValueError("The development question file does not contain the selected questions")
    scorer = metrics if metrics is not None else build_metrics()
    rows = []
    for number, case in enumerate(selected, 1):
        if progress:
            progress(number - 1, len(selected), case["question_id"])
        record = records.get(case["question_id"])
        if record is None or not record.get("answer"):
            raise ValueError(f"No saved answer for {case['question_id']}")
        hits = cited_hits(record)
        exposed = retrieved_hits(record)
        methods = {"cited": hits}
        if exposed:
            methods["retrieved"] = exposed
        scored = score_methods(case["question"], case["reference"],
                               methods, scorer, question_id=case["question_id"])
        available = sum(value[name] is not None for value in scored.values()
                        for name in ("context_precision", "context_recall"))
        total = 2 * len(scored)
        rows.append({"case": {key: value for key, value in case.items() if key != "reference"},
                     "judge": {"status": "judged" if available == total else "partial" if available else "unavailable",
                               "model": RAGAS_JUDGE_MODEL},
                     "metrics": scored, "hits": methods})
    if progress:
        progress(len(selected), len(selected), None)
    run_id = "retrieval-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]
    result = {"run_id": run_id, "answer_run_id": answer_run.get("run_id"),
              "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "question_count": len(rows), "mode": "pilot" if question_count == 2 else "full",
              "summary": _summary(rows), "rows": rows,
              "metric_implementation": "ragas.metrics.collections.ContextPrecision/ContextRecall",
              "ranking_scope": "saved_cited_and_agent_exposure_v2",
              "scope": "Cited evidence is ordered by first citation. Agent-retrieved evidence includes uncited SQL/PDF sources in tool-call exposure order; multiple searches do not form one global ranking. Both use official RAGAS LLM estimates."}
    destination = root / "evaluation/retrieval_runs" / f"{run_id}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(destination)
    return result


def retry_unavailable(question_id: str, *, root: Path = ROOT, metrics=None) -> dict:
    """Rescore only missing metrics for one question in the current saved run."""
    answer_run_id = json.loads((root / "evaluation/dev_run.json").read_text()).get("run_id")
    result = latest_result(root, answer_run_id)
    if result is None:
        raise ValueError("Run retrieval evaluation for the current saved answers first")
    row = next((item for item in result["rows"]
                if item["case"]["question_id"] == question_id), None)
    case = next((item for item in cases(root) if item["question_id"] == question_id), None)
    if row is None or case is None:
        raise ValueError("Choose a question in the current retrieval run")
    missing_by_method = {method: [name for name in ("context_precision", "context_recall")
                                  if scored[name] is None]
                         for method, scored in row["metrics"].items()}
    missing = {method: names for method, names in missing_by_method.items() if names}
    if not missing:
        raise ValueError("This question has no unavailable retrieval metric")
    for method, names in missing.items():
        current = row["metrics"][method]
        rescored = score_methods(case["question"], case["reference"],
                                 {method: row["hits"][method]}, metrics,
                                 metric_names=names, question_id=question_id)[method]
        for name in names:
            current[name] = rescored[name]
            detail = "passage_judgments" if name == "context_precision" else "recall_claims"
            current[detail] = rescored[detail]
            if name == "context_recall":
                current["claim_chunk_link_error"] = rescored["claim_chunk_link_error"]
            current["errors"].pop(name, None)
            if name in rescored["errors"]:
                current["errors"][name] = rescored["errors"][name]
    available = sum(value[name] is not None for value in row["metrics"].values()
                    for name in ("context_precision", "context_recall"))
    total = 2 * len(row["metrics"])
    row["judge"]["status"] = "judged" if available == total else "partial" if available else "unavailable"
    result["summary"] = _summary(result["rows"])
    result["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    destination = root / "evaluation/retrieval_runs" / f"{result['run_id']}.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(destination)
    return result
