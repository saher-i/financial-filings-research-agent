"""Run official RAGAS metrics on saved answers without rerunning the answer agent.

The supplied answer key is never sent to RAGAS.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from src.answer_agent.verification import _source_context
from src.answer_agent.models import QUALITY_CANDIDATE_MODEL
from src.ingestion_pipeline.embedding.client import DEFAULT_MODEL as NOMIC_EMBEDDING_MODEL

RAGAS_JUDGE_MODEL = QUALITY_CANDIDATE_MODEL
RAGAS_EMBEDDING_MODEL = NOMIC_EMBEDDING_MODEL
PILOT_IDS = ("q_001", "q_012")
RAGAS_RESULT_VERSION = "ragas-v5"
LOGGER = logging.getLogger(__name__)


def build_metrics():
    """Construct official RAGAS scorers with OpenAI-compatible Fireworks clients."""
    try:
        from openai import AsyncOpenAI
        from ragas.embeddings.openai_provider import OpenAIEmbeddings
        from ragas.llms import llm_factory
        from ragas.metrics.collections import AnswerRelevancy, Faithfulness
    except ImportError as exc:
        raise RuntimeError("Install the RAGAS dependency before running generation evaluation") from exc

    key = os.getenv("FIREWORKS_API_KEY")
    if not key:
        raise RuntimeError("Set FIREWORKS_API_KEY before running generation evaluation")
    endpoint = "https://api.fireworks.ai/inference/v1"
    llm_client = AsyncOpenAI(api_key=key, base_url=endpoint, timeout=60.0, max_retries=1)
    embedding_client = AsyncOpenAI(api_key=key, base_url=endpoint, timeout=60.0, max_retries=1)
    llm = llm_factory(RAGAS_JUDGE_MODEL, client=llm_client, max_tokens=16384)

    class TracedFaithfulness(Faithfulness):
        """Expose the official metric's statements and NLI verdicts for inspection."""

        async def ascore(self, **kwargs):
            self.claim_trace = []
            return await super().ascore(**kwargs)

        async def _create_verdicts(self, statements, context):
            verdicts = await super()._create_verdicts(statements, context)
            self.claim_trace = [item.model_dump() for item in verdicts.statements]
            return verdicts

    class NomicQueryEmbeddings(OpenAIEmbeddings):
        """Apply Nomic's query prefix to both sides of RAGAS question similarity."""

        async def aembed_text(self, text, **kwargs):
            vector = await super().aembed_text(f"search_query: {text}", **kwargs)
            self.original_question = text
            self.original_vector = vector
            return vector

        async def aembed_texts(self, texts, **kwargs):
            questions = list(texts)
            vectors = await super().aembed_texts([f"search_query: {text}" for text in questions], **kwargs)
            self.generated_questions = questions
            self.generated_vectors = vectors
            return vectors

    embeddings = NomicQueryEmbeddings(client=embedding_client, model=RAGAS_EMBEDDING_MODEL)
    return TracedFaithfulness(llm=llm), AnswerRelevancy(llm=llm, embeddings=embeddings)


def _relevancy_trace(embeddings) -> list[dict]:
    """Show the question similarities already used by official RAGAS."""
    original = getattr(embeddings, "original_vector", None)
    questions = getattr(embeddings, "generated_questions", None)
    vectors = getattr(embeddings, "generated_vectors", None)
    if (original is None or not questions or vectors is None or len(questions) != len(vectors)
            or any(len(vector) != len(original) for vector in vectors)):
        return []
    norm = math.sqrt(sum(float(value) ** 2 for value in original))
    trace = []
    for question, vector in zip(questions, vectors):
        denominator = norm * math.sqrt(sum(float(value) ** 2 for value in vector))
        similarity = sum(float(left) * float(right) for left, right in zip(original, vector)) / denominator if denominator else None
        trace.append({"question": question, "similarity": round(similarity, 3) if similarity is not None else None})
    return trace


def _sql_context(excerpt: str) -> str:
    """Explain SQL revenue units without changing or dropping the original rows."""
    try:
        rows = json.loads(excerpt)
    except (TypeError, ValueError):
        return excerpt
    if not isinstance(rows, list):
        return excerpt
    descriptions = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        revenue = row.get("revenue")
        if isinstance(revenue, (int, float)) and not isinstance(revenue, bool):
            billions = Decimal(str(revenue)) / Decimal(1_000_000_000)
            labels = [f"{key}={row[key]}" for key in
                      ("company_ticker", "fiscal_year", "period_end", "region", "segment")
                      if row.get(key) is not None]
            descriptions.append(f"{', '.join(labels)}: revenue ${billions:f} billion USD")
    return excerpt + ("\nReadable revenue values (derived from the SQL rows above):\n" +
                      "\n".join(descriptions) if descriptions else "")


def _contexts(record: dict) -> list[str]:
    """Give RAGAS the gathered SQL/PDF evidence as its retrieved contexts."""
    sources = record.get("sources") or []
    if not isinstance(sources, list):
        sources = []
    contexts = []
    for item in _source_context(sources, str(record.get("answer") or "")):
        excerpt = _sql_context(item["excerpt"]) if item["location"] == "financial database" else item["excerpt"]
        contexts.append(f"{item['location']}\n{excerpt}")
    return contexts


def _error(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}".strip()
    key = os.getenv("FIREWORKS_API_KEY")
    return message.replace(key, "[redacted]")[:400] if key else message[:400]


async def score_answer(record: dict, faithfulness, response_relevancy,
                       only: str | None = None) -> dict:
    """Keep metric failures independent; never substitute a custom score."""
    if only not in (None, "faithfulness", "response_relevancy"):
        raise ValueError("Unknown generation metric")
    if not record.get("answer"):
        return {"faithfulness": None, "response_relevancy": None,
                "errors": {"answer": "No generated answer"}}
    question = record["question"]
    answer = record["answer"]
    result = {"faithfulness": None, "response_relevancy": None, "errors": {}}
    contexts = _contexts(record) if only in (None, "faithfulness") else []
    if only in (None, "faithfulness") and contexts:
        started = time.monotonic()
        try:
            score = await faithfulness.ascore(user_input=question, response=answer,
                                             retrieved_contexts=contexts)
            result["faithfulness"] = round(float(score.value), 3)
            if getattr(faithfulness, "claim_trace", None):
                result["faithfulness_claims"] = faithfulness.claim_trace
        except Exception as exc:
            result["errors"]["faithfulness"] = _error(exc)
            LOGGER.warning("generation_metric_failed question_id=%s metric=faithfulness model=%s elapsed_ms=%d error=%s detail=%s",
                           record.get("id"), RAGAS_JUDGE_MODEL, round((time.monotonic()-started)*1000),
                           type(exc).__name__, _error(exc))
        else:
            LOGGER.info("generation_metric_ok question_id=%s metric=faithfulness model=%s elapsed_ms=%d score=%s",
                        record.get("id"), RAGAS_JUDGE_MODEL, round((time.monotonic()-started)*1000), result["faithfulness"])
    elif only in (None, "faithfulness"):
        result["errors"]["faithfulness"] = "No retrieved contexts"
    if only in (None, "response_relevancy"):
        started = time.monotonic()
        try:
            embeddings = getattr(response_relevancy, "embeddings", None)
            if embeddings is not None:
                for key in ("original_vector", "generated_questions", "generated_vectors"):
                    if hasattr(embeddings, key):
                        delattr(embeddings, key)
            score = await response_relevancy.ascore(user_input=question, response=answer)
            result["response_relevancy"] = round(float(score.value), 3)
            trace = _relevancy_trace(embeddings)
            if trace:
                result["response_relevancy_questions"] = trace
        except Exception as exc:
            result["errors"]["response_relevancy"] = _error(exc)
            LOGGER.warning("generation_metric_failed question_id=%s metric=response_relevancy model=%s elapsed_ms=%d error=%s detail=%s",
                           record.get("id"), RAGAS_JUDGE_MODEL, round((time.monotonic()-started)*1000),
                           type(exc).__name__, _error(exc))
        else:
            LOGGER.info("generation_metric_ok question_id=%s metric=response_relevancy model=%s elapsed_ms=%d score=%s",
                        record.get("id"), RAGAS_JUDGE_MODEL, round((time.monotonic()-started)*1000), result["response_relevancy"])
    return result


async def _evaluate(records: list[dict], faithfulness, response_relevancy, progress) -> list[dict]:
    scored = []
    for number, record in enumerate(records, 1):
        scores = await score_answer(record, faithfulness, response_relevancy)
        scored.append({"id": record["id"], "question": record.get("question"),
                       "answer": record.get("answer"), "scores": scores})
        if progress:
            progress(number, len(records), record["id"])
    return scored


def evaluate_saved_answers(root: Path, progress=None, metrics=None, *, pilot: bool = True) -> dict:
    source = root / "evaluation/dev_run.json"
    if not source.is_file():
        raise FileNotFoundError("Run Answer Evaluation before scoring generation")
    run = json.loads(source.read_text())
    records = run.get("questions", [])
    if pilot:
        records = [record for record in records if record.get("id") in PILOT_IDS]
        if len(records) != len(PILOT_IDS):
            raise ValueError("The saved answer run does not contain both pilot questions")
    if not records:
        raise ValueError("The saved answer run contains no questions")
    faithfulness, response_relevancy = metrics or build_metrics()
    scored = asyncio.run(_evaluate(records, faithfulness, response_relevancy, progress))
    summary = {}
    for name in ("faithfulness", "response_relevancy"):
        values = [row["scores"][name] for row in scored if row["scores"][name] is not None]
        summary[name] = {"mean": round(sum(values) / len(values), 3) if values else None,
                         "scored": len(values)}
    result = {"source_run_id": run.get("run_id"),
              "mode": "pilot" if pilot else "full",
              "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "metric_implementation": "ragas.metrics.collections",
              "judge_model": RAGAS_JUDGE_MODEL, "embedding_model": RAGAS_EMBEDDING_MODEL,
              "total": len(scored), "summary": summary, "rows": scored}
    destination = _result_path(root, run.get("run_id", "saved-run"), pilot)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(destination)
    return result


def _result_path(root: Path, run_id: str, pilot: bool) -> Path:
    mode = "pilot" if pilot else "full"
    return root / "evaluation/generation_runs" / f"{run_id}-{RAGAS_RESULT_VERSION}-{mode}.json"


def retry_saved_metric(root: Path, question_id: str, metric: str, metrics=None) -> dict:
    """Recheck one failed score in the current saved run, preserving all others."""
    if metric not in {"faithfulness", "response_relevancy"}:
        raise ValueError("Unknown generation metric")
    source = root / "evaluation/dev_run.json"
    if not source.is_file():
        raise FileNotFoundError("Run Answer Evaluation first")
    run = json.loads(source.read_text())
    record = next((item for item in run.get("questions", []) if item.get("id") == question_id), None)
    if record is None:
        raise ValueError("Question not found in saved answers")
    candidates = [_result_path(root, run.get("run_id", "saved-run"), pilot) for pilot in (True, False)]
    existing = [path for path in candidates if path.is_file() and
                any(row.get("id") == question_id for row in json.loads(path.read_text()).get("rows", []))]
    if not existing:
        raise FileNotFoundError("Score this question before retrying a metric")
    destination = max(existing, key=lambda path: path.stat().st_mtime_ns)
    result = json.loads(destination.read_text())
    row = next(row for row in result["rows"] if row["id"] == question_id)
    if row["scores"].get(metric) is not None or not row["scores"].get("errors", {}).get(metric):
        raise ValueError("Only a failed metric can be retried")
    faithfulness, relevancy = metrics or build_metrics()
    LOGGER.info("generation_metric_retry_started question_id=%s metric=%s model=%s",
                question_id, metric, RAGAS_JUDGE_MODEL)
    attempt = asyncio.run(score_answer(record, faithfulness, relevancy, only=metric))
    row["scores"][metric] = attempt[metric]
    row["scores"].get("errors", {}).pop(metric, None)
    if metric in attempt["errors"]:
        row["scores"].setdefault("errors", {})[metric] = attempt["errors"][metric]
    for key in ("faithfulness_claims", "response_relevancy_questions"):
        if key in attempt:
            row["scores"][key] = attempt[key]
    values = [item["scores"][metric] for item in result["rows"]
              if item["scores"].get(metric) is not None]
    result["summary"][metric] = {"mean": round(sum(values)/len(values), 3) if values else None,
                                 "scored": len(values)}
    result["evaluated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(destination)
    return {"question_id": question_id, "metric": metric, "score": attempt[metric],
            "error": attempt["errors"].get(metric)}


def load_pilot(root: Path) -> dict | None:
    path = root / "evaluation/dev_run.json"
    if not path.is_file():
        return None
    run_id = json.loads(path.read_text()).get("run_id", "saved-run")
    destination = _result_path(root, run_id, True)
    return json.loads(destination.read_text()) if destination.is_file() else None


def pilot_ready(root: Path) -> bool:
    result = load_pilot(root)
    return bool(result and result.get("total") == len(PILOT_IDS)
                and all(result["summary"][name]["scored"] == len(PILOT_IDS)
                        for name in ("faithfulness", "response_relevancy")))


def load_latest(root: Path) -> dict | None:
    path = root / "evaluation/dev_run.json"
    if not path.is_file():
        return None
    run_id = json.loads(path.read_text()).get("run_id", "saved-run")
    destinations = [_result_path(root, run_id, pilot) for pilot in (True, False)]
    existing = [path for path in destinations if path.is_file()]
    return json.loads(max(existing, key=lambda path: path.stat().st_mtime_ns).read_text()) if existing else None


def saved_cited_source(root: Path, question_id: str, source_id: str) -> dict | None:
    """Resolve only a cited source from the current saved answer run."""
    if not isinstance(question_id, str) or not isinstance(source_id, str):
        return None
    path = root / "evaluation/dev_run.json"
    if not path.is_file():
        return None
    run = json.loads(path.read_text())
    record = next((row for row in run.get("questions", []) if row.get("id") == question_id), None)
    if not record or f"[{source_id}]" not in str(record.get("answer") or ""):
        return None
    return next((source for source in record.get("sources", []) if source.get("id") == source_id), None)
