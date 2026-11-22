"""Serve the customer chat and developer workspace locally.

Run from the repository root:
    APP_MODE=developer .venv/bin/python -m devui.server
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit
from uuid import uuid4

from src.config import load_local_fireworks_key
from src.diagnostics.logging import configure_logging
from devui.ingestion_views import (
    STRATEGIES, VARIANTS_DIR, browse_chunks, chunk_path, chunk_state,
    cited_page_highlights, document_page, filing_path,
    variant_catalog, variant_id, variant_path,
)

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent / "static"
QDRANT_STORE = ROOT / "data" / "index" / "qdrant"
JOB_LOCK = threading.Lock()
JOB = {"state": "idle", "stage": None, "strategy": None, "message": ""}
EVAL_JOB = {"state": "idle", "message": ""}
EVAL_RESULT: dict | None = None
ANSWER_RUN_JOB = {"state": "idle", "message": ""}
GENERATION_EVAL_JOB = {"state": "idle", "message": ""}
BENCHMARK_EVAL_JOB = {"state": "idle", "message": ""}
CHAT_LOCK = threading.Lock()
MODEL_CATALOG: list[dict] = []
MODEL_LOCK = threading.Lock()


def developer_enabled() -> bool:
    return os.getenv("APP_MODE", "customer").lower() == "developer"


def run_chat(question: str, progress=None, answer_style: str = "concise",
             model: str | None = None, history: list[dict] | None = None,
             verify: bool = True) -> dict:
    """Keep the HTTP contract separate from the answer agent implementation."""
    from src.answer_agent.graph import answer_question

    return answer_question(question, progress=progress, answer_style=answer_style,
                           model=model, history=history, verify=verify)


def evaluation_model(requested: object) -> str:
    """Allow the configured default or a discovered tool-capable model."""
    from src.answer_agent.models import selectable_models, validate_model
    from src.answer_agent.provider import DEFAULT_MODEL

    default = os.getenv("FIREWORKS_CHAT_MODEL", DEFAULT_MODEL)
    selected = validate_model(requested if requested is not None else default)
    with MODEL_LOCK:
        allowed = {entry["id"] for entry in selectable_models(default, MODEL_CATALOG)}
    if selected not in allowed:
        raise ValueError("Refresh the Fireworks model list and choose a listed tool-capable model")
    return selected


def run_benchmark_answers(model: str | None = None) -> None:
    """Run all included evaluation questions while holding the chat/Qdrant owner."""
    global ANSWER_RUN_JOB
    from eval_pipeline.answers import run_development

    def progress(done: int, total: int, question_id: str) -> None:
        global ANSWER_RUN_JOB
        with JOB_LOCK:
            ANSWER_RUN_JOB = {"state": "running", "done": done, "total": total,
                               "message": f"Finished {question_id} ({done}/{total})"}

    try:
        result = run_development(lambda question: run_chat(question, model=model, verify=True),
                                 progress=progress, model=model, preserve_previous=True)
        with JOB_LOCK:
            ANSWER_RUN_JOB = {"state": "complete", "done": result["total"],
                               "total": result["total"],
                               "message": f"Saved {result['completed']} of {result['total']} generated answers"}
    except Exception:
        logging.getLogger(__name__).exception("Benchmark answer generation failed")
        with JOB_LOCK:
            ANSWER_RUN_JOB = {"state": "error", "message": "Evaluation stopped; inspect the server log"}
    finally:
        CHAT_LOCK.release()


def run_benchmark_answer_retry(question_id: str, model: str) -> None:
    """Regenerate one answer without rerunning or grading the other nine."""
    global ANSWER_RUN_JOB
    from eval_pipeline.answers import retry_answer

    try:
        outcome = retry_answer(question_id,
                               lambda question: run_chat(question, model=model, verify=True),
                               model=model)
        record = outcome["record"]
        with JOB_LOCK:
            ANSWER_RUN_JOB = {
                "state": "complete" if outcome["updated"] else "error",
                "done": 1, "total": 1,
                "message": (f"Saved new answer for {question_id} ({record.get('answer_status', 'answered')})."
                            if outcome["updated"] else
                            f"Could not retry {question_id}: {record.get('error', 'unknown error')}")}
    except Exception:
        logging.getLogger(__name__).exception("Benchmark answer retry failed question_id=%s", question_id)
        with JOB_LOCK:
            ANSWER_RUN_JOB = {"state": "error", "message": f"Could not retry {question_id}; inspect server logs"}
    finally:
        CHAT_LOCK.release()


def run_retrieval_eval(question_count: int = 10) -> None:
    global EVAL_JOB, EVAL_RESULT
    from devui.retrieval_eval import evaluate

    def progress(done: int, total: int, question_id: str | None) -> None:
        global EVAL_JOB
        with JOB_LOCK:
            EVAL_JOB = {"state": "running",
                        "done": done, "total": total,
                        "message": f"Scoring {question_id}…" if question_id else "Finishing results…"}

    try:
        result = evaluate(root=ROOT, progress=progress, question_count=question_count)
        with JOB_LOCK:
            EVAL_RESULT = result
            EVAL_JOB = {"state": "complete",
                        "done": result["question_count"], "total": result["question_count"],
                        "message": "Evaluation complete"}
    except Exception as exc:
        logging.getLogger(__name__).exception("Retrieval evaluation failed")
        with JOB_LOCK:
            EVAL_JOB = {"state": "error", "done": 0, "total": question_count,
                        "message": f"Evaluation stopped: {exc}"}


def run_retrieval_retry(question_id: str) -> None:
    global EVAL_JOB, EVAL_RESULT
    from devui.retrieval_eval import retry_unavailable

    try:
        result = retry_unavailable(question_id, root=ROOT)
        with JOB_LOCK:
            EVAL_RESULT = result
            EVAL_JOB = {"state": "complete", "done": 1, "total": 1,
                        "message": f"Finished retrying {question_id}"}
    except Exception:
        logging.getLogger(__name__).exception("Retrieval retry failed for %s", question_id)
        with JOB_LOCK:
            EVAL_JOB = {"state": "error", "message": f"Could not retry {question_id}; inspect server logs"}


def run_benchmark_eval() -> None:
    global BENCHMARK_EVAL_JOB
    from eval_pipeline.benchmark import evaluate

    def progress(done: int, total: int, question_id: str | None) -> None:
        global BENCHMARK_EVAL_JOB
        with JOB_LOCK:
            BENCHMARK_EVAL_JOB = {"state": "running", "done": done, "total": total,
                                  "message": f"Grading {question_id}…" if question_id else "Saving results…"}

    try:
        result = evaluate(progress=progress)
        with JOB_LOCK:
            BENCHMARK_EVAL_JOB = {"state": "complete", "done": len(result["rows"]),
                                  "total": len(result["rows"]), "message": "Benchmark grading complete"}
    except Exception:
        logging.getLogger(__name__).exception("Benchmark evaluation failed")
        with JOB_LOCK:
            BENCHMARK_EVAL_JOB = {"state": "error", "message": "Benchmark grading failed; inspect server logs"}


def run_benchmark_retry(question_id: str) -> None:
    global BENCHMARK_EVAL_JOB
    from eval_pipeline.benchmark import retry_unavailable

    try:
        verdict = retry_unavailable(question_id)
        with JOB_LOCK:
            BENCHMARK_EVAL_JOB = {"state": "complete", "done": 1, "total": 1,
                                  "message": f"Retried {question_id}: {verdict['status']}"}
    except Exception:
        logging.getLogger(__name__).exception("Benchmark retry failed for %s", question_id)
        with JOB_LOCK:
            BENCHMARK_EVAL_JOB = {"state": "error", "message": f"Could not retry {question_id}; inspect server logs"}


def run_generation_eval(pilot: bool = True) -> None:
    global GENERATION_EVAL_JOB
    from eval_pipeline.generation import evaluate_saved_answers

    def progress(done: int, total: int, question_id: str) -> None:
        with JOB_LOCK:
            GENERATION_EVAL_JOB.update(done=done, total=total,
                                       message=f"Scored {question_id}")

    try:
        result = evaluate_saved_answers(ROOT, progress=progress, pilot=pilot)
        faith_scored = result["summary"]["faithfulness"]["scored"]
        relevance_scored = result["summary"]["response_relevancy"]["scored"]
        with JOB_LOCK:
            GENERATION_EVAL_JOB.update(state="complete", done=result["total"],
                                       total=result["total"],
                                       message=f"{result['mode'].title()}: Faithfulness {faith_scored}/{result['total']}, Response Relevancy {relevance_scored}/{result['total']} scored. Inspect any errors below.")
    except Exception:
        logging.getLogger(__name__).exception("Generation evaluation failed")
        with JOB_LOCK:
            GENERATION_EVAL_JOB.update(state="error", message="Evaluation failed; inspect server log")


def run_generation_metric_retry(question_id: str, metric: str) -> None:
    """Retry only one unavailable RAGAS metric without rerunning answers."""
    from eval_pipeline.generation import retry_saved_metric

    try:
        outcome = retry_saved_metric(ROOT, question_id, metric)
        with JOB_LOCK:
            GENERATION_EVAL_JOB.update(
                state="complete", done=1, total=1,
                message=(f"Retried {metric} for {question_id}: "
                         f"{'scored' if outcome['score'] is not None else 'still unavailable; inspect the error below'}"))
    except Exception:
        logging.getLogger(__name__).exception("Generation metric retry failed question_id=%s metric=%s",
                                              question_id, metric)
        with JOB_LOCK:
            GENERATION_EVAL_JOB.update(state="error", message="Metric retry failed; inspect server log")


def embedding_path(strategy: str) -> Path:
    if strategy not in STRATEGIES:
        raise ValueError("Unknown chunking strategy")
    return ROOT / "data" / "index" / f"{strategy}_embeddings.db"


def vector_state(strategy: str) -> dict:
    """Report whether embeddings and the active Qdrant index match current chunks."""
    from src.ingestion_pipeline.embedding.build import file_hash

    chunks = ROOT / "data" / "index" / f"{strategy}_chunks.jsonl"
    source_hash = file_hash(chunks) if chunks.is_file() else None
    embedding = {"ready": False, "count": 0}
    path = embedding_path(strategy)
    if path.is_file():
        try:
            with sqlite3.connect(path) as db:
                metadata = dict(db.execute("SELECT key, value FROM metadata"))
                count = db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            embedding = {"ready": source_hash is not None and metadata.get("source_hash") == source_hash
                         and count == int(metadata["count"]), "count": count,
                         "dimension": int(metadata["dimension"]), "model": metadata["model"]}
        except (sqlite3.DatabaseError, KeyError, ValueError):
            embedding = {"ready": False, "count": 0}
    index = {"ready": False, "count": 0}
    manifest_path = QDRANT_STORE / f"active_{strategy}.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
            index = {"ready": embedding["ready"] and manifest["source_hash"] == source_hash
                     and manifest["model"] == embedding["model"]
                     and manifest["count"] == embedding["count"],
                     "count": manifest["count"], "dimension": manifest["dimension"],
                     "collection": manifest["collection"]}
            if (index["ready"] and JOB["state"] != "running"
                    and EVAL_JOB["state"] != "running" and not CHAT_LOCK.locked()):
                from qdrant_client import QdrantClient

                client = QdrantClient(path=str(QDRANT_STORE))
                try:
                    index["ready"] = (client.collection_exists(manifest["collection"])
                                      and client.count(manifest["collection"], exact=True).count == manifest["count"])
                finally:
                    client.close()
        except (ValueError, KeyError, ImportError, OSError, RuntimeError):
            index = {"ready": False, "count": 0}
    return {"embedding": embedding, "index": index}


def embedding_preview(strategy: str, offset: int = 0, limit: int = 5) -> dict:
    import numpy as np

    if not vector_state(strategy)["embedding"]["ready"]:
        raise FileNotFoundError("Generate embeddings for the current chunks first")
    with sqlite3.connect(embedding_path(strategy)) as db:
        total = db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        rows = db.execute("SELECT chunk_id, vector FROM embeddings ORDER BY chunk_id LIMIT ? OFFSET ?",
                          (limit, offset)).fetchall()
    return {"total": total, "offset": offset, "items": [
        {"chunk_id": identifier, "dimension": len(vector) // 4,
         "values": np.frombuffer(vector, dtype="<f4").round(5).tolist()}
        for identifier, vector in rows]}


def qdrant_points(strategy: str, cursor: str | None = None, limit: int = 5) -> dict:
    from qdrant_client import QdrantClient

    info = vector_state(strategy)["index"]
    if not info["ready"]:
        raise FileNotFoundError("Build the Qdrant index for the current chunks first")
    client = QdrantClient(path=str(QDRANT_STORE))
    try:
        points, next_cursor = client.scroll(info["collection"], offset=cursor, limit=limit,
                                            with_payload=True, with_vectors=True)
    finally:
        client.close()
    return {"collection": info["collection"], "count": info["count"],
            "next_cursor": str(next_cursor) if next_cursor is not None else None,
            "points": [{"point_id": str(point.id), "chunk_id": point.payload["chunk_id"],
                        "ticker": point.payload["ticker"], "fiscal_year": point.payload["fiscal_year"],
                        "filename": point.payload["filename"], "pages": point.payload["pages"],
                        "text": point.payload["text"], "dimension": len(point.vector),
                        "values": [round(value, 5) for value in point.vector]}
                       for point in points]}


def build_chunk_set(strategy: str, identifier: str | None = None,
                target: int = 3600, maximum: int = 4800, overlap: int = 400) -> None:
    global JOB
    started = time.monotonic()
    try:
        command = [sys.executable, "-m", "src.ingestion_pipeline.chunking.main", "--strategy", strategy]
        if identifier:
            VARIANTS_DIR.mkdir(parents=True, exist_ok=True)
            command += ["--target", str(target), "--maximum", str(maximum), "--overlap", str(overlap),
                        "--output", str(variant_path(identifier))]
        process = subprocess.run(
            command,
            cwd=ROOT, capture_output=True, text=True, timeout=1200, check=False,
        )
        message = (process.stdout if process.returncode == 0 else process.stderr).strip()[-3000:]
        state = "complete" if process.returncode == 0 else "failed"
        if state == "complete" and identifier:
            metadata = {"strategy": strategy, "target": target, "maximum": maximum,
                        "overlap": overlap, "build_seconds": round(time.monotonic() - started, 2)}
            destination = VARIANTS_DIR / f"{identifier}.json"
            temporary = destination.with_suffix(".tmp")
            temporary.write_text(json.dumps(metadata))
            temporary.replace(destination)
    except (OSError, subprocess.TimeoutExpired) as exc:
        state, message = "failed", str(exc)
    with JOB_LOCK:
        JOB = {"state": state, "stage": "chunking", "strategy": strategy, "variant_id": identifier,
               "run_id": JOB.get("run_id"), "message": message,
               "build_seconds": round(time.monotonic() - started, 2)}


def run_vector_stage(stage: str, strategy: str, run_id: int) -> None:
    global JOB
    started = time.monotonic()
    command = [sys.executable, "-m", f"src.ingestion_pipeline.{stage}.main",
               "--strategy", strategy]
    if stage == "indexing":
        command.append("build")
    try:
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                 timeout=1800, check=False)
        state = "complete" if process.returncode == 0 else "failed"
        message = (process.stdout if state == "complete" else process.stderr).strip()[-3000:]
    except (OSError, subprocess.TimeoutExpired) as exc:
        state, message = "failed", str(exc)
    with JOB_LOCK:
        JOB = {"state": state, "stage": stage, "strategy": strategy, "run_id": run_id,
               "message": message, "build_seconds": round(time.monotonic() - started, 2)}


class Handler(BaseHTTPRequestHandler):
    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _event(self, name: str, payload: dict) -> None:
        body = f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()
        try:
            self.wfile.write(body)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _file(self, path: Path, content_type: str) -> None:
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _hidden(self) -> None:
        self._json(404, {"error": "Not found"})

    def do_GET(self) -> None:
        route = urlsplit(self.path)
        public_paths = {
            "/": ("chat.html", "text/html; charset=utf-8"),
            "/chat.css": ("chat.css", "text/css; charset=utf-8"),
            "/chat.js": ("chat.js", "text/javascript; charset=utf-8"),
        }
        if route.path in public_paths:
            name, kind = public_paths[route.path]
            return self._file(ASSETS / name, kind)
        if route.path == "/api/chat/models":
            from src.answer_agent.models import selectable_models
            from src.answer_agent.provider import DEFAULT_MODEL

            default = os.getenv("FIREWORKS_CHAT_MODEL", DEFAULT_MODEL)
            with MODEL_LOCK:
                models = selectable_models(default, MODEL_CATALOG)
            return self._json(200, {"models": models, "default": default,
                                    "developer_mode": developer_enabled()})
        if route.path == "/api/chat/history":
            from devui.observability import recent_chats

            return self._json(200, {"chats": recent_chats()})
        if route.path.startswith("/api/chat/conversations/"):
            from devui.observability import conversation_by_id

            identifier = route.path.removeprefix("/api/chat/conversations/")
            if not re.fullmatch(r"[0-9a-f]{12}", identifier):
                return self._hidden()
            conversation = conversation_by_id(identifier)
            return self._json(200, {"conversation": conversation}) if conversation else self._hidden()
        if route.path.startswith("/api/chat/history/"):
            from devui.observability import chat_by_id

            run_id = route.path.removeprefix("/api/chat/history/")
            if not re.fullmatch(r"[0-9a-f]{12}", run_id):
                return self._hidden()
            run = chat_by_id(run_id)
            return self._json(200, {"chat": run}) if run else self._hidden()
        if route.path.startswith("/filings/"):
            filename = unquote(route.path.removeprefix("/filings/"))
            path = filing_path(filename)
            return self._file(path, "application/pdf") if path and path.is_file() else self._hidden()
        if route.path == "/api/chat/document-highlights":
            params = parse_qs(route.query)
            try:
                result = cited_page_highlights(params.get("filename", [""])[0],
                                               int(params.get("page", ["0"])[0]),
                                               params.get("chunk_id", [""])[0])
                return self._json(200, result)
            except (ValueError, FileNotFoundError):
                return self._hidden()
        if route.path == "/api/chat/document-image":
            params = parse_qs(route.query)
            path = filing_path(params.get("filename", [""])[0])
            if path is None or not path.is_file():
                return self._hidden()
            try:
                page = int(params.get("page", ["0"])[0])
                with pymupdf.open(path) as pdf:
                    if not 1 <= page <= len(pdf):
                        return self._hidden()
                    image = pdf[page - 1].get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False).tobytes("png")
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(image)))
                self.end_headers()
                self.wfile.write(image)
                return
            except ValueError:
                return self._hidden()
        if not developer_enabled():
            return self._hidden()
        if route.path == "/dev/chunks":
            self.send_response(308)
            self.send_header("Location", "/ingestion-pipeline")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        paths = {
            "/ingestion-pipeline": ("ingestion-pipeline.html", "text/html; charset=utf-8"),
            "/retrieval-evaluation": ("retrieval-evaluation.html", "text/html; charset=utf-8"),
            "/generation-evaluation": ("generation-evaluation.html", "text/html; charset=utf-8"),
            "/benchmark-evaluation": ("benchmark-evaluation.html", "text/html; charset=utf-8"),
            "/agent-workflow": ("agent-workflow.html", "text/html; charset=utf-8"),
            "/dev/agent-workflow.css": ("agent-workflow.css", "text/css; charset=utf-8"),
            "/dev/agent-workflow.js": ("agent-workflow.js", "text/javascript; charset=utf-8"),
            "/dev/chunks.css": ("chunks.css", "text/css; charset=utf-8"),
            "/dev/chunks.js": ("chunks.js", "text/javascript; charset=utf-8"),
            "/dev/retrieval-evaluation.css": ("retrieval-evaluation.css", "text/css; charset=utf-8"),
            "/dev/generation-evaluation.css": ("generation-evaluation.css", "text/css; charset=utf-8"),
            "/dev/retrieval-evaluation.js": ("retrieval-evaluation.js", "text/javascript; charset=utf-8"),
            "/dev/generation-evaluation.js": ("generation-evaluation.js", "text/javascript; charset=utf-8"),
            "/dev/eval-progress.js": ("eval-progress.js", "text/javascript; charset=utf-8"),
            "/dev/benchmark-evaluation.js": ("benchmark-evaluation.js", "text/javascript; charset=utf-8"),
            "/dev/model-picker.js": ("model-picker.js", "text/javascript; charset=utf-8"),
            "/dev/dev-theme.css": ("dev-theme.css", "text/css; charset=utf-8"),
        }
        if route.path == "/api/dev/agent-workflow":
            from devui.observability import recent_runs

            return self._json(200, {"runs": recent_runs()})
        if route.path == "/api/dev/models":
            from src.answer_agent.models import selectable_models
            from src.answer_agent.provider import DEFAULT_MODEL

            default = os.getenv("FIREWORKS_CHAT_MODEL", DEFAULT_MODEL)
            with MODEL_LOCK:
                models = selectable_models(default, MODEL_CATALOG)
            return self._json(200, {"models": models,
                                    "default": default,
                                    "key_available": bool(os.getenv("FIREWORKS_API_KEY"))})
        if route.path == "/api/dev/benchmark-evaluation":
            path = ROOT / "evaluation/benchmark_run.json"
            answers_path = ROOT / "evaluation/dev_run.json"
            try:
                result = json.loads(path.read_text()) if path.is_file() else None
            except (OSError, ValueError):
                result = None
            try:
                answer_run = json.loads(answers_path.read_text()) if answers_path.is_file() else {}
                answer_run_id = answer_run.get("run_id")
                answer_rows = [{key: row.get(key) for key in
                                ("id", "question", "answer", "answer_status", "failure_reason",
                                 "error", "elapsed_seconds")}
                               for row in answer_run.get("questions", [])]
            except (OSError, ValueError):
                answer_run_id = None
                answer_rows = []
            with JOB_LOCK:
                job = BENCHMARK_EVAL_JOB.copy()
                answer_job = ANSWER_RUN_JOB.copy()
            return self._json(200, {"job": job, "answer_job": answer_job, "result": result,
                                    "answer_run_available": answer_run_id is not None,
                                    "answer_run_id": answer_run_id,
                                    "answer_rows": answer_rows,
                                    "key_available": bool(os.getenv("FIREWORKS_API_KEY"))})
        if route.path == "/api/dev/generation-evaluation/source":
            from eval_pipeline.generation import saved_cited_source

            parameters = parse_qs(route.query)
            question_id = parameters.get("question_id", [""])[0]
            source_id = parameters.get("source_id", [""])[0]
            if not re.fullmatch(r"q_\d{3}", question_id) or not re.fullmatch(r"S\d+", source_id):
                return self._hidden()
            source = saved_cited_source(ROOT, question_id, source_id)
            return self._json(200, {"source": source}) if source else self._hidden()
        if route.path == "/api/dev/generation-evaluation":
            from eval_pipeline.generation import load_latest, pilot_ready
            from importlib.util import find_spec

            with JOB_LOCK:
                job = GENERATION_EVAL_JOB.copy()
            return self._json(200, {"job": job, "result": load_latest(ROOT),
                                    "pilot_ready": pilot_ready(ROOT),
                                    "answer_run_available": (ROOT / "evaluation/dev_run.json").is_file(),
                                    "ragas_available": find_spec("ragas") is not None,
                                    "key_available": bool(os.getenv("FIREWORKS_API_KEY"))})
        if route.path == "/api/dev/retrieval-evaluation":
            from devui.retrieval_eval import cases, latest_result

            with JOB_LOCK:
                job, result = EVAL_JOB.copy(), EVAL_RESULT
            answer_path = ROOT / "evaluation/dev_run.json"
            answer_run_id = json.loads(answer_path.read_text()).get("run_id") if answer_path.is_file() else None
            if result is not None and result.get("answer_run_id") != answer_run_id:
                result = None
            if result is None and answer_run_id:
                result = latest_result(ROOT, answer_run_id)
            return self._json(200, {"job": job, "result": result, "cases": cases(),
                                    "answer_run_available": bool(answer_run_id),
                                    "judge_key_available": bool(os.getenv("FIREWORKS_API_KEY"))})
        if route.path == "/api/dev/status":
            with JOB_LOCK:
                job = JOB.copy()
            return self._json(200, {
                "chunk_sets": chunk_state(),
                "vectors": {strategy: vector_state(strategy) for strategy in STRATEGIES},
                "job": job,
                "embedding_key_available": bool(os.getenv("FIREWORKS_API_KEY")),
            })
        if route.path == "/api/dev/embeddings":
            params = parse_qs(route.query)
            try:
                strategy = params.get("strategy", ["structure"])[0]
                offset = int(params.get("offset", ["0"])[0])
                if strategy not in STRATEGIES or not 0 <= offset <= 100000:
                    raise ValueError("Invalid embedding selection")
                return self._json(200, embedding_preview(strategy, offset))
            except (ValueError, FileNotFoundError) as exc:
                return self._json(400, {"error": str(exc)})
        if route.path == "/api/dev/qdrant-points":
            params = parse_qs(route.query)
            try:
                strategy = params.get("strategy", ["structure"])[0]
                if strategy not in STRATEGIES:
                    raise ValueError("Invalid index selection")
                cursor = params.get("cursor", [None])[0]
                if cursor and not re.fullmatch(r"[0-9a-fA-F-]{36}", cursor):
                    raise ValueError("Invalid Qdrant cursor")
                return self._json(200, qdrant_points(strategy, cursor))
            except (ValueError, FileNotFoundError, ImportError) as exc:
                return self._json(400, {"error": str(exc)})
        if route.path == "/api/dev/variants":
            return self._json(200, {"variants": variant_catalog()})
        if route.path == "/api/dev/chunks":
            params = parse_qs(route.query)
            identifier = params.get("variant", params.get("strategy", ["structure"]))[0]
            try:
                page = int(params["page"][0]) if "page" in params and params["page"][0] else None
                offset = int(params.get("offset", ["0"])[0])
                limit = int(params.get("limit", ["10"])[0])
                query = params.get("query", [""])[0]
                if (page is not None and not 1 <= page <= 1000) or not 0 <= offset <= 100000 or not 1 <= limit <= 20 or len(query) > 100:
                    raise ValueError("Invalid chunk filter")
                return self._json(200, browse_chunks(identifier, params.get("filename", [""])[0], page, query, offset, limit))
            except (ValueError, FileNotFoundError) as exc:
                return self._json(400, {"error": str(exc)})
        if route.path == "/api/dev/document-page":
            params = parse_qs(route.query)
            try:
                filename = params.get("filename", [""])[0]
                identifier = params.get("variant", ["structure"])[0]
                page = int(params.get("page", ["0"])[0])
                return self._json(200, document_page(identifier, filename, page))
            except (ValueError, FileNotFoundError) as exc:
                return self._json(400, {"error": str(exc)})
        if route.path == "/api/dev/document-image":
            params = parse_qs(route.query)
            path = filing_path(params.get("filename", [""])[0])
            if path is None:
                return self._hidden()
            try:
                page = int(params.get("page", ["0"])[0])
                with pymupdf.open(path) as pdf:
                    if not 1 <= page <= len(pdf):
                        raise ValueError("Page is outside this filing")
                    image = pdf[page - 1].get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False).tobytes("png")
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "private, max-age=3600")
                self.send_header("Content-Length", str(len(image)))
                self.end_headers()
                self.wfile.write(image)
                return
            except ValueError as exc:
                return self._json(400, {"error": str(exc)})
        if route.path.startswith("/api/dev/filings/"):
            filename = unquote(route.path.removeprefix("/api/dev/filings/"))
            path = filing_path(filename)
            if path:
                return self._file(path, "application/pdf")
            return self._hidden()
        if route.path in paths:
            name, kind = paths[route.path]
            return self._file(ASSETS / name, kind)
        return self._hidden()

    def _body(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                raise ValueError("Request body must be 1–4096 bytes")
            value = json.loads(self.rfile.read(length))
            if not isinstance(value, dict):
                raise ValueError("Expected a JSON object")
            return value
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})
            return None

    def do_DELETE(self) -> None:
        origin = self.headers.get("Origin")
        if origin and origin not in {f"http://{self.headers.get('Host')}",
                                     f"https://{self.headers.get('Host')}"}:
            return self._json(403, {"error": "Origin mismatch"})
        prefix = "/api/chat/conversations/"
        if not self.path.startswith(prefix):
            return self._hidden()
        identifier = self.path.removeprefix(prefix)
        if not re.fullmatch(r"[0-9a-f]{12}", identifier):
            return self._hidden()
        if not CHAT_LOCK.acquire(blocking=False):
            return self._json(409, {"error": "Wait for the current answer to finish"})
        try:
            from devui.observability import delete_conversation

            return self._json(200, {"deleted": True}) if delete_conversation(identifier) else self._hidden()
        finally:
            CHAT_LOCK.release()

    def do_POST(self) -> None:
        global JOB, EVAL_JOB, ANSWER_RUN_JOB, GENERATION_EVAL_JOB, BENCHMARK_EVAL_JOB
        origin = self.headers.get("Origin")
        if origin and origin not in {f"http://{self.headers.get('Host')}", f"https://{self.headers.get('Host')}"}:
            return self._json(403, {"error": "Origin mismatch"})
        if self.path in {"/api/chat", "/api/chat/stream"}:
            streaming = self.path == "/api/chat/stream"
            body = self._body()
            if body is None:
                return
            question = body.get("question")
            if not isinstance(question, str) or not 1 <= len(question.strip()) <= 2000:
                return self._json(400, {"error": "Question must be 1–2000 characters"})
            answer_style = body.get("answer_style", "concise")
            if answer_style not in {"concise", "detailed"}:
                return self._json(400, {"error": "Choose concise or detailed answer style"})
            conversation_id = body.get("conversation_id")
            if conversation_id is not None and not (isinstance(conversation_id, str)
                    and re.fullmatch(r"[0-9a-f]{12}", conversation_id)):
                return self._json(400, {"error": "Invalid conversation"})
            history = None
            if conversation_id:
                from devui.observability import conversation_context

                history = conversation_context(conversation_id)
                if history is None:
                    return self._json(404, {"error": "Conversation not found"})
            else:
                conversation_id = uuid4().hex[:12]
            try:
                model = evaluation_model(body.get("model"))
            except ValueError as exc:
                return self._json(400, {"error": str(exc)})
            if not os.getenv("FIREWORKS_API_KEY"):
                return self._json(503, {"error": "Configure FIREWORKS_API_KEY and restart the server"})
            with JOB_LOCK:
                busy = JOB["state"] == "running" or EVAL_JOB["state"] == "running"
            if busy:
                return self._json(409, {"error": "An index or developer evaluation is running; try again shortly"})
            if not CHAT_LOCK.acquire(blocking=False):
                return self._json(409, {"error": "Another question is being answered; try again shortly"})
            progress_events: list[dict] = []
            started = time.perf_counter()

            def on_progress(update: dict) -> None:
                progress_events.append(update)
                if streaming:
                    self._event("progress", update)

            try:
                if streaming:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Accel-Buffering", "no")
                    self.end_headers()
                    self.close_connection = True
                options = {"progress": on_progress, "answer_style": answer_style, "model": model}
                if history:
                    options["history"] = history
                result = run_chat(question.strip(), **options)
                if not isinstance(result, dict) or not isinstance(result.get("answer"), str):
                    raise ValueError("The agent returned an invalid answer")
                payload = {"answer": result["answer"], "sources": result.get("sources", []),
                           "trace": result.get("trace", []),
                           "model": result.get("model", model),
                           "conversation_id": conversation_id,
                           "total_ms": round((time.perf_counter() - started) * 1000, 1)}
                if result.get("verification") is not None:
                    payload["verification"] = result["verification"]
                if result.get("answer_status") is not None:
                    payload["answer_status"] = result["answer_status"]
                try:
                    from devui.observability import record_run

                    payload["run_id"] = record_run(question.strip(), progress_events,
                                                   result, payload["total_ms"],
                                                   conversation_id=conversation_id)
                except OSError:
                    logging.getLogger(__name__).exception("Could not save local agent workflow")
                return self._event("answer", payload) if streaming else self._json(200, payload)
            except Exception:
                logging.getLogger(__name__).exception("Chat request failed")
                payload = {"error": "The assistant could not answer this question. Check the server log and retry."}
                return self._event("error", payload) if streaming else self._json(500, payload)
            finally:
                CHAT_LOCK.release()
        if not developer_enabled():
            return self._hidden()
        body = self._body()
        if body is None:
            return
        if self.path == "/api/dev/models/refresh":
            from src.answer_agent.models import discover_models

            if body:
                return self._json(400, {"error": "No options are accepted for model discovery"})
            key = os.getenv("FIREWORKS_API_KEY")
            if not key:
                return self._json(409, {"error": "Set FIREWORKS_API_KEY in .env and restart the server"})
            try:
                models = discover_models(key)
            except Exception:
                logging.getLogger(__name__).exception("Fireworks model discovery failed")
                return self._json(502, {"error": "Could not load Fireworks models; check server log and retry"})
            with MODEL_LOCK:
                MODEL_CATALOG[:] = models
            return self._json(200, {"models": models})
        if self.path == "/api/dev/benchmark-evaluation/generate":
            if set(body) - {"model"}:
                return self._json(400, {"error": "Choose an answer model"})
            try:
                model = evaluation_model(body.get("model"))
            except ValueError as exc:
                return self._json(400, {"error": str(exc)})
            if not os.getenv("FIREWORKS_API_KEY"):
                return self._json(409, {"error": "Set FIREWORKS_API_KEY in .env and restart the server"})
            if not vector_state("structure")["index"]["ready"]:
                return self._json(409, {"error": "Build the Structure aware Qdrant index first"})
            with JOB_LOCK:
                if (JOB["state"] == "running" or EVAL_JOB["state"] == "running"
                        or BENCHMARK_EVAL_JOB["state"] == "running"):
                    return self._json(409, {"error": "Another developer job is already running"})
                if not CHAT_LOCK.acquire(blocking=False):
                    return self._json(409, {"error": "Another chat or benchmark answer run is running"})
                ANSWER_RUN_JOB = {"state": "running", "done": 0, "total": 10,
                                   "message": "Starting all ten development questions…"}
            threading.Thread(target=run_benchmark_answers,
                             args=(model,), daemon=True).start()
            return self._json(202, {"job": ANSWER_RUN_JOB})
        if self.path == "/api/dev/benchmark-evaluation/retry-answer":
            if set(body) != {"question_id", "model"} or not isinstance(body["question_id"], str):
                return self._json(400, {"error": "Choose a saved question and answer model"})
            question_id = body["question_id"]
            try:
                model = evaluation_model(body["model"])
                answer_run = json.loads((ROOT / "evaluation/dev_run.json").read_text())
            except ValueError as exc:
                return self._json(400, {"error": str(exc)})
            except OSError:
                return self._json(409, {"error": "Generate the ten answers first"})
            if len(answer_run.get("questions", [])) != 10 or not any(
                    row.get("id") == question_id for row in answer_run["questions"]):
                return self._json(409, {"error": "Retry a question in the current ten-answer run"})
            if not os.getenv("FIREWORKS_API_KEY"):
                return self._json(409, {"error": "Set FIREWORKS_API_KEY in .env and restart the server"})
            with JOB_LOCK:
                if (ANSWER_RUN_JOB["state"] == "running" or BENCHMARK_EVAL_JOB["state"] == "running"
                        or GENERATION_EVAL_JOB["state"] == "running" or EVAL_JOB["state"] == "running"
                        or JOB["state"] == "running"):
                    return self._json(409, {"error": "Another developer job is already running"})
                if not CHAT_LOCK.acquire(blocking=False):
                    return self._json(409, {"error": "Another chat or answer run is running"})
                ANSWER_RUN_JOB = {"state": "running", "done": 0, "total": 1,
                                  "message": f"Retrying answer for {question_id}…"}
            threading.Thread(target=run_benchmark_answer_retry,
                             args=(question_id, model), daemon=True).start()
            return self._json(202, {"job": ANSWER_RUN_JOB})
        if self.path == "/api/dev/generation-evaluation/run":
            from eval_pipeline.generation import pilot_ready
            if set(body) != {"mode"} or body["mode"] not in {"pilot", "full"}:
                return self._json(400, {"error": "Choose the two-question pilot or full run"})
            pilot = body["mode"] == "pilot"
            if not pilot and not pilot_ready(ROOT):
                return self._json(409, {"error": "Run a successful two-question pilot first"})
            if not (ROOT / "evaluation/dev_run.json").is_file():
                return self._json(409, {"error": "Generate the ten answers in Benchmark Evaluation first"})
            if not os.getenv("FIREWORKS_API_KEY"):
                return self._json(409, {"error": "Set FIREWORKS_API_KEY in .env"})
            from importlib.util import find_spec
            if find_spec("ragas") is None:
                return self._json(409, {"error": "Install dependencies from starter/requirements.txt and restart the server"})
            with JOB_LOCK:
                if GENERATION_EVAL_JOB["state"] == "running" or ANSWER_RUN_JOB["state"] == "running":
                    return self._json(409, {"error": "An evaluation is already running"})
                GENERATION_EVAL_JOB = {"state": "running", "mode": body["mode"],
                                       "done": 0, "total": 2 if pilot else 10,
                                       "message": "Scoring saved answers…"}
            threading.Thread(target=run_generation_eval, args=(pilot,), daemon=True).start()
            return self._json(202, {"job": GENERATION_EVAL_JOB})
        if self.path == "/api/dev/generation-evaluation/retry":
            from eval_pipeline.generation import load_latest
            if set(body) != {"question_id", "metric"} or not isinstance(body["question_id"], str) or \
                    not re.fullmatch(r"q_\d{3}", body["question_id"]) or \
                    not isinstance(body["metric"], str) or \
                    body["metric"] not in {"faithfulness", "response_relevancy"}:
                return self._json(400, {"error": "Choose a question and one failed metric"})
            result = load_latest(ROOT)
            row = next((item for item in (result or {}).get("rows", [])
                        if item.get("id") == body["question_id"]), None)
            if not row or row.get("scores", {}).get(body["metric"]) is not None or \
                    not row.get("scores", {}).get("errors", {}).get(body["metric"]):
                return self._json(409, {"error": "Only an unavailable metric in the current saved run can be retried"})
            if not os.getenv("FIREWORKS_API_KEY"):
                return self._json(409, {"error": "Set FIREWORKS_API_KEY in .env"})
            with JOB_LOCK:
                if GENERATION_EVAL_JOB["state"] == "running" or ANSWER_RUN_JOB["state"] == "running":
                    return self._json(409, {"error": "An evaluation is already running"})
                GENERATION_EVAL_JOB = {"state": "running", "mode": "retry", "question_id": body["question_id"],
                                       "metric": body["metric"], "done": 0, "total": 1,
                                       "message": f"Retrying {body['metric']} for {body['question_id']}…"}
            threading.Thread(target=run_generation_metric_retry,
                             args=(body["question_id"], body["metric"]), daemon=True).start()
            return self._json(202, {"job": GENERATION_EVAL_JOB})
        if self.path == "/api/dev/retrieval-evaluation/run":
            question_count = body.get("question_count", 10)
            if type(question_count) is not int or question_count not in (2, 10):
                return self._json(400, {"error": "Choose the two-question pilot or all ten questions"})
            if not (ROOT / "evaluation/dev_run.json").is_file():
                return self._json(409, {"error": "Generate the ten benchmark answers first"})
            if not os.getenv("FIREWORKS_API_KEY"):
                return self._json(409, {"error": "Set FIREWORKS_API_KEY in the server environment"})
            with JOB_LOCK:
                if JOB["state"] == "running" or EVAL_JOB["state"] == "running" or CHAT_LOCK.locked():
                    return self._json(409, {"error": "Another developer job is already running"})
                EVAL_JOB = {"state": "running",
                            "done": 0, "total": question_count, "message": "Starting retrieval evaluation…"}
            threading.Thread(target=run_retrieval_eval,
                             args=(question_count,), daemon=True).start()
            return self._json(202, {"job": EVAL_JOB})
        if self.path == "/api/dev/retrieval-evaluation/retry":
            from devui.retrieval_eval import latest_result

            question_id = body.get("question_id")
            if not isinstance(question_id, str) or not re.fullmatch(r"q_\d{3}", question_id):
                return self._json(400, {"error": "Choose a question with an unavailable metric"})
            answer_path = ROOT / "evaluation/dev_run.json"
            try:
                answer_run_id = json.loads(answer_path.read_text()).get("run_id")
            except (OSError, ValueError):
                return self._json(409, {"error": "Generate benchmark answers first"})
            result = latest_result(ROOT, answer_run_id)
            row = next((item for item in (result or {}).get("rows", [])
                        if item["case"]["question_id"] == question_id), None)
            if row is None or all(row["metrics"]["cited"][name] is not None
                                  for name in ("context_precision", "context_recall")):
                return self._json(409, {"error": "Only unavailable metrics in the current retrieval run can be retried"})
            if not os.getenv("FIREWORKS_API_KEY"):
                return self._json(409, {"error": "Set FIREWORKS_API_KEY for the RAGAS judge"})
            with JOB_LOCK:
                if JOB["state"] == "running" or EVAL_JOB["state"] == "running" or CHAT_LOCK.locked():
                    return self._json(409, {"error": "Another developer job is already running"})
                EVAL_JOB = {"state": "running", "done": 0, "total": 1,
                            "message": f"Retrying unavailable metrics for {question_id}…"}
            threading.Thread(target=run_retrieval_retry, args=(question_id,), daemon=True).start()
            return self._json(202, {"job": EVAL_JOB})
        if self.path == "/api/dev/benchmark-evaluation/run":
            if not (ROOT / "evaluation/dev_run.json").is_file():
                return self._json(409, {"error": "Generate the ten answers first"})
            if not os.getenv("FIREWORKS_API_KEY"):
                return self._json(409, {"error": "Set FIREWORKS_API_KEY for the LLM-judged questions"})
            with JOB_LOCK:
                if BENCHMARK_EVAL_JOB["state"] == "running" or ANSWER_RUN_JOB["state"] == "running":
                    return self._json(409, {"error": "Wait for the current answer or benchmark evaluation"})
                BENCHMARK_EVAL_JOB = {"state": "running", "done": 0, "total": 10,
                                      "message": "Starting benchmark grading…"}
            threading.Thread(target=run_benchmark_eval, daemon=True).start()
            return self._json(202, {"job": BENCHMARK_EVAL_JOB})
        if self.path == "/api/dev/benchmark-evaluation/retry":
            question_id = body.get("question_id")
            if not isinstance(question_id, str) or not re.fullmatch(r"q_\d{3}", question_id):
                return self._json(400, {"error": "Choose an unavailable question"})
            path = ROOT / "evaluation/benchmark_run.json"
            try:
                result = json.loads(path.read_text())
                answer_run = json.loads((ROOT / "evaluation/dev_run.json").read_text())
                row = next((item for item in result["rows"] if item["id"] == question_id), None)
                record = next((item for item in answer_run["questions"] if item["id"] == question_id), None)
            except (OSError, ValueError, KeyError):
                return self._json(409, {"error": "Run Benchmark Evaluation first"})
            if (result.get("answer_run_id") != answer_run.get("run_id") or row is None
                    or row.get("status") != "unavailable" or not record or not record.get("answer")):
                return self._json(409, {"error": "This unavailable verdict has no saved answer to regrade"})
            if not os.getenv("FIREWORKS_API_KEY"):
                return self._json(409, {"error": "Set FIREWORKS_API_KEY for the LLM judge"})
            with JOB_LOCK:
                if BENCHMARK_EVAL_JOB["state"] == "running" or ANSWER_RUN_JOB["state"] == "running":
                    return self._json(409, {"error": "Wait for the current answer or benchmark evaluation"})
                BENCHMARK_EVAL_JOB = {"state": "running", "done": 0, "total": 1,
                                      "message": f"Retrying {question_id}…"}
            threading.Thread(target=run_benchmark_retry, args=(question_id,), daemon=True).start()
            return self._json(202, {"job": BENCHMARK_EVAL_JOB})
        if self.path in {"/api/dev/embedding/build", "/api/dev/index/build"}:
            strategy = body.get("strategy")
            if strategy not in STRATEGIES:
                return self._json(400, {"error": "Choose structure or semantic"})
            stage = "embedding" if self.path == "/api/dev/embedding/build" else "indexing"
            current = vector_state(strategy)
            if not chunk_path(strategy).is_file():
                return self._json(409, {"error": "Build this chunk set first"})
            if stage == "embedding" and not os.getenv("FIREWORKS_API_KEY"):
                return self._json(409, {"error": "Set FIREWORKS_API_KEY before generating embeddings"})
            if stage == "indexing" and not current["embedding"]["ready"]:
                return self._json(409, {"error": "Generate embeddings for these chunks first"})
            if stage == "indexing":
                try:
                    import qdrant_client  # noqa: F401
                except ImportError:
                    return self._json(409, {"error": "Install qdrant-client and restart the server"})
            with JOB_LOCK:
                if JOB["state"] == "running" or EVAL_JOB["state"] == "running" or CHAT_LOCK.locked():
                    return self._json(409, {"error": "Another ingestion step is already running"})
                run_id = time.time_ns()
                JOB = {"state": "running", "stage": stage, "strategy": strategy,
                       "run_id": run_id, "message": "Generating embeddings…" if stage == "embedding"
                       else "Writing Qdrant points…"}
            threading.Thread(target=run_vector_stage, args=(stage, strategy, run_id), daemon=True).start()
            return self._json(202, {"job": JOB})
        if self.path in {"/api/dev/build", "/api/dev/variant/build"}:
            strategy = body.get("strategy")
            if strategy not in STRATEGIES:
                return self._json(400, {"error": "Choose structure or semantic"})
            identifier = None
            settings = (3600, 4800, 400)
            if self.path == "/api/dev/variant/build":
                settings = (body.get("target"), body.get("maximum"), body.get("overlap"))
                if any(type(value) is not int for value in settings):
                    return self._json(400, {"error": "Target, maximum, and overlap must be integers"})
                try:
                    identifier = variant_id(strategy, *settings)
                except ValueError as exc:
                    return self._json(400, {"error": str(exc)})
            if strategy == "semantic" and not os.getenv("FIREWORKS_API_KEY"):
                return self._json(409, {"error": "Set FIREWORKS_API_KEY before building semantic chunks"})
            with JOB_LOCK:
                if JOB["state"] == "running" or EVAL_JOB["state"] == "running" or CHAT_LOCK.locked():
                    return self._json(409, {"error": "Another ingestion step is already running"})
                JOB = {"state": "running", "stage": "chunking", "strategy": strategy, "variant_id": identifier,
                       "run_id": time.time_ns(),
                       "message": "Extracting and chunking filings…"}
            threading.Thread(target=build_chunk_set, args=(strategy, identifier, *settings), daemon=True).start()
            return self._json(202, {"job": JOB})
        return self._hidden()


def main() -> None:
    configure_logging()
    load_local_fireworks_key()
    port = int(os.getenv("PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Research chat: http://127.0.0.1:{port}/")
    if developer_enabled():
        print(f"Developer ingestion UI: http://127.0.0.1:{port}/ingestion-pipeline")
    server.serve_forever()


if __name__ == "__main__":
    main()
