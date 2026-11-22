"""Local, developer-only records of actual chat runs."""

from __future__ import annotations

import json
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.ingestion_pipeline import ROOT

TRACE_FILE = ROOT / "logs" / "agent_runs.jsonl"
_WRITE_LOCK = threading.Lock()


def record_run(question: str, events: list[dict], result: dict,
               total_ms: float, path: Path = TRACE_FILE,
               conversation_id: str | None = None) -> str:
    """Save a completed run locally; never persist credentials or model prompts."""
    run_id = uuid4().hex[:12]
    record = {
        "id": run_id,
        "conversation_id": conversation_id or run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": question,
        "answer": result["answer"],
        "answer_status": result.get("answer_status", "answered"),
        "answer_style": result.get("answer_style", "concise"),
        "model": result.get("model"),
        "rejected_draft": result.get("draft", "") if result.get("answer_status") == "citation_failed" else "",
        "total_ms": total_ms,
        "events": events,
        "trace": result.get("trace", []),
        "sources": result.get("sources", []),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK, path.open("a") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return run_id


def recent_runs(limit: int = 20, path: Path = TRACE_FILE) -> list[dict]:
    """Read a bounded set of recent runs for the local workflow page."""
    if not path.is_file():
        return []
    with path.open() as stream:
        lines = deque(stream, maxlen=limit)
    return [json.loads(line) for line in reversed(lines) if line.strip()]


def recent_chats(limit: int = 20, path: Path = TRACE_FILE) -> list[dict]:
    """One sidebar entry per conversation, newest activity first."""
    runs = recent_runs(max(limit * 20, 100), path)
    groups: dict[str, dict] = {}
    for run in runs:
        identifier = run.get("conversation_id") or run["id"]
        if identifier not in groups:
            groups[identifier] = {"id": identifier, "created_at": run.get("created_at"),
                                  "question": run.get("question"),
                                  "answer_status": run.get("answer_status"), "turns": 0}
        groups[identifier]["question"] = run.get("question") or groups[identifier]["question"]
        groups[identifier]["turns"] += 1
    return list(groups.values())[:limit]


def conversation_by_id(conversation_id: str, path: Path = TRACE_FILE) -> dict | None:
    """Return all saved turns in one recent local conversation."""
    turns = [run for run in reversed(recent_runs(500, path))
             if (run.get("conversation_id") or run.get("id")) == conversation_id]
    if not turns:
        return None
    return {"id": conversation_id, "question": turns[0]["question"], "turns": turns}


def conversation_context(conversation_id: str, path: Path = TRACE_FILE) -> list[dict] | None:
    """Bounded prior dialogue for reference resolution; current evidence is re-retrieved."""
    import re

    conversation = conversation_by_id(conversation_id, path)
    if conversation is None:
        return None
    messages = []
    for turn in conversation["turns"][-4:]:
        if turn.get("answer_status") != "answered":
            continue
        question = str(turn.get("question", ""))[:1000]
        answer = re.sub(r"\[S\d+\]", "", str(turn.get("answer", "")))[:1800]
        messages.extend(({"role": "user", "content": question},
                         {"role": "assistant", "content": answer}))
    return messages


def delete_conversation(conversation_id: str, path: Path = TRACE_FILE) -> bool:
    """Remove all turns of one local conversation, preserving other run records."""
    if not path.is_file():
        return False
    temporary = path.with_suffix(path.suffix + ".tmp")
    removed = False
    with _WRITE_LOCK:
        try:
            with path.open() as source, temporary.open("w") as target:
                for line in source:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if (record.get("conversation_id") or record.get("id")) == conversation_id:
                        removed = True
                    else:
                        target.write(line)
            if removed:
                temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return removed


def chat_by_id(run_id: str, path: Path = TRACE_FILE) -> dict | None:
    """Load one of the recent chats for display without another model call."""
    for run in recent_runs(100, path):
        if run.get("id") == run_id:
            return run
    return None
