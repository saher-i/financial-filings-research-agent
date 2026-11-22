"""Regenerate and grade all ten answers with the optional review enabled.

Run after stopping the local server:
    .venv/bin/python -m scripts.run_verified_benchmark

The prior answer file is saved as previous_dev_answers.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval_pipeline.answers import run_development
from eval_pipeline.benchmark import evaluate
from src.answer_agent.graph import answer_question
from src.config import load_local_fireworks_key


ROOT = Path(__file__).resolve().parents[1]
def main() -> None:
    load_local_fireworks_key()
    baseline = json.loads((ROOT / "evaluation/dev_run.json").read_text())
    model = baseline.get("model")
    previous_counts = json.loads((ROOT / "evaluation/benchmark_run.json").read_text())["counts"]

    def progress(done: int, total: int, question_id: str) -> None:
        print(f"Answered {question_id} ({done}/{total})", flush=True)

    run_development(lambda question: answer_question(question, model=model, verify=True),
                    root=ROOT, progress=progress, model=model, preserve_previous=True)
    result = evaluate(ROOT)
    run = json.loads((ROOT / "evaluation/dev_run.json").read_text())
    reviews = {}
    for record in run["questions"]:
        review = record.get("verification") or {}
        reviews[record["id"]] = {
            "answer_status": record.get("answer_status"),
            "review_status": review.get("status"),
            "review_calls": review.get("review_calls", 0),
            "elapsed_seconds": record.get("elapsed_seconds"),
        }
    print(json.dumps({"previous_benchmark": previous_counts,
                      "verified_benchmark": result["counts"], "reviews": reviews}, indent=2))
    print("Saved new answers to dev_answers.json; prior answers are in previous_dev_answers.json")


if __name__ == "__main__":
    main()
