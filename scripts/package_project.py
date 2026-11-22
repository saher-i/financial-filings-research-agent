"""Package the runnable project without local credentials or caches."""

import argparse
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = ("README.md", "report.md", "setup.sh", ".env.example", "dev_answers.json")
SOURCE_DIRS = ("src", "devui", "eval_pipeline", "tests", "questions", "scripts", "starter")
DATA_FILES = ("data/README.md", "data/manifest.json", "data/financials.db",
              "data/index/structure_chunks.jsonl", "data/index/structure_embeddings.db",
              "data/index/structure_lexical.db",
              "data/index/qdrant/active_structure.json", "data/index/qdrant/meta.json",
              "evaluation/dev_run.json", "evaluation/benchmark_run.json")


def _final_result(root: Path, directory: str, answer_run_id: str,
                  question_count: int) -> Path:
    candidates = []
    for path in (root / "evaluation" / directory).glob("*.json"):
        result = json.loads(path.read_text())
        if result.get("mode") != "full":
            continue
        if result.get("source_run_id", result.get("answer_run_id")) != answer_run_id:
            continue
        if directory == "generation_runs":
            scored = result.get("summary", {})
            complete = all(scored.get(metric, {}).get("scored") == question_count
                           for metric in ("faithfulness", "response_relevancy"))
        else:
            scored = result.get("summary", {}).get("cited", {})
            complete = all(scored.get(metric) == question_count
                           for metric in ("precision_scored", "recall_scored"))
        if complete:
            candidates.append(path)
    if not candidates:
        raise ValueError(f"No complete {directory} result for the active answer run")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def project_files(root: Path = ROOT) -> list[Path]:
    question_ids = {item["id"] for item in json.loads(
        (root / "questions/dev_questions.json").read_text())}
    answers_path = root / "dev_answers.json"
    if not answers_path.is_file():
        raise ValueError("Run all ten development questions before packaging: dev_answers.json is missing")
    answers = json.loads(answers_path.read_text())
    if set(answers) != question_ids or any(not isinstance(value, str) or not value.strip()
                                            for value in answers.values()):
        raise ValueError("dev_answers.json must contain a nonempty generated answer for every development question")
    records_path = root / "evaluation/dev_run.json"
    if not records_path.is_file():
        raise ValueError("evaluation/dev_run.json is missing")
    answer_run = json.loads(records_path.read_text())
    records = answer_run.get("questions", [])
    if {row.get("id") for row in records if row.get("answer") == answers.get(row.get("id"))} != question_ids:
        raise ValueError("evaluation/dev_run.json must contain the same generated answers as dev_answers.json")
    answer_run_id = answer_run.get("run_id")
    if not answer_run_id:
        raise ValueError("evaluation/dev_run.json must identify the active run")
    benchmark_path = root / "evaluation/benchmark_run.json"
    if not benchmark_path.is_file():
        raise ValueError("evaluation/benchmark_run.json is missing")
    benchmark = json.loads(benchmark_path.read_text())
    if benchmark.get("answer_run_id") != answer_run_id or len(benchmark.get("rows", [])) != len(question_ids):
        raise ValueError("Benchmark result must match the active answer run")
    final_results = [_final_result(root, directory, answer_run_id, len(question_ids))
                     for directory in ("generation_runs", "retrieval_runs")]
    pdfs = list((root / "data/pdfs").glob("*.pdf"))
    if len(pdfs) != 6:
        raise ValueError("The project archive needs all six PDF filings")
    qdrant_files = [path for path in (root / "data/index/qdrant/collection").rglob("*") if path.is_file()]
    if not any(path.name == "storage.sqlite" for path in qdrant_files):
        raise ValueError("Build the Structure aware Qdrant index before packaging")
    files = [root / name for name in (*ROOT_FILES, *DATA_FILES)]
    files.extend(final_results)
    files.extend(path for directory in SOURCE_DIRS for path in (root / directory).rglob("*") if path.is_file())
    files.extend(pdfs)
    files.extend(qdrant_files)
    missing = [str(path.relative_to(root)) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing project files: " + ", ".join(missing))
    kept = []
    for path in files:
        relative = path.relative_to(root)
        if path.is_symlink() or any(part in {"__pycache__", ".git", ".venv", ".env", ".local"}
                                   for part in relative.parts):
            continue
        if path.suffix in {".pyc", ".pyo"} or path.name in {".DS_Store", ".lock"}:
            continue
        kept.append(path)
    return sorted(set(kept))


def package(destination: Path, root: Path = ROOT) -> tuple[int, int]:
    files = project_files(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, arcname=path.relative_to(root).as_posix())
    return len(files), destination.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/project.zip")
    args = parser.parse_args()
    try:
        count, size = package(args.output)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"Cannot package project: {exc}\n")
    print(f"Packaged {count} files ({size / 1_000_000:.1f} MB) at {args.output}")


if __name__ == "__main__":
    main()
