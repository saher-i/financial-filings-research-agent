"""Project packaging requires generated answers and excludes local secrets."""

import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from scripts.package_project import DATA_FILES, ROOT_FILES, package


class PackageProjectTests(unittest.TestCase):
    def test_requires_generated_answers_and_excludes_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "questions").mkdir()
            (root / "questions/dev_questions.json").write_text(json.dumps([
                {"id": "q_1", "question": "Question"}]))
            with self.assertRaisesRegex(ValueError, "dev_answers.json is missing"):
                package(root / "project.zip", root)
            for name in (*ROOT_FILES, *DATA_FILES):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    path.write_text("placeholder")
            (root / "dev_answers.json").write_text('{"q_1":"Agent answer"}')
            (root / "evaluation/dev_run.json").write_text(json.dumps({"run_id": "final-run", "questions": [
                {"id": "q_1", "answer": "Agent answer"}]}))
            (root / "evaluation/benchmark_run.json").write_text(json.dumps({
                "answer_run_id": "final-run", "rows": [{"id": "q_1"}]}))
            generation = root / "evaluation/generation_runs/final-full.json"
            generation.parent.mkdir(parents=True)
            generation.write_text(json.dumps({"mode": "full", "source_run_id": "final-run",
                "summary": {"faithfulness": {"scored": 1}, "response_relevancy": {"scored": 1}}}))
            retrieval = root / "evaluation/retrieval_runs/final.json"
            retrieval.parent.mkdir(parents=True)
            retrieval.write_text(json.dumps({"mode": "full", "answer_run_id": "final-run",
                "summary": {"cited": {"precision_scored": 1, "recall_scored": 1}}}))
            (generation.parent / "old-pilot.json").write_text(json.dumps({"mode": "pilot"}))
            for index in range(6):
                path = root / f"data/pdfs/file{index}.pdf"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"%PDF")
            index = root / "data/index/qdrant/collection/test/storage.sqlite"
            index.parent.mkdir(parents=True)
            index.write_bytes(b"sqlite")
            (root / "src").mkdir()
            (root / "src/main.py").write_text("print('safe')")
            (root / "eval_pipeline").mkdir()
            (root / "eval_pipeline/retrieval.py").write_text("# evaluator")
            (root / "src/.env").write_text("SECRET=never-package")
            (root / "data/index/qdrant/.lock").write_text("lock")
            destination = root / "project.zip"
            package(destination, root)
            with ZipFile(destination) as archive:
                names = set(archive.namelist())
            self.assertIn("src/main.py", names)
            self.assertIn("dev_answers.json", names)
            self.assertIn("evaluation/generation_runs/final-full.json", names)
            self.assertIn("evaluation/retrieval_runs/final.json", names)
            self.assertIn("report.md", names)
            self.assertNotIn("output/pdf/10k_research_report.pdf", names)
            self.assertIn("eval_pipeline/retrieval.py", names)
            self.assertNotIn("evaluation/generation_runs/old-pilot.json", names)
            self.assertNotIn("src/.env", names)
            self.assertNotIn("data/index/qdrant/.lock", names)


if __name__ == "__main__":
    unittest.main()
