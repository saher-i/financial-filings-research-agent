"""The development runner must save agent outputs without feeding it answers."""

import json
import tempfile
import unittest
from pathlib import Path

import httpx2
from openai import APIConnectionError

from eval_pipeline.answers import retry_answer, run_development


class DevelopmentEvaluationTests(unittest.TestCase):
    def test_retry_replaces_only_selected_answer_and_preserves_prior_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "questions").mkdir()
            (root / "questions/dev_questions.json").write_text(json.dumps([
                {"id": f"q_{index:03}", "question": f"Question {index}", "required_modalities": []}
                for index in range(1, 11)]))
            run_development(lambda question: {"answer": f"Original {question}", "sources": []}, root=root)
            before = json.loads((root / "evaluation/dev_run.json").read_text())
            outcome = retry_answer("q_003", lambda _: {"answer": "Retried answer", "sources": []}, root=root)
            after = json.loads((root / "evaluation/dev_run.json").read_text())
            self.assertTrue(outcome["updated"])
            self.assertNotEqual(before["run_id"], after["run_id"])
            self.assertEqual(after["questions"][2]["answer"], "Retried answer")
            self.assertEqual(after["questions"][0], before["questions"][0])
            self.assertEqual(after["questions"][3:], before["questions"][3:])
            self.assertEqual(json.loads((root / "dev_answers.json").read_text())["q_003"], "Retried answer")
            self.assertEqual(json.loads((root / "evaluation/runs" / before["run_id"] / "dev_run.json").read_text()), before)

    def test_records_real_agent_outputs_and_source_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "questions").mkdir()
            (root / "questions/dev_questions.json").write_text(json.dumps([
                {"id": "q_1", "question": "How much?", "required_modalities": ["sql", "pdf"]}]))
            (root / "questions/dev_questions_with_answers.json").write_text(json.dumps([
                {"id": "q_1", "gold_answer": "reference only"}]))
            seen = []

            def answerer(question):
                seen.append(question)
                return {"answer": "Generated answer", "sources": [
                    {"type": "sql", "sql": "SELECT 1"},
                    {"type": "pdf", "filename": "filing.pdf", "pages": [1]}], "trace": []}

            result = run_development(answerer, root=root)
            self.assertEqual(seen, ["How much?"])
            self.assertEqual(result["source_coverage_count"], 1)
            self.assertEqual(json.loads((root / "dev_answers.json").read_text()), {"q_1": "Generated answer"})
            record = json.loads((root / "evaluation/dev_run.json").read_text())["questions"][0]
            self.assertNotIn("reference_answer", record)
            self.assertTrue(record["source_coverage"])

    def test_partial_run_does_not_replace_full_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "questions").mkdir()
            (root / "dev_answers.json").write_text('{"previous": "kept"}')
            (root / "evaluation").mkdir()
            (root / "evaluation/dev_run.json").write_text('{"questions": [{"id": "previous"}]}')
            question = {"id": "q_1", "question": "Question", "required_modalities": ["sql"]}
            (root / "questions/dev_questions.json").write_text(json.dumps([question]))
            (root / "questions/dev_questions_with_answers.json").write_text(json.dumps([
                {"id": "q_1", "gold_answer": "expected"}]))
            run_development(lambda _: {"answer": "Result", "sources": []}, root=root, ids={"q_1"})
            self.assertEqual((root / "dev_answers.json").read_text(), '{"previous": "kept"}')
            self.assertEqual((root / "evaluation/dev_run.json").read_text(), '{"questions": [{"id": "previous"}]}')
            self.assertTrue((root / "evaluation/partial_dev_run.json").is_file())

    def test_full_run_in_separate_directory_preserves_submitted_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "questions").mkdir()
            (root / "questions/dev_questions.json").write_text(json.dumps([
                {"id": "q_1", "question": "Question", "required_modalities": []}]))
            (root / "dev_answers.json").write_text('{"q_1":"Submitted"}')
            output = root / "evaluation/verified_comparison"
            run_development(lambda _: {"answer": "Reviewed", "sources": [],
                "verification": {"status": "supported", "review_calls": 1}},
                root=root, output_dir=output)
            self.assertEqual(json.loads((root / "dev_answers.json").read_text()), {"q_1": "Submitted"})
            self.assertEqual(json.loads((output / "dev_answers.json").read_text()), {"q_1": "Reviewed"})
            record = json.loads((output / "dev_run.json").read_text())["questions"][0]
            self.assertEqual(record["verification"]["status"], "supported")

    def test_reviewed_full_run_saves_previous_answers_before_replacing_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "questions").mkdir()
            (root / "questions/dev_questions.json").write_text(json.dumps([
                {"id": "q_1", "question": "Question", "required_modalities": []}]))
            (root / "dev_answers.json").write_text('{"q_1":"Previous answer"}')
            run_development(lambda _: {"answer": "New answer", "sources": []},
                            root=root, preserve_previous=True)
            self.assertEqual(json.loads((root / "previous_answers.json").read_text()),
                             {"q_1": "Previous answer"})
            self.assertEqual(json.loads((root / "dev_answers.json").read_text()),
                             {"q_1": "New answer"})

    def test_full_runs_keep_separate_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "questions").mkdir()
            (root / "questions/dev_questions.json").write_text(json.dumps([
                {"id": "q_1", "question": "Question", "required_modalities": []}]))
            (root / "questions/dev_questions_with_answers.json").write_text(json.dumps([
                {"id": "q_1", "gold_answer": "Reference"}]))
            run_development(lambda _: {"answer": "First", "sources": [],
                                       "model_diagnostics": [{"finish_reason": "stop"}]}, root=root,
                            model="first-model")
            first_id = json.loads((root / "evaluation/dev_run.json").read_text())["run_id"]
            run_development(lambda _: {"answer": "Second", "sources": []}, root=root,
                            model="second-model")
            self.assertEqual(len(list((root / "evaluation/runs").glob("*/dev_run.json"))), 2)
            prior = json.loads((root / "evaluation/runs" / first_id / "dev_run.json").read_text())
            self.assertEqual(prior["questions"][0]["answer"], "First")
            self.assertEqual(prior["questions"][0]["model_diagnostics"][0]["finish_reason"], "stop")

    def test_api_connection_failure_keeps_previous_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "questions").mkdir()
            (root / "questions/dev_questions.json").write_text(json.dumps([
                {"id": "q_1", "question": "Question", "required_modalities": ["sql"]}]))
            (root / "questions/dev_questions_with_answers.json").write_text(json.dumps([
                {"id": "q_1", "gold_answer": "Reference"}]))
            previous = '{"q_1": "Prior generated answer"}'
            (root / "dev_answers.json").write_text(previous)

            def unavailable(_):
                raise APIConnectionError(request=httpx2.Request("POST", "https://api.fireworks.ai"))

            with self.assertRaisesRegex(RuntimeError, "Fireworks request failed"):
                run_development(unavailable, root=root)
            self.assertEqual((root / "dev_answers.json").read_text(), previous)
            self.assertFalse((root / "evaluation/dev_run.json").exists())


if __name__ == "__main__":
    unittest.main()
