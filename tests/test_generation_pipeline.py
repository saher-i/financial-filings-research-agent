"""Official RAGAS adapter evaluates saved answers without calling the agent."""

import json
import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from eval_pipeline.generation import (_relevancy_trace, _sql_context, build_metrics, score_answer,
                                      evaluate_saved_answers, load_latest, pilot_ready,
                                      saved_cited_source, retry_saved_metric)


class Faithfulness:
    def __init__(self):
        self.calls = []

    async def ascore(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(value=0.75)


class ResponseRelevancy:
    def __init__(self):
        self.calls = []

    async def ascore(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(value=0.9)


class GenerationPipelineTests(unittest.TestCase):
    def test_failed_faithfulness_logs_question_and_bounded_reason(self):
        class FailingFaithfulness:
            async def ascore(self, **kwargs):
                raise ValueError("The output is incomplete due to a max_tokens length limit.")

        record = {"id": "q_006", "question": "What risks?", "answer": "Private answer [S1].",
                  "sources": [{"id": "S1", "type": "pdf", "filename": "filing.pdf",
                               "text": "Risk evidence."}]}
        with self.assertLogs('eval_pipeline.generation', level='WARNING') as logs:
            result = asyncio.run(score_answer(record, FailingFaithfulness(), ResponseRelevancy(),
                                              only="faithfulness"))
        self.assertIsNone(result['faithfulness'])
        self.assertIn('question_id=q_006', logs.output[0])
        self.assertIn('max_tokens length limit', logs.output[0])
        self.assertNotIn('Private answer', logs.output[0])

    def test_retry_updates_only_failed_metric_and_preserves_other_score(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "evaluation").mkdir()
            record = {"id": "q_012", "question": "Why did revenue rise?",
                      "answer": "Revenue rose [S1].", "sources": [{"id": "S1", "type": "pdf",
                      "filename": "filing.pdf", "text": "Revenue rose."}]}
            (root / "evaluation/dev_run.json").write_text(json.dumps({
                "run_id": "run-retry", "questions": [record]}))
            relevancy = ResponseRelevancy()
            faith = Faithfulness()
            result = evaluate_saved_answers(root, metrics=(faith, relevancy), pilot=False)
            path = next((root / "evaluation/generation_runs").glob("*.json"))
            result["rows"][0]["scores"]["faithfulness"] = None
            result["rows"][0]["scores"]["errors"] = {"faithfulness": "TimeoutError"}
            path.write_text(json.dumps(result))
            faith.calls.clear()
            relevancy.calls.clear()
            outcome = retry_saved_metric(root, "q_012", "faithfulness", metrics=(faith, relevancy))
            saved = json.loads(path.read_text())
            self.assertEqual(outcome["score"], 0.75)
            self.assertEqual(saved["rows"][0]["scores"]["response_relevancy"], 0.9)
            self.assertNotIn("faithfulness", saved["rows"][0]["scores"]["errors"])
            self.assertEqual(len(faith.calls), 1)
            self.assertEqual(relevancy.calls, [])

    def test_saved_source_opens_only_sources_cited_by_that_answer(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "evaluation").mkdir()
            cited = {"id": "S1", "type": "sql", "rows": [{"revenue": 42}]}
            unused = {"id": "S2", "type": "pdf", "text": "Other passage"}
            (root / "evaluation/dev_run.json").write_text(json.dumps({"questions": [
                {"id": "q_012", "answer": "Revenue was 42 [S1].", "sources": [cited, unused]}]}))
            self.assertEqual(saved_cited_source(root, "q_012", "S1"), cited)
            self.assertIsNone(saved_cited_source(root, "q_012", "S2"))
            self.assertIsNone(saved_cited_source(root, "q_001", "S1"))

    def test_relevancy_trace_shows_the_official_metric_embedding_comparisons(self):
        embeddings = SimpleNamespace(original_vector=[1.0, 0.0],
                                     generated_questions=["Same question?", "Different question?"],
                                     generated_vectors=[[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(_relevancy_trace(embeddings), [
            {"question": "Same question?", "similarity": 1.0},
            {"question": "Different question?", "similarity": 0.0}])

    def test_score_answer_saves_relevancy_questions_without_changing_metric_score(self):
        class TracedRelevancy:
            embeddings = SimpleNamespace()

            async def ascore(self, **kwargs):
                self.embeddings.original_vector = [1.0, 0.0]
                self.embeddings.generated_questions = ["What changed?"]
                self.embeddings.generated_vectors = [[0.8, 0.6]]
                return SimpleNamespace(value=0.8)

        record = {"question": "What changed?", "answer": "Revenue rose [S1].",
                  "sources": [{"id": "S1", "type": "pdf", "filename": "filing.pdf", "text": "Revenue rose."}]}
        result = asyncio.run(score_answer(record, Faithfulness(), TracedRelevancy()))
        self.assertEqual(result["response_relevancy"], 0.8)
        self.assertEqual(result["response_relevancy_questions"],
                         [{"question": "What changed?", "similarity": 0.8}])

    def test_sql_context_keeps_original_row_and_explains_billions_exactly(self):
        excerpt = '[{"company_ticker":"AAPL","fiscal_year":2025,"period_end":"2025-09-27","revenue":416161000000}]'
        context = _sql_context(excerpt)
        self.assertIn(excerpt, context)
        self.assertIn("AAPL", context)
        self.assertIn("2025-09-27", context)
        self.assertIn("$416.161 billion USD", context)

    def test_official_metrics_use_async_embeddings_and_sufficient_output_budget(self):
        with patch.dict(os.environ, {"FIREWORKS_API_KEY": "dummy"}):
            faithfulness, relevancy = build_metrics()
        self.assertTrue(relevancy.embeddings.is_async)
        self.assertEqual(relevancy.embeddings.model, "nomic-ai/nomic-embed-text-v1.5")
        self.assertEqual(faithfulness.llm.model, "accounts/fireworks/models/glm-5p3")
        self.assertEqual(faithfulness.llm.model_args["max_tokens"], 16384)

        with patch("ragas.embeddings.openai_provider.OpenAIEmbeddings.aembed_text",
                   new_callable=AsyncMock) as embed_text:
            asyncio.run(relevancy.embeddings.aembed_text("Revenue growth?"))
        embed_text.assert_awaited_once_with("search_query: Revenue growth?")

    def test_official_metric_adapters_receive_saved_answer_and_context(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "evaluation").mkdir()
            (root / "evaluation/dev_run.json").write_text(json.dumps({
                "run_id": "run-test", "questions": [{"id": "q1", "question": "Did revenue rise?",
                 "answer": "Revenue rose [S1].", "sources": [{"id": "S1", "type": "pdf",
                 "filename": "filing.pdf", "text": "Revenue rose."}]}]}))
            faithfulness, relevancy = Faithfulness(), ResponseRelevancy()
            result = evaluate_saved_answers(root, metrics=(faithfulness, relevancy), pilot=False)
            self.assertEqual(result["metric_implementation"], "ragas.metrics.collections")
            self.assertEqual(result["summary"]["faithfulness"], {"mean": 0.75, "scored": 1})
            self.assertEqual(result["summary"]["response_relevancy"], {"mean": 0.9, "scored": 1})
            self.assertIn("Revenue rose.", faithfulness.calls[0]["retrieved_contexts"][0])
            self.assertEqual(relevancy.calls[0], {"user_input": "Did revenue rise?",
                                                   "response": "Revenue rose [S1]."})
            self.assertEqual(load_latest(root)["source_run_id"], "run-test")

    def test_pilot_uses_two_saved_questions_and_unlocks_full_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "evaluation").mkdir()
            records = [{"id": question_id, "question": question_id,
                        "answer": "Revenue rose [S1].", "sources": [{"id": "S1", "type": "pdf",
                        "filename": "filing.pdf", "text": "Revenue rose."}]}
                       for question_id in ("q_001", "q_012", "q_008")]
            (root / "evaluation/dev_run.json").write_text(json.dumps({
                "run_id": "run-pilot", "questions": records}))
            self.assertFalse(pilot_ready(root))
            result = evaluate_saved_answers(root, metrics=(Faithfulness(), ResponseRelevancy()))
            self.assertEqual([row["id"] for row in result["rows"]], ["q_001", "q_012"])
            self.assertTrue(pilot_ready(root))


if __name__ == "__main__":
    unittest.main()
