"""Evaluation metrics must separate measured facts from optional model judgments."""

import json
import unittest

from eval_pipeline.generation_metrics import judge_record, measured_metrics
from eval_pipeline.generation_metrics import _parse_review


class Provider:
    def complete(self, messages, tools, *, max_tokens):
        assert "expected_answer" in messages[1]["content"]
        return {"tool_calls": [{"function": {"name": "submit_evaluation", "arguments": json.dumps({
            "status": "partial", "coverage": "partial", "faithfulness": "pass",
            "relevance": "pass", "citation_support": "pass",
            "findings": ["The comparison is missing one year."]})}}]}


class AnswerMetricTests(unittest.TestCase):
    def test_measured_metrics_do_not_call_citation_presence_support(self):
        row = {"answer": "Supported [S1] but unknown [S9]", "answer_status": "answered",
               "sources": [{"id": "S1", "type": "sql"}], "source_coverage": True,
               "elapsed_seconds": 2.4,
               "trace": [{"tool": "query_financials"}, {"tool": "calculate"}],
               "model_diagnostics": [{"prompt_tokens": 10, "completion_tokens": 8},
                                     {"prompt_tokens": 5, "completion_tokens": 3, "retry": True}]}
        metrics = measured_metrics(row)
        self.assertEqual(metrics["citation_validity"], .5)
        self.assertEqual(metrics["cited_source_count"], 2)
        self.assertEqual(metrics["input_tokens"], 15)
        self.assertEqual(metrics["output_tokens"], 11)
        self.assertEqual(metrics["tool_calls"], 2)
        self.assertEqual(metrics["retries"], 1)

    def test_judge_is_structured_and_evidence_aware(self):
        row = {"question": "What changed?", "answer": "It rose [S1].",
               "sources": [{"id": "S1", "type": "pdf", "filename": "filing.pdf",
                            "pages": [2], "text": "It rose."}]}
        review = judge_record(row, "It rose in both years.", provider=Provider())
        self.assertEqual(review["status"], "partial")
        self.assertEqual(review["faithfulness"], "pass")

    def test_benchmark_judge_receives_cited_passage_beyond_review_excerpt(self):
        class CapturingProvider(Provider):
            def complete(self, messages, tools, *, max_tokens):
                self.evidence = json.loads(messages[1]["content"])["cited_evidence"]
                return super().complete(messages, tools, max_tokens=max_tokens)

        provider = CapturingProvider()
        passage = "Earlier text. " * 300 + "Revenue $120,810 million."
        row = {"question": "What was revenue?", "answer": "$120,810 million [S1].",
               "sources": [{"id": "S1", "type": "pdf", "text": passage}]}
        judge_record(row, "$120,810 million", provider=provider)
        self.assertIn("Revenue $120,810 million", provider.evidence[0]["excerpt"])

    def test_judge_failure_records_validation_stage_without_answer_text(self):
        class InvalidProvider:
            def complete(self, messages, tools, *, max_tokens):
                return {"tool_calls": [{"function": {
                    "name": "submit_evaluation", "arguments": '{"status":"correct"}'}}]}

        row = {"id": "q_025", "question": "Services?", "answer": "Private answer [S1]",
               "sources": [{"id": "S1", "type": "sql", "text": "Revenue"}]}
        with self.assertLogs("eval_pipeline.generation_metrics", level="WARNING") as logs:
            review = judge_record(row, "Expected", provider=InvalidProvider())
        self.assertEqual(review["status"], "unavailable")
        self.assertEqual(review["error_stage"], "retry_parse")
        self.assertEqual(review["error_detail"], "Judge returned invalid dimension ratings")
        self.assertNotIn("Private answer", "\n".join(logs.output))

    def test_optional_findings_shape_does_not_discard_valid_scores(self):
        ratings = {"status": "partial", "coverage": "partial", "faithfulness": "pass",
                   "relevance": "pass", "citation_support": "pass"}
        review = _parse_review({"content": json.dumps({**ratings, "findings": "Missing detail."})})
        self.assertEqual(review["findings"], ["Missing detail."])
        long_findings = ["detail " * 100] * 6
        review = _parse_review({"content": json.dumps({**ratings, "findings": long_findings})})
        self.assertEqual(len(review["findings"]), 5)
        self.assertTrue(all(len(item) <= 300 for item in review["findings"]))
        with self.assertRaisesRegex(ValueError, "without findings"):
            _parse_review({"content": json.dumps(ratings)})


if __name__ == "__main__":
    unittest.main()
