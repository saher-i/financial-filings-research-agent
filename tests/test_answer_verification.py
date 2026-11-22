"""Review is bounded, source-aware, and separate from the agent's research loop."""

import importlib.util
import json
import unittest

from src.answer_agent.graph import answer_question
from src.answer_agent.verification import _review_result, _source_context, review_answer


class Provider:
    model = "test-provider"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def complete(self, messages, tools, *, max_tokens=None):
        self.calls.append((messages, tools, max_tokens))
        return next(self.responses)


def call(name, arguments):
    return {"id": "call", "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments)}}


class VerificationTests(unittest.TestCase):
    def test_malformed_review_gets_one_bounded_retry(self):
        provider = Provider([
            {"role": "assistant", "content": "not JSON", "tool_calls": [],
             "diagnostics": {"finish_reason": "length"}},
            {"role": "assistant", "content": "", "tool_calls": [
                call("submit_answer_review", {"status": "supported", "findings": []})],
             "diagnostics": {"finish_reason": "tool_calls"}},
        ])
        result = review_answer("What changed?", "Revenue rose [S1].",
                               [{"id": "S1", "type": "sql", "rows": [{"revenue": 2}]}], provider)
        self.assertEqual(result["status"], "supported")
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(result["attempts"][0]["error"],
                         "Review model did not return a valid structured review")
        self.assertEqual(result["attempts"][1]["status"], "valid")

    def test_twice_malformed_review_stays_unavailable_with_diagnostics(self):
        provider = Provider([
            {"role": "assistant", "content": "bad", "tool_calls": [],
             "diagnostics": {"finish_reason": "length"}},
            {"role": "assistant", "content": "still bad", "tool_calls": [],
             "diagnostics": {"finish_reason": "stop"}},
        ])
        result = review_answer("What changed?", "Revenue rose [S1].",
                               [{"id": "S1", "type": "sql", "rows": [{"revenue": 2}]}], provider)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(result["attempts"][1]["finish_reason"], "stop")
        self.assertIn("valid structured review", result["reason"])

    @unittest.skipUnless(importlib.util.find_spec("langgraph"), "LangGraph dependency not installed")
    def test_missing_required_evidence_gets_one_targeted_graph_pass(self):
        provider = Provider([
            {"role": "assistant", "content": "", "tool_calls": [
                call("search_filings", {"query": "filing growth"})]},
            {"role": "assistant", "content": "Sales increased [S1].", "tool_calls": []},
            {"role": "assistant", "content": "", "tool_calls": [
                call("submit_answer_review", {"status": "needs_revision",
                     "findings": ["The requested management explanation is missing."],
                     "evidence_gap": "management's explanation for the sales increase"})]},
            {"role": "assistant", "content": "", "tool_calls": [
                call("search_filings", {"query": "management sales increase explanation"})]},
            {"role": "assistant", "content": "Sales increased [S1]. Management cites demand [S2].",
             "tool_calls": []},
            {"role": "assistant", "content": "", "tool_calls": [
                call("submit_answer_review", {"status": "supported", "findings": []})]},
        ])
        searches = []
        def pdf(query, **kwargs):
            searches.append(query)
            return {"results": [{"id": str(len(searches)), "ticker": "AAPL",
                                 "fiscal_year": 2025, "filename": "filing.pdf", "pages": [1],
                                 "text": "Sales increased." if len(searches) == 1
                                         else "Management cites demand."}]}
        result = answer_question("How did sales change, and why?", verify=True,
                                 provider=provider, schema={"tables": {}},
                                 sql_tool=lambda query: None, pdf_search=pdf)
        self.assertEqual(searches, ["filing growth", "management sales increase explanation"])
        self.assertEqual(result["answer_status"], "answered")
        self.assertEqual(result["verification"]["status"], "supported_after_recovery")
        self.assertEqual(result["verification"]["recovery_tool_calls"], 1)
        self.assertEqual(len(provider.calls), 6)
        self.assertNotIn("submit_research_plan",
                         {tool["function"]["name"] for tool in provider.calls[3][1]})

    @unittest.skipUnless(importlib.util.find_spec("langgraph"), "LangGraph dependency not installed")
    def test_recovery_without_new_evidence_stops_after_one_attempt(self):
        provider = Provider([
            {"role": "assistant", "content": "", "tool_calls": [
                call("search_filings", {"query": "revenue explanation"})]},
            {"role": "assistant", "content": "Revenue changed [S1].", "tool_calls": []},
            {"role": "assistant", "content": "", "tool_calls": [
                call("submit_answer_review", {"status": "needs_revision",
                     "findings": ["A requested cause lacks evidence."],
                     "evidence_gap": "cause of the revenue change"})]},
            {"role": "assistant", "content": "The cause is unavailable [S1].", "tool_calls": []},
            {"role": "assistant", "content": "", "tool_calls": [
                call("submit_answer_review", {"status": "supported", "findings": []})]},
        ])
        result = answer_question("Why did revenue change?", verify=True,
                                 provider=provider, schema={"tables": {}},
                                 sql_tool=lambda query: None,
                                 pdf_search=lambda query, **kwargs: {"results": [
                                     {"id": "one", "ticker": "AAPL", "fiscal_year": 2025,
                                      "filename": "filing.pdf", "pages": [1],
                                      "text": "Revenue changed."}]})
        self.assertEqual(result["answer_status"], "answered")
        self.assertEqual(result["verification"]["recovery_tool_calls"], 0)
        self.assertEqual(result["trace"][-2]["status"], "no_new_evidence")
        self.assertEqual(len(provider.calls), 5)

    def test_rejects_malformed_or_unsubstantiated_review(self):
        with self.assertRaisesRegex(ValueError, "invalid findings"):
            _review_result({"content": json.dumps({"status": "needs_revision", "findings": [42]})})
        with self.assertRaisesRegex(ValueError, "cannot have findings"):
            _review_result({"content": json.dumps({"status": "supported", "findings": ["missing"]})})

    def test_source_context_includes_uncited_relevant_passages(self):
        sources = [{"id": "S1", "type": "pdf", "filename": "filing.pdf", "pages": [10],
                    "text": "Component shortages."},
                   {"id": "S2", "type": "pdf", "filename": "filing.pdf", "pages": [8],
                    "text": "Natural disasters and public health emergencies."}]
        context = _source_context(sources, "Components were at risk [S1].")
        self.assertEqual([item["id"] for item in context], ["S1", "S2"])
        self.assertIn("Natural disasters", context[1]["excerpt"])

    @unittest.skipUnless(importlib.util.find_spec("langgraph"), "LangGraph dependency not installed")
    def test_review_repairs_missing_risk_category_once(self):
        provider = Provider([
            {"role": "assistant", "content": "", "tool_calls": [
                call("search_filings", {"query": "supply chain risks"})]},
            {"role": "assistant", "content": "Component shortages [S1].", "tool_calls": []},
            {"role": "assistant", "content": "", "tool_calls": [
                call("submit_answer_review", {"status": "needs_revision",
                     "findings": ["Include natural-disaster and public-health disruption risks from S2."]})]},
            {"role": "assistant", "content": "Component shortages [S1]. Natural disasters and public-health emergencies can disrupt supply [S2].", "tool_calls": []},
            {"role": "assistant", "content": "", "tool_calls": [
                call("submit_answer_review", {"status": "supported", "findings": []})]},
        ])
        result = answer_question("What supply chain risks were disclosed?", verify=True,
                                 provider=provider, schema={"tables": {}},
                                 sql_tool=lambda query: None,
                                 pdf_search=lambda query, **kwargs: {"results": [
                                     {"id": "one", "ticker": "AAPL", "fiscal_year": 2025,
                                      "filename": "filing.pdf", "pages": [10],
                                      "text": "Component shortages."},
                                     {"id": "two", "ticker": "AAPL", "fiscal_year": 2025,
                                      "filename": "filing.pdf", "pages": [8],
                                      "text": "Natural disasters and public-health emergencies."}]})
        self.assertEqual(result["answer_status"], "answered")
        self.assertIn("Natural disasters", result["answer"])
        self.assertEqual(result["verification"]["status"], "supported_after_revision")
        self.assertEqual(result["verification"]["review_calls"], 2)
        self.assertEqual([entry["tool"] for entry in result["trace"]][-3:],
                         ["answer_review", "answer_revision", "answer_review"])
        self.assertEqual(len(provider.calls), 5)

    @unittest.skipUnless(importlib.util.find_spec("langgraph"), "LangGraph dependency not installed")
    def test_unavailable_review_keeps_cited_answer_but_flags_it(self):
        provider = Provider([
            {"role": "assistant", "content": "", "tool_calls": [
                call("query_financials", {"sql": "SELECT 42 AS revenue"})]},
            {"role": "assistant", "content": "Revenue was 42 [S1].", "tool_calls": []},
            {"role": "assistant", "content": "not JSON", "tool_calls": []},
        ])
        result = answer_question("What was the revenue?", verify=True,
                                 provider=provider, schema={"tables": {}},
                                 sql_tool=lambda query: {"sql": query, "columns": ["revenue"],
                                                         "rows": [{"revenue": 42}]},
                                 pdf_search=lambda query, **kwargs: {"results": []})
        self.assertEqual(result["answer_status"], "needs_verification")
        self.assertEqual(result["answer"], "Revenue was 42 [S1].")
        self.assertEqual(result["verification"]["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
