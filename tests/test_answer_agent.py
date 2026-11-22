"""Exercise agent tool routing and citation handling without paid API calls."""

import importlib.util
import json
import unittest
from types import SimpleNamespace

from src.answer_agent.graph import (_finish, _has_unverified_numeric_claim,
                                    _normalize_citations, _retry_messages,
                                    answer_question, build_graph)
from src.answer_agent.tools.execute import _execute_calls
from src.answer_agent.tools.pages import read_pages
from src.answer_agent.prompts.system import system_prompt
from src.answer_agent.provider import FireworksChatProvider
from src.answer_agent.graph import validate_research_plan
from src.answer_agent.checks import _format_calculated_precision


class FakeProvider:
    model = "fake-test-provider"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []
        self.tool_sets = []

    def complete(self, messages, tools):
        self.messages.append(messages)
        self.tool_sets.append(tools)
        if tools:
            assert {tool["function"]["name"] for tool in tools} in (
                {"query_financials", "search_filings", "read_pages", "calculate"},
                {"query_financials", "search_filings", "read_pages", "calculate",
                 "submit_research_plan"})
        return next(self.responses)


def tool_call(identifier, name, arguments):
    return {"id": identifier, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


class AnswerAgentTests(unittest.TestCase):
    def test_empty_answer_retry_selects_relevant_middle_evidence(self):
        sources = [{"id": f"S{index}", "type": "pdf", "text": "Unrelated filing text"}
                   for index in range(1, 15)]
        sources[6]["text"] = "Cybersecurity response measures and incident controls."
        state = {"question": "What cybersecurity response measures are disclosed?",
                 "draft": "", "sources": sources,
                 "messages": [{"role": "system", "content": "Rules"}],
                 "prior_messages": [], "trace": []}
        retry = _retry_messages(state)[-1]["content"]
        self.assertIn("[S7]", retry)
        self.assertNotIn("[S14]", retry)

    def test_direct_reported_value_is_not_blocked_by_incorrect_plan(self):
        state = {"draft": "Revenue was $416.161 billion [S1].",
                 "sources": [{"id": "S1", "type": "sql", "rows": [{"revenue": 416161000000}]}],
                 "trace": [], "pending_calls": [], "steps": 2,
                 "research_plan": {"needs_calculation": True, "calculation_type": "grouped_change"}}
        self.assertEqual(_finish(state, 8)["answer_status"], "answered")

    def test_each_derived_number_needs_matching_calculation(self):
        sources = [{"id": "S1", "type": "sql", "rows": [{"prior": 80, "current": 100}]}]
        state = {"draft": "Revenue rose 25% and margin rose 5% [S1].", "sources": sources,
                 "trace": [{"tool": "calculate", "source_ids": ["S1"],
                            "result": {"operation": "percent_change", "result": "25"}}],
                 "pending_calls": [], "steps": 2, "research_plan": None}
        self.assertEqual(_finish(state, 8)["answer_status"], "calculation_failed")
        state["trace"].append({"tool": "calculate", "source_ids": ["S1"],
                               "result": {"operation": "percentage", "result": "5"}})
        self.assertEqual(_finish(state, 8)["answer_status"], "answered")

    def test_ratio_needs_cited_calculator_result_unless_reported_directly(self):
        state = {"draft": "The ratio was 2.01x [S1].",
                 "sources": [{"id": "S1", "type": "sql", "rows": [{"income": 201, "cost": 100}]}],
                 "trace": [], "pending_calls": [], "steps": 2}
        self.assertEqual(_finish(state, 8)["answer_status"], "calculation_failed")
        state["trace"] = [{"tool": "calculate", "source_ids": ["S1"],
                           "result": {"operation": "ratio", "result": "2.01"}}]
        self.assertEqual(_finish(state, 8)["answer_status"], "answered")
        state["trace"] = []
        state["sources"][0]["rows"][0]["reported_ratio"] = 2.01
        self.assertEqual(_finish(state, 8)["answer_status"], "answered")

    def test_cited_ratio_can_be_expressed_as_a_percentage(self):
        sources = [{"id": "S1", "type": "sql", "rows": [{"part": 22, "whole": 100}]}]
        trace = [{"tool": "calculate", "source_ids": ["S1"],
                  "result": {"operation": "ratio", "result": "0.22"}}]
        state = {"draft": "The share was 22% [S1].", "sources": sources,
                 "trace": trace, "pending_calls": [], "steps": 2}
        self.assertEqual(_finish(state, 8)["answer_status"], "answered")
        state["draft"] = "The share was 23% [S1]."
        self.assertEqual(_finish(state, 8)["answer_status"], "calculation_failed")
        state["draft"] = "The share was 22% without a citation."
        self.assertEqual(_finish(state, 8)["answer_status"], "citation_failed")

    def test_same_number_in_wrong_sql_unit_does_not_support_claim(self):
        state = {"draft": "Revenue was $2 billion [S1].",
                 "sources": [{"id": "S1", "type": "sql", "rows": [{"revenue": 2}]}],
                 "trace": [], "pending_calls": [], "steps": 2}
        self.assertEqual(_finish(state, 8)["answer_status"], "calculation_failed")

    def test_signed_derived_result_requires_matching_sign(self):
        state = {"draft": "Revenue declined -3.8% [S1].",
                 "sources": [{"id": "S1", "type": "sql", "rows": [{"prior": 100, "current": 96.2}]}],
                 "trace": [{"tool": "calculate", "source_ids": ["S1"],
                            "result": {"operation": "percent_change", "result": "-3.8"}}],
                 "pending_calls": [], "steps": 2}
        self.assertEqual(_finish(state, 8)["answer_status"], "answered")
        state["draft"] = "Revenue increased 3.8% [S1]."
        self.assertEqual(_finish(state, 8)["answer_status"], "calculation_failed")

    def test_long_raw_calculator_decimal_uses_display_rounding(self):
        trace = [{"tool": "calculate", "result": {"operation": "percentage",
                  "result": "51.30766281892916471440132896", "rounded_1": "51.3"}}]
        answer = "The share was 51.30766281892916471440132896% [S1]."
        self.assertEqual(_format_calculated_precision(answer, trace),
                         "The share was 51.3% [S1].")
        self.assertEqual(_format_calculated_precision("The filing reported 51.30766% [S1].", trace),
                         "The filing reported 51.30766% [S1].")

    def test_multi_source_calculation_needs_every_operand_cited(self):
        state = {"draft": "The share was 25% [S1].",
                 "sources": [{"id": "S1", "type": "sql", "rows": [{"part": 25}]},
                             {"id": "S2", "type": "sql", "rows": [{"whole": 100}]}],
                 "trace": [{"tool": "calculate", "source_ids": ["S1", "S2"],
                            "result": {"operation": "percentage", "result": "25"}}],
                 "pending_calls": [], "steps": 2}
        self.assertEqual(_finish(state, 8)["answer_status"], "calculation_failed")
        state["draft"] = "The share was 25% [S1][S2]."
        self.assertEqual(_finish(state, 8)["answer_status"], "answered")

    def test_unplanned_numeric_fallback_accepts_sourced_value_and_year(self):
        sources = [{"id": "S1", "type": "sql", "rows": [{"revenue": 416161000000,
                                                          "fiscal_year": 2025}]}]
        draft = "Apple's FY2025 revenue was $416.161 billion [S1]."
        self.assertFalse(_has_unverified_numeric_claim(draft, sources))
        state = {"draft": draft, "sources": sources, "trace": [], "pending_calls": [],
                 "steps": 2, "research_plan": None}
        self.assertEqual(_finish(state, 8)["answer_status"], "answered")

    def test_unplanned_numeric_fallback_withholds_unsourced_change(self):
        sources = [{"id": "S1", "type": "sql", "rows": [{"earlier": 66952000000,
                                                          "later": 64377000000}]}]
        draft = "Greater China declined by $2.575 billion, or 3.8% [S1]."
        self.assertTrue(_has_unverified_numeric_claim(draft, sources))
        state = {"draft": draft, "sources": sources, "trace": [], "pending_calls": [],
                 "steps": 2, "research_plan": None}
        self.assertEqual(_finish(state, 8)["answer_status"], "calculation_failed")

    def test_research_plan_validates_schema_and_calculation_consistency(self):
        plan = {"needs_sql": True, "needs_pdf": False, "needs_calculation": True,
                "calculation_type": "grouped_change", "required_facts": ["Segment changes"]}
        self.assertEqual(validate_research_plan(json.dumps(plan)), plan)
        with self.assertRaises(ValueError):
            validate_research_plan({**plan, "calculation_type": None})
        with self.assertRaises(ValueError):
            validate_research_plan({**plan, "required_facts": []})
        with self.assertRaises(ValueError):
            validate_research_plan("not JSON")

    @unittest.skipUnless(importlib.util.find_spec("langgraph"), "LangGraph dependency not installed")
    def test_plan_and_pdf_can_be_selected_on_first_turn(self):
        provider = FakeProvider([
            {"role": "assistant", "content": "", "tool_calls": [
                tool_call("plan", "submit_research_plan", {
                    "needs_sql": False, "needs_pdf": True, "needs_calculation": False,
                    "calculation_type": None, "required_facts": ["Disclosed supplier risk"]}),
                tool_call("pdf", "search_filings", {"query": "supplier risk", "ticker": "AAPL"})]},
            {"role": "assistant", "content": "The filing describes supplier risk [S1].", "tool_calls": []},
        ])
        result = answer_question("What supplier risk does Apple disclose?", provider=provider,
                                 schema={"tables": {}},
                                 sql_tool=lambda _: self.fail("SQL must not be forced"),
                                 pdf_search=lambda query, **kwargs: {"results": [{
                                     "id": "risk", "ticker": "AAPL", "fiscal_year": 2025,
                                     "filename": "AAPL_FY2025_10-K.pdf", "pages": [6],
                                     "text": "The company faces supplier risk."}]})
        self.assertEqual(result["answer_status"], "answered")
        self.assertEqual(result["plan_status"], "valid")
        self.assertEqual([entry["tool"] for entry in result["trace"]], ["search_filings"])
        self.assertEqual(len(provider.messages), 2)

    @unittest.skipUnless(importlib.util.find_spec("langgraph"), "LangGraph dependency not installed")
    def test_malformed_plan_does_not_block_evidence_tool(self):
        provider = FakeProvider([
            {"role": "assistant", "content": "", "tool_calls": [
                tool_call("plan", "submit_research_plan", {"needs_sql": "yes"}),
                tool_call("sql", "query_financials", {"sql": "SELECT 42 AS value"})]},
            {"role": "assistant", "content": "The value is 42 [S1].", "tool_calls": []},
        ])
        result = answer_question("What is the value?", provider=provider,
                                 schema={"tables": {}},
                                 sql_tool=lambda sql: {"sql": sql, "columns": ["value"],
                                                       "rows": [{"value": 42}]},
                                 pdf_search=lambda query, **kwargs: {"results": []})
        self.assertEqual(result["answer_status"], "answered")
        self.assertEqual(result["plan_status"], "invalid")
        self.assertIsNone(result["research_plan"])

    @unittest.skipUnless(importlib.util.find_spec("langgraph"), "LangGraph dependency not installed")
    def test_unplanned_derived_percent_requests_calculator_recovery(self):
        provider = FakeProvider([
            {"role": "assistant", "content": "", "tool_calls": [
                tool_call("sql", "query_financials", {"sql": "SELECT 100 AS newer, 80 AS older"})]},
            {"role": "assistant", "content": "Revenue grew 25% [S1].", "tool_calls": []},
            {"role": "assistant", "content": "", "tool_calls": [
                tool_call("calc", "calculate", {"operation": "percent_change", "operands": [
                    {"source_id": "S1", "row": 0, "field": "newer"},
                    {"source_id": "S1", "row": 0, "field": "older"}]})]},
            {"role": "assistant", "content": "Revenue grew 25% [S1].", "tool_calls": []},
        ])
        result = answer_question("How much did revenue grow?", provider=provider,
                                 schema={"tables": {}},
                                 sql_tool=lambda sql: {"sql": sql, "columns": ["newer", "older"],
                                                       "rows": [{"newer": 100, "older": 80}]},
                                 pdf_search=lambda query, **kwargs: {"results": []})
        self.assertEqual(result["answer_status"], "answered")
        self.assertIn("Use calculate with sourced operands", provider.messages[2][-1]["content"])
        self.assertEqual(result["trace"][-1]["tool"], "calculate")

    @unittest.skipUnless(importlib.util.find_spec("langgraph"), "LangGraph dependency not installed")
    def test_agent_drafts_when_evidence_cap_is_reached(self):
        provider = FakeProvider([
            {"role": "assistant", "content": "", "tool_calls": [tool_call(
                "c1", "query_financials", {"sql": "SELECT 42 AS value"})]},
            {"role": "assistant", "content": "The value is 42 [S1].", "tool_calls": []},
        ])
        graph = build_graph(provider, lambda sql: {"sql": sql, "columns": ["value"],
                                                   "rows": [{"value": 42}]},
                            lambda query, **kwargs: {"results": []}, max_sources=1)
        result = graph.invoke({"question": "What is the value?", "prior_messages": [],
                               "messages": [{"role": "system", "content": "Test"},
                                            {"role": "user", "content": "What is the value?"}],
                               "sources": [], "trace": [], "pending_calls": [], "draft": "", "answer": "",
                               "answer_status": "", "steps": 0, "needs_revision": False,
                               "model_diagnostics": [], "failure_reason": None})
        self.assertEqual(provider.tool_sets[1], [])
        self.assertEqual(result["answer_status"], "answered")

    @unittest.skipUnless(importlib.util.find_spec("langgraph"), "LangGraph dependency not installed")
    def test_followup_sees_prior_turn_but_cites_new_evidence(self):
        provider = FakeProvider([
            {"role": "assistant", "content": "", "tool_calls": [tool_call(
                "c1", "query_financials", {"sql": "SELECT 42 AS change"})]},
            {"role": "assistant", "content": "It rose by 42 [S1].", "tool_calls": []},
        ])
        history = [{"role": "user", "content": "What was Apple's revenue?"},
                   {"role": "assistant", "content": "It was 100."}]
        result = answer_question("How much did it grow?", history=history, provider=provider,
                                 schema={"tables": {}},
                                 sql_tool=lambda sql: {"sql": sql, "columns": ["change"],
                                                       "rows": [{"change": 42}]},
                                 pdf_search=lambda query, **kwargs: {"results": []})
        self.assertEqual(result["answer_status"], "answered")
        self.assertEqual(provider.messages[0][1:4], [*history,
            {"role": "user", "content": "How much did it grow?"}])
        self.assertEqual(result["sources"][0]["id"], "S1")

    def test_deepseek_gets_larger_first_response_budget_and_diagnostics(self):
        options = []
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Answer", tool_calls=[]),
                                     finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20,
                                  completion_tokens_details=SimpleNamespace(reasoning_tokens=5)))
        provider = FireworksChatProvider.__new__(FireworksChatProvider)
        provider.model = "accounts/fireworks/models/deepseek-v4p1-flash"
        provider.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: (options.append(kwargs), response)[1])))
        result = provider.complete([{"role": "user", "content": "Question"}], [])
        self.assertEqual(options[0]["max_tokens"], 3000)
        self.assertEqual(result["diagnostics"]["reasoning_tokens"], 5)

    def test_tool_results_have_distinct_inspectable_sources(self):
        calls = [tool_call("call-1", "query_financials", {"sql": "SELECT revenue FROM income_statements"}),
                 tool_call("call-2", "search_filings", {"query": "Greater China", "ticker": "AAPL", "fiscal_year": 2025})]
        state = {"messages": [{"role": "assistant", "tool_calls": calls}],
                 "sources": [], "trace": [], "pending_calls": calls}

        def sql(_):
            return {"sql": "SELECT revenue FROM income_statements", "columns": ["revenue"],
                    "rows": [{"revenue": 100}], "source": {"database": "financials.db"}}

        def pdf(query, ticker=None, fiscal_year=None):
            self.assertEqual((query, ticker, fiscal_year), ("Greater China", "AAPL", 2025))
            return {"results": [{"id": "chunk-1", "ticker": "AAPL", "fiscal_year": 2025,
                                 "filename": "AAPL_FY2025_10-K.pdf", "pages": [24],
                                 "text": "Greater China sales declined due to iPhone."}]}

        update = _execute_calls(state, sql, pdf, 10)
        self.assertEqual([source["id"] for source in update["sources"]], ["S1", "S2"])
        self.assertEqual(update["sources"][0]["type"], "sql")
        self.assertEqual(update["sources"][1]["pages"], [24])
        self.assertEqual([message["role"] for message in update["messages"][-2:]], ["tool", "tool"])
        self.assertEqual([row["source_ids"] for row in update["trace"]], [["S1"], ["S2"]])

    def test_short_table_hit_opens_original_page_for_margin_values(self):
        call = tool_call("margin", "search_filings", {"query": "Services gross margin"})
        preview = ("Gross Margin\nProducts and Services gross margin and gross margin percentage "
                   "for 2025, 2024 and 2023 were as follows (dollars in millions):")
        def pdf(_query, **_kwargs):
            return {"results": [{"id": "margin-heading", "ticker": "AAPL", "fiscal_year": 2025,
                                 "filename": "AAPL_FY2025_10-K.pdf", "pages": [25], "text": preview}]}
        result = _execute_calls({"messages": [], "sources": [], "trace": [],
                                 "pending_calls": [call]}, None, pdf, 5)
        self.assertIn("75.4%", result["sources"][0]["text"])
        self.assertIn("36.8%", result["messages"][-1]["content"])
        self.assertEqual(result["trace"][0]["expanded_pages"][0]["page"], 25)

    def test_short_passage_expands_without_table_phrasing(self):
        from src.answer_agent.tools.pages import expand_table_preview
        hit = {"text": "The results are summarized here.", "pages": [4],
               "filename": "AAPL_FY2025_10-K.pdf", "anchors": [{"kind": "paragraph"}]}
        text, expanded = expand_table_preview(hit, lambda filename, page: {
            "text": "The results are summarized here. Revenue increased to $10 billion."})
        self.assertTrue(expanded)
        self.assertIn("$10 billion", text)

    def test_repeated_identical_filing_search_reuses_result(self):
        calls = [tool_call("first", "search_filings", {"query": "Apple supply risk"}),
                 tool_call("second", "search_filings", {"query": "Apple supply risk"})]
        searches = []
        def pdf(query, **kwargs):
            searches.append(query)
            return {"results": [{"id": "chunk-1", "ticker": "AAPL", "fiscal_year": 2025,
                                 "filename": "AAPL_FY2025_10-K.pdf", "pages": [10],
                                 "text": "Supply-chain risk."}]}
        result = _execute_calls({"messages": [], "sources": [], "trace": [],
                                 "pending_calls": calls}, None, pdf, 10)
        self.assertEqual(searches, ["Apple supply risk"])
        self.assertEqual(len(result["sources"]), 1)
        self.assertTrue(result["trace"][1]["cached"])
        self.assertEqual(result["messages"][0]["content"], result["messages"][1]["content"])

    def test_tool_call_budget_returns_a_response_for_each_requested_call(self):
        calls = [tool_call("first", "query_financials", {"sql": "SELECT 1"}),
                 tool_call("second", "query_financials", {"sql": "SELECT 2"})]
        queries = []
        def sql(query):
            queries.append(query)
            return {"sql": query, "columns": ["n"], "rows": [{"n": 1}]}
        result = _execute_calls({"messages": [], "sources": [], "trace": [],
                                 "pending_calls": calls}, sql, None, 10,
                                max_tool_calls=1)
        self.assertEqual(queries, ["SELECT 1"])
        self.assertEqual(len(result["messages"]), 2)
        self.assertIn("Tool call limit", result["trace"][1]["error"])

    def test_uncited_or_unknown_source_answer_is_withheld(self):
        state = {"draft": "Revenue was 100.", "sources": [{"id": "S1"}],
                 "pending_calls": [], "steps": 2}
        self.assertIn("valid source citations", _finish(state, 6)["answer"])
        state["draft"] = "Revenue was 100 [S2]."
        self.assertIn("valid source citations", _finish(state, 6)["answer"])
        state["draft"] = "Revenue was 100 [S1]."
        self.assertEqual(_finish(state, 6)["answer"], state["draft"])

    def test_normalizes_page_and_range_citations(self):
        self.assertEqual(_normalize_citations("Pages 【S1‑S2】 and [S5, p. 5]."),
                         "Pages [S1][S2] and [S5] (p. 5).")
        self.assertEqual(_normalize_citations("Evidence [\u200bS2][\ufeffS3]."),
                         "Evidence [S2][S3].")
        self.assertEqual(_normalize_citations("Evidence [S1\u200b] and [S\u200c2]."),
                         "Evidence [S1] and [S2].")
        self.assertEqual(_normalize_citations("Derived values [S1, S2; calculate]."),
                         "Derived values [S1][S2].")

    @unittest.skipUnless(importlib.util.find_spec("langgraph"), "LangGraph dependency not installed")
    def test_empty_model_reply_retries_once_with_compact_evidence(self):
        provider = FakeProvider([
            {"role": "assistant", "content": "", "tool_calls": [
                tool_call("c1", "query_financials", {"sql": "SELECT 42 AS revenue"})]},
            {"role": "assistant", "content": "", "tool_calls": [],
             "diagnostics": {"finish_reason": "length", "completion_tokens": 2000}},
            {"role": "assistant", "content": "Revenue was 42 [S1].", "tool_calls": [],
             "diagnostics": {"finish_reason": "stop", "completion_tokens": 20}},
        ])
        result = answer_question("Revenue?", provider=provider, schema={"tables": {}},
                                 sql_tool=lambda query: {"sql": query, "columns": ["revenue"],
                                                         "rows": [{"revenue": 42}]},
                                 pdf_search=lambda query, **kwargs: {"results": []})
        self.assertEqual(result["answer_status"], "answered")
        self.assertIn("42 [S1]", result["answer"])
        self.assertTrue(result["model_diagnostics"][-1]["retry"])
        self.assertEqual(result["model_diagnostics"][-2]["finish_reason"], "length")
        self.assertEqual(len(provider.messages[-1]), 3)

    def test_answer_length_changes_format_rule(self):
        concise = system_prompt({"tables": {}}, "concise")
        detailed = system_prompt({"tables": {}}, "detailed")
        self.assertIn("120 words", concise)
        self.assertIn("350 words", detailed)
        self.assertIn("Cite evidence next to each factual claim", concise)
        self.assertIn("Cite evidence next to each factual claim", detailed)

    def test_original_page_recovers_values_missing_from_chunk(self):
        result = read_pages("GOOGL_FY2025_10-K.pdf", 36)
        self.assertIn("36,147", result["text"])
        self.assertIn("40,367", result["text"])
        with self.assertRaises(ValueError):
            read_pages("../.env", 1)
        with self.assertRaises(ValueError):
            read_pages("GOOGL_FY2025_10-K.pdf", 36, 39)

    def test_page_reader_can_supply_agent_evidence(self):
        call = tool_call("page-1", "read_pages", {"filename": "GOOGL_FY2025_10-K.pdf", "page": 36})
        result = _execute_calls({"messages": [], "sources": [], "trace": [],
                                 "pending_calls": [call]}, None, None, 10)
        self.assertEqual(result["trace"][0]["tool"], "read_pages")
        self.assertEqual(result["sources"][0]["pages"], [36])
        self.assertIn("40,367", result["sources"][0]["text"])

    def test_cited_change_answer_does_not_force_another_search(self):
        state = {"question": "How did YouTube ads revenue change between FY2024 and FY2025?",
                 "draft": "It rose $4.2 billion [S1].", "sources": [{"id": "S1", "type": "sql",
                                                               "rows": [{"change_billions": 4.2}]}],
                 "pending_calls": [], "steps": 3}
        self.assertEqual(_finish(state, 8)["answer_status"], "answered")

    @unittest.skipUnless(importlib.util.find_spec("langgraph"), "LangGraph dependency not installed")
    def test_graph_handles_mixed_evidence_and_multi_step_tools(self):
        responses = [
            {"role": "assistant", "content": "", "tool_calls": [
                tool_call("c1", "query_financials", {"sql": "SELECT 100 AS revenue FROM income_statements LIMIT 1"})]},
            {"role": "assistant", "content": "", "tool_calls": [
                tool_call("c2", "search_filings", {"query": "Greater China", "ticker": "AAPL"})]},
            {"role": "assistant", "content": "Revenue was $100 [S1]. The filing gives context [S2].",
             "tool_calls": []},
        ]
        provider = FakeProvider(responses)
        progress = []
        result = answer_question("What happened?", provider=provider,
                                 schema={"tables": {"income_statements": ["revenue"]}},
                                 sql_tool=lambda query: {"sql": query, "columns": ["revenue"],
                                                         "rows": [{"revenue": 100}]},
                                 pdf_search=lambda query, **kwargs: {"results": [{
                                     "id": "c", "ticker": "AAPL", "fiscal_year": 2025,
                                     "filename": "AAPL_FY2025_10-K.pdf", "pages": [24],
                                     "text": "Greater China declined."}]}, progress=progress.append)
        self.assertIn("[S1]", result["answer"])
        self.assertEqual(result["answer_status"], "answered")
        self.assertEqual([source["type"] for source in result["sources"]], ["sql", "pdf"])
        self.assertEqual([entry["tool"] for entry in result["trace"]],
                         ["query_financials", "search_filings"])
        self.assertEqual(len(provider.messages), 3)
        self.assertIn("sql", [item["stage"] for item in progress])
        self.assertIn("filings", [item["stage"] for item in progress])
        starts = {item["id"] for item in progress if item["phase"] == "start"}
        completions = {item["id"] for item in progress if item["phase"] == "complete"}
        self.assertEqual(starts, completions)
        self.assertEqual(len(starts), 5)  # Three model calls and two evidence tools.
        self.assertTrue(all(item["duration_ms"] >= 0 for item in progress
                            if item["phase"] == "complete"))

    @unittest.skipUnless(importlib.util.find_spec("langgraph"), "LangGraph dependency not installed")
    def test_graph_caps_repeated_tool_requests(self):
        class Repeater:
            model = "fake-test-provider"

            def complete(self, messages, tools):
                return {"role": "assistant", "content": "", "tool_calls": [
                    tool_call("repeat", "query_financials", {"sql": "SELECT 1 FROM companies"})]}

        result = answer_question("Check companies", provider=Repeater(),
                                 schema={"tables": {"companies": []}}, max_steps=2,
                                 sql_tool=lambda query: {"sql": query, "columns": ["1"], "rows": [{"1": 1}]},
                                 pdf_search=lambda query, **kwargs: {"results": []})
        self.assertIn("step limit", result["answer"])
        self.assertEqual(len(result["trace"]), 1)

    @unittest.skipUnless(importlib.util.find_spec("langgraph"), "LangGraph dependency not installed")
    def test_graph_requests_citation_repair_after_uncited_draft(self):
        provider = FakeProvider([
            {"role": "assistant", "content": "", "tool_calls": [
                tool_call("c1", "query_financials", {"sql": "SELECT revenue FROM income_statements LIMIT 1"})]},
            {"role": "assistant", "content": "Revenue was 100.", "tool_calls": []},
            {"role": "assistant", "content": "Revenue was 100 【S1】.", "tool_calls": []},
        ])
        result = answer_question("Revenue?", provider=provider, schema={"tables": {}},
                                 sql_tool=lambda query: {"sql": query, "columns": ["revenue"],
                                                         "rows": [{"revenue": 100}]},
                                 pdf_search=lambda query, **kwargs: {"results": []})
        self.assertEqual(result["answer"], "Revenue was 100 [S1].")
        self.assertEqual(result["answer_status"], "answered")
        self.assertIn("exact citations", provider.messages[-1][-1]["content"])
        self.assertNotIn("tool_calls", provider.messages[-1][-2])
        self.assertEqual({tool["function"]["name"] for tool in provider.tool_sets[0]},
                         {"query_financials", "search_filings", "read_pages", "calculate",
                          "submit_research_plan"})

    @unittest.skipUnless(importlib.util.find_spec("langgraph"), "LangGraph dependency not installed")
    def test_mixed_question_requests_filing_before_final_answer(self):
        provider = FakeProvider([
            {"role": "assistant", "content": "", "tool_calls": [
                tool_call("plan", "submit_research_plan", {
                    "needs_sql": True, "needs_pdf": True, "needs_calculation": False,
                    "calculation_type": None, "required_facts": ["Services revenue", "10-K explanation"]}),
                tool_call("c1", "query_financials", {"sql": "SELECT revenue FROM segment_revenue"})]},
            {"role": "assistant", "content": "Services grew [S1].", "tool_calls": []},
            {"role": "assistant", "content": "", "tool_calls": [
                tool_call("c2", "search_filings", {"query": "Apple Services strategy"})]},
            {"role": "assistant", "content": "Services grew [S1]. The filing discusses strategy [S2].",
             "tool_calls": []},
        ])
        result = answer_question("What does Apple's 10-K say about Services revenue growth?",
                                 provider=provider, schema={"tables": {}},
                                 sql_tool=lambda query: {"sql": query, "columns": ["revenue"],
                                                         "rows": [{"revenue": 100}]},
                                 pdf_search=lambda query, **kwargs: {"results": [{
                                     "id": "p1", "ticker": "AAPL", "fiscal_year": 2025,
                                     "filename": "AAPL_FY2025_10-K.pdf", "pages": [25],
                                     "text": "Services strategy."}]})
        self.assertEqual(result["answer_status"], "answered")
        self.assertEqual([source["type"] for source in result["sources"]], ["sql", "pdf"])
        self.assertIn("Search the relevant 10-K", provider.messages[2][-1]["content"])
        self.assertEqual(result["plan_status"], "valid")
        self.assertEqual(result["research_plan"]["required_facts"],
                         ["Services revenue", "10-K explanation"])

    @unittest.skipUnless(importlib.util.find_spec("langgraph"), "LangGraph dependency not installed")
    def test_graph_reports_citation_failure_after_repair_limit(self):
        provider = FakeProvider([
            {"role": "assistant", "content": "", "tool_calls": [
                tool_call("c1", "query_financials", {"sql": "SELECT revenue FROM income_statements LIMIT 1"})]},
            {"role": "assistant", "content": "Revenue was 100.", "tool_calls": []},
            {"role": "assistant", "content": "Revenue was 100.", "tool_calls": []},
        ])
        result = answer_question("Revenue?", provider=provider, schema={"tables": {}}, max_steps=3,
                                 sql_tool=lambda query: {"sql": query, "columns": ["revenue"],
                                                         "rows": [{"revenue": 100}]},
                                 pdf_search=lambda query, **kwargs: {"results": []})
        self.assertEqual(result["answer_status"], "citation_failed")
        self.assertEqual(result["draft"], "Revenue was 100.")
        self.assertEqual(provider.tool_sets[-1], [])
        self.assertIn("final response", provider.messages[-1][-1]["content"])


if __name__ == "__main__":
    unittest.main()
