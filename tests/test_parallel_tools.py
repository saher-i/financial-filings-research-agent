"""Independent evidence calls overlap without changing source or tool-call order."""

import json
import threading
import time
import unittest

from src.answer_agent.tools.execute import _execute_calls


def call(identifier, name, arguments):
    return {"id": identifier, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


def state(calls):
    return {"messages": [], "sources": [], "trace": [], "pending_calls": calls, "steps": 1}


class ParallelToolTests(unittest.TestCase):
    def test_independent_sql_and_pdf_overlap_but_keep_source_order(self):
        both_started = threading.Barrier(2)
        calls = [call("sql", "query_financials", {"sql": "SELECT revenue"}),
                 call("pdf", "search_filings", {"query": "margin"})]

        def sql(query):
            both_started.wait(timeout=2)
            return {"sql": query, "columns": ["revenue"], "rows": [{"revenue": 100}]}

        def pdf(query, **_kwargs):
            both_started.wait(timeout=2)
            return {"results": [{"id": "chunk", "ticker": "AAPL", "fiscal_year": 2025,
                                 "filename": "AAPL_FY2025_10-K.pdf", "pages": [4],
                                 "text": "Revenue increased."}]}

        result = _execute_calls(state(calls), sql, pdf, 8)
        self.assertEqual([source["id"] for source in result["sources"]], ["S1", "S2"])
        self.assertEqual([entry["call_id"] for entry in result["trace"]], ["sql", "pdf"])
        self.assertEqual([entry["source_ids"] for entry in result["trace"]], [["S1"], ["S2"]])
        self.assertEqual([message["tool_call_id"] for message in result["messages"]], ["sql", "pdf"])

    def test_calculation_waits_for_retrieved_evidence(self):
        calls = [call("sql", "query_financials", {"sql": "SELECT values"}),
                 call("calc", "calculate", {"operation": "difference", "operands": [
                     {"source_id": "S1", "row": 0, "field": "new"},
                     {"source_id": "S1", "row": 0, "field": "old"}]})]

        def sql(query):
            return {"sql": query, "columns": ["new", "old"],
                    "rows": [{"new": 120, "old": 100}]}

        result = _execute_calls(state(calls), sql, None, 8)
        self.assertEqual(result["trace"][1]["result"]["result"], "20")
        self.assertEqual(result["trace"][1]["source_ids"], ["S1"])

    def test_parallelism_is_bounded(self):
        calls = [call(str(i), "query_financials", {"sql": f"SELECT {i}"}) for i in range(5)]
        lock = threading.Lock()
        active = 0
        maximum = 0

        def sql(query):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.03)
            with lock:
                active -= 1
            return {"sql": query, "columns": ["n"], "rows": [{"n": 1}]}

        result = _execute_calls(state(calls), sql, None, 8, max_parallel_calls=2)
        self.assertEqual(maximum, 2)
        self.assertEqual([source["id"] for source in result["sources"]],
                         [f"S{i}" for i in range(1, 6)])

    def test_failed_parallel_call_keeps_result_mapping(self):
        calls = [call("bad", "query_financials", {"sql": "BAD"}),
                 call("good", "query_financials", {"sql": "GOOD"})]

        def sql(query):
            if query == "BAD":
                raise ValueError("bad query")
            return {"sql": query, "columns": ["n"], "rows": [{"n": 7}]}

        result = _execute_calls(state(calls), sql, None, 8)
        self.assertEqual(result["sources"][0]["id"], "S1")
        self.assertIn("bad query", result["messages"][0]["content"])
        self.assertEqual(json.loads(result["messages"][1]["content"])["source_id"], "S1")
        self.assertEqual(result["trace"][0]["tool"], "query_financials")
        self.assertEqual(result["trace"][1]["source_ids"], ["S1"])


if __name__ == "__main__":
    unittest.main()
