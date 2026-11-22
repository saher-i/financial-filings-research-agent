import unittest
import json

from src.answer_agent.tools.calculator import calculate
from src.answer_agent.tools.execute import _execute_calls
from src.answer_agent.graph import _finish


class CalculatorTests(unittest.TestCase):
    def setUp(self):
        self.sources = [
            {"id": "S1", "type": "sql", "rows": [
                {"fiscal_year": 2024, "segment_name": "Google Services", "revenue": 304930000000},
                {"fiscal_year": 2025, "segment_name": "Google Services", "revenue": 342721000000},
                {"fiscal_year": 2024, "segment_name": "Google Cloud", "revenue": 43229000000},
                {"fiscal_year": 2025, "segment_name": "Google Cloud", "revenue": 58705000000},
                {"fiscal_year": 2024, "segment_name": "Other Bets", "revenue": 1648000000},
                {"fiscal_year": 2025, "segment_name": "Other Bets", "revenue": 1537000000},
                {"fiscal_year": 2024, "segment_name": "Hedging gains (losses)", "revenue": 211000000},
                {"fiscal_year": 2025, "segment_name": "Hedging gains (losses)", "revenue": -127000000},
            ]},
            {"id": "S2", "type": "sql", "rows": [{"revenue_change": 52818000000}]},
        ]

    def test_grouped_change_reconciles_actual_segment_rows(self):
        result = calculate(self.sources, {"operation": "grouped_change", "source_id": "S1",
            "base_year": 2024, "new_year": 2025,
            "expected_total": {"source_id": "S2", "row": 0, "field": "revenue_change"}})
        changes = {row["group"]: row["change"] for row in result["groups"]}
        self.assertEqual(changes["Google Services"], "37791000000")
        self.assertEqual(changes["Google Cloud"], "15476000000")
        self.assertEqual(changes["Other Bets"], "-111000000")
        self.assertEqual(changes["Hedging gains (losses)"], "-338000000")
        self.assertEqual(result["total_change"], "52818000000")
        self.assertEqual(result["total_change_billions"], "52.818")
        self.assertEqual(result["value_unit"], "usd")
        self.assertTrue(result["unit_inferred"])
        self.assertTrue(result["reconciled"])

    def test_percent_change_uses_sql_rows(self):
        result = calculate(self.sources, {"operation": "percent_change", "operands": [
            {"source_id": "S1", "row": 1, "field": "revenue"},
            {"source_id": "S1", "row": 0, "field": "revenue"}]})
        self.assertEqual(result["result"], "12.3933361755156921260617191")

    def test_grouped_change_supports_non_dollar_quantities(self):
        sources = [{"id": "S1", "type": "sql", "rows": [
            {"fiscal_year": 2024, "category": "A", "units_sold": 10},
            {"fiscal_year": 2025, "category": "A", "units_sold": 14}]}]
        result = calculate(sources, {"operation": "grouped_change", "source_id": "S1",
            "base_year": 2024, "new_year": 2025, "group_field": "category",
            "value_field": "units_sold", "value_unit": "raw"})
        self.assertEqual(result["groups"][0]["change"], "4")
        self.assertIsNone(result["groups"][0]["change_billions"])

        omitted = calculate(sources, {"operation": "grouped_change", "source_id": "S1",
            "base_year": 2024, "new_year": 2025, "group_field": "category",
            "value_field": "units_sold"})
        self.assertEqual(omitted["value_unit"], "raw")
        self.assertTrue(omitted["unit_inferred"])
        self.assertIsNone(omitted["total_change_billions"])

        declared_usd = calculate(sources, {"operation": "grouped_change", "source_id": "S1",
            "base_year": 2024, "new_year": 2025, "group_field": "category",
            "value_field": "units_sold", "value_unit": "usd"})
        self.assertEqual(declared_usd["value_unit"], "usd")
        self.assertFalse(declared_usd["unit_inferred"])
        self.assertEqual(declared_usd["total_change_billions"], "0")

        with self.assertRaisesRegex(ValueError, "value_unit"):
            calculate(sources, {"operation": "grouped_change", "source_id": "S1",
                "base_year": 2024, "new_year": 2025, "value_unit": "euros"})

    def test_percent_change_rejects_reversed_years_or_mixed_segments(self):
        with self.assertRaisesRegex(ValueError, "newer fiscal year"):
            calculate(self.sources, {"operation": "percent_change", "operands": [
                {"source_id": "S1", "row": 0, "field": "revenue"},
                {"source_id": "S1", "row": 1, "field": "revenue"}]})
        with self.assertRaisesRegex(ValueError, "same segment_name"):
            calculate(self.sources, {"operation": "percent_change", "operands": [
                {"source_id": "S1", "row": 3, "field": "revenue"},
                {"source_id": "S1", "row": 0, "field": "revenue"}]})

    def test_pdf_table_values_can_be_calculated_only_when_present_near_anchor(self):
        pdf = [{"id": "S14", "type": "pdf", "text":
                "Year Ended December 31,\n2024\n2025\nYouTube ads\n36,147\n40,367\nGoogle Network\n30,359\n29,792"}]
        result = calculate(pdf, {"operation": "percent_change", "operands": [
            {"source_id": "S14", "value": "40,367", "anchor": "YouTube ads"},
            {"source_id": "S14", "value": "36,147", "anchor": "YouTube ads"}]})
        self.assertEqual(result["rounded_1"], "11.7")
        self.assertEqual(result["source_ids"], ["S14"])
        with self.assertRaisesRegex(ValueError, "not found near"):
            calculate(pdf, {"operation": "percent_change", "operands": [
                {"source_id": "S14", "value": "99,999", "anchor": "YouTube ads"},
                {"source_id": "S14", "value": "36,147", "anchor": "YouTube ads"}]})
        with self.assertRaisesRegex(ValueError, "not found near"):
            calculate(pdf, {"operation": "percent_change", "operands": [
                {"source_id": "S14", "value": "40,367", "anchor": "Other Bets"},
                {"source_id": "S14", "value": "36,147", "anchor": "YouTube ads"}]})
        with self.assertRaisesRegex(ValueError, "same metric anchor"):
            calculate(pdf, {"operation": "percent_change", "operands": [
                {"source_id": "S14", "value": "40,367", "anchor": "YouTube ads"},
                {"source_id": "S14", "value": "36,147", "anchor": "Tube ads"}]})

    def test_pdf_declared_millions_are_scaled_to_usd_for_difference(self):
        sources = [{"id": "S7", "type": "pdf", "text":
                    "The following table presents revenue (in millions):\n"
                    "Subscription revenue\n$ 1,250\n$ 1,500\n"}]
        result = calculate(sources, {"operation": "difference", "value_unit": "usd",
            "operands": [{"source_id": "S7", "value": "1,500", "anchor": "Subscription revenue"},
                         {"source_id": "S7", "value": "1,250", "anchor": "Subscription revenue"}]})
        self.assertEqual(result["result"], "250000000")
        self.assertEqual(result["operands"][0]["written_unit"], "millions")
        self.assertEqual(result["result_billions"], "0.25")
        state = {"draft": "Revenue increased by $250 million [S7].", "sources": sources,
                 "trace": [{"tool": "calculate", "source_ids": result["source_ids"],
                            "result": result}], "pending_calls": [], "steps": 2}
        self.assertEqual(_finish(state, 8)["answer_status"], "answered")
        raw = calculate(sources, {"operation": "difference", "value_unit": "raw",
            "operands": [{"source_id": "S7", "value": "1,500", "anchor": "Subscription revenue"},
                         {"source_id": "S7", "value": "1,250", "anchor": "Subscription revenue"}]})
        self.assertEqual(raw["result"], "250")

    def test_rejects_unseen_numbers_and_incomplete_group(self):
        with self.assertRaisesRegex(ValueError, "returned SQL value"):
            calculate(self.sources, {"operation": "sum", "operands": [
                {"source_id": "S1", "row": 99, "field": "revenue"}]})
        with self.assertRaisesRegex(ValueError, "Both years"):
            calculate([{"id": "S1", "type": "sql", "rows": self.sources[0]["rows"][:1]}],
                      {"operation": "grouped_change", "source_id": "S1", "base_year": 2024,
                       "new_year": 2025})

    def test_agent_tool_call_records_calculated_sql_evidence(self):
        sources = [{**source, "key": f"sql:source-{index}"}
                   for index, source in enumerate(self.sources)]
        call = {"id": "calc-1", "function": {"name": "calculate", "arguments": json.dumps({
            "operation": "grouped_change", "source_id": "S1", "base_year": 2024, "new_year": 2025,
            "expected_total": {"source_id": "S2", "row": 0, "field": "revenue_change"}})}}
        result = _execute_calls({"sources": sources, "messages": [], "trace": [],
                                 "pending_calls": [call], "steps": 2}, None, None, 24)
        output = json.loads(result["messages"][-1]["content"])
        self.assertTrue(output["reconciled"])
        self.assertEqual(result["trace"][-1]["source_ids"], ["S1", "S2"])
        self.assertEqual(result["trace"][-1]["result"]["total_change_billions"], "52.818")

    def test_repeated_grouped_change_ignores_unused_operand_list(self):
        sources = [{**source, "key": f"sql:source-{index}"}
                   for index, source in enumerate(self.sources)]
        def call(identifier, operands):
            return {"id": identifier, "function": {"name": "calculate", "arguments": json.dumps({
                "operation": "grouped_change", "source_id": "S1", "base_year": 2024,
                "new_year": 2025, "operands": operands})}}
        result = _execute_calls({"sources": sources, "messages": [], "trace": [],
                                 "pending_calls": [call("first", []), call("second", [{"row": 1}])],
                                 "steps": 2}, None, None, 24)
        self.assertEqual(result["trace"][0]["result"]["total_change_billions"], "52.818")
        self.assertTrue(result["trace"][1]["cached"])

    def test_segment_breakdown_requires_a_successful_calculator_call(self):
        state = {"question": "Break down the revenue increase by segment", "draft": "Up by $1 [S1].",
                 "sources": self.sources, "pending_calls": [], "steps": 4, "trace": [],
                 "research_plan": {"needs_calculation": True, "calculation_type": "grouped_change"}}
        self.assertEqual(_finish(state, 8)["answer_status"], "calculation_failed")
        state["trace"] = [{"tool": "calculate", "arguments": {"operation": "grouped_change"},
                           "source_ids": ["S1"], "result": {"result": "1"}}]
        self.assertEqual(_finish(state, 8)["answer_status"], "answered")

    def test_incorrect_plan_cannot_authorize_unsourced_percentage(self):
        state = {"question": "How much did revenue grow?", "draft": "It grew 12.4% [S1].",
                 "sources": self.sources, "pending_calls": [], "steps": 4, "trace": [],
                 "research_plan": {"needs_calculation": False, "calculation_type": None}}
        self.assertEqual(_finish(state, 8)["answer_status"], "calculation_failed")

    def test_services_growth_comparison_does_not_require_all_segment_reconciliation(self):
        state = {"question": "Apple's Services segment has been growing. Calculate Services growth for each "
                             "of the last two years and compare it with iPhone growth.",
                 "draft": "Services grew 13.5% [S1].", "sources": [{"id": "S1"}],
                 "pending_calls": [], "steps": 4, "trace": [
                     {"tool": "calculate", "arguments": {"operation": "percent_change"},
                      "source_ids": ["S1"], "result": {"result": "13.5"}}]}
        self.assertEqual(_finish(state, 8)["answer_status"], "answered")


if __name__ == "__main__":
    unittest.main()
