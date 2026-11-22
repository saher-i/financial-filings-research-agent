"""Read-only financial evidence and query boundaries."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.financial_sql import execute_readonly_sql, get_schema


class FinancialSqlTests(unittest.TestCase):
    def test_schema_and_annual_revenue(self):
        schema = get_schema()
        self.assertEqual(set(schema["tables"]), {"companies", "income_statements", "balance_sheets",
                                                 "segment_revenue", "geographic_revenue"})
        self.assertIn("AAPL", {item["ticker"] for item in schema["companies"]})
        result = execute_readonly_sql(
            "SELECT revenue FROM income_statements "
            "WHERE company_ticker='AAPL' AND fiscal_year=2025 AND period_type='FY'")
        self.assertEqual(result["rows"], [{"revenue": 416161000000}])
        self.assertEqual(result["source"]["tables"], ["income_statements"])

    def test_cte_aggregation_and_row_cap(self):
        result = execute_readonly_sql(
            "WITH annual AS (SELECT company_ticker, revenue FROM income_statements "
            "WHERE fiscal_year=2025 AND period_type='FY') "
            "SELECT company_ticker, revenue FROM annual ORDER BY revenue DESC", max_rows=2)
        self.assertEqual(result["row_count"], 2)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["rows"][0]["company_ticker"], "AAPL")
        count = execute_readonly_sql("SELECT COUNT(*) AS n FROM companies")
        self.assertEqual(count["rows"], [{"n": 3}])

    def test_financial_calculation(self):
        result = execute_readonly_sql(
            "SELECT company_ticker, ROUND(total_current_assets * 1.0 / "
            "NULLIF(total_current_liabilities, 0), 2) AS current_ratio "
            "FROM balance_sheets WHERE fiscal_year=2025 AND period_type='FY' "
            "ORDER BY current_ratio DESC")
        self.assertEqual(result["rows"][0], {"company_ticker": "GOOGL", "current_ratio": 2.01})

    def test_rejects_nonfinancial_and_mutating_sql(self):
        for sql in ("DELETE FROM companies", "PRAGMA table_info(companies)",
                    "SELECT name FROM sqlite_master", "SELECT load_extension('x') FROM companies",
                    "SELECT 1", "SELECT ticker FROM companies; DELETE FROM companies"):
            with self.subTest(sql=sql), self.assertRaises(ValueError):
                execute_readonly_sql(sql)

    def test_database_cannot_be_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "financials.db"
            with sqlite3.connect(path) as db:
                db.execute("CREATE TABLE companies(ticker TEXT)")
                db.execute("INSERT INTO companies VALUES ('AAPL')")
            with self.assertRaises(ValueError):
                execute_readonly_sql("WITH x AS (SELECT ticker FROM companies) "
                                     "SELECT ticker FROM x; UPDATE companies SET ticker='XX'", path)
            with sqlite3.connect(path) as db:
                self.assertEqual(db.execute("SELECT ticker FROM companies").fetchone()[0], "AAPL")


if __name__ == "__main__":
    unittest.main()
