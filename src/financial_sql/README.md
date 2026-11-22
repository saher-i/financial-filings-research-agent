# Financial SQL evidence tool

The five tables in `data/financials.db` hold company details and selected annual financial numbers. `tool.py` lets the agent inspect the table names and columns, then run a query for exact numbers or calculations.

The public functions are:

- `get_schema()` — returns the five allowed tables, their columns, companies, and available fiscal years.
- `execute_readonly_sql(sql)` — returns column names, rows, row count, the SQL used, and which tables supplied the rows.

For example:

```python
from src.financial_sql import execute_readonly_sql

result = execute_readonly_sql(
    "SELECT revenue FROM income_statements "
    "WHERE company_ticker='AAPL' AND fiscal_year=2025 AND period_type='FY'"
)
print(result["rows"])
```

Amounts are stored as whole U.S. dollars. Use `period_type = 'FY'` for annual rows. The tool accepts one `SELECT` or `WITH` query, reads only the five supplied tables, and returns at most 100 rows by default. A read-only database connection, a SQLite authorizer, and a time limit prevent a query from changing the database or running indefinitely. The tool does not decide which SQL to write; the agent will choose queries based on a user's question.
