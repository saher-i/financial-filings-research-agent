"""Schemas exposed to the chat model for tool selection."""

TOOLS = [
    {"type": "function", "function": {
        "name": "query_financials",
        "description": "Run one read-only SELECT query over the supplied financial SQLite tables. Retrieve reported values or evidence-backed operands; call calculate when a derived result is needed.",
        "parameters": {"type": "object", "properties": {"sql": {"type": "string"}},
                       "required": ["sql"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "search_filings",
        "description": "Search the supplied 10-K PDFs for disclosures relevant to the question. Returns original PDF passages with page numbers.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "ticker": {"type": "string", "enum": ["AAPL", "MSFT", "GOOGL"]},
            "fiscal_year": {"type": "integer", "enum": [2024, 2025]}},
            "required": ["query"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "read_pages",
        "description": "Read one or two physical pages of a supplied 10-K PDF when a search passage omits nearby table rows or context. Use the page number shown on a retrieved passage; do not guess filenames or pages.",
        "parameters": {"type": "object", "properties": {
            "filename": {"type": "string", "enum": [
                f"{ticker}_FY{year}_10-K.pdf" for ticker in ("AAPL", "MSFT", "GOOGL")
                for year in (2024, 2025)]},
            "page": {"type": "integer"}, "end_page": {"type": "integer"}},
            "required": ["filename", "page"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "calculate",
        "description": "Compute derived arithmetic from gathered evidence. SQL operands use source_id, zero-based row index, and field. PDF operands use source_id, the exact written value, and a nearby text anchor from that cited passage; read the full page when a table is clipped. For percent_change, pass the newer value first and older value second, in matching units. Set value_unit to usd for dollar amounts or raw for other quantities. PDF dollar values are scaled from an explicit nearby table unit such as in millions. Omitted grouped revenue uses the database's USD unit; other grouped fields remain raw. Never provide remembered or invented operands.",
        "parameters": {"type": "object", "properties": {
            "operation": {"type": "string", "enum": ["sum", "difference", "ratio", "percentage", "percent_change", "grouped_change"]},
            "value_unit": {"type": "string", "enum": ["usd", "raw"]},
            "operands": {"type": "array", "items": {"type": "object", "properties": {
                "source_id": {"type": "string"}, "row": {"type": "integer"}, "field": {"type": "string"},
                "value": {"type": "string"}, "anchor": {"type": "string"}},
                "required": ["source_id"], "additionalProperties": False}},
            "source_id": {"type": "string"}, "base_year": {"type": "integer"},
            "new_year": {"type": "integer"}, "year_field": {"type": "string"},
            "group_field": {"type": "string"}, "value_field": {"type": "string"},
            "expected_total": {"type": "object", "properties": {
                "source_id": {"type": "string"}, "row": {"type": "integer"}, "field": {"type": "string"}},
                "required": ["source_id", "row", "field"], "additionalProperties": False},
        }, "required": ["operation"], "additionalProperties": False}}},
]
