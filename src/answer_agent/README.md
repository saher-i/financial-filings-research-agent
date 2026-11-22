# Answering agent

The agent answers one question by deciding whether to call the read-only financial SQL tool, search the PDFs, or both. It may use several tools when a question has several parts. A LangGraph loop records each tool call and stops after at most eight model turns by default. The last turn uses the evidence already gathered and does not offer new tool calls. The chat API can pass up to four prior user/assistant exchanges for follow-up references; previous citation labels are removed, and the current turn must gather and cite fresh evidence.

The SQL tool returns exact rows from `data/financials.db`. The PDF tool combines keyword and Qdrant vector search with reciprocal rank fusion. It opens the original page for short, single-page passages that may omit nearby context; the trace records the expansion. The agent can search again for distinct parts of a broad question. Each returned source gets a short ID such as `[S1]`; the answer must cite those IDs. The API result also contains the SQL query or PDF filename, page numbers and source text so a user can inspect the evidence. If citations are missing, the agent requests a cited revision within its step limit. If citations still fail, it withholds the draft and returns `answer_status: citation_failed`.

The default chat model is Fireworks' serverless `accounts/fireworks/models/deepseek-v4p1-flash`. The chat and developer evaluation pages also offer lower-cost Nemotron Lightning 3.5 30B, GPT OSS 120B, and GLM-5.3 (`accounts/fireworks/models/glm-5p3`) as a higher-cost quality candidate. GLM-5.3 supports serverless function calling, but has not been shown to outperform DeepSeek on this project's questions; compare them in Answer Evaluation before choosing. Set `FIREWORKS_CHAT_MODEL` to change the server default. The agent reads `FIREWORKS_API_KEY` from the process environment or the ignored root `.env` file. Query embeddings use the model recorded in the active Qdrant index.

The calculator reads numeric operands from gathered SQL rows or numbers anchored to text in a gathered PDF passage. It uses decimal arithmetic for differences, ratios, percentages, sums, and grouped changes. A grouped change covers every returned SQL category and can reconcile with a separately retrieved total. The caller identifies whether grouped values are USD or unscaled quantities. Directly reported values need evidence but no calculator call; derived values do. Numeric checks verify presence and units, while the model review checks whether cited evidence supports the meaning of each claim. Normal chat and benchmark answers both use that review.

## Where to read the code

1. `prompts/system.py` contains the answer rules and inserts the allowed database schema.
2. `tools/schemas.py` lists the SQL, filing-search, page-reading, and calculator tools the model can choose.
3. `tools/financial.py` calls the read-only SQL implementation in `src/financial_sql/tool.py`.
4. `tools/filings.py` calls hybrid PDF retrieval.
5. `tools/calculator.py` computes from gathered SQL rows or anchored PDF numbers.
6. `tools/pages.py` reads one or two physical PDF pages when a retrieved chunk omits a table or nearby context.
7. `tools/execute.py` checks tool arguments, runs calls, and attaches source IDs.
8. `models.py` discovers eligible answer models for developer evaluation.
9. `graph.py` owns the LangGraph model/tool loop, citation check, and bounded review recovery.

Inspect the chosen SQL rows and calculator trace before accepting a derived number. The final citation check verifies source IDs but does not prove that each claim is supported.

Install `starter/requirements.txt` after the dependency update. With the structure index already built, call:

```python
from src.answer_agent import answer_question

result = answer_question("What was Apple's FY2025 revenue?")
print(result["answer"])
print(result["sources"])
```

For a one-off terminal question, stop the developer server first so the CLI can open the local Qdrant store, then run:

```bash
.venv/bin/python -m src.cli ask "What was Apple's FY2025 revenue?"
```

This is a prototype. It bounds tool/model turns and returns evidence, but source IDs alone do not prove that every generated claim follows from its source. The development question evaluation must inspect answers and citations before treating them as reliable.
