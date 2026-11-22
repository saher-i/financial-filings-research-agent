# Filing retrieval

This stage searches the chunks in two ways at the same time:

1. `rewrite.py` asks the small Fireworks Nemotron Lightning model for a short, spelling-aware keyword query. It preserves explicit years and company names, caps the response, and falls back to the original query on failure. `lexical.py` uses SQLite FTS5 on that query. It builds `data/index/<strategy>_lexical.db` from the current chunk file when needed.
2. `semantic.py` embeds the question with Fireworks and searches the active Qdrant collection for similar chunk vectors.
3. `hybrid.py` combines the two ranked lists with reciprocal rank fusion (RRF). A chunk found by both methods receives a contribution from each list. Raw BM25 and vector scores are not compared directly.

The developer retrieval diagnostic also compares three optional post-ranking variants over the **same fused candidates**: a local query-term coverage reranker, a distinct-PDF-page preference, and both together. They make no additional inference calls. The default answer search remains plain RRF until a same-corpus comparison shows a reliable benefit. The local reranker is a cheap baseline, not a model-based cross-encoder.

The agent first writes a filing search query. Query reconstruction then changes only the lexical branch; semantic retrieval still embeds the agent's original phrase. The workflow trace and retrieval evaluation show both phrases and the rewrite duration. This extra model call can add latency and does not guarantee spelling recovery; compare it with the original-query baseline before treating it as an accuracy improvement.

The search checks that Qdrant and the lexical index belong to the same chunk file. Results include the chunk text, ticker, year, PDF filename, page numbers, and source positions. The command also shows the separate lexical and semantic candidates so you can see why a chunk was ranked.

From the project root, stop the developer server first because local Qdrant allows one process to open the store at a time. With `FIREWORKS_API_KEY` set, run:

```bash
.venv/bin/python -m src.cli search "supply chain risks" --ticker AAPL --year 2025
```

Use `--strategy semantic` to search semantic chunks. `--limit` controls fused results; `--candidates` controls how many results each method contributes. Lexical indexing runs locally and makes no API call. Semantic search embeds the question once through Fireworks.

This retrieves PDF evidence. The developer UI at `http://127.0.0.1:8000/retrieval-evaluation` runs a passage-level check against five filing questions from `questions/dev_questions.json`; see `devui/README.md` for the score definitions and limitations. The financial SQL tool and answer-generating agent use these results to answer questions.

For every question, the UI reports LLM-estimated context precision and recall for keyword, vector, plain RRF, local reranking, page diversity, and reranking plus diversity. All post-ranking variants share one query embedding and candidate pool per question. These scores use the supplied answers and a model judge, so inspect the question-level lists and calibrate against human labels before treating small score changes as reliable.
