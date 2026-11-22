# 10-K research assistant

A local assistant for questions about Apple, Microsoft, and Alphabet financials and FY2024/FY2025 10-K filings. It combines read-only SQL queries with PDF search and returns source links.

## Run locally

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/getting-started/installation/), and a Fireworks API key. Run these commands from the extracted project root:

```bash
./setup.sh
cp .env.example .env
```

Edit `.env` and set **`FIREWORKS_API_KEY`**. This is the only required environment variable. The server reads the key locally; do not put it in a URL or browser form. `FIREWORKS_CHAT_MODEL` optionally changes the default answer model, `PORT` changes the local port (default `8000`), and `APP_MODE=developer` enables the evaluation workspace. The default answer model is DeepSeek V4.1 Flash through Fireworks.

The repository includes the SQLite database, six PDFs, and the prepared Structure aware search index. If the index is missing, rebuild it after setting the key:

```bash
.venv/bin/python -m src.ingestion_pipeline.chunking.main --strategy structure
.venv/bin/python -m src.ingestion_pipeline.embedding.main --strategy structure
.venv/bin/python -m src.ingestion_pipeline.indexing.main --strategy structure build
```

Start the interactive chat with dev mode:

```bash
APP_MODE=developer .venv/bin/python -m devui.server
```

Open **http://127.0.0.1:8000/** and ask an ad hoc question. The HTTP API is also available:

```bash
curl -s http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"What was Apple total net revenue in fiscal year 2025?"}'
```

The response has a top-level `answer` plus source and execution details. To use the command line instead:

```bash
.venv/bin/python -m src.cli ask "What was Apple total net revenue in fiscal year 2025?"
```

Open `/benchmark-evaluation`, `/generation-evaluation`, or `/retrieval-evaluation` on the same local host. The developer workspace also includes ingestion status and the agent workflow. A new Benchmark answer run uses `verify=True` for all ten questions and writes the new answers to `dev_answers.json`. Run Benchmark grading afterward, followed by Generation and Retrieval Evaluation if you want metrics for the new answers.

You can run the same reviewed benchmark from the command line instead. Stop the server first:

```bash
.venv/bin/python -m scripts.run_verified_benchmark
```

This replaces the active answers and benchmark scores and makes additional Fireworks calls. If you create another run, update the report after its Benchmark, Generation, and Retrieval Evaluations finish so the reported scores match the active answers.

## Project layout

- `dev_answers.json`: the ten final generated answers, keyed by question ID.
- `report.md`: the short report.
- `evaluation/`: the final saved answer run and its Benchmark, Generation, and Retrieval Evaluation results.
- `src/`, `eval_pipeline/`, and `devui/`: the agent, evaluation code, local API, and interactive pages.
- `data/`: the financial SQLite database, six filings, and prepared Structure aware index.
- `questions/`: the development questions and expected answers used by local evaluation.

To run the existing offline tests:

```bash
.venv/bin/python -m unittest discover -s tests
```
