# Local research UI

The customer research chat is available at `http://127.0.0.1:8000/`. Start it from the project root with:

```bash
.venv/bin/python -m devui.server
```

The page accepts ad hoc questions, shows answers and the SQL/PDF evidence used, and can open original filing PDFs at cited pages. Choose **Concise** or **Elaborate** and an **Agent model** beside the question box before sending. The ready choices are DeepSeek V4.1 Flash (default), Nemotron Lightning 3.5 30B (lower cost), GPT OSS 120B, and GLM-5.3 (a higher-cost quality candidate). GLM-5.3's quality on these questions is unmeasured. Follow-up questions stay in the same scrollable conversation; the agent receives up to four prior question-answer pairs as context and gathers fresh evidence for each turn. The left sidebar restores a whole saved conversation without another model call and offers a delete button for it. The same server exposes `POST http://localhost:8000/api/chat` with JSON `{"question":"...","answer_style":"concise","model":"accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b"}` and returns JSON with a top-level `answer` plus `sources`, `trace`, and `conversation_id`. Send that `conversation_id` with the next question to continue the conversation. `answer_style` may be `concise` or `detailed`; omitting it uses `concise`. Omit `model` for the configured default. `GET /api/chat/models` lists selectable agent models. The server reads `FIREWORKS_API_KEY` from `.env` at startup if it is not already exported. Restart after changing `.env`.

## Developer ingestion workspace

This page is for building PDF chunks, embedding finished chunks, and inspecting the local Qdrant index. It is available only when `APP_MODE=developer`. In the default mode, the page and `/api/dev/*` routes return 404. The server listens on `127.0.0.1`. This mode setting is not a user login system.

From the project root, run:

```bash
APP_MODE=developer .venv/bin/python -m devui.server
```

Open `http://127.0.0.1:8000/ingestion-pipeline`.

At startup, this developer server loads `FIREWORKS_API_KEY` from the repository's `.env` file if the key is not already exported in its environment. Restart the server after changing `.env`; the browser page alone cannot reload server environment settings. The key is never returned by the UI API.

## What you can do

1. Press **Build chunks** under Structure aware or Semantic. The server runs `python -m src.ingestion_pipeline.chunking.main --strategy structure` or `semantic` in the background. The output is `data/index/<strategy>_chunks.jsonl`.
2. Press **View chunks** to see the saved text, chunk counts, PDF filenames, and page numbers. Filter by filing, page, or text.
3. Press **Show on PDF** on a chunk. Yellow marks only the source text for that selected chunk on the current page. A long source block split into smaller parts can have an approximate highlight box.
4. Use the sliders to change target length, maximum length, or overlap. Press **Build saved trial** to write a separate chunk file under `data/index/variants/`. The **Chunk sets** table includes the two default sets and any saved slider trials. Trials stay on this computer until you remove their files.
5. Under **Search preparation**, press **Generate Embeddings** for a baseline chunk set. The server calls Fireworks in the background. **View embeddings** shows the saved numbers for each finished chunk.
6. Press **Build Qdrant index**. The status changes to **Indexed** when the collection contains all chunks. **View in Qdrant explorer** shows points, vector values, text, and PDF source details from the local Qdrant store.

The sliders do nothing until you press **Build saved trial**. A semantic chunk build and either **Generate Embeddings** action need `FIREWORKS_API_KEY` in the server environment. A structure-aware chunk build does not call a model. Install `qdrant-client` from `starter/requirements.txt` before building an index. Search preparation uses the two baseline chunk sets.

Qdrant runs in local mode under `data/index/qdrant/`; it has no separate web dashboard. The Qdrant explorer reads the actual saved collection.

## Retrieval evaluation

Open `http://127.0.0.1:8000/retrieval-evaluation` after generating the ten benchmark answers. Select the two-question pilot (q_006 and q_012) or all ten. The page evaluates the SQL results and PDF passages **cited in the saved final answers**, in first-citation order. It does not rerun the agent or search the PDF index. A source cited repeatedly appears once. SQL-only questions are included.

Open `/benchmark-evaluation` to generate all ten development answers into `dev_answers.json` and `evaluation/dev_run.json`, then apply the per-question `evaluation` method in `questions/dev_questions_with_answers.json`. `fuzzy_numeric` normalizes billion/million units and accepts small rounding differences; `exact_match_entity` requires the expected company and a rounded numeric result; `llm_judge` compares the saved answer with the supplied answer and cited evidence using a Fireworks model. Each verdict is shown per question and saved separately to `evaluation/benchmark_run.json`. These are local implementations of the methods named in the question file, not a supplied hidden evaluator or a human adjudication. The page warns when the scores belong to an older answer run.

Each evaluation page shows progress while a job is running. On Benchmark Evaluation, a question with an unavailable judgment and a saved answer has a **Retry this judgment** button. Retry grades only that question, preserves other saved verdicts, and updates the summary counts; it does not rerun the answering agent.

For each question, the backend passes its question, supplied reference answer, and cited SQL/PDF evidence to the official RAGAS `ContextPrecision` and `ContextRecall` classes. RAGAS uses a Fireworks-hosted LLM to label source usefulness and reference-claim support, then computes its ranking-sensitive precision and claim-coverage recall. No human passage labels are required. Failed precision or recall calls are excluded from that metric's average separately. The server needs `FIREWORKS_API_KEY`. This is **cited-evidence precision and recall**: uncited retrieved passages are not measured. Citation order is used as the ranking, so precision does not measure the original search ranking.

New retrieval runs save each cited source's RAGAS Context Precision verdict and reason plus Context Recall claim attributions. A separate LLM annotation records which cited SQL source or PDF chunk IDs support each reference claim; this annotation does not alter either official RAGAS score. The UI links claims to evidence and shows Relevant or Not relevant beside each citation; judge reasons, evidence text, and score calculations are expandable. Older retrieval runs need a rerun to show this evaluation.

## Customer chat and final answer evaluation

In normal mode, `/` is the customer chat and `POST /api/chat` accepts `{"question":"..."}`. The response contains an `answer`, SQL/PDF `sources`, a tool `trace`, and a `conversation_id`. Send that ID with a follow-up question to reuse bounded conversational context; each answer retrieves fresh evidence and gets its own source IDs. The page scrolls through the thread above a fixed composer, restores recent conversations, and can delete one locally. The page links citations to the returned sources and opens original PDF pages. Developer routes are hidden in normal mode.

In developer mode, use the Generate step on `/benchmark-evaluation` to run the agent on all ten included evaluation questions. Progress and saved answers appear as each question finishes. The run uses live Fireworks calls and can take several minutes. Grade the saved run from the same page after generation finishes.

Open `/generation-evaluation` to score the **saved** answer run without regenerating answers. First click **Test 2 questions**. When both official RAGAS Faithfulness and Response Relevancy return scores on both pilot questions, **Evaluate all 10** becomes available. Fireworks supplies the judge and embeddings. A failed metric can be retried for that question alone; its other score stays saved. Unavailable metric scores and error details are shown separately; results are saved under `evaluation/generation_runs/`. Install `starter/requirements.txt` first. These are model-based estimates and should be calibrated against human review.

For debugging, start the server in a terminal with `APP_MODE=developer .venv/bin/python -m devui.server` and watch its output. Generation logs identify the question, metric, model, elapsed time, and success or exception type. Retrieval judge logs also identify the attempt, passage count, output length, and whether a response was empty, truncated, or invalid. Logs omit raw passages and API keys.

Choose the **Answer model** on Benchmark Evaluation before generating answers. The selected model changes answer generation; retrieval uses the configured index. A new ten-question run updates the active answer files and keeps archived run history.

The customer chat uses `POST /api/chat/stream` for live, user-friendly progress messages from the agent. This is a line-delimited event stream with `progress`, `answer`, and `error` events. Each model or evidence step reports its measured duration when it completes; the final answer shows the total server processing time. The JSON `/api/chat` response also includes `total_ms` and retains its required top-level `answer`.

## Inspect one agent workflow

In developer mode, ask a question at `/`, then open `/agent-workflow` and select that run. The page shows the answer model, actual model turns, SQL and filing tool calls, calculator inputs and results, source IDs, durations, the final answer, and the evidence returned. It does not show hidden model reasoning. Runs are saved locally in ignored `logs/agent_runs.jsonl`; no external tracing account is needed. The developer page and its JSON endpoint return 404 in normal customer mode.
