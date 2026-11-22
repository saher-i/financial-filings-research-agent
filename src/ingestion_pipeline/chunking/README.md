# PDF chunks

This code reads the six 10-K PDF files. It saves the text in smaller pieces called **chunks**. We use chunks so we can find written information in the PDFs later. Financial numbers are already in `data/financials.db`.

## What happens

1. `main.py` reads the list of PDFs in `data/manifest.json`.
2. `extract.py` reads text and tries to find tables on each page. It saves where each piece of text was found.
3. `build.py` puts nearby text pieces together. It can start a new chunk at a heading, a table, or a size limit.
4. `main.py` writes one chunk per line to a `.jsonl` file.

Each chunk has its text, company, year, PDF filename, page numbers, and positions on the page. A chunk can include text from more than one page. The positions let the developer UI mark the chunk's source text on the PDF.

The extracted text is also saved in `data/index/extractions/`. The next run can reuse it when the PDF has not changed.

## The two methods

**Structure aware:** Uses headings, tables, nearby text, and size rules. It does not call an AI model. We start with this method because the PDFs have many headings and tables.

**Semantic:** Uses the same rules. It also tries to start a new chunk when nearby text changes topic. To do this, it asks the Fireworks model `nomic-ai/nomic-embed-text-v1.5` for **embeddings**. An embedding is a list of numbers made from text. The code compares those lists. It saves them in `data/index/embedding_cache.db` for reuse. This method needs `FIREWORKS_API_KEY`.

First, check that the chunk text and PDF positions look right. Retrieval Evaluation measures the passages sent to the agent; Benchmark Evaluation and Generation Evaluation inspect saved answers.

## Size settings

- **Target: 3,600 characters.** The code tries to keep a chunk near this size. Headings and tables can make it shorter.
- **Maximum: 4,800 characters.** Long text is split before grouping. A very long table row can still go over this size.
- **Overlap: up to 400 characters.** Some text can appear in two chunks when a chunk ends because of size. This does not happen at every heading or table.

These are starting settings. Pages do not force a new chunk.

## Build chunks

Run this command from the project root to make structure-aware chunks:

```bash
.venv/bin/python -m src.ingestion_pipeline.chunking.main --strategy structure
```

It writes `data/index/structure_chunks.jsonl`.

To make semantic chunks, set `FIREWORKS_API_KEY` in your shell first. Then run:

```bash
.venv/bin/python -m src.ingestion_pipeline.chunking.main --strategy semantic
```

It writes `data/index/semantic_chunks.jsonl`.

Files in `data/index/` stay on your computer and are ignored by Git. This folder only makes chunks. The semantic embeddings here decide where chunks start. The separate `../embedding/` and `../indexing/` stages prepare the finished chunks for search.

## See the chunks

Start the developer UI:

```bash
APP_MODE=developer .venv/bin/python -m devui.server
```

Open `http://127.0.0.1:8000/ingestion-pipeline`. Choose a method and a PDF. Read a chunk, then press **Show on PDF**. Yellow marks where its text came from. The sliders can make new chunk files with different size settings. This page does not calculate retrieval scores.

## Code files

| File | What it does |
| --- | --- |
| `models.py` | Defines a text piece and a chunk. |
| `extract.py` | Reads PDF text and tables. |
| `build.py` | Makes chunks. |
| `semantic.py` | Finds extra places to start semantic chunks. |
| `semantic_embeddings.py` | Gets and saves text embeddings only for semantic chunk boundaries. |
| `main.py` | Runs the stages and saves the chunks. |

To continue the pipeline, run the steps in `../embedding/README.md` and then `../indexing/README.md`.
