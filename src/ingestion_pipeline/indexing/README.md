# Index finished chunks with Qdrant

This stage joins each saved chunk with its embedding and stores the result in a local Qdrant collection. Qdrant keeps the vector and the chunk's text, company, year, PDF filename, page numbers, and source positions together. It can search vectors and filter by company or year.

The build checks that the embeddings were made from the current chunk file. If the chunks change, run the embedding step again. A new collection becomes active only after all points are saved. The local files in `data/index/qdrant/` are ignored by Git.

## Run

Install the updated dependencies if your virtual environment was set up before Qdrant was added:

```bash
uv pip install --python .venv/bin/python 'qdrant-client>=1.20,<2'
```

From the project root, after embedding the matching chunk set:

```bash
.venv/bin/python -m src.ingestion_pipeline.indexing.main --strategy structure build
.venv/bin/python -m src.ingestion_pipeline.indexing.main --strategy structure inspect
```

Use `--strategy semantic` for semantic chunks. Question-time search now lives in `src/retrieval_pipeline/`.

This stage builds the vector search index. The developer retrieval evaluation is available at `/retrieval-evaluation`; the full answer pipeline comes later.
