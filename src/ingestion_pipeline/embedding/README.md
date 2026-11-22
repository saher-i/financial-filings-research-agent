# Embed finished chunks

This stage turns each saved chunk into a list of numbers. The numbers let us compare a question with the chunk later. It works on either chunk file made by `chunking/`.

The semantic chunking step already calls an embedding model to decide where some chunks start. This stage is different: it embeds the **finished chunks** so they can be searched.

## What the code does

1. `build.py` reads the chunk JSONL file and checks for duplicate IDs.
2. `client.py` sends batches to Fireworks using `nomic-ai/nomic-embed-text-v1.5`. Each document starts with `search_document:` and includes the ticker and filing year.
3. `build.py` saves the returned vectors in a local SQLite staging file. Its metadata records the model, vector size, chunk count, and a hash of the source file. The searchable vector store is Qdrant, built in the next stage.
4. It also saves vectors in `chunk_embedding_cache.db`, so a later run can reuse unchanged text. An incomplete run does not replace the previous finished file.

## Run

Set `FIREWORKS_API_KEY` in your shell. From the project root:

```bash
.venv/bin/python -m src.ingestion_pipeline.embedding.main --strategy structure
.venv/bin/python -m src.ingestion_pipeline.embedding.main --strategy semantic
```

These commands read `data/index/structure_chunks.jsonl` and `data/index/semantic_chunks.jsonl`. They write `structure_embeddings.db` and `semantic_embeddings.db` in the same directory. Start with one strategy if you only want to try one. Fireworks calls cost money; rerunning unchanged inputs reuses saved vectors.

The next stage is `../indexing/`. The files in `data/index/` are generated locally and ignored by Git.
