"""Fireworks client for document and query embeddings."""

import os

DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
BGE_MODEL = "accounts/fireworks/models/bge-m3"
TRIAL_MODELS = {DEFAULT_MODEL: "Nomic Embed Text v1.5", BGE_MODEL: "BGE-M3"}


def validate_embedding_model(model: str | None) -> str:
    selected = model or DEFAULT_MODEL
    if selected not in TRIAL_MODELS:
        raise ValueError("Choose a supported embedding model")
    return selected


class FireworksEmbedder:
    def __init__(self, model: str = DEFAULT_MODEL):
        from openai import OpenAI

        key = os.getenv("FIREWORKS_API_KEY")
        if not key:
            raise RuntimeError("Set FIREWORKS_API_KEY before requesting embeddings")
        self.model = model
        self.client = OpenAI(api_key=key, base_url="https://api.fireworks.ai/inference/v1")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model, input=texts)
        items = sorted(response.data, key=lambda item: item.index)
        if len(items) != len(texts) or [item.index for item in items] != list(range(len(texts))):
            raise ValueError("Embedding response did not contain one vector per input, in order")
        return [item.embedding for item in items]


def document_input(chunk: dict, model: str = DEFAULT_MODEL) -> str:
    """Include searchable filing context and the model's expected retrieval prefix."""
    body = f"{chunk['ticker']} FY{chunk['fiscal_year']} 10-K filing\n{chunk['text']}"
    return "search_document: " + body if model == DEFAULT_MODEL else body


def query_input(question: str, model: str = DEFAULT_MODEL) -> str:
    return f"search_query: {question}" if model == DEFAULT_MODEL else question
