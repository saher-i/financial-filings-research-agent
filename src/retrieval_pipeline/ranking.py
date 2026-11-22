"""Optional, local post-ranking baselines over a fixed fused candidate pool."""

from __future__ import annotations

import math
import re


STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "between", "by", "did", "do", "does",
    "each", "for", "from", "had", "has", "how", "in", "is", "its", "of", "on",
    "or", "the", "their", "this", "to", "was", "were", "what", "which", "with",
})


def _terms(text: str) -> set[str]:
    return {term for term in re.findall(r"[a-z0-9]+", text.casefold())
            if len(term) > 1 and term not in STOPWORDS}


def local_rerank(candidates: list[dict], question: str, limit: int = 5) -> list[dict]:
    """Blend RRF with candidate-local weighted term coverage; no model call."""
    if limit < 1:
        raise ValueError("limit must be positive")
    terms = _terms(question)
    if not candidates or not terms:
        return [dict(hit) for hit in candidates[:limit]]
    document_terms = [_terms(hit.get("text", "")) for hit in candidates]
    count = len(candidates)
    weights = {term: math.log((count + 1) / (1 + sum(term in text for text in document_terms))) + 1
               for term in terms}
    denominator = sum(weights.values())
    top_rrf = max(float(hit.get("rrf_score", 0)) for hit in candidates) or 1.0
    ranked = []
    for index, (hit, words) in enumerate(zip(candidates, document_terms)):
        coverage = sum(weight for term, weight in weights.items() if term in words) / denominator
        score = 0.8 * float(hit.get("rrf_score", 0)) / top_rrf + 0.2 * coverage
        ranked.append({**hit, "local_rerank_score": score, "term_coverage": coverage,
                       "original_rank": index + 1})
    return sorted(ranked, key=lambda hit: (-hit["local_rerank_score"], hit["original_rank"], hit["id"]))[:limit]


def diversify_pages(candidates: list[dict], limit: int = 5) -> list[dict]:
    """Prefer a new PDF page; backfill original order when unique pages run out."""
    if limit < 1:
        raise ValueError("limit must be positive")
    chosen, deferred = [], []
    seen_pages: set[tuple[str, int]] = set()
    for hit in candidates:
        pages = {(hit.get("filename", ""), page) for page in hit.get("pages", [])}
        if pages and pages & seen_pages:
            deferred.append(hit)
        else:
            chosen.append(hit)
            seen_pages.update(pages)
        if len(chosen) >= limit:
            break
    if len(chosen) < limit:
        chosen.extend(deferred[:limit - len(chosen)])
    return [dict(hit) for hit in chosen]
