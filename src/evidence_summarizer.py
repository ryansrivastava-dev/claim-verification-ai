"""Extractive evidence summarization for retrieved passages.

This avoids paid APIs while still giving the app a real intermediate evidence
analysis step. It selects the most claim-relevant sentences from each retrieved
source and labels whether the passage looks directly relevant, partly relevant,
or weak.
"""

from __future__ import annotations

import re
from typing import Any


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with", "by",
    "from", "is", "are", "was", "were", "be", "being", "been", "has", "have", "had", "that",
    "this", "it", "as", "at", "into", "than", "then", "current", "currently", "today", "now",
}


def _sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", str(text)).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if len(part.strip()) >= 35]


def _terms(text: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", str(text).lower())
    return {token for token in tokens if token not in _STOPWORDS and len(token) > 2}


def _sentence_score(sentence: str, claim_terms: set[str]) -> float:
    if not claim_terms:
        return 0.0
    lower = sentence.lower()
    overlap = sum(1 for term in claim_terms if term in lower)
    return overlap / len(claim_terms)


def summarize_evidence_item(claim: str, item: dict[str, Any], max_sentences: int = 3) -> dict[str, Any]:
    """Add an extractive summary to one evidence item."""
    claim_terms = _terms(claim)
    sentences = _sentences(str(item.get("text", "")))

    if not sentences:
        summary = str(item.get("text", ""))[:450]
        coverage = 0.0
    else:
        ranked = sorted(
            ((sentence, _sentence_score(sentence, claim_terms)) for sentence in sentences),
            key=lambda pair: pair[1],
            reverse=True,
        )
        selected = [sentence for sentence, score in ranked[:max_sentences] if score > 0]
        if not selected:
            selected = [ranked[0][0]]
        summary = " ".join(selected)[:900]
        coverage = ranked[0][1]

    if coverage >= 0.50:
        relevance_label = "Directly relevant"
    elif coverage >= 0.25:
        relevance_label = "Partly relevant"
    else:
        relevance_label = "Weak or background evidence"

    enriched = dict(item)
    enriched.update(
        {
            "summary": summary,
            "summary_coverage": float(coverage),
            "evidence_relevance_label": relevance_label,
        }
    )
    return enriched


def summarize_evidence_batch(claim: str, evidence: list[dict[str, Any]], max_items: int = 8) -> list[dict[str, Any]]:
    """Summarize the strongest retrieved evidence items."""
    return [summarize_evidence_item(claim, item) for item in evidence[:max_items]]
