"""Evidence synthesis and re-retrieval decision logic.

The synthesis step makes the app more like a real fact-checking workflow by
turning many retrieved passages into a short source-grounded analysis and by
identifying when another search is needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any

try:
    from entailment_checker import extract_simple_relation
except ImportError:  # pragma: no cover
    from src.entailment_checker import extract_simple_relation


@dataclass(frozen=True)
class EvidenceSynthesis:
    """Human-readable summary of the verification evidence."""

    evidence_strength: str
    source_count: int
    strong_source_count: int
    analysis: str
    needs_more_search: bool
    rereview_query: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _important_terms(text: str) -> list[str]:
    stop = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by", "from",
        "is", "are", "was", "were", "current", "currently", "today", "now", "this", "that", "has",
        "have", "had", "does", "did", "not",
    }
    terms = []
    for token in re.findall(r"[A-Za-z0-9]+", text.lower()):
        if token not in stop and len(token) > 2 and token not in terms:
            terms.append(token)
    return terms[:8]


def synthesize_evidence(
    claim: str,
    evidence: list[dict[str, Any]],
    predicted_label: str,
    confidence: float,
    explanation: str,
    profile: Any | None = None,
) -> EvidenceSynthesis:
    """Create a concise synthesis from evidence summaries and prediction signals."""
    source_count = len({item.get("url") or item.get("doc_id") for item in evidence})
    strong_source_count = sum(1 for item in evidence if float(item.get("trust_score", 0.0)) >= 0.80)
    top_items = evidence[:3]

    if confidence >= 0.78 and strong_source_count >= 1 and top_items:
        strength = "Strong"
    elif confidence >= 0.62 and top_items:
        strength = "Moderate"
    elif top_items:
        strength = "Weak"
    else:
        strength = "Insufficient"

    source_phrases = []
    for item in top_items:
        title = item.get("title", "Untitled source")
        label = item.get("evidence_relevance_label", "Retrieved evidence")
        summary = item.get("summary") or item.get("text", "")[:220]
        source_phrases.append(f"{label}: {title} — {summary}")

    if source_phrases:
        evidence_sentence = " ".join(source_phrases)
    else:
        evidence_sentence = "No usable public evidence passages were retrieved."

    current_note = ""
    if profile is not None and getattr(profile, "needs_current_source", False):
        current_note = " The claim is time-sensitive, so current or official evidence is required."

    analysis = (
        f"The system labeled the claim as {predicted_label} with {strength.lower()} evidence. "
        f"{explanation}{current_note} Top evidence considered: {evidence_sentence}"
    )

    needs_more = predicted_label == "Not Enough Evidence" or confidence < 0.62 or source_count < 2
    rereview_query = None
    if needs_more:
        terms = " ".join(_important_terms(claim))
        relation = extract_simple_relation(claim)
        if profile is not None and getattr(profile, "needs_current_source", False):
            rereview_query = f"{terms} current official source"
        elif relation is not None:
            subject, predicate = relation
            rereview_query = f"{subject} facts known as {predicate} reliable source"
        else:
            rereview_query = f"{terms} reliable source evidence"

    return EvidenceSynthesis(
        evidence_strength=strength,
        source_count=source_count,
        strong_source_count=strong_source_count,
        analysis=analysis,
        needs_more_search=needs_more,
        rereview_query=rereview_query,
    )
