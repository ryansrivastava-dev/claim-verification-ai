"""Web-grounded claim verification pipeline.

The pipeline searches live public evidence sources, ranks passages, and assigns
an evidence label. It is not a memorized-answer system: every result should be
read with the cited evidence passages.
"""

from __future__ import annotations

import re
from typing import Any

try:
    from claim_type_classifier import detect_claim_type
    from web_retriever import MultiSourceEvidenceRetriever
except ImportError:  # pragma: no cover
    from src.claim_type_classifier import detect_claim_type
    from src.web_retriever import MultiSourceEvidenceRetriever


_NEGATION_CUES = [
    "does not",
    "did not",
    "do not",
    "is not",
    "are not",
    "was not",
    "were not",
    "no evidence",
    "not associated",
    "no association",
    "failed to",
    "unrelated",
    "false",
    "incorrect",
]

_SUPPORT_CUES = [
    "is",
    "are",
    "was",
    "were",
    "increases",
    "decreases",
    "causes",
    "associated with",
    "located in",
    "capital of",
]


def _contains_phrase(text: str, phrases: list[str]) -> bool:
    lower = str(text).lower()
    return any(phrase in lower for phrase in phrases)


def _claim_terms(claim: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", claim.lower())
    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "being",
        "been",
        "that",
        "this",
        "it",
    }
    return {token for token in tokens if token not in stopwords and len(token) > 2}


def _coverage_ratio(claim: str, evidence_text: str) -> float:
    terms = _claim_terms(claim)
    if not terms:
        return 0.0
    evidence_lower = evidence_text.lower()
    covered = sum(1 for term in terms if term in evidence_lower)
    return covered / len(terms)


def predict_from_evidence(claim: str, evidence: list[dict[str, Any]]) -> tuple[str, float, str]:
    """Assign an evidence label from retrieved passages.

    This deliberately avoids claiming certainty. It produces labels based on
    source-backed relevance signals and simple contradiction cues.
    """
    if not evidence:
        return "Not Enough Evidence", 0.50, "No public evidence passages were retrieved for the claim."

    best = evidence[0]
    best_text = str(best.get("text", ""))
    best_score = float(best.get("score", 0.0))
    relevance = float(best.get("relevance_score", 0.0))
    coverage = _coverage_ratio(claim, best_text)

    if relevance < 0.08 or coverage < 0.30:
        return (
            "Not Enough Evidence",
            0.55,
            "The retrieved sources do not cover enough of the claim to support a reliable label.",
        )

    claim_negative = _contains_phrase(claim, _NEGATION_CUES)
    evidence_negative = _contains_phrase(best_text, _NEGATION_CUES)

    if claim_negative != evidence_negative and relevance >= 0.18 and coverage >= 0.45:
        return (
            "Possibly Refuted",
            min(0.84, 0.54 + best_score),
            "The strongest evidence is relevant, but its wording appears to conflict with the claim.",
        )

    if relevance >= 0.18 and coverage >= 0.45:
        return (
            "Likely Supported",
            min(0.90, 0.58 + best_score),
            "The strongest retrieved evidence closely matches the claim and does not show an obvious contradiction.",
        )

    return (
        "Not Enough Evidence",
        min(0.72, 0.50 + best_score),
        "Some related evidence was found, but the match is not strong enough for a confident label.",
    )


def run_web_fact_check(
    claim: str,
    method: str = "Hybrid",
    top_k: int = 5,
    alpha: float = 0.5,
    max_pages: int = 6,
    use_web: bool = True,
    use_wikipedia: bool = True,
    use_openalex: bool = True,
    retriever: Any | None = None,
) -> dict[str, Any]:
    """Run live source search, passage retrieval, and evidence labeling."""
    claim = str(claim).strip()
    if not claim:
        raise ValueError("Claim cannot be empty.")

    profile = detect_claim_type(claim)
    retriever = retriever or MultiSourceEvidenceRetriever(
        use_web=use_web,
        use_wikipedia=use_wikipedia,
        use_openalex=use_openalex,
    )
    evidence = retriever.retrieve(
        claim=claim,
        method=method,
        top_k=top_k,
        alpha=alpha,
        max_pages=max_pages,
    )
    label, confidence, explanation = predict_from_evidence(claim, evidence)

    if profile.needs_current_source and label != "Not Enough Evidence":
        explanation = (
            f"{explanation} This claim appears time-sensitive, so the source dates should be checked."
        )

    return {
        "claim": claim,
        "predicted_label": label,
        "confidence": confidence,
        "explanation": explanation,
        "claim_profile": {
            "category": profile.category,
            "needs_current_source": profile.needs_current_source,
            "reason": profile.reason,
        },
        "retrieval_method": method,
        "top_evidence": evidence,
        "source_mode": "Live public evidence retrieval",
    }
