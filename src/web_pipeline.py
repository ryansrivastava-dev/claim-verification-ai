"""Web-grounded claim verification pipeline.

The pipeline searches live public evidence sources, ranks passages, and assigns
an evidence label. It is not a memorized-answer system: every result should be
read with the cited evidence passages.
"""

from __future__ import annotations

import re
from typing import Any

try:
    from claim_type_classifier import ClaimProfile, detect_claim_type
    from evidence_summarizer import summarize_evidence_batch
    from evidence_synthesizer import synthesize_evidence
    from query_planner import plan_search_queries
    from report_builder import build_markdown_report
    from entailment_checker import add_entailment_scores
    from web_retriever import MultiSourceEvidenceRetriever
except ImportError:  # pragma: no cover
    from src.claim_type_classifier import ClaimProfile, detect_claim_type
    from src.evidence_summarizer import summarize_evidence_batch
    from src.evidence_synthesizer import synthesize_evidence
    from src.query_planner import plan_search_queries
    from src.report_builder import build_markdown_report
    from src.entailment_checker import add_entailment_scores
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

_HISTORICAL_ROLE_CUES = [
    "former",
    "previous",
    "served as",
    "was the",
    "was president",
    "was ceo",
    "from 19",
    "from 20",
    "until 20",
    "46th president",
    "45th president",
]

_CURRENT_EVIDENCE_CUES = [
    "current",
    "currently",
    "incumbent",
    "present",
    "serves as",
    "serving as",
    "is the",
    "official",
    "administration",
]


def _contains_phrase(text: str, phrases: list[str]) -> bool:
    lower = str(text).lower()
    return any(phrase in lower for phrase in phrases)


def _normalize_words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", str(text).lower())


def _contains_entity(text: str, entity: str) -> bool:
    """Return True when all meaningful entity words appear in the text."""
    text_lower = str(text).lower()
    words = [word for word in _normalize_words(entity) if len(word) > 1]
    if not words:
        return False
    return all(word in text_lower for word in words)


def _claim_terms(claim: str) -> set[str]:
    tokens = _normalize_words(claim)
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
        "current",
        "currently",
        "today",
        "now",
    }
    return {token for token in tokens if token not in stopwords and len(token) > 2}


def _coverage_ratio(claim: str, evidence_text: str) -> float:
    terms = _claim_terms(claim)
    if not terms:
        return 0.0
    evidence_lower = evidence_text.lower()
    covered = sum(1 for term in terms if term in evidence_lower)
    return covered / len(terms)


def _build_retrieval_query(claim: str, profile: ClaimProfile) -> str:
    """Rewrite current-sensitive claims so search targets current sources.

    Without this, a query like 'Joe Biden is the current president' may retrieve
    old biography pages about Joe Biden instead of current officeholder pages.
    """
    role_claim = profile.current_role_claim
    if not role_claim:
        if profile.needs_current_source:
            return f"{claim} current latest official source"
        return claim

    scope_part = f" of {role_claim.scope}" if role_claim.scope else ""
    return f"current {role_claim.role}{scope_part} official source"


def _predict_current_role_claim(
    claim: str,
    evidence: list[dict[str, Any]],
    profile: ClaimProfile,
) -> tuple[str, float, str] | None:
    """Handle claims where 'current' changes the truth value.

    The main rule is conservative: a current-role claim cannot be supported by
    an old biography match. It needs role-specific current evidence. If strong
    current/official evidence points to a different roleholder, label it refuted.
    """
    role_claim = profile.current_role_claim
    if role_claim is None:
        return None

    subject = role_claim.subject
    role = role_claim.role
    scope = role_claim.scope

    if not evidence:
        return (
            "Not Enough Evidence",
            0.50,
            "No public evidence was retrieved for this current-role claim.",
        )

    support_hits: list[dict[str, Any]] = []
    refute_hits: list[dict[str, Any]] = []
    historical_hits: list[dict[str, Any]] = []

    for item in evidence[:5]:
        combined_text = " ".join(
            [
                str(item.get("title", "")),
                str(item.get("source", "")),
                str(item.get("url", "")),
                str(item.get("text", "")),
            ]
        ).lower()
        relevance = float(item.get("relevance_score", 0.0))
        score = float(item.get("score", 0.0))
        trust = float(item.get("trust_score", 0.0))

        has_subject = _contains_entity(combined_text, subject)
        has_role = role.lower() in combined_text
        has_scope = not scope or _contains_entity(combined_text, scope)
        has_current_cue = _contains_phrase(combined_text, _CURRENT_EVIDENCE_CUES)
        has_historical_cue = _contains_phrase(combined_text, _HISTORICAL_ROLE_CUES)
        strong_source = trust >= 0.80 or any(
            domain in combined_text
            for domain in [
                "whitehouse.gov",
                ".gov",
                "official",
                "apple.com/leadership",
                "about.google",
                "microsoft.com/en-us/leadership",
            ]
        )

        if has_subject and has_role and has_scope and has_historical_cue and not has_current_cue:
            historical_hits.append(item)
            continue

        if has_subject and has_role and has_scope and (has_current_cue or strong_source) and relevance >= 0.10:
            support_hits.append(item)
            continue

        if (
            has_role
            and has_scope
            and not has_subject
            and strong_source
            and (has_current_cue or score >= 0.45 or relevance >= 0.15)
        ):
            refute_hits.append(item)

    if support_hits:
        best = support_hits[0]
        return (
            "Supported by Evidence",
            min(0.88, 0.62 + float(best.get("score", 0.0))),
            "Current-role evidence from a strong source matches the claimed person or organization.",
        )

    if refute_hits:
        best = refute_hits[0]
        scope_text = f" for {scope}" if scope else ""
        return (
            "Contradicted by Evidence",
            min(0.88, 0.60 + float(best.get("score", 0.0))),
            (
                f"The claim says {subject} is the current {role}{scope_text}, but the strongest current/official "
                "evidence points to the role without matching that claimed subject."
            ),
        )

    if historical_hits:
        best = historical_hits[0]
        return (
            "Contradicted by Evidence",
            min(0.84, 0.56 + float(best.get("score", 0.0))),
            (
                "The strongest matching evidence describes the claimed person in a past or former role, "
                "which does not support a claim about who holds the role currently."
            ),
        )

    return (
        "Not Enough Evidence",
        0.58,
        "The retrieved sources did not provide strong current-role evidence, so the app will not treat old matches as support.",
    )


def predict_from_evidence(
    claim: str,
    evidence: list[dict[str, Any]],
    profile: ClaimProfile | None = None,
) -> tuple[str, float, str]:
    """Assign an evidence label from retrieved passages.

    This deliberately avoids claiming certainty. It produces labels based on
    source-backed relevance signals and simple contradiction cues.
    """
    if profile and profile.current_role_claim:
        current_result = _predict_current_role_claim(claim, evidence, profile)
        if current_result is not None:
            return current_result

    if not evidence:
        return "Not Enough Evidence", 0.50, "No public evidence passages were retrieved for the claim."

    best = evidence[0]
    best_text = str(best.get("text", ""))
    best_score = float(best.get("score", 0.0))
    relevance = float(best.get("relevance_score", 0.0))
    coverage = _coverage_ratio(claim, best_text)

    if profile and profile.needs_current_source:
        combined = " ".join(
            [
                str(best.get("title", "")),
                str(best.get("source", "")),
                str(best.get("url", "")),
                best_text,
            ]
        ).lower()
        if not _contains_phrase(combined, _CURRENT_EVIDENCE_CUES) and float(best.get("trust_score", 0.0)) < 0.85:
            return (
                "Not Enough Evidence",
                0.56,
                "This is time-sensitive, and the strongest source does not clearly provide current evidence.",
            )

    if relevance < 0.08 or coverage < 0.30:
        return (
            "Not Enough Evidence",
            0.55,
            "The retrieved sources do not cover enough of the claim to support a reliable label.",
        )

    checked = [
        item for item in evidence[:5]
        if item.get("entailment_label") in {"entailment", "contradiction", "neutral"}
    ]
    support_hits = [
        item for item in checked
        if item.get("entailment_label") == "entailment"
        and float(item.get("entailment_confidence", 0.0)) >= 0.55
        and float(item.get("relevance_score", 0.0)) >= 0.08
    ]
    contradiction_hits = [
        item for item in checked
        if item.get("entailment_label") == "contradiction"
        and float(item.get("entailment_confidence", 0.0)) >= 0.55
        and float(item.get("relevance_score", 0.0)) >= 0.08
    ]

    if support_hits and contradiction_hits:
        best_support = max(support_hits, key=lambda item: float(item.get("entailment_confidence", 0.0)))
        best_contra = max(contradiction_hits, key=lambda item: float(item.get("entailment_confidence", 0.0)))
        if abs(float(best_support.get("entailment_confidence", 0.0)) - float(best_contra.get("entailment_confidence", 0.0))) < 0.12:
            return (
                "Conflicting Evidence",
                0.64,
                "Retrieved sources include both supporting and contradicting evidence, so the app does not force a one-sided label.",
            )

    if contradiction_hits:
        best_contra = max(contradiction_hits, key=lambda item: float(item.get("entailment_confidence", 0.0)))
        nli_conf = float(best_contra.get("entailment_confidence", 0.0))
        return (
            "Contradicted by Evidence",
            min(0.88, 0.52 + 0.32 * nli_conf + 0.08 * float(best_contra.get("trust_score", 0.0))),
            "The strongest relevant evidence contradicts the claim rather than merely mentioning similar words.",
        )

    if support_hits:
        best_support = max(support_hits, key=lambda item: float(item.get("entailment_confidence", 0.0)))
        nli_conf = float(best_support.get("entailment_confidence", 0.0))
        return (
            "Supported by Evidence",
            min(0.88, 0.52 + 0.30 * nli_conf + 0.08 * float(best_support.get("trust_score", 0.0))),
            "The strongest relevant evidence directly supports the claim relationship, not just the same keywords.",
        )

    claim_negative = _contains_phrase(claim, _NEGATION_CUES)
    evidence_negative = _contains_phrase(best_text, _NEGATION_CUES)

    if claim_negative != evidence_negative and relevance >= 0.18 and coverage >= 0.45:
        return (
            "Contradicted by Evidence",
            min(0.78, 0.50 + 0.20 * best_score),
            "The strongest evidence is relevant, but its wording appears to conflict with the claim.",
        )

    return (
        "Not Enough Evidence",
        min(0.70, 0.50 + 0.18 * best_score),
        "Related evidence was found, but the entailment check did not confirm direct support for the exact claim.",
    )




def _dedupe_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dedupe evidence from multiple planned queries while preserving score order."""
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in sorted(evidence, key=lambda x: float(x.get("score", 0.0)), reverse=True):
        key = f"{item.get('url', '')}|{item.get('passage_id', '')}|{item.get('text', '')[:80]}".lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _retrieve_with_plan(
    retriever: Any,
    search_plan: Any,
    method: str,
    top_k: int,
    alpha: float,
    max_pages: int,
) -> list[dict[str, Any]]:
    """Run retrieval across a small set of planned queries and merge the results."""
    all_evidence: list[dict[str, Any]] = []
    queries = search_plan.generated_queries or [search_plan.primary_query]
    # Use the first two queries by default to improve quality without making the app too slow.
    for query in queries[:2]:
        retrieved = retriever.retrieve(
            claim=query,
            method=method,
            top_k=top_k,
            alpha=alpha,
            max_pages=max_pages,
        )
        for item in retrieved:
            enriched = dict(item)
            enriched["planned_query"] = query
            all_evidence.append(enriched)
    return _dedupe_evidence(all_evidence)[:top_k]


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
    search_plan = plan_search_queries(claim, profile=profile, max_queries=4)
    retrieval_query = search_plan.primary_query or _build_retrieval_query(claim, profile)

    retriever = retriever or MultiSourceEvidenceRetriever(
        use_web=use_web,
        use_wikipedia=use_wikipedia,
        use_openalex=use_openalex,
    )

    evidence = _retrieve_with_plan(
        retriever=retriever,
        search_plan=search_plan,
        method=method,
        top_k=top_k,
        alpha=alpha,
        max_pages=max_pages,
    )
    evidence = add_entailment_scores(claim, evidence, max_items=min(5, top_k))

    label, confidence, explanation = predict_from_evidence(claim, evidence, profile=profile)

    if profile.needs_current_source:
        explanation = (
            f"{explanation} Current-source handling applied: old biography matches are not enough for time-sensitive claims."
        )

    summarized_evidence = summarize_evidence_batch(claim, evidence, max_items=max(8, top_k))
    synthesis = synthesize_evidence(
        claim=claim,
        evidence=summarized_evidence,
        predicted_label=label,
        confidence=confidence,
        explanation=explanation,
        profile=profile,
    )

    # One targeted re-retrieval pass when the first evidence set is weak. This mirrors
    # human fact-checking without creating an expensive or infinite search loop.
    if synthesis.needs_more_search and synthesis.rereview_query and retriever is not None:
        second_pass = retriever.retrieve(
            claim=synthesis.rereview_query,
            method=method,
            top_k=top_k,
            alpha=alpha,
            max_pages=max(2, max_pages // 2),
        )
        if second_pass:
            evidence = _dedupe_evidence(evidence + second_pass)[:top_k]
            evidence = add_entailment_scores(claim, evidence, max_items=min(5, top_k))
            label, confidence, explanation = predict_from_evidence(claim, evidence, profile=profile)
            if profile.needs_current_source:
                explanation = (
                    f"{explanation} Current-source handling applied: old biography matches are not enough for time-sensitive claims."
                )
            summarized_evidence = summarize_evidence_batch(claim, evidence, max_items=max(8, top_k))
            synthesis = synthesize_evidence(
                claim=claim,
                evidence=summarized_evidence,
                predicted_label=label,
                confidence=confidence,
                explanation=explanation,
                profile=profile,
            )

    profile_dict = {
        "category": profile.category,
        "needs_current_source": profile.needs_current_source,
        "reason": profile.reason,
        "current_role_claim": (
            None
            if profile.current_role_claim is None
            else {
                "subject": profile.current_role_claim.subject,
                "role": profile.current_role_claim.role,
                "scope": profile.current_role_claim.scope,
            }
        ),
    }

    result = {
        "claim": claim,
        "retrieval_query": retrieval_query,
        "predicted_label": label,
        "confidence": confidence,
        "explanation": explanation,
        "claim_profile": profile_dict,
        "search_plan": search_plan.to_dict(),
        "retrieval_method": method,
        "top_evidence": summarized_evidence,
        "synthesis": synthesis.to_dict(),
        "workflow_steps": [
            {"step": "Claim analysis", "status": "complete", "detail": profile.reason},
            {"step": "Query planning", "status": "complete", "detail": retrieval_query},
            {"step": "Evidence retrieval", "status": "complete", "detail": f"Retrieved {len(evidence)} ranked evidence passages."},
            {"step": "Evidence summarization", "status": "complete", "detail": f"Summarized {len(summarized_evidence)} passages."},
            {"step": "Evidence synthesis", "status": "complete", "detail": synthesis.evidence_strength},
            {"step": "Verdict evaluation", "status": "complete", "detail": label},
        ],
        "source_mode": "Live public evidence retrieval",
    }
    result["fact_check_report"] = build_markdown_report(result)
    return result
