"""Transparent query planning for web-grounded fact checking.

This module makes the project feel closer to a real fact-checking workflow:
it does not send the raw claim to search once and stop. Instead, it produces a
small set of targeted search queries based on claim type, time sensitivity, and
important claim terms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any

try:
    from claim_type_classifier import ClaimProfile, detect_claim_type
    from entailment_checker import extract_simple_relation
except ImportError:  # pragma: no cover
    from src.claim_type_classifier import ClaimProfile, detect_claim_type
    from src.entailment_checker import extract_simple_relation


@dataclass(frozen=True)
class SearchPlan:
    """Search strategy produced before retrieval begins."""

    claim: str
    primary_query: str
    generated_queries: list[str]
    strategy_notes: list[str]
    needs_current_source: bool
    claim_category: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "by", "from", "is", "are", "was", "were", "be", "being", "been", "has", "have",
    "had", "that", "this", "it", "as", "at", "into", "than", "then", "current",
    "currently", "today", "now", "latest", "claim", "says", "said",
}


def _clean_query(query: str) -> str:
    query = re.sub(r"\s+", " ", str(query)).strip(" .,:;!?\"'")
    return query.strip()


def _important_terms(claim: str, max_terms: int = 8) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", claim.lower())
    terms: list[str] = []
    for token in tokens:
        if token in _STOPWORDS or len(token) <= 2:
            continue
        if token not in terms:
            terms.append(token)
    return terms[:max_terms]


def _dedupe_queries(queries: list[str], max_queries: int) -> list[str]:
    seen: set[str] = set()
    clean: list[str] = []
    for query in queries:
        query = _clean_query(query)
        if not query:
            continue
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        clean.append(query)
        if len(clean) >= max_queries:
            break
    return clean



def _relation_queries(claim: str) -> list[str]:
    """Generate extra queries for simple subject-predicate claims.

    This improves cases where the raw claim retrieves related but incomplete
    evidence. For example, "Mars is blue" should also search for canonical Mars
    facts, not only pages that happen to mention blue sunsets.
    """
    relation = extract_simple_relation(claim)
    if relation is None:
        return []
    subject, predicate = relation
    subject = _clean_query(subject)
    predicate = _clean_query(predicate)
    if not subject or not predicate:
        return []

    queries = [
        f"{subject} facts reliable source",
        f"{subject} {predicate} true false",
        f"{subject} known as evidence",
    ]

    predicate_lower = predicate.lower()
    if predicate_lower.startswith(("in ", "at ", "located")):
        queries.insert(0, f"where is {subject} located official source")
    elif len(predicate.split()) <= 3:
        queries.insert(0, f"what is {subject} known as")

    return queries


def plan_search_queries(claim: str, profile: ClaimProfile | None = None, max_queries: int = 4) -> SearchPlan:
    """Create a small, transparent search plan for a claim.

    The goal is not to be magical. It is to avoid the common failure where the
    app searches only the exact user claim and retrieves related but wrong pages.
    """
    claim = _clean_query(claim)
    if not claim:
        raise ValueError("Claim cannot be empty.")

    profile = profile or detect_claim_type(claim)
    notes: list[str] = []
    queries: list[str] = []

    relation_queries = _relation_queries(claim)

    role_claim = profile.current_role_claim
    if role_claim:
        scope_part = f" of {role_claim.scope}" if role_claim.scope else ""
        primary = f"current {role_claim.role}{scope_part} official source"
        queries.extend(
            [
                primary,
                f"who is the current {role_claim.role}{scope_part}",
                f"{role_claim.role}{scope_part} official government website current",
                f"{role_claim.subject} {role_claim.role}{scope_part} current fact check",
            ]
        )
        notes.append(
            "Current-role claim detected; search targets the role and official current sources before biographies."
        )
    elif profile.needs_current_source:
        primary = f"{claim} current latest official source"
        queries.extend([primary, f"{claim} latest reliable source", f"{claim} official source"])
        notes.append("Time-sensitive wording detected; queries include current/latest source terms.")
    elif profile.category in {"health/medical", "science/technology", "climate/environment"}:
        primary = claim
        terms = " ".join(_important_terms(claim))
        queries.extend(
            [
                claim,
                f"{claim} fact check",
                f"{terms} scientific evidence study",
                f"{terms} official scientific source",
                f"{terms} NIH CDC WHO NOAA NASA",
                f"{claim} systematic review evidence",
            ]
        )
        notes.append("Science, climate, or health claim detected; queries prioritize scholarly, official, and domain-specific evidence.")
    else:
        primary = claim
        terms = " ".join(_important_terms(claim))
        queries.extend([claim, f"{terms} official source", f"{claim} fact check"])
        notes.append("General fact claim detected; queries search direct wording and official/source-backed phrasing.")

    if relation_queries:
        # Insert relation-focused queries near the front, after the user's original claim.
        queries = [queries[0]] + relation_queries + queries[1:] if queries else relation_queries
        notes.append("Simple claim relationship detected; added entity-focused searches to reduce keyword-match false positives.")

    queries = _dedupe_queries(queries, max_queries=max_queries)
    primary = queries[0] if queries else primary

    return SearchPlan(
        claim=claim,
        primary_query=primary,
        generated_queries=queries,
        strategy_notes=notes,
        needs_current_source=profile.needs_current_source,
        claim_category=profile.category,
    )
