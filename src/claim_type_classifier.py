"""Lightweight claim type and freshness detection."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ClaimProfile:
    """A simple description of what kind of claim the user entered."""

    category: str
    needs_current_source: bool
    reason: str


_CURRENT_PATTERNS = [
    r"\bcurrent\b",
    r"\btoday\b",
    r"\byesterday\b",
    r"\bthis week\b",
    r"\bthis month\b",
    r"\bright now\b",
    r"\blatest\b",
    r"\bcurrently\b",
    r"\bprice\b",
    r"\bstock\b",
    r"\bschedule\b",
    r"\bscore\b",
    r"\bweather\b",
    r"\bCEO\b",
    r"\bpresident\b",
]

_CATEGORY_KEYWORDS = [
    ("medical/scientific", ["disease", "blood pressure", "stroke", "vaccine", "study", "protein", "cell", "virus", "bacteria", "climate"]),
    ("sports/current events", ["won", "score", "game", "season", "team", "player", "match", "tournament"]),
    ("business/organization", ["company", "CEO", "stock", "market", "revenue", "founded", "headquarters"]),
    ("geography/history", ["capital", "located", "country", "city", "war", "founded", "born", "died"]),
    ("general knowledge", []),
]


def detect_claim_type(claim: str) -> ClaimProfile:
    """Classify a claim into a broad category and flag freshness needs."""
    claim_text = str(claim).strip()
    lower = claim_text.lower()

    needs_current = any(re.search(pattern, claim_text, flags=re.IGNORECASE) for pattern in _CURRENT_PATTERNS)

    category = "general knowledge"
    for candidate, keywords in _CATEGORY_KEYWORDS:
        if any(keyword.lower() in lower for keyword in keywords):
            category = candidate
            break

    if needs_current:
        reason = "The claim contains time-sensitive wording, so the answer may require recently updated sources."
    elif category == "medical/scientific":
        reason = "The claim appears scientific or medical, so high-quality evidence is especially important."
    else:
        reason = "The claim appears suitable for source-grounded retrieval, but coverage depends on available public sources."

    return ClaimProfile(category=category, needs_current_source=needs_current, reason=reason)
