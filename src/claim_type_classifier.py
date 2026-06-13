"""Lightweight claim type and freshness detection."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CurrentRoleClaim:
    """Parsed structure for claims like 'X is the current president'."""

    subject: str
    role: str
    scope: str


@dataclass(frozen=True)
class ClaimProfile:
    """A simple description of what kind of claim the user entered."""

    category: str
    needs_current_source: bool
    reason: str
    current_role_claim: CurrentRoleClaim | None = None


_CURRENT_PATTERNS = [
    r"\bcurrent\b",
    r"\bcurrently\b",
    r"\bnow\b",
    r"\btoday\b",
    r"\byesterday\b",
    r"\bthis week\b",
    r"\bthis month\b",
    r"\bright now\b",
    r"\blatest\b",
    r"\bincumbent\b",
    r"\bpresent\b",
    r"\bprice\b",
    r"\bstock\b",
    r"\bschedule\b",
    r"\bscore\b",
    r"\bweather\b",
    r"\bceo\b",
    r"\bpresident\b",
    r"\bprime minister\b",
    r"\bmayor\b",
    r"\bgovernor\b",
    r"\bhead coach\b",
]

_ROLE_WORDS = [
    "president",
    "ceo",
    "chief executive officer",
    "prime minister",
    "mayor",
    "governor",
    "head coach",
    "chairman",
    "leader",
]

_CATEGORY_KEYWORDS = [
    ("current public office / leadership", ["current", "currently", "president", "ceo", "prime minister", "mayor", "governor", "head coach"]),
    ("medical/scientific", ["disease", "blood pressure", "stroke", "vaccine", "study", "protein", "cell", "virus", "bacteria", "climate"]),
    ("sports/current events", ["won", "score", "game", "season", "team", "player", "match", "tournament"]),
    ("business/organization", ["company", "stock", "market", "revenue", "founded", "headquarters"]),
    ("geography/history", ["capital", "located", "country", "city", "war", "founded", "born", "died"]),
    ("general knowledge", []),
]


def _clean_piece(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip(" .,:;!?\"'")
    return text.strip()


def extract_current_role_claim(claim: str) -> CurrentRoleClaim | None:
    """Extract claims such as 'Joe Biden is the current president'.

    This does not try to solve the claim. It only identifies claims where the
    word 'current' changes the truth value and the verifier must require fresh
    role evidence rather than old biography pages.
    """
    text = _clean_piece(claim)
    if not text:
        return None

    role_pattern = "|".join(re.escape(role) for role in sorted(_ROLE_WORDS, key=len, reverse=True))

    patterns = [
        # Joe Biden is the current president of the United States
        rf"^(?P<subject>.+?)\s+(?:is|are)\s+(?:the\s+)?(?:current|currently\s+the|incumbent|present)\s+(?P<role>{role_pattern})(?:\s+of\s+(?P<scope>.+?))?$",
        # Joe Biden is currently president of the United States
        rf"^(?P<subject>.+?)\s+(?:is|are)\s+currently\s+(?:the\s+)?(?P<role>{role_pattern})(?:\s+of\s+(?P<scope>.+?))?$",
        # The current president of the United States is Joe Biden
        rf"^(?:the\s+)?(?:current|incumbent|present)\s+(?P<role>{role_pattern})(?:\s+of\s+(?P<scope>.+?))?\s+(?:is|are)\s+(?P<subject>.+?)$",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        subject = _clean_piece(match.group("subject"))
        role = _clean_piece(match.group("role")).lower()
        scope = _clean_piece(match.groupdict().get("scope") or "")
        if subject and role:
            # Useful default for common short claim: 'X is the current president'.
            if role == "president" and not scope:
                scope = "United States"
            return CurrentRoleClaim(subject=subject, role=role, scope=scope)
    return None


def detect_claim_type(claim: str) -> ClaimProfile:
    """Classify a claim into a broad category and flag freshness needs."""
    claim_text = str(claim).strip()
    lower = claim_text.lower()

    current_role_claim = extract_current_role_claim(claim_text)
    needs_current = bool(current_role_claim) or any(
        re.search(pattern, claim_text, flags=re.IGNORECASE) for pattern in _CURRENT_PATTERNS
    )

    category = "general knowledge"
    for candidate, keywords in _CATEGORY_KEYWORDS:
        if any(keyword.lower() in lower for keyword in keywords):
            category = candidate
            break

    if current_role_claim:
        reason = (
            "The claim is about a current office or leadership role, so the system requires fresh, "
            "role-specific evidence rather than older biographical matches."
        )
        category = "current public office / leadership"
    elif needs_current:
        reason = "The claim contains time-sensitive wording, so the answer requires recently updated sources."
    elif category == "medical/scientific":
        reason = "The claim appears scientific or medical, so high-quality evidence is especially important."
    else:
        reason = "The claim appears suitable for source-grounded retrieval, but coverage depends on available public sources."

    return ClaimProfile(
        category=category,
        needs_current_source=needs_current,
        reason=reason,
        current_role_claim=current_role_claim,
    )
