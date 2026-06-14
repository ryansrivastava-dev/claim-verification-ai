"""Lightweight claim type and current-context detection.

The classifier is intentionally transparent. It does not decide whether a claim
is true. It only decides what kind of claim the system is handling so retrieval
and UI explanations can be more accurate.
"""

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


# Words that make a claim time-sensitive even when it is not a role claim.
# Keep public-office titles out of this list; those are handled by role-specific
# patterns below so claims like "George Washington was president" are not treated
# as current just because they contain the word "president".
_TIME_SENSITIVE_PATTERNS = [
    r"\bcurrent\b",
    r"\bcurrently\b",
    r"\bnow\b",
    r"\btoday\b",
    r"\byesterday\b",
    r"\bthis week\b",
    r"\bthis month\b",
    r"\bthis year\b",
    r"\bright now\b",
    r"\blatest\b",
    r"\brecent\b",
    r"\bnewest\b",
    r"\bincumbent\b",
    r"\bpresent\b",
    r"\bprice\b",
    r"\bstock\b",
    r"\bschedule\b",
    r"\bscore\b",
    r"\bstandings\b",
    r"\bweather\b",
    r"\bforecast\b",
    r"\branking\b",
]

_ROLE_WORDS = [
    "chief executive officer",
    "prime minister",
    "president",
    "governor",
    "head coach",
    "chairman",
    "chancellor",
    "speaker",
    "senator",
    "mayor",
    "ceo",
    "leader",
]

# Ordered from most specific to broadest. Earlier categories win.
_CATEGORY_RULES: list[tuple[str, list[str]]] = [
    (
        "climate/environment",
        [
            "climate",
            "global warming",
            "warming",
            "greenhouse gas",
            "carbon dioxide",
            "carbon emissions",
            "emissions",
            "sea level",
            "drought",
            "heat wave",
            "wildfire",
            "pollution",
            "environment",
            "renewable",
            "fossil fuel",
            "temperature rise",
            "extreme weather",
        ],
    ),
    (
        "health/medical",
        [
            "disease",
            "blood pressure",
            "stroke",
            "vaccine",
            "vaccination",
            "medicine",
            "medical",
            "drug",
            "cancer",
            "covid",
            "virus",
            "bacteria",
            "antibiotic",
            "diabetes",
            "infection",
            "heart disease",
            "clinical trial",
            "symptom",
            "treatment",
        ],
    ),
    (
        "science/technology",
        [
            "study",
            "research",
            "protein",
            "cell",
            "dna",
            "gene",
            "planet",
            "space",
            "nasa",
            "satellite",
            "telescope",
            "artificial intelligence",
            "machine learning",
            "software",
            "algorithm",
            "computer",
            "data center",
            "quantum",
        ],
    ),
    (
        "politics/government",
        [
            "election",
            "senate",
            "congress",
            "supreme court",
            "law",
            "bill",
            "policy",
            "vote",
            "voted",
            "administration",
            "president",
            "governor",
            "mayor",
            "prime minister",
            "minister",
            "parliament",
        ],
    ),
    (
        "business/economics",
        [
            "company",
            "stock",
            "market",
            "revenue",
            "profit",
            "inflation",
            "gdp",
            "unemployment",
            "economy",
            "business",
            "headquarters",
            "founded",
            "ceo",
            "acquisition",
            "merger",
            "price",
            "sales",
        ],
    ),
    (
        "sports",
        [
            "won",
            "score",
            "game",
            "season",
            "team",
            "player",
            "match",
            "tournament",
            "championship",
            "coach",
            "nba",
            "nfl",
            "mlb",
            "nhl",
            "soccer",
            "football",
            "basketball",
        ],
    ),
    (
        "geography/place",
        [
            "capital",
            "located",
            "country",
            "city",
            "state",
            "province",
            "continent",
            "border",
            "river",
            "mountain",
            "population",
            "area",
        ],
    ),
    (
        "history",
        [
            "war",
            "revolution",
            "treaty",
            "ancient",
            "empire",
            "dynasty",
            "born",
            "died",
            "assassinated",
            "founded in",
            "created in",
            "independence",
            "in 19",
            "in 20",
        ],
    ),
    (
        "entertainment/culture",
        [
            "movie",
            "film",
            "actor",
            "actress",
            "singer",
            "song",
            "album",
            "artist",
            "oscar",
            "grammy",
            "netflix",
            "disney",
            "book",
            "novel",
        ],
    ),
    (
        "education",
        [
            "university",
            "college",
            "school",
            "student",
            "tuition",
            "admission",
            "acceptance rate",
            "ranking",
            "degree",
            "campus",
        ],
    ),
    ("general knowledge", []),
]


def _clean_piece(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip(" .,:;!?\"'")
    return text.strip()


def _role_pattern() -> str:
    return "|".join(re.escape(role) for role in sorted(_ROLE_WORDS, key=len, reverse=True))


def extract_current_role_claim(claim: str) -> CurrentRoleClaim | None:
    """Extract present-tense leadership/office claims.

    This catches both explicit current claims ("X is the current president") and
    present-tense role claims ("Tim Cook is CEO of Apple"). It deliberately does
    not match past-tense claims like "X was president".
    """
    text = _clean_piece(claim)
    if not text:
        return None

    role_pattern = _role_pattern()
    patterns = [
        # Joe Biden is the current president of the United States
        rf"^(?P<subject>.+?)\s+(?:is|are)\s+(?:the\s+)?(?:current|currently\s+the|incumbent|present)\s+(?P<role>{role_pattern})(?:\s+of\s+(?P<scope>.+?))?\??$",
        # Joe Biden is currently president of the United States
        rf"^(?P<subject>.+?)\s+(?:is|are)\s+currently\s+(?:the\s+)?(?P<role>{role_pattern})(?:\s+of\s+(?P<scope>.+?))?\??$",
        # Tim Cook is CEO of Apple / Donald Trump is president of the United States
        rf"^(?P<subject>.+?)\s+(?:is|are)\s+(?:the\s+)?(?P<role>{role_pattern})(?:\s+of\s+(?P<scope>.+?))\??$",
        # The current president of the United States is Joe Biden
        rf"^(?:the\s+)?(?:current|incumbent|present)\s+(?P<role>{role_pattern})(?:\s+of\s+(?P<scope>.+?))?\s+(?:is|are)\s+(?P<subject>.+?)\??$",
        # Who is the current CEO of Apple?
        rf"^who\s+(?:is|are)\s+(?:the\s+)?(?:current|incumbent|present)?\s*(?P<role>{role_pattern})(?:\s+of\s+(?P<scope>.+?))?\??$",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        subject = _clean_piece(match.groupdict().get("subject") or "unknown")
        role = _clean_piece(match.group("role")).lower()
        scope = _clean_piece(match.groupdict().get("scope") or "")
        if role:
            if role == "president" and not scope:
                scope = "United States"
            return CurrentRoleClaim(subject=subject, role=role, scope=scope)
    return None


def _contains_time_sensitive_language(text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in _TIME_SENSITIVE_PATTERNS)


def _detect_category(lower: str) -> str:
    for candidate, keywords in _CATEGORY_RULES:
        if any(keyword in lower for keyword in keywords):
            return candidate
    return "general knowledge"


def detect_claim_type(claim: str) -> ClaimProfile:
    """Classify a claim into a broad category and flag current-source needs."""
    claim_text = str(claim).strip()
    lower = claim_text.lower()

    current_role_claim = extract_current_role_claim(claim_text)
    needs_current = bool(current_role_claim) or _contains_time_sensitive_language(claim_text)
    category = _detect_category(lower)

    if current_role_claim:
        category = "current public office / leadership"
        reason = (
            "The claim is about a current office or leadership role, so the system requires fresh, "
            "role-specific evidence rather than older biographical matches."
        )
    elif needs_current:
        reason = "The claim contains time-sensitive wording, so the answer requires recently updated sources."
    elif category == "climate/environment":
        reason = "The claim appears related to climate or environmental evidence, so official and scientific sources are prioritized."
    elif category == "health/medical":
        reason = "The claim appears health or medical related, so high-quality medical and scientific evidence is especially important."
    elif category == "science/technology":
        reason = "The claim appears scientific or technical, so source-backed evidence and scholarly sources are useful."
    else:
        reason = "The claim appears suitable for source-grounded retrieval, but coverage depends on available public sources."

    return ClaimProfile(
        category=category,
        needs_current_source=needs_current,
        reason=reason,
        current_role_claim=current_role_claim,
    )
