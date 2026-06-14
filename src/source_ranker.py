"""Transparent source-quality scoring for evidence retrieval.

These scores are ranking/display signals only. A high source score does not prove
that the claim is true; it means the source type is generally more appropriate
for evidence-grounded fact checking.
"""

from __future__ import annotations

from urllib.parse import urlparse


_HIGH_TRUST_SUFFIXES = (".gov", ".edu")
_HIGH_TRUST_DOMAINS = (
    "nih.gov",
    "ncbi.nlm.nih.gov",
    "cdc.gov",
    "who.int",
    "nasa.gov",
    "noaa.gov",
    "fda.gov",
    "bls.gov",
    "census.gov",
    "supremecourt.gov",
    "congress.gov",
    "whitehouse.gov",
    "state.gov",
    "house.gov",
    "senate.gov",
    "justice.gov",
    "apple.com",
    "microsoft.com",
    "about.google",
)
_SCHOLARLY_DOMAINS = (
    "openalex.org",
    "crossref.org",
    "nature.com",
    "science.org",
    "nejm.org",
    "thelancet.com",
    "sciencedirect.com",
    "springer.com",
    "wiley.com",
    "jstor.org",
    "acm.org",
    "ieee.org",
    "arxiv.org",
)
_REFERENCE_DOMAINS = (
    "wikipedia.org",
    "wikidata.org",
    "britannica.com",
)
_HEALTH_REFERENCE_DOMAINS = (
    "mayoclinic.org",
    "clevelandclinic.org",
    "healthline.com",
)
_NEWS_DOMAINS = (
    "apnews.com",
    "reuters.com",
    "bbc.com",
    "bbc.co.uk",
    "npr.org",
    "pbs.org",
    "nytimes.com",
    "washingtonpost.com",
    "cnn.com",
    "theguardian.com",
    "politico.com",
)


def _domain(url: str) -> str:
    """Return a normalized domain from a URL."""
    parsed = urlparse(url or "")
    return parsed.netloc.lower().replace("www.", "")


def source_category(url: str, source_name: str = "") -> str:
    """Classify the source type for display in the app and report."""
    domain = _domain(url)
    source_lower = source_name.lower()

    if domain.endswith(_HIGH_TRUST_SUFFIXES) or any(domain.endswith(item) for item in _HIGH_TRUST_DOMAINS):
        return "Official / government / institutional"
    if "openalex" in source_lower or any(domain.endswith(item) for item in _SCHOLARLY_DOMAINS):
        return "Scholarly"
    if "wikipedia" in source_lower or any(domain.endswith(item) for item in _REFERENCE_DOMAINS):
        return "Reference"
    if any(domain.endswith(item) for item in _HEALTH_REFERENCE_DOMAINS):
        return "Health reference"
    if any(domain.endswith(item) for item in _NEWS_DOMAINS):
        return "News"
    if domain:
        return "General web"
    return "Unknown source type"


def source_trust_score(url: str, source_name: str = "") -> float:
    """Return a simple source quality signal from 0 to 1.

    The score is deliberately transparent and conservative. It rewards official,
    educational, government, scholarly, and reference sources, but it does not
    assert that any source is always correct.
    """
    category = source_category(url, source_name)
    if category == "Official / government / institutional":
        return 0.95
    if category == "Scholarly":
        return 0.86
    if category == "Reference":
        return 0.82
    if category == "Health reference":
        return 0.78
    if category == "News":
        return 0.72
    if category == "General web":
        return 0.60
    return 0.45


def trust_label(score: float) -> str:
    """Convert a numeric source score into a readable label."""
    score = float(score)
    if score >= 0.90:
        return "High trust signal"
    if score >= 0.78:
        return "Strong trust signal"
    if score >= 0.68:
        return "Moderate trust signal"
    if score >= 0.55:
        return "Basic trust signal"
    return "Low/unknown trust signal"


def evidence_strength_label(score: float, trust: float, relevance: float) -> str:
    """Human-readable explanation of evidence strength."""
    score = float(score)
    trust = float(trust)
    relevance = float(relevance)
    if score >= 0.75 and trust >= 0.78 and relevance >= 0.35:
        return "Strong evidence candidate"
    if score >= 0.55 and relevance >= 0.20:
        return "Moderate evidence candidate"
    if relevance >= 0.08:
        return "Weak/background evidence"
    return "Low relevance evidence"


def blend_relevance_and_trust(relevance: float, trust: float, trust_weight: float = 0.15) -> float:
    """Combine retrieval relevance with source trust for evidence ranking."""
    trust_weight = max(0.0, min(1.0, trust_weight))
    return (1.0 - trust_weight) * float(relevance) + trust_weight * float(trust)


def source_quality_report(url: str, source_name: str, relevance: float, final_score: float) -> dict[str, str | float]:
    """Return source quality metadata for UI/reporting."""
    trust = source_trust_score(url, source_name)
    return {
        "source_category": source_category(url, source_name),
        "trust_label": trust_label(trust),
        "evidence_strength_label": evidence_strength_label(final_score, trust, relevance),
        "trust_score": trust,
    }
