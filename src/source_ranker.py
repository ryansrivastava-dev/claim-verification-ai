"""Transparent source-quality scoring for evidence retrieval.

These scores are only ranking signals. They do not prove that a source is true.
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
)
_MEDIUM_TRUST_DOMAINS = (
    "wikipedia.org",
    "wikidata.org",
    "britannica.com",
    "openalex.org",
    "crossref.org",
    "nature.com",
    "science.org",
    "nejm.org",
    "thelancet.com",
    "mayoclinic.org",
    "clevelandclinic.org",
    "history.com",
)


def source_trust_score(url: str, source_name: str = "") -> float:
    """Return a simple trust score from 0 to 1 for ranking display.

    This is deliberately transparent and conservative. It rewards official,
    educational, government, scientific, and reference sources, but it does not
    assert that any source is always correct.
    """
    parsed = urlparse(url or "")
    domain = parsed.netloc.lower().replace("www.", "")
    source_lower = source_name.lower()

    if domain.endswith(_HIGH_TRUST_SUFFIXES):
        return 0.95
    if any(domain.endswith(item) for item in _HIGH_TRUST_DOMAINS):
        return 0.94
    if any(domain.endswith(item) for item in _MEDIUM_TRUST_DOMAINS):
        return 0.82
    if "openalex" in source_lower:
        return 0.82
    if "wikipedia" in source_lower:
        return 0.82
    if domain:
        return 0.60
    return 0.45


def blend_relevance_and_trust(relevance: float, trust: float, trust_weight: float = 0.15) -> float:
    """Combine retrieval relevance with source trust for evidence ranking."""
    trust_weight = max(0.0, min(1.0, trust_weight))
    return (1.0 - trust_weight) * float(relevance) + trust_weight * float(trust)
