"""Claim-evidence entailment checks.

This module prevents a common fact-checking failure: treating related evidence
as supporting evidence. Retrieval can find pages that mention the same entities
without proving the exact claim. The entailment layer checks whether an evidence
passage supports, contradicts, or is neutral toward the claim.

The implementation uses an optional lightweight NLI cross-encoder when it can be
loaded. If the model is unavailable, the deterministic fallback stays
conservative: it only supports a claim when the passage directly expresses the
same relationship. The fallback also includes a small generic contradiction
check for mutually exclusive descriptors, so obvious evidence like "Mars is
called the Red Planet" can refute "Mars is blue" without turning the app into
a brittle keyword matcher.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache
import math
import os
import re
from typing import Any


NLI_MODEL_NAME = os.getenv("NLI_MODEL_NAME", "cross-encoder/nli-MiniLM2-L6-H768")
NLI_DISABLED = os.getenv("DISABLE_NLI_MODEL", "0").lower() in {"1", "true", "yes"}

# Fallback-only semantic contrast groups. These are not the main reasoning system;
# the NLI model is preferred when available. The groups provide a transparent
# safety net for common mutually exclusive descriptors when the deployed app
# cannot load the NLI model. This is intentionally broader than a one-off list
# for any single example.
_MUTUALLY_EXCLUSIVE_DESCRIPTOR_GROUPS: list[set[str]] = [
    {"red", "blue", "green", "yellow", "orange", "white", "black", "brown", "gray", "grey", "purple", "pink"},
    {"hot", "warm", "cold", "cool", "freezing"},
    {"increase", "increases", "increased", "rise", "rises", "rising", "decrease", "decreases", "decreased", "fall", "falls", "falling"},
    {"legal", "illegal", "lawful", "unlawful"},
    {"true", "false", "correct", "incorrect"},
    {"country", "state", "province", "territory", "city", "capital"},
    {"cheese", "rock", "rocky", "metal", "gas", "ice"},
]

_ENTITY_DESCRIPTOR_PATTERNS = [
    r"{subject}.{{0,180}}(?:called|known as|referred to as|nicknamed)\s+(?:the\s+)?(?P<descriptor>[a-z][a-z\- ]{{1,60}})",
    r"{subject}.{{0,180}}(?:is|are|was|were|appears|appear|looks|look)\s+(?:mostly\s+|often\s+|commonly\s+|generally\s+|a\s+|an\s+|the\s+)?(?P<descriptor>[a-z][a-z\- ]{{1,60}})",
]


@dataclass(frozen=True)
class EntailmentResult:
    """Result from comparing one evidence passage against one claim."""

    label: str  # entailment | contradiction | neutral
    confidence: float
    explanation: str
    method: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    highest = max(values)
    exps = [math.exp(v - highest) for v in values]
    total = sum(exps)
    if total == 0:
        return [0.0 for _ in exps]
    return [v / total for v in exps]


@lru_cache(maxsize=1)
def _load_nli_model():  # pragma: no cover - depends on optional network/model cache
    if NLI_DISABLED:
        raise RuntimeError("NLI model disabled by DISABLE_NLI_MODEL.")
    from sentence_transformers import CrossEncoder

    return CrossEncoder(NLI_MODEL_NAME)


def _important_terms(text: str) -> list[str]:
    stop = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by", "from",
        "is", "are", "was", "were", "be", "being", "been", "has", "have", "had", "does", "did",
        "not", "that", "this", "it", "as", "at", "into", "current", "currently", "today", "now",
        "there", "substantial", "likely", "claim", "says", "said",
    }
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z0-9]+", str(text).lower()):
        if token not in stop and len(token) > 2 and token not in terms:
            terms.append(token)
    return terms


def _term_coverage(claim: str, evidence: str) -> float:
    terms = _important_terms(claim)
    if not terms:
        return 0.0
    lower = str(evidence).lower()
    return sum(1 for term in terms if term in lower) / len(terms)


def _simple_relation(claim: str) -> tuple[str, str] | None:
    """Extract broad subject/predicate for short copular claims.

    This is not meant to solve all language. It is used only by the conservative
    fallback to avoid supporting a claim when the evidence merely mentions the
    same words in a different context.
    """
    text = re.sub(r"\s+", " ", str(claim).strip(" .!?"))
    patterns = [
        r"^(.+?)\s+(?:is|are|was|were)\s+(?:commonly\s+|often\s+|widely\s+)?(?:called|known as|referred to as|nicknamed)\s+(?:a|an|the)?\s*(.+)$",
        r"^(.+?)\s+(?:is|are|was|were)\s+(?:a|an|the)?\s*(.+)$",
        r"^(.+?)\s+(?:has|have|had)\s+(.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            subject = match.group(1).strip(" ,")
            predicate = match.group(2).strip(" ,")
            if 1 <= len(subject.split()) <= 8 and 1 <= len(predicate.split()) <= 10:
                return subject, predicate
    return None



def extract_simple_relation(claim: str) -> tuple[str, str] | None:
    """Public wrapper used by query planning and tests."""
    return _simple_relation(claim)


def _normalize_descriptor_terms(text: str) -> set[str]:
    """Normalize a short descriptor phrase into comparable terms."""
    aliases = {
        "reddish": "red",
        "bluish": "blue",
        "grey": "gray",
        "rocky": "rock",
    }
    terms: set[str] = set()
    for token in re.findall(r"[a-z]+", str(text).lower()):
        if token in {"the", "a", "an", "of", "to", "in", "on", "at", "and", "or", "planet", "body", "object"}:
            continue
        if len(token) <= 1:
            continue
        terms.add(aliases.get(token, token))
    return terms


def _find_contrast_group(term: str) -> set[str] | None:
    for group in _MUTUALLY_EXCLUSIVE_DESCRIPTOR_GROUPS:
        if term in group:
            return group
    return None


def _extract_evidence_descriptors(subject_terms: list[str], evidence_lower: str) -> list[str]:
    """Extract descriptors explicitly attached to the claim subject in evidence."""
    if not subject_terms:
        return []
    subject_pattern = r"\b" + r"\W+".join(map(re.escape, subject_terms)) + r"\b"
    descriptors: list[str] = []
    for pattern_template in _ENTITY_DESCRIPTOR_PATTERNS:
        pattern = pattern_template.format(subject=subject_pattern)
        for match in re.finditer(pattern, evidence_lower, flags=re.DOTALL):
            descriptor = re.sub(r"\s+", " ", match.group("descriptor")).strip(" .,:;!?\"'")
            if descriptor and descriptor not in descriptors:
                descriptors.append(descriptor)
    return descriptors


def _fallback_descriptor_contradiction(subject: str, predicate: str, evidence_lower: str) -> EntailmentResult | None:
    """Detect clear descriptor conflicts for simple subject-predicate claims.

    Example: claim "Mars is blue" vs evidence "Mars is called the Red Planet".
    This is a fallback guardrail, not the primary verifier.
    """
    subject_terms = _important_terms(subject)
    predicate_terms = _normalize_descriptor_terms(predicate)
    if not subject_terms or not predicate_terms:
        return None

    evidence_descriptors = _extract_evidence_descriptors(subject_terms, evidence_lower)
    if not evidence_descriptors:
        return None

    evidence_terms = set()
    for descriptor in evidence_descriptors:
        evidence_terms.update(_normalize_descriptor_terms(descriptor))

    for predicate_term in predicate_terms:
        group = _find_contrast_group(predicate_term)
        if not group:
            continue
        conflicting_terms = (group - {predicate_term}) & evidence_terms
        if conflicting_terms:
            conflict = sorted(conflicting_terms)[0]
            return EntailmentResult(
                "contradiction",
                0.72,
                (
                    "The evidence explicitly attaches a mutually exclusive descriptor "
                    f"('{conflict}') to the claim subject instead of the claimed descriptor "
                    f"('{predicate_term}')."
                ),
                "nli-fallback-structured-contradiction",
            )
    return None



def _capital_relation_entails(claim_lower: str, evidence_lower: str) -> bool:
    """Detect swapped capital relation: 'capital of X is Y' vs 'Y is the capital of X'."""
    match = re.search(r"capital of ([a-z .'-]+?) is ([a-z .'-]+)$", claim_lower.strip(" .!?"))
    if not match:
        return False
    place = match.group(1).strip()
    capital = match.group(2).strip()
    return capital in evidence_lower and "capital" in evidence_lower and place in evidence_lower


def _coverage_entails_non_copular(claim: str, evidence: str, coverage: float) -> EntailmentResult | None:
    """Conservative high-coverage support for claims that are not simple X-is-Y relations."""
    relation = _simple_relation(claim)
    if relation is not None:
        return None
    if coverage >= 0.60 and _has_negation(claim) == _has_negation(evidence):
        return EntailmentResult(
            "entailment",
            min(0.70, 0.52 + 0.20 * coverage),
            "Most key claim terms appear in the evidence with no obvious contradiction.",
            "conservative-fallback-high-coverage",
        )
    return None

def _has_negation(text: str) -> bool:
    return bool(
        re.search(
            r"\b(no|not|never|false|incorrect|does not|did not|is not|are not|was not|were not|failed to)\b",
            str(text).lower(),
        )
    )


def _fallback_entailment(claim: str, evidence: str) -> EntailmentResult:
    claim_clean = re.sub(r"\s+", " ", str(claim).strip())
    evidence_clean = re.sub(r"\s+", " ", str(evidence).strip())
    claim_lower = claim_clean.lower().strip(" .!?\"")
    evidence_lower = evidence_clean.lower()

    if not evidence_clean:
        return EntailmentResult("neutral", 0.50, "No evidence text was available.", "conservative-fallback")

    coverage = _term_coverage(claim_clean, evidence_clean)

    # Direct statement match is the only high-confidence support in the fallback.
    if claim_lower and claim_lower in evidence_lower:
        return EntailmentResult("entailment", 0.86, "The evidence directly contains the claim wording.", "conservative-fallback")

    if _capital_relation_entails(claim_lower, evidence_lower):
        return EntailmentResult(
            "entailment",
            0.70,
            "The evidence states the same capital relationship in reversed wording.",
            "conservative-fallback-relation",
        )

    high_coverage_support = _coverage_entails_non_copular(claim_clean, evidence_clean, coverage)
    if high_coverage_support is not None:
        return high_coverage_support

    if coverage < 0.45:
        return EntailmentResult("neutral", 0.56, "The evidence does not cover enough key claim terms.", "conservative-fallback")

    claim_neg = _has_negation(claim_clean)
    evidence_neg = _has_negation(evidence_clean)
    if claim_neg != evidence_neg and coverage >= 0.60:
        return EntailmentResult(
            "contradiction",
            0.70,
            "The claim and evidence use opposing negation cues.",
            "conservative-fallback",
        )

    relation = _simple_relation(claim_clean)
    if relation:
        subject, predicate = relation
        subject_terms = _important_terms(subject)
        predicate_terms = _important_terms(predicate)
        if subject_terms and predicate_terms:
            subject_present = all(term in evidence_lower for term in subject_terms)
            predicate_present = all(term in evidence_lower for term in predicate_terms)
            # Related wording alone is not enough. The fallback requires the subject
            # and predicate to appear near each other in a short span.
            if subject_present and predicate_present:
                subject_pattern = r"\b" + r"\W+".join(map(re.escape, subject_terms)) + r"\b"
                predicate_pattern = r"\b" + r"\W+".join(map(re.escape, predicate_terms)) + r"\b"
                relation_window_pattern = (
                    subject_pattern
                    + r".{0,120}\b(?:is|are|was|were|appears|appear|looks|look|called|known as|referred to as)\b(?P<window>.{0,160})"
                )
                for relation_match in re.finditer(relation_window_pattern, evidence_lower, flags=re.DOTALL):
                    window = relation_match.group("window")
                    if all(term in window for term in predicate_terms):
                        return EntailmentResult(
                            "entailment",
                            0.68,
                            "The evidence directly states the same subject-predicate relationship as the claim.",
                            "conservative-fallback",
                        )
        descriptor_contradiction = _fallback_descriptor_contradiction(subject, predicate, evidence_lower)
        if descriptor_contradiction is not None:
            return descriptor_contradiction

        return EntailmentResult(
            "neutral",
            0.62,
            "The evidence is related, but it does not directly establish the claim relationship.",
            "conservative-fallback",
        )

    return EntailmentResult(
        "neutral",
        0.58,
        "The evidence is related, but the fallback checker cannot confirm direct support.",
        "conservative-fallback",
    )


def check_entailment(claim: str, evidence: str, *, prefer_model: bool = True) -> EntailmentResult:
    """Classify whether evidence entails, contradicts, or is neutral to a claim."""
    if prefer_model and not NLI_DISABLED:
        try:  # pragma: no cover - model availability varies by environment
            model = _load_nli_model()
            raw_scores = model.predict([(str(evidence), str(claim))], convert_to_numpy=True)
            scores = raw_scores[0].tolist() if hasattr(raw_scores[0], "tolist") else list(raw_scores[0])
            probs = _softmax([float(score) for score in scores])
            id2label = getattr(getattr(model, "model", None), "config", None)
            id2label = getattr(id2label, "id2label", None) or {}
            labels = [str(id2label.get(i, i)).lower() for i in range(len(probs))]
            normalized = []
            for label in labels:
                if "entail" in label:
                    normalized.append("entailment")
                elif "contra" in label:
                    normalized.append("contradiction")
                elif "neutral" in label:
                    normalized.append("neutral")
                else:
                    normalized.append(label)
            best_idx = max(range(len(probs)), key=lambda i: probs[i])
            label = normalized[best_idx]
            if label not in {"entailment", "contradiction", "neutral"}:
                # Common NLI fallback order for sentence-transformers cross-encoders.
                order = ["contradiction", "entailment", "neutral"]
                label = order[best_idx] if best_idx < len(order) else "neutral"
            return EntailmentResult(
                label=label,
                confidence=float(probs[best_idx]),
                explanation=f"NLI model judged the evidence as {label} for the claim.",
                method=NLI_MODEL_NAME,
            )
        except Exception:
            pass
    return _fallback_entailment(claim, evidence)


def add_entailment_scores(
    claim: str,
    evidence: list[dict[str, Any]],
    *,
    max_items: int = 5,
    prefer_model: bool = True,
) -> list[dict[str, Any]]:
    """Attach entailment labels to the top retrieved evidence passages."""
    enriched: list[dict[str, Any]] = []
    for index, item in enumerate(evidence):
        updated = dict(item)
        if index < max_items:
            result = check_entailment(claim, str(item.get("text", "")), prefer_model=prefer_model)
            updated.update(
                {
                    "entailment_label": result.label,
                    "entailment_confidence": result.confidence,
                    "entailment_method": result.method,
                    "entailment_explanation": result.explanation,
                }
            )
        else:
            updated.update(
                {
                    "entailment_label": "not_checked",
                    "entailment_confidence": 0.0,
                    "entailment_method": "not_checked",
                    "entailment_explanation": "Only top-ranked passages are checked to keep the app responsive.",
                }
            )
        enriched.append(updated)
    return enriched
