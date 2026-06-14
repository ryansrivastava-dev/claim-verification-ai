"""Claim-evidence entailment checks.

This module prevents a common fact-checking failure: treating related evidence
as supporting evidence. Retrieval can find pages that mention the same entities
without proving the exact claim. The entailment layer checks whether an evidence
passage supports, contradicts, or is neutral toward the claim.

The implementation uses an optional lightweight NLI cross-encoder when it can be
loaded. If the model is unavailable, the deterministic fallback stays
conservative: it only supports a claim when the passage directly expresses the
same relationship, and otherwise returns neutral instead of overclaiming.
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
                if re.search(subject_pattern + r".{0,120}" + predicate_pattern, evidence_lower, re.DOTALL):
                    return EntailmentResult(
                        "entailment",
                        0.68,
                        "The evidence directly links the claim subject and predicate.",
                        "conservative-fallback",
                    )
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
