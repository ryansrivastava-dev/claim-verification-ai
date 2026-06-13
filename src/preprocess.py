"""Text preprocessing utilities for scientific claim verification.

This file is intentionally beginner-readable. It contains small functions that
can be imported by data loading, retrieval, training, and tests.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"


def clean_text(text: Any, lowercase: bool = False) -> str:
    """Normalize text without changing its meaning.

    Parameters
    ----------
    text:
        Input value. Non-string values are converted to strings unless missing.
    lowercase:
        Whether to lowercase the final text. For scientific text, keeping case
        can preserve acronyms, so the default is False.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower() if lowercase else text


def split_into_sentences(text: str) -> list[str]:
    """Split text into simple sentence-like chunks.

    This avoids heavy NLP dependencies so the project stays easy to run.
    """
    text = clean_text(text)
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?])\s+", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def normalize_label(label: Any) -> str:
    """Map common SciFact-style labels into three project labels."""
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return "NOT_ENOUGH_INFO"
    value = str(label).strip().upper().replace("-", "_").replace(" ", "_")
    supported = {"SUPPORT", "SUPPORTED", "SUPPORTS", "TRUE", "1"}
    refuted = {"REFUTE", "REFUTED", "REFUTES", "CONTRADICT", "CONTRADICTS", "FALSE", "0"}
    nei = {"NOT_ENOUGH_INFO", "NEI", "INSUFFICIENT", "UNKNOWN", "NO_EVIDENCE", "NONE"}
    if value in supported:
        return "SUPPORTED"
    if value in refuted:
        return "REFUTED"
    if value in nei:
        return "NOT_ENOUGH_INFO"
    return value


def _as_list(value: Any) -> list[Any]:
    """Convert common dataset values into a list."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        # Try JSON-like lists first.
        if value.startswith("[") and value.endswith("]"):
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, list) else [parsed]
            except Exception:
                pass
        return [piece.strip() for piece in re.split(r"[,;]", value) if piece.strip()]
    return [value]


def extract_evidence_doc_ids(row: pd.Series) -> list[str]:
    """Extract evidence/cited document IDs from several possible SciFact schemas."""
    candidates: list[Any] = []
    for col in ["evidence_doc_id", "evidence_doc_ids", "cited_doc_ids", "doc_id", "document_id"]:
        if col in row:
            candidates.extend(_as_list(row[col]))

    # Some SciFact mirrors store evidence as a list/dict of evidence sets.
    evidence = row.get("evidence") if "evidence" in row else None
    if isinstance(evidence, dict):
        for value in evidence.values():
            candidates.extend(_as_list(value))
    elif isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, dict):
                for key in ["doc_id", "document_id", "evidence_doc_id"]:
                    if key in item:
                        candidates.extend(_as_list(item[key]))
            else:
                candidates.extend(_as_list(item))

    cleaned: list[str] = []
    for item in candidates:
        if item is None or (isinstance(item, float) and pd.isna(item)):
            continue
        text = str(item).strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def make_passages(corpus_df: pd.DataFrame) -> pd.DataFrame:
    """Convert a document-level corpus into sentence/passage rows.

    Expected output columns:
    - passage_id
    - doc_id
    - sent_id
    - title
    - text
    """
    passages: list[dict[str, Any]] = []

    for row_index, row in corpus_df.reset_index(drop=True).iterrows():
        doc_id = row.get("doc_id", row.get("id", row.get("document_id", row_index)))
        title = clean_text(row.get("title", ""))

        abstract = row.get("abstract", row.get("text", row.get("body", "")))
        if isinstance(abstract, list):
            sentences = [clean_text(sentence) for sentence in abstract if clean_text(sentence)]
        else:
            sentences = split_into_sentences(clean_text(abstract))

        if not sentences and title:
            sentences = [title]

        for sent_id, sentence in enumerate(sentences):
            passage_text = clean_text(f"{title}. {sentence}" if title and title not in sentence else sentence)
            if not passage_text:
                continue
            passages.append(
                {
                    "passage_id": f"{doc_id}_sent{sent_id}",
                    "doc_id": str(doc_id),
                    "sent_id": sent_id,
                    "title": title,
                    "text": passage_text,
                }
            )

    return pd.DataFrame(passages)


def create_claim_evidence_pairs(
    claims_df: pd.DataFrame,
    passages_df: pd.DataFrame,
    max_passages_per_claim: int = 3,
) -> pd.DataFrame:
    """Create lightweight training pairs from gold evidence document links.

    This uses available gold document IDs when present. It does not create fake
    labels or pretend that unannotated evidence is definitive.
    """
    rows: list[dict[str, Any]] = []
    passages_by_doc = {doc_id: group for doc_id, group in passages_df.groupby("doc_id")}

    for _, claim_row in claims_df.iterrows():
        label = normalize_label(claim_row.get("label", claim_row.get("evidence_label")))
        claim = clean_text(claim_row.get("claim", ""))
        claim_id = claim_row.get("claim_id", claim_row.get("id", ""))
        evidence_doc_ids = _as_list(claim_row.get("evidence_doc_ids", []))

        if evidence_doc_ids:
            used = 0
            for doc_id in evidence_doc_ids:
                doc_id = str(doc_id)
                if doc_id not in passages_by_doc:
                    continue
                for _, passage_row in passages_by_doc[doc_id].head(max_passages_per_claim).iterrows():
                    rows.append(
                        {
                            "claim_id": claim_id,
                            "claim": claim,
                            "label": label,
                            "passage_id": passage_row["passage_id"],
                            "doc_id": passage_row["doc_id"],
                            "evidence_text": passage_row["text"],
                            "split": claim_row.get("split", "unknown"),
                        }
                    )
                    used += 1
                if used >= max_passages_per_claim:
                    break
        else:
            rows.append(
                {
                    "claim_id": claim_id,
                    "claim": claim,
                    "label": label,
                    "passage_id": "",
                    "doc_id": "",
                    "evidence_text": "",
                    "split": claim_row.get("split", "unknown"),
                }
            )

    return pd.DataFrame(rows)


def preprocess_files(processed_dir: Path = PROCESSED_DIR) -> None:
    """Run preprocessing on files created by load_data.py."""
    claims_path = processed_dir / "claims.csv"
    corpus_path = processed_dir / "corpus.csv"
    if not claims_path.exists() or not corpus_path.exists():
        raise FileNotFoundError(
            "Expected data/processed/claims.csv and data/processed/corpus.csv. "
            "Run python src/load_data.py first."
        )

    claims_df = pd.read_csv(claims_path)
    corpus_df = pd.read_csv(corpus_path)

    claims_df["claim"] = claims_df["claim"].apply(clean_text)
    if "label" in claims_df.columns:
        claims_df["label"] = claims_df["label"].apply(normalize_label)
    claims_df["evidence_doc_ids"] = claims_df.apply(lambda row: json.dumps(extract_evidence_doc_ids(row)), axis=1)

    passages_df = make_passages(corpus_df)
    pairs_df = create_claim_evidence_pairs(claims_df, passages_df)

    claims_df.to_csv(processed_dir / "claims.csv", index=False)
    passages_df.to_csv(processed_dir / "evidence_passages.csv", index=False)
    pairs_df.to_csv(processed_dir / "claim_evidence_pairs.csv", index=False)

    print(f"Saved cleaned claims: {len(claims_df):,}")
    print(f"Saved evidence passages: {len(passages_df):,}")
    print(f"Saved claim/evidence training pairs: {len(pairs_df):,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess SciFact claims and evidence passages.")
    parser.add_argument("--processed_dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()
    preprocess_files(args.processed_dir)


if __name__ == "__main__":
    main()
