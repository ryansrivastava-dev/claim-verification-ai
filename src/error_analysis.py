"""Create a CSV for reviewing incorrect predictions."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

try:
    from pipeline import build_retriever, predict_label
except ImportError:  # pragma: no cover
    from src.pipeline import build_retriever, predict_label

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"


def guess_error_type(true_label: str, pred_label: str, retrieved: list[dict[str, Any]], gold_doc_ids: list[str]) -> str:
    """Suggest an error type for manual review.

    This is only a rough template. Human review is better for final analysis.
    """
    retrieved_docs = {str(item.get("doc_id", "")) for item in retrieved}
    gold_docs = {str(item) for item in gold_doc_ids if str(item).strip()}
    if gold_docs and not (retrieved_docs & gold_docs):
        return "Retrieval failure"
    if true_label != pred_label:
        return "Classification failure"
    return "Needs manual review"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create error analysis CSV.")
    parser.add_argument("--retriever", default="hybrid", choices=["tfidf", "bm25", "dense", "hybrid"])
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--sample_size", type=int, default=50)
    args = parser.parse_args()

    claims_path = PROCESSED_DIR / "claims.csv"
    passages_path = PROCESSED_DIR / "evidence_passages.csv"
    model_path = MODEL_DIR / "tfidf_logreg.joblib"
    if not claims_path.exists() or not passages_path.exists():
        raise FileNotFoundError("Run python src/load_data.py first.")
    if not model_path.exists():
        raise FileNotFoundError("Run python src/train_classifier.py first.")

    claims_df = pd.read_csv(claims_path).fillna("")
    passages_df = pd.read_csv(passages_path).fillna("")
    retriever = build_retriever(args.retriever, passages_df)
    classifier = joblib.load(model_path)

    rows: list[dict[str, Any]] = []
    for _, row in claims_df.iterrows():
        claim = str(row["claim"])
        true_label = str(row.get("label", ""))
        if not true_label:
            continue
        retrieved = retriever.retrieve(claim, top_k=args.top_k)
        pred_label, _ = predict_label(claim, retrieved, classifier, "tfidf_logistic_regression")
        if pred_label != true_label:
            gold_doc_ids = row.get("evidence_doc_ids", [])
            rows.append(
                {
                    "claim": claim,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "top_evidence": " || ".join(item.get("text", "") for item in retrieved),
                    "likely_error_type": guess_error_type(true_label, pred_label, retrieved, gold_doc_ids),
                    "manual_notes": "",
                }
            )

    random.seed(42)
    if len(rows) > args.sample_size:
        rows = random.sample(rows, args.sample_size)

    output_path = PROCESSED_DIR / "error_analysis_sample.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved {len(rows)} error analysis rows to {output_path}")
    print("Suggested categories: Retrieval failure, Classification failure, Ambiguous claim, Insufficient evidence, Label confusion.")


if __name__ == "__main__":
    main()
