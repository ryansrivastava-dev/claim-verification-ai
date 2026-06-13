"""Evaluate end-to-end retrieval + classification systems."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

try:
    from pipeline import build_retriever, predict_label
    from preprocess import _as_list
except ImportError:  # pragma: no cover
    from src.pipeline import build_retriever, predict_label
    from src.preprocess import _as_list

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"


def has_gold_hit(evidence_doc_ids: Any, retrieved: list[dict[str, Any]]) -> float:
    gold = {str(x) for x in _as_list(evidence_doc_ids)}
    if not gold:
        return float("nan")
    return 1.0 if any(str(item.get("doc_id", "")) in gold for item in retrieved) else 0.0


def plot_pipeline_results(results_df: pd.DataFrame, output_path: Path) -> None:
    if results_df.empty or "macro_f1" not in results_df:
        return
    plt.figure(figsize=(8, 5))
    plt.bar(results_df["system"], results_df["macro_f1"])
    plt.ylabel("Macro F1")
    plt.xlabel("End-to-end system")
    plt.title("End-to-end claim verification comparison")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate end-to-end systems.")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_claims", type=int, default=0)
    parser.add_argument("--classifier", default="tfidf_logistic_regression")
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
    if args.max_claims:
        claims_df = claims_df.head(args.max_claims)
    classifier = joblib.load(model_path)

    systems = ["tfidf", "bm25", "dense", "hybrid"]
    rows: list[dict[str, Any]] = []
    notes: list[str] = []

    for system in systems:
        try:
            retriever = build_retriever(system, passages_df)
        except Exception as exc:
            notes.append(f"{system} skipped: {type(exc).__name__}: {exc}")
            continue

        y_true: list[str] = []
        y_pred: list[str] = []
        retrieval_hits: list[float] = []
        for _, row in tqdm(claims_df.iterrows(), total=len(claims_df), desc=system):
            if not row.get("label"):
                continue
            retrieved = retriever.retrieve(str(row["claim"]), top_k=args.top_k)
            pred, _ = predict_label(str(row["claim"]), retrieved, classifier, args.classifier)
            y_true.append(str(row["label"]))
            y_pred.append(str(pred))
            retrieval_hits.append(has_gold_hit(row.get("evidence_doc_ids", []), retrieved))

        if y_true:
            valid_hits = pd.Series(retrieval_hits).dropna()
            rows.append(
                {
                    "system": system,
                    "accuracy": float(accuracy_score(y_true, y_pred)),
                    "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
                    f"evidence_hit@{args.top_k}": float(valid_hits.mean()) if not valid_hits.empty else None,
                    "evaluated_claims": len(y_true),
                }
            )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(PROCESSED_DIR / "pipeline_results.csv", index=False)
    (PROCESSED_DIR / "pipeline_notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_pipeline_results(results_df, FIGURES_DIR / "pipeline_comparison.png")
    print(results_df.to_string(index=False))
    if notes:
        print("Notes:")
        for note in notes:
            print(f"- {note}")


if __name__ == "__main__":
    main()
