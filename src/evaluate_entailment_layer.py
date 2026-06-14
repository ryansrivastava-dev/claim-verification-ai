"""Evaluate the claim-evidence entailment layer on labeled claim/evidence pairs.

Input CSV columns:
- claim
- evidence
- label: entailment, contradiction, or neutral

Example:
python src/evaluate_entailment_layer.py --input data/raw/entailment_pairs.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

try:
    from entailment_checker import check_entailment
except ImportError:  # pragma: no cover
    from src.entailment_checker import check_entailment


LABELS = ["entailment", "contradiction", "neutral"]


def normalize_entailment_label(value: Any) -> str:
    raw = str(value).strip().lower()
    if raw in {"supports", "supported", "support", "entails"}:
        return "entailment"
    if raw in {"contradicts", "contradicted", "refutes", "refuted"}:
        return "contradiction"
    if raw in {"not enough evidence", "nei", "unrelated", "background"}:
        return "neutral"
    return raw


def evaluate_entailment(input_path: Path, output_dir: Path = Path("reports"), prefer_model: bool = True) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"Entailment evaluation CSV not found: {input_path}")
    df = pd.read_csv(input_path).fillna("")
    missing = {"claim", "evidence", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"Entailment CSV is missing columns: {sorted(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, row in df.iterrows():
        claim = str(row["claim"]).strip()
        evidence = str(row["evidence"]).strip()
        gold = normalize_entailment_label(row["label"])
        if not claim or not evidence:
            continue
        result = check_entailment(claim, evidence, prefer_model=prefer_model)
        rows.append(
            {
                "claim": claim,
                "evidence": evidence,
                "true_label": gold,
                "predicted_label": result.label,
                "confidence": result.confidence,
                "method": result.method,
                "explanation": result.explanation,
                "correct": gold == result.label,
            }
        )

    predictions = pd.DataFrame(rows)
    predictions.to_csv(output_dir / "entailment_predictions.csv", index=False)
    y_true = predictions["true_label"].tolist() if not predictions.empty else []
    y_pred = predictions["predicted_label"].tolist() if not predictions.empty else []

    metrics = {
        "input_file": str(input_path),
        "num_pairs": int(len(predictions)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)) if y_true else 0.0,
        "macro_recall": float(recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)) if y_true else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)) if y_true else 0.0,
        "classification_report": classification_report(y_true, y_pred, labels=LABELS, zero_division=0, output_dict=True) if y_true else {},
        "note": "Use this to evaluate whether the entailment layer distinguishes direct support from related evidence.",
    }
    (output_dir / "entailment_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if y_true:
        cm = confusion_matrix(y_true, y_pred, labels=LABELS)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm)
        ax.set_xticks(range(len(LABELS)))
        ax.set_xticklabels(LABELS, rotation=30, ha="right")
        ax.set_yticks(range(len(LABELS)))
        ax.set_yticklabels(LABELS)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Entailment Layer Confusion Matrix")
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(figures_dir / "entailment_confusion_matrix.png", dpi=180)
        plt.close(fig)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the entailment layer.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--disable-model", action="store_true", help="Force deterministic fallback instead of optional NLI model")
    args = parser.parse_args()
    metrics = evaluate_entailment(Path(args.input), Path(args.output_dir), prefer_model=not args.disable_model)
    print(json.dumps({k: metrics[k] for k in ["num_pairs", "macro_precision", "macro_recall", "macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
