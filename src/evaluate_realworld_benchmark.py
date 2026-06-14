"""Evaluate the live web-grounded pipeline on a labeled benchmark CSV.

This script does not include unverified metrics. It only writes results after running
claims through the actual pipeline.

Input CSV requirements:
- claim: text claim to verify
- label: gold label using one of Supported, Refuted, Not Enough Evidence, etc.

Example:
python src/evaluate_realworld_benchmark.py --input data/raw/benchmark_claims.csv --limit 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

try:
    from web_pipeline import run_web_fact_check
except ImportError:  # pragma: no cover
    from src.web_pipeline import run_web_fact_check


LABEL_MAP = {
    "supported": "Supported",
    "likely supported": "Supported",
    "supported by evidence": "Supported",
    "refuted": "Refuted",
    "likely refuted": "Refuted",
    "possibly refuted": "Refuted",
    "contradicted by evidence": "Refuted",
    "not enough evidence": "Not Enough Evidence",
    "nei": "Not Enough Evidence",
    "conflicting evidence": "Conflicting Evidence",
    "conflicting evidence/cherrypicking": "Conflicting Evidence",
    "cherrypicking": "Conflicting Evidence",
}


def normalize_label(label: Any) -> str:
    """Normalize benchmark and model labels into comparable buckets."""
    raw = str(label).strip()
    return LABEL_MAP.get(raw.lower(), raw)


def evaluate(input_path: Path, limit: int | None = None, output_dir: Path = Path("reports")) -> dict[str, Any]:
    """Run the live pipeline over a labeled benchmark CSV and save metrics."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input benchmark file not found: {input_path}")

    df = pd.read_csv(input_path).fillna("")
    required = {"claim", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Benchmark CSV is missing required columns: {sorted(missing)}")

    if limit:
        df = df.head(limit)

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, row in df.iterrows():
        claim = str(row["claim"]).strip()
        gold = normalize_label(row["label"])
        if not claim:
            continue
        try:
            result = run_web_fact_check(claim=claim, top_k=5, max_pages=5)
            predicted = normalize_label(result.get("predicted_label", ""))
            confidence = float(result.get("confidence", 0.0))
            explanation = result.get("explanation", "")
            top_source = ""
            top_url = ""
            if result.get("top_evidence"):
                top_source = result["top_evidence"][0].get("title", "")
                top_url = result["top_evidence"][0].get("url", "")
        except Exception as exc:  # network failures should be inspectable
            predicted = "Pipeline Error"
            confidence = 0.0
            explanation = f"{type(exc).__name__}: {exc}"
            top_source = ""
            top_url = ""

        rows.append(
            {
                "claim": claim,
                "true_label": gold,
                "predicted_label": predicted,
                "confidence": confidence,
                "correct": gold == predicted,
                "top_source": top_source,
                "top_url": top_url,
                "explanation": explanation,
            }
        )
        print(f"[{len(rows)}/{len(df)}] {predicted} | gold={gold} | {claim[:80]}")

    results = pd.DataFrame(rows)
    results_path = output_dir / "benchmark_predictions.csv"
    results.to_csv(results_path, index=False)

    y_true = results["true_label"].tolist()
    y_pred = results["predicted_label"].tolist()
    labels = sorted(set(y_true) | set(y_pred))

    metrics = {
        "input_file": str(input_path),
        "num_claims": int(len(results)),
        "accuracy": float(accuracy_score(y_true, y_pred)) if results.shape[0] else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if results.shape[0] else 0.0,
        "labels": labels,
        "classification_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
    }

    metrics_path = output_dir / "benchmark_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report_rows = []
    for label_name, values in metrics["classification_report"].items():
        if isinstance(values, dict):
            report_rows.append({"label": label_name, **values})
    if report_rows:
        pd.DataFrame(report_rows).to_csv(output_dir / "classification_report.csv", index=False)

    evaluation_summary = pd.DataFrame(
        [
            {
                "System": "Modular live evidence pipeline",
                "Evidence source": "Web/Wikipedia/OpenAlex",
                "Model": "Retrieval + entailment + evidence synthesis",
                "Accuracy": metrics["accuracy"],
                "Macro F1": metrics["macro_f1"],
                "Claims evaluated": metrics["num_claims"],
            }
        ]
    )
    evaluation_summary.to_csv(output_dir / "evaluation_summary.csv", index=False)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), max(5, len(labels) * 1.2)))
    im = ax.imshow(cm)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            ax.text(c, r, str(cm[r, c]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(figures_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the live fact-checking pipeline on a labeled CSV.")
    parser.add_argument("--input", required=True, help="CSV file with columns: claim,label")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of rows to evaluate")
    parser.add_argument("--output-dir", default="reports", help="Directory for metrics and prediction files")
    args = parser.parse_args()

    metrics = evaluate(Path(args.input), limit=args.limit, output_dir=Path(args.output_dir))
    print(json.dumps({"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "num_claims": metrics["num_claims"]}, indent=2))


if __name__ == "__main__":
    main()
