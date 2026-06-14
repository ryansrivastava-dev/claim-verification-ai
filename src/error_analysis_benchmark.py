"""Create an error-analysis CSV from live benchmark predictions.

Input is usually reports/benchmark_predictions.csv created by evaluate_realworld_benchmark.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def infer_error_type(row: pd.Series) -> str:
    """Heuristic category to speed up manual error review."""
    predicted = str(row.get("predicted_label", "")).lower()
    true = str(row.get("true_label", "")).lower()
    explanation = str(row.get("explanation", "")).lower()
    top_url = str(row.get("top_url", "")).lower()

    if predicted == "pipeline error" or "error" in explanation:
        return "Pipeline/runtime failure"
    if "not enough" in predicted and "not enough" not in true:
        return "Retrieval or entailment too conservative"
    if "supported" in predicted and "refuted" in true:
        return "False support / entailment failure"
    if "refuted" in predicted and "supported" in true:
        return "False contradiction / entailment failure"
    if not top_url:
        return "Retrieval failure"
    if any(term in explanation for term in ["current", "biography", "old"]):
        return "Current-context issue"
    return "Needs manual review"


def build_error_analysis(input_path: Path, output_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {input_path}")
    df = pd.read_csv(input_path).fillna("")
    required = {"claim", "true_label", "predicted_label", "correct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {sorted(missing)}")

    work = df.copy()
    correct = work["correct"].astype(str).str.lower().isin({"true", "1", "yes"}) | (work["correct"] == True)
    errors = work[~correct].copy()
    if errors.empty:
        output = pd.DataFrame(
            columns=[
                "claim",
                "true_label",
                "predicted_label",
                "top_source",
                "top_url",
                "likely_error_type",
                "manual_notes",
            ]
        )
    else:
        errors["likely_error_type"] = errors.apply(infer_error_type, axis=1)
        errors["manual_notes"] = ""
        cols = [
            col
            for col in ["claim", "true_label", "predicted_label", "top_source", "top_url", "explanation", "likely_error_type", "manual_notes"]
            if col in errors.columns
        ]
        output = errors[cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build error analysis from benchmark predictions.")
    parser.add_argument("--input", default="reports/benchmark_predictions.csv")
    parser.add_argument("--output", default="reports/benchmark_error_analysis.csv")
    args = parser.parse_args()
    output = build_error_analysis(Path(args.input), Path(args.output))
    print(f"Saved {len(output)} error rows to {args.output}")


if __name__ == "__main__":
    main()
