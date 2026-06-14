"""Plot confidence calibration from benchmark predictions.

Reads reports/benchmark_predictions.csv or any CSV with columns:
- confidence
- correct

Example:
python src/plot_confidence_calibration.py --input reports/benchmark_predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def build_calibration_table(input_path: Path, output_dir: Path = Path("reports"), bins: int = 5) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {input_path}")
    df = pd.read_csv(input_path).fillna("")
    missing = {"confidence", "correct"} - set(df.columns)
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {sorted(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    work = df.copy()
    work["confidence"] = pd.to_numeric(work["confidence"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    work["correct"] = work["correct"].astype(str).str.lower().isin({"true", "1", "yes"}) | (work["correct"] == True)
    work["bin"] = pd.cut(work["confidence"], bins=bins, labels=False, include_lowest=True)

    rows = []
    for bin_id, group in work.groupby("bin", dropna=True):
        rows.append(
            {
                "bin": int(bin_id),
                "count": int(len(group)),
                "average_confidence": float(group["confidence"].mean()),
                "actual_accuracy": float(group["correct"].mean()),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "confidence_calibration.csv", index=False)

    if not table.empty:
        plt.figure(figsize=(6, 5))
        plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
        plt.plot(table["average_confidence"], table["actual_accuracy"], marker="o", label="System")
        plt.xlabel("Average predicted confidence")
        plt.ylabel("Actual accuracy")
        plt.title("Confidence Calibration")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "confidence_calibration.png", dpi=180)
        plt.close()
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Create confidence calibration plot from predictions.")
    parser.add_argument("--input", default="reports/benchmark_predictions.csv")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--bins", type=int, default=5)
    args = parser.parse_args()
    table = build_calibration_table(Path(args.input), Path(args.output_dir), bins=args.bins)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
