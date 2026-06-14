"""Run ablation studies for the web-grounded fact-checking pipeline.

The script compares retrieval/ranking configurations on the same labeled CSV.
It does not contain precomputed or invented metrics; it writes results only after
running the actual pipeline.

Input CSV columns:
- claim
- label

Example:
python src/evaluate_ablation_study.py --input data/raw/benchmark_claims.csv --limit 20
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

try:
    from evaluate_realworld_benchmark import normalize_label
    from web_pipeline import run_web_fact_check
except ImportError:  # pragma: no cover
    from src.evaluate_realworld_benchmark import normalize_label
    from src.web_pipeline import run_web_fact_check


@dataclass(frozen=True)
class AblationConfig:
    name: str
    method: str
    alpha: float = 0.5
    trust_weight: float = 0.15


DEFAULT_CONFIGS = [
    AblationConfig("TF-IDF only", "TF-IDF", alpha=0.5, trust_weight=0.15),
    AblationConfig("BM25 only", "BM25", alpha=0.5, trust_weight=0.15),
    AblationConfig("Hybrid alpha 0.25", "Hybrid", alpha=0.25, trust_weight=0.15),
    AblationConfig("Hybrid alpha 0.50", "Hybrid", alpha=0.50, trust_weight=0.15),
    AblationConfig("Hybrid alpha 0.75", "Hybrid", alpha=0.75, trust_weight=0.15),
    AblationConfig("Hybrid no source credibility", "Hybrid", alpha=0.50, trust_weight=0.00),
    AblationConfig("Hybrid stronger source credibility", "Hybrid", alpha=0.50, trust_weight=0.30),
]


def exact_mcnemar_p_value(system_a_correct: list[bool], system_b_correct: list[bool]) -> dict[str, float | int]:
    """Exact McNemar test using the binomial distribution.

    b = cases where A is correct and B is wrong
    c = cases where A is wrong and B is correct
    The two-sided exact p-value is 2 * P(X <= min(b, c)) for X ~ Binomial(b+c, 0.5).
    """
    if len(system_a_correct) != len(system_b_correct):
        raise ValueError("Correctness lists must have the same length.")

    b = sum(1 for a, b_ok in zip(system_a_correct, system_b_correct) if a and not b_ok)
    c = sum(1 for a, b_ok in zip(system_a_correct, system_b_correct) if (not a) and b_ok)
    n = b + c
    if n == 0:
        return {"a_correct_b_wrong": b, "a_wrong_b_correct": c, "p_value": 1.0}

    tail = sum(math.comb(n, i) * (0.5 ** n) for i in range(0, min(b, c) + 1))
    p_value = min(1.0, 2.0 * tail)
    return {"a_correct_b_wrong": b, "a_wrong_b_correct": c, "p_value": float(p_value)}


def _load_claims(input_path: Path, limit: int | None) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Benchmark CSV not found: {input_path}")
    df = pd.read_csv(input_path).fillna("")
    missing = {"claim", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"Benchmark CSV is missing columns: {sorted(missing)}")
    df = df[df["claim"].astype(str).str.strip().astype(bool)].copy()
    if limit:
        df = df.head(limit)
    return df


def evaluate_configs(
    input_path: Path,
    output_dir: Path = Path("reports"),
    limit: int | None = None,
    top_k: int = 5,
    max_pages: int = 5,
) -> dict[str, Any]:
    """Evaluate all ablation configurations and save metrics/predictions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = _load_claims(input_path, limit)
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    correctness_by_config: dict[str, list[bool]] = {}

    for config in DEFAULT_CONFIGS:
        y_true: list[str] = []
        y_pred: list[str] = []
        config_correct: list[bool] = []
        print(f"\nRunning config: {config.name}")
        for index, row in df.iterrows():
            claim = str(row["claim"]).strip()
            gold = normalize_label(row["label"])
            try:
                result = run_web_fact_check(
                    claim=claim,
                    method=config.method,
                    alpha=config.alpha,
                    trust_weight=config.trust_weight,
                    top_k=top_k,
                    max_pages=max_pages,
                )
                predicted = normalize_label(result.get("predicted_label", ""))
                confidence = float(result.get("confidence", 0.0))
                top_url = result.get("top_evidence", [{}])[0].get("url", "") if result.get("top_evidence") else ""
                explanation = result.get("explanation", "")
            except Exception as exc:
                predicted = "Pipeline Error"
                confidence = 0.0
                top_url = ""
                explanation = f"{type(exc).__name__}: {exc}"

            correct = gold == predicted
            y_true.append(gold)
            y_pred.append(predicted)
            config_correct.append(correct)
            rows.append(
                {
                    "config": config.name,
                    "method": config.method,
                    "alpha": config.alpha,
                    "trust_weight": config.trust_weight,
                    "claim": claim,
                    "true_label": gold,
                    "predicted_label": predicted,
                    "confidence": confidence,
                    "correct": correct,
                    "top_url": top_url,
                    "explanation": explanation,
                }
            )
            print(f"[{config.name}] {predicted} | gold={gold} | {claim[:70]}")

        correctness_by_config[config.name] = config_correct
        summary_rows.append(
            {
                "config": config.name,
                "method": config.method,
                "alpha": config.alpha,
                "trust_weight": config.trust_weight,
                "claims_evaluated": len(y_true),
                "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if y_true else 0.0,
            }
        )

    predictions = pd.DataFrame(rows)
    summary = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False)
    predictions.to_csv(output_dir / "ablation_predictions.csv", index=False)
    summary.to_csv(output_dir / "ablation_results.csv", index=False)

    significance_rows = []
    baseline = "BM25 only" if "BM25 only" in correctness_by_config else next(iter(correctness_by_config))
    for config_name, config_correct in correctness_by_config.items():
        if config_name == baseline:
            continue
        stats = exact_mcnemar_p_value(correctness_by_config[baseline], config_correct)
        significance_rows.append({"baseline": baseline, "comparison": config_name, **stats})
    significance = pd.DataFrame(significance_rows)
    significance.to_csv(output_dir / "significance_tests.csv", index=False)

    if not summary.empty:
        plt.figure(figsize=(10, 5))
        plt.bar(summary["config"], summary["macro_f1"])
        plt.ylabel("Macro F1")
        plt.xlabel("Configuration")
        plt.title("Ablation Study: Retrieval and Source Credibility")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(figures_dir / "ablation_macro_f1.png", dpi=180)
        plt.close()

    metadata = {
        "input_file": str(input_path),
        "claims_evaluated": int(len(df)),
        "top_k": top_k,
        "max_pages": max_pages,
        "configs": [config.__dict__ for config in DEFAULT_CONFIGS],
        "note": "Metrics were generated by running the live pipeline. Do not compare to papers unless the benchmark and settings match.",
    }
    (output_dir / "ablation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"summary": summary.to_dict(orient="records"), "significance": significance_rows, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation studies on a labeled claim CSV.")
    parser.add_argument("--input", required=True, help="CSV with claim,label columns")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-pages", type=int, default=5)
    args = parser.parse_args()

    result = evaluate_configs(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        limit=args.limit,
        top_k=args.top_k,
        max_pages=args.max_pages,
    )
    print(json.dumps({"configs_evaluated": len(result["summary"]), "metadata": result["metadata"]}, indent=2))


if __name__ == "__main__":
    main()
