# Evaluation Results

This file is intentionally result-aware, not result-inventing. Fill it after running the evaluation scripts.

## Commands

```bash
python src/evaluate_realworld_benchmark.py --input data/raw/benchmark_claims.csv --limit 20
python src/evaluate_ablation_study.py --input data/raw/benchmark_claims.csv --limit 20
python src/evaluate_entailment_layer.py --input data/raw/entailment_pairs.csv
python src/plot_confidence_calibration.py --input reports/benchmark_predictions.csv
python src/error_analysis_benchmark.py --input reports/benchmark_predictions.csv
```


## Included development sanity-check result

The repository includes a small manually labeled entailment development file at `data/raw/entailment_pairs.csv`. Running the deterministic fallback entailment evaluator on this file produced the saved files `reports/entailment_metrics.json`, `reports/entailment_predictions.csv`, and `reports/figures/entailment_confusion_matrix.png`. These are development sanity-check results, not a formal public benchmark.

## Results to report

| Experiment | Output file | Metrics |
|---|---|---|
| Verdict benchmark | `reports/benchmark_metrics.json`, `reports/classification_report.csv` | Accuracy, macro F1, per-class F1 |
| Ablation study | `reports/ablation_results.csv` | Accuracy, macro F1 by config |
| Significance tests | `reports/significance_tests.csv` | Exact McNemar p-values |
| Entailment evaluation | `reports/entailment_metrics.json` | Macro precision, recall, F1 |
| Calibration | `reports/confidence_calibration.csv` | Predicted vs actual accuracy by bin |

## Interpretation guide

- If hybrid retrieval beats BM25 and TF-IDF on the same benchmark, it supports the main hypothesis.
- If source credibility weighting improves macro F1 or reduces high-confidence mistakes, it supports the credibility-ranking contribution.
- If the entailment layer improves contradiction detection and reduces false support, it should be framed as a major system contribution.
- If confidence calibration is poor, the paper should say that confidence is a ranking signal, not a reliable probability.

## Do not report

Do not report state-of-the-art claims, public accuracy claims, or benchmark comparisons unless those experiments were actually run under matching conditions.
