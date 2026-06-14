# Evaluation Results

This project is set up to report real benchmark results only after the evaluation scripts are run.

## How to run a benchmark

Prepare a CSV file at `data/raw/benchmark_claims.csv` with this format:

```csv
claim,label
"The capital of France is Paris.",Supported
"The Eiffel Tower is located in Berlin.",Refuted
```

Then run:

```bash
python src/evaluate_realworld_benchmark.py --input data/raw/benchmark_claims.csv --limit 50
```

The script saves:

- `reports/benchmark_predictions.csv`
- `reports/benchmark_metrics.json`
- `reports/evaluation_leaderboard.csv`
- `reports/figures/confusion_matrix.png`

## Metrics reported

- Accuracy
- Macro F1
- Per-class precision, recall, and F1
- Confusion matrix

## Honesty rule

Do not write benchmark numbers in the README or paper unless they come from the saved output files above.
