# Error Analysis Log

This file is for recording real errors after running the benchmark or manually testing the app. Do not fill this with invented results.

## Error categories

- **Outdated source:** The system relied on a past source for a current claim.
- **Weak evidence:** Retrieved sources were related but did not directly verify the claim.
- **Ambiguous wording:** The claim could be interpreted in multiple ways.
- **Entity confusion:** Sources discussed a similarly named person, place, organization, or event.
- **Source quality issue:** A low-quality or irrelevant source was ranked too highly.
- **Insufficient evidence:** The public sources available were not enough to verify the claim.
- **Label boundary issue:** The answer fell between Supported, Refuted, Conflicting Evidence, or Not Enough Evidence.

## Manual review table

| Claim | True label | Predicted label | What went wrong | Fix or next step |
|---|---|---|---|---|
| Joe Biden is the current president | Refuted | Previously likely supported | Old biography evidence matched Biden + president but did not prove current office | Added freshness guardrail for time-sensitive claims |

## How to generate more examples

Run:

```bash
python src/evaluate_realworld_benchmark.py --input data/raw/benchmark_claims.csv --limit 50
```

Then review:

```text
reports/benchmark_predictions.csv
reports/benchmark_metrics.json
reports/figures/confusion_matrix.png
```
