# Error Analysis Plan

Error analysis should be filled after running the benchmark and ablation scripts. The goal is to explain why the system fails, not just list incorrect predictions.

## Failure categories

| Category | Meaning | Possible fix |
|---|---|---|
| Retrieval failure | The correct evidence was not retrieved. | Better query planning, more sources, dense retrieval. |
| Source credibility failure | A weak source was ranked too highly. | Adjust credibility weighting, block low-quality domains. |
| Entailment failure | Evidence was related but did not truly support/refute the claim. | Improve NLI model or fallback rules. |
| Ambiguous claim | The claim was vague or depended on interpretation. | Add claim clarification or subclaim decomposition. |
| Current-context failure | Old evidence was mistaken for current evidence. | Improve current-source handling and official-source preference. |
| Insufficient evidence | Public sources did not contain enough information. | Return Not Enough Evidence with clearer explanation. |
| Label mapping issue | Dataset labels do not map cleanly to app labels. | Normalize labels carefully and document mapping. |

## Manual review table

After running `src/evaluate_realworld_benchmark.py`, copy representative errors here.

| Claim | True label | Predicted label | Top source | Error type | What happened | Fix idea |
|---|---|---|---|---|---|---|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Research use

For a paper, include 3–5 specific errors and explain whether they were caused by retrieval, source ranking, entailment, ambiguity, or label mismatch.
