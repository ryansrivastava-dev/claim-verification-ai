# Evidence-Based Fact Verification AI — One-Page Summary

## Problem

Online information changes quickly, and people often see claims without knowing what evidence supports or refutes them. A useful fact-checking system should not just output an answer; it should show the sources, search strategy, and reasoning path behind the answer.

## What I built

I built a modular AI fact-checking system that verifies claims using live public evidence. The app accepts a factual claim, plans search queries, retrieves evidence from public sources, ranks sources by relevance and credibility signals, summarizes evidence passages, synthesizes the findings, handles time-sensitive claims, and generates a transparent citation-backed report.

## System workflow

```text
Claim → Claim Type Detection → Query Planning → Live Evidence Retrieval → Source Ranking → Evidence Summarization → Evidence Entailment Check → Evidence Synthesis → Current-Context Handling → Verdict + Citations + Report
```

## Methods

- TF-IDF retrieval
- BM25 retrieval
- Hybrid retrieval
- Source credibility scoring
- Extractive evidence summarization
- Claim-evidence entailment checking
- Evidence synthesis
- Current-fact handling
- Downloadable Markdown and PDF reports

## Evaluation

The project includes scripts for real benchmark evaluation using labeled CSV files. The system reports accuracy, macro F1, per-class metrics, prediction files, and confusion matrices only after experiments are actually run. No fabricated metrics are included.

## Skills demonstrated

- Python software engineering
- NLP and information retrieval
- Web evidence retrieval
- Source ranking and credibility scoring
- Streamlit app development
- Modular system design
- Testing with pytest
- Research writing and reproducibility
- Ethical AI design

## Future work

- Evaluate on AVeriTeC, SciFact, FEVER, or LIAR
- Add optional small-language-model reasoning
- Expand entailment and contradiction checking
- Add source-date extraction
- Add a public benchmark summary after running real experiments
- Add a short walkthrough video or GIF
