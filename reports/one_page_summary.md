# One-Page Project Summary

## Project

**Evidence-Based Fact Verification AI** is a modular AI system that checks public factual claims using live evidence retrieval, source ranking, entailment checking, and transparent reports.

## Problem

Many AI systems answer factual questions without showing evidence. This can create false confidence, especially when a claim is current, ambiguous, or only partially supported by sources. This project addresses that problem by making the verification process visible and citation-backed.

## What I built

I built a Streamlit application and Python research pipeline that:

- Detects the type of claim being checked.
- Plans search queries instead of only searching the raw claim.
- Retrieves public evidence from web, Wikipedia, and OpenAlex sources.
- Ranks evidence using TF-IDF, BM25, hybrid scoring, and source credibility signals.
- Summarizes evidence passages.
- Checks whether evidence entails, contradicts, or is neutral toward the claim.
- Handles time-sensitive claims with current-context logic.
- Generates downloadable Markdown and PDF reports with citations.
- Includes benchmark, ablation, entailment, calibration, and error-analysis scripts.

## Methods

The system compares sparse retrieval methods and hybrid keyword retrieval. It also evaluates source credibility weighting and a claim-evidence entailment layer. The original research pipeline includes dense retrieval for SciFact-style experiments.

## Evaluation

The repository includes scripts for:

- Accuracy and macro F1 on labeled claim benchmarks.
- Ablation studies comparing TF-IDF, BM25, hybrid alpha values, and source credibility weighting.
- Exact McNemar significance tests.
- Entailment-layer precision, recall, and macro F1.
- Confidence calibration curves.
- Error analysis templates.

## Results

Final results should be added only after running the included evaluation scripts on labeled benchmark data. The README and paper draft intentionally avoid fabricated metrics.

## Skills demonstrated

Python, NLP, information retrieval, natural language inference, Streamlit deployment, source ranking, evaluation design, error analysis, reproducibility, and research communication.

## Future work

Run larger public benchmark evaluations on SciFact, FEVER, or AVeriTeC; add dense retrieval to the live app; improve confidence calibration; expand source credibility evaluation; and test whether hybrid retrieval significantly improves accuracy over sparse baselines.
