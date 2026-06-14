# Evidence-Based Fact Verification AI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)

A modular AI fact-checking system that verifies public factual claims using live evidence retrieval, source credibility scoring, claim-evidence entailment checking, evidence synthesis, current-context handling, benchmark evaluation scripts, and citation-backed reports.

This project began as **Hybrid Retrieval for Scientific Claim Verification** and expanded into a broader web-grounded verification system. It still includes the original SciFact research pipeline, while the deployed Streamlit app checks public claims using live sources.

---

## Project summary

The system is designed around one research question:

> How do sparse retrieval, dense retrieval, and hybrid retrieval affect the accuracy and explainability of an AI system for claim verification?

The current app implements a live modular pipeline:

```text
Claim
→ Claim type detection
→ Query planning
→ Live public evidence retrieval
→ Source credibility scoring
→ Evidence summarization
→ Evidence synthesis
→ Claim-evidence entailment checking
→ Current-context handling for time-sensitive claims
→ Verdict + citations + downloadable report
```

The goal is not to be a perfect truth engine. The goal is to make verification transparent by showing the search plan, retrieved evidence, source quality signals, entailment decisions, and final reasoning.

---

## What makes this project different

- **Modular fact-checking workflow:** separates query planning, retrieval, ranking, summarization, synthesis, entailment checking, and verdict generation.
- **Live evidence retrieval:** retrieves public evidence at runtime instead of relying only on model memory.
- **Hybrid retrieval:** compares TF-IDF, BM25, and hybrid keyword ranking in the live app, with dense retrieval included in the research pipeline.
- **Claim-evidence entailment layer:** reduces false positives where a source is related to a claim but does not actually support it.
- **Source credibility scoring:** labels sources as official/institutional, scholarly, reference, health reference, news, or general web.
- **Current-context handling:** prevents old biography pages from being treated as proof of current facts.
- **Downloadable reports:** exports Markdown and PDF reports with citations.
- **Research evaluation support:** includes benchmark, ablation, entailment, calibration, and error-analysis scripts.

---

## Live app features

- Enter a public factual claim.
- Choose ranking method: `Hybrid`, `BM25`, or `TF-IDF`.
- Enable/disable public evidence sources: general web, Wikipedia, and OpenAlex.
- View the system's search plan and verification workflow.
- Inspect evidence cards with source type, trust signal, relevance, summary, and entailment label.
- Download a Markdown or PDF fact-check report.
- View benchmark/evaluation outputs after running evaluation scripts.

The app returns labels such as:

- `Supported by Evidence`
- `Contradicted by Evidence`
- `Conflicting Evidence`
- `Not Enough Evidence`

---

## System architecture

![System architecture](reports/figures/system_architecture.png)

---

## Evidence sources

The app retrieves evidence live from:

- General web search through DuckDuckGo-compatible search packages when available.
- Wikipedia through the public MediaWiki API.
- OpenAlex for scholarly/scientific works.

The system works best for documented public claims. It cannot reliably verify private, undocumented, hyper-local, very recent, or highly disputed claims.

---

## Methods compared

### TF-IDF retrieval

A sparse keyword baseline that converts claims and evidence passages into weighted term vectors.

### BM25 retrieval

A classic information retrieval method that ranks documents by keyword relevance.

### Hybrid retrieval

The live app uses:

```text
hybrid_score = alpha * bm25_score + (1 - alpha) * tfidf_score
```

Source credibility is then blended into the ranking with a configurable trust weight.

### Dense retrieval

The research pipeline includes Sentence-Transformers dense retrieval for SciFact-style experiments. The live app prioritizes lightweight deployment.

### Entailment checking

The entailment layer classifies claim-evidence relationships as:

- `entailment`
- `contradiction`
- `neutral`

This helps distinguish direct support from merely related evidence.

---

## Evaluation plan

This repository does **not** fabricate results. Metrics should only be reported after running scripts on labeled data.

### 1. Verdict benchmark

Uses a CSV with `claim,label` columns.

```bash
python src/evaluate_realworld_benchmark.py --input data/raw/benchmark_claims.csv --limit 20
```

Outputs:

```text
reports/benchmark_predictions.csv
reports/benchmark_metrics.json
reports/classification_report.csv
reports/evaluation_summary.csv
reports/figures/confusion_matrix.png
```

### 2. Retrieval and credibility ablation

Compares TF-IDF, BM25, hybrid alpha values, and source credibility weighting.

```bash
python src/evaluate_ablation_study.py --input data/raw/benchmark_claims.csv --limit 20
```

Outputs:

```text
reports/ablation_predictions.csv
reports/ablation_results.csv
reports/significance_tests.csv
reports/figures/ablation_macro_f1.png
```

The significance file uses an exact McNemar test for paired system comparisons.

### 3. Entailment layer evaluation

Uses a CSV with `claim,evidence,label` columns.

```bash
python src/evaluate_entailment_layer.py --input data/raw/entailment_pairs.csv
```

Outputs:

```text
reports/entailment_predictions.csv
reports/entailment_metrics.json
reports/figures/entailment_confusion_matrix.png
```

### 4. Confidence calibration

After generating benchmark predictions:

```bash
python src/plot_confidence_calibration.py --input reports/benchmark_predictions.csv
```

Outputs:

```text
reports/confidence_calibration.csv
reports/figures/confidence_calibration.png
```

### 5. Benchmark error analysis

After generating benchmark predictions:

```bash
python src/error_analysis_benchmark.py --input reports/benchmark_predictions.csv
```

Outputs:

```text
reports/benchmark_error_analysis.csv
```

### 6. Original SciFact research pipeline

```bash
python src/load_data.py
python src/preprocess.py
python src/evaluate_retrieval.py
python src/train_classifier.py
python src/evaluate_classifier.py
python src/evaluate_pipeline.py
```

---

## Bundled evaluation files

The repo includes two small development CSVs:

```text
data/raw/benchmark_claims.csv
data/raw/entailment_pairs.csv
```

These are useful for smoke testing and development. For a formal research paper, run a larger public benchmark such as SciFact, FEVER, or AVeriTeC and report the exact setup.

---

## Installation

```bash
git clone <your-repo-url>
cd claim-verification-ai
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the app

```bash
streamlit run streamlit_app.py
```

For Streamlit Community Cloud:

```text
Branch: main
Main file path: streamlit_app.py
```

---

## Run tests

```bash
pytest
```

---

## Limitations

- Live search results can change over time.
- Source credibility scores are transparent heuristics, not proof that a source is correct.
- Entailment predictions can still fail on complex wording or incomplete evidence.
- Confidence is a system signal, not a calibrated probability unless calibration is evaluated.
- The app should support human judgment, not replace expert review.

---

## Ethical considerations

This is a research prototype. It should not be used as a medical, legal, financial, or scientific authority. It can retrieve incomplete or misleading sources and can produce incorrect labels. The intended use is transparent, citation-first support for human review.

---

## College portfolio blurb

Built a modular, web-grounded AI fact verification system that plans search queries, retrieves public evidence, ranks source credibility, summarizes findings, checks claim-evidence entailment, handles time-sensitive claims, and generates citation-backed reports. The project includes a deployed Streamlit interface and research evaluation scripts for benchmark metrics, ablation studies, statistical significance, entailment evaluation, calibration, and error analysis.
