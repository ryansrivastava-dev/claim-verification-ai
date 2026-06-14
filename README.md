# Evidence-Based Fact Verification AI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)

A modular AI fact-checking system that verifies claims using live public evidence, source ranking, evidence summarization, evidence synthesis, time-sensitive claim handling, and transparent citation-backed reports.

The project began as **Hybrid Retrieval for Scientific Claim Verification** and expanded into a broader web-grounded verification system. It still includes the SciFact research pipeline, but the deployed Streamlit app checks public claims using live evidence retrieval instead of a toy demo corpus.

---

## Why this project is special

Most simple fact-checking apps do one search and return a label. This project uses a modular workflow that is closer to real automated fact-checking research:

```text
Claim
→ Claim type detection
→ Query planning
→ Live public evidence retrieval
→ Source credibility scoring
→ Evidence summarization
→ Evidence synthesis
→ Freshness guardrail for current facts
→ Evidence-backed verdict
→ Downloadable citation report
```

The goal is not to be a perfect truth engine. The goal is to make verification transparent: the app shows what it searched, what it found, how sources were ranked, and why it produced a label.

---

## Live app features

- User enters any public factual claim.
- System detects the claim category, including climate/environment, health/medical, science/technology, politics/government, business/economics, sports, geography/place, history, entertainment/culture, education, and time-sensitive claims.
- Query planner creates targeted search queries instead of only searching the raw claim.
- Evidence is retrieved from public sources at runtime.
- Sources are ranked with TF-IDF, BM25, or hybrid keyword retrieval.
- Sources receive credibility signals such as official, scholarly, reference, news, or general web.
- Evidence passages are summarized using an extractive summarizer.
- Evidence synthesis explains the overall finding.
- A freshness guardrail prevents old biography pages from being treated as proof of current facts. The app now avoids mislabeling climate claims as geography/history just because they mention a location.
- App returns labels such as:
  - `Likely Supported`
  - `Likely Refuted`
  - `Not Enough Evidence`
- Users can download a Markdown or PDF fact-check report.
- The technical details view is readable by default, with raw JSON hidden under an advanced expander.
- The Streamlit app includes three sections:
  - Live Fact Check
  - Evaluation Results
  - Project Architecture

---

## System architecture

![System architecture](reports/figures/system_architecture.png)

---

## Evidence sources

The deployed app retrieves evidence live from public sources:

- **General web search** using DuckDuckGo-compatible search packages when available.
- **Wikipedia** through the public MediaWiki API.
- **OpenAlex** for scholarly/scientific works.

The system can check many public, documented claims. It cannot guarantee verification of private, undocumented, hyper-local, very recent, or disputed claims.

---

## Research question

How do sparse retrieval, dense retrieval, and hybrid retrieval affect the accuracy and explainability of an AI system for claim verification?

## Hypothesis

A hybrid retrieval system combining keyword-based retrieval with sentence-embedding retrieval will retrieve better evidence and improve end-to-end claim verification performance compared with either retrieval method alone.

---

## Methods compared

### Sparse retrieval

- **TF-IDF:** Converts claims and evidence passages into weighted keyword vectors.
- **BM25:** A strong keyword-search baseline commonly used in information retrieval.

### Dense retrieval

- Uses Sentence-Transformers such as `all-MiniLM-L6-v2` for semantic evidence search in the full research pipeline.

### Hybrid retrieval

The live app combines BM25 and TF-IDF:

```text
hybrid_score = alpha * bm25_score + (1 - alpha) * tfidf_score
```

The research pipeline also includes dense retrieval code for scientific claim verification experiments.

---

## Evaluation

This repository does **not** fabricate results. Metrics should only be added after running evaluation scripts.

Metrics supported:

- Accuracy
- Macro F1
- Per-class F1
- Confusion matrix
- Evidence retrieval quality when gold evidence is available
- Recall@k and MRR for retrieval experiments

Run live benchmark evaluation with a labeled CSV:

```bash
python src/evaluate_realworld_benchmark.py --input data/raw/benchmark_claims.csv --limit 50
```

Expected CSV format:

```csv
claim,label
"The capital of France is Paris.",Supported
"The Eiffel Tower is located in Berlin.",Refuted
```

Outputs:

```text
reports/benchmark_predictions.csv
reports/benchmark_metrics.json
reports/evaluation_leaderboard.csv
reports/figures/confusion_matrix.png
```

The Streamlit app automatically displays these files in the **Evaluation Results** tab when they exist.

---

## Installation

Clone the repo:

```bash
git clone <your-repo-url>
cd claim-verification-ai
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

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

## Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

For Streamlit Community Cloud:

```text
Branch: main
Main file path: streamlit_app.py
```

---

## Run the original SciFact research pipeline

```bash
python src/load_data.py
python src/preprocess.py
python src/evaluate_retrieval.py
python src/train_classifier.py
python src/evaluate_classifier.py
python src/evaluate_pipeline.py
```

---

## Run tests

```bash
pytest
```

---

## Repository structure

```text
claim-verification-ai/
├── README.md
├── requirements.txt
├── streamlit_app.py
├── app/
├── data/
├── notebooks/
├── reports/
│   ├── figures/
│   │   └── system_architecture.png
│   ├── paper_draft.md
│   ├── one_page_summary.md
│   ├── evaluation_results.md
│   └── error_analysis.md
├── src/
│   ├── query_planner.py
│   ├── web_retriever.py
│   ├── source_ranker.py
│   ├── evidence_summarizer.py
│   ├── evidence_synthesizer.py
│   ├── web_pipeline.py
│   ├── report_builder.py
│   ├── evaluate_realworld_benchmark.py
│   └── ...
└── tests/
```

---

## Ethical considerations

- This is not a guaranteed truth engine.
- The system depends on retrieved public sources.
- It can make incorrect predictions if evidence is missing, misleading, outdated, or ambiguous.
- Confidence is a signal, not proof.
- High-stakes domains such as medical, legal, financial, or political claims require expert review.
- The tool is designed to support human judgment, not replace it.

---

## Limitations

- Live web search may miss important sources.
- Search APIs and public endpoints can fail or change.
- Some claims require specialized databases that are not included.
- Very recent claims may not have reliable indexed evidence yet.
- Rule-based synthesis is transparent and lightweight, but less powerful than a carefully evaluated LLM reasoning system.
- Benchmark numbers must be generated by running the included scripts.

---

## Future work

- Evaluate on AVeriTeC, SciFact, FEVER, or LIAR.
- Add optional local small-language-model reasoning for query planning and evidence synthesis.
- Add stronger temporal filtering for current-event claims.
- Add source date extraction and publication-date scoring.
- Add a saved benchmark leaderboard after real experiments are run.
- Add a short demo GIF to the README.
- Improve contradiction detection beyond keyword cues.

---

## College portfolio blurb

Built a modular AI fact-checking system that verifies claims using live public evidence. The system plans search queries, retrieves and ranks sources, summarizes evidence, synthesizes findings, handles time-sensitive claims, and generates transparent citation-backed reports through a Streamlit app. The repository also includes a scientific claim verification research pipeline, evaluation scripts, notebooks, tests, and a research paper draft.
