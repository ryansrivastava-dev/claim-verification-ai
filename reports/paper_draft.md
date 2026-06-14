# Evidence-Based Fact Verification AI: A Modular Web-Grounded Claim Verification System

## Abstract

This project presents a modular, evidence-grounded system for public claim verification. Given a claim, the system plans search queries, retrieves live public evidence, ranks sources using relevance and credibility signals, summarizes evidence passages, checks claim-evidence entailment, synthesizes findings, and produces a citation-backed verdict. The project compares sparse retrieval methods such as TF-IDF and BM25, hybrid keyword retrieval, source credibility weighting, and an entailment layer designed to prevent related evidence from being mistaken for direct support. The system includes a Streamlit application and reproducible evaluation scripts for benchmark accuracy, macro F1, ablation studies, McNemar significance testing, entailment precision/recall, confidence calibration, and error analysis. Results should be filled only after running the included scripts on labeled benchmark data. The project is intended as a transparent research prototype, not a replacement for expert fact-checking.

## 1. Introduction

Misinformation spreads quickly online, and many claims require evidence-based verification rather than unsupported model-generated answers. Automated fact-checking systems can help users inspect public evidence, but they must be transparent about sources, uncertainty, and limitations. This project builds a modular claim verification system that focuses on evidence retrieval and explainability.

The central research question is:

> How do sparse retrieval, dense retrieval, and hybrid retrieval affect the accuracy and explainability of an AI system for claim verification?

The hypothesis is that hybrid retrieval and source-aware ranking can improve evidence quality compared with single-method retrieval, and that an entailment layer can reduce false positives from merely related evidence.

## 2. Related Work

Automated fact-checking builds on evidence retrieval, natural language inference, and claim classification. SciFact focuses on scientific claim verification. FEVER introduced large-scale fact verification against Wikipedia evidence. AVeriTeC focuses on real-world claim verification using web evidence. LIAR provides short political claims labeled for truthfulness. Recent systems use retrieval-augmented generation and modular fact-checking pipelines to improve interpretability.

This project is inspired by those directions but remains a lightweight, reproducible research prototype suitable for normal development environments. It does not claim state-of-the-art performance unless evaluation results are generated and compared under matching conditions.

## 3. System Design

The system follows this pipeline:

```text
Claim → Claim Type Detection → Query Planning → Evidence Retrieval → Source Credibility Scoring → Evidence Summarization → Evidence Synthesis → Entailment Checking → Verdict + Report
```

### Claim type detection

The claim type detector identifies broad topics such as health/medical, climate/environment, politics/government, business/economics, sports, geography/place, history, and current public office/leadership claims. This helps the search planner produce better queries and helps the interface explain how the claim was processed.

### Query planning

The query planner generates multiple evidence-seeking queries rather than searching only the raw claim. For simple subject-predicate claims, it can generate entity-focused queries to look for canonical facts and counter-evidence.

### Evidence retrieval

The live app retrieves from public sources including general web search, Wikipedia, and OpenAlex. The original research pipeline also includes SciFact loading, preprocessing, sparse retrieval, dense retrieval, hybrid retrieval, classifier training, and evaluation.

### Source credibility scoring

Each source receives a transparent trust signal based on source type, such as official/government/institutional, scholarly, reference, health reference, news, or general web. This score is blended with retrieval relevance in the live ranking. The source credibility component is evaluated separately in ablation studies.

### Entailment layer

The entailment layer checks whether evidence supports, contradicts, or is neutral toward the claim. This is included because retrieval relevance alone can mistake related sources for supporting evidence. For example, a source that mentions Mars and blue sunsets should not support the claim “Mars is blue.”

### Evidence synthesis and report generation

The system synthesizes retrieved evidence into a short analysis and generates a downloadable report containing the claim, label, confidence, search plan, evidence summaries, citations, and limitations.

## 4. Experiments

The project includes scripts for four main experiments.

### 4.1 Verdict benchmark

Run:

```bash
python src/evaluate_realworld_benchmark.py --input data/raw/benchmark_claims.csv --limit 20
```

Metrics:

- Accuracy
- Macro F1
- Per-class F1
- Confusion matrix

### 4.2 Retrieval and credibility ablation

Run:

```bash
python src/evaluate_ablation_study.py --input data/raw/benchmark_claims.csv --limit 20
```

This compares:

- TF-IDF only
- BM25 only
- Hybrid alpha 0.25
- Hybrid alpha 0.50
- Hybrid alpha 0.75
- Hybrid with no source credibility weighting
- Hybrid with stronger source credibility weighting

The script also writes exact McNemar tests comparing paired correctness across systems.

### 4.3 Entailment evaluation

Run:

```bash
python src/evaluate_entailment_layer.py --input data/raw/entailment_pairs.csv
```

Metrics:

- Macro precision
- Macro recall
- Macro F1
- Entailment confusion matrix

### 4.4 Confidence calibration

Run:

```bash
python src/plot_confidence_calibration.py --input reports/benchmark_predictions.csv
```

The calibration plot compares predicted confidence against observed accuracy.

## 5. Results

Results must be filled after running the evaluation scripts. Do not fabricate metrics.

| Experiment | Metric | Result |
|---|---:|---:|
| Verdict benchmark | Accuracy | TBD |
| Verdict benchmark | Macro F1 | TBD |
| Best ablation config | Macro F1 | TBD |
| Entailment layer | Macro F1 | TBD |
| Calibration | Notes | TBD |

## 6. Error Analysis

The error analysis should categorize failures into:

- Retrieval failure
- Source credibility failure
- Entailment failure
- Ambiguous claim
- Current-context failure
- Insufficient evidence
- Label mapping issue

The project includes `reports/error_analysis.md` and scripts to create prediction-level outputs for manual review.

## 7. Discussion

The key research value of this project is not simply the final label. It is the ability to inspect how retrieval, source quality, and entailment influence the final verdict. Hybrid retrieval can improve ranking quality, but it must be evaluated against BM25 and TF-IDF on the same benchmark. Source credibility weighting is useful only if ablation results show improved accuracy or better error behavior. The entailment layer is a major contribution because it addresses a common failure mode in retrieval-based fact-checking: related evidence being treated as support.

## 8. Limitations

- Live web retrieval changes over time, making exact reproducibility difficult.
- Small benchmarks can overstate performance.
- Source credibility scoring is heuristic and must be evaluated.
- The entailment layer can fail on complex phrasing.
- Confidence scores should not be interpreted as probabilities unless calibrated.
- The app is not appropriate as a final authority for medical, legal, financial, or high-stakes decisions.

## 9. Conclusion

This project builds a modular, transparent claim verification system that retrieves live evidence, ranks source credibility, checks entailment, synthesizes findings, and generates citation-backed reports. The strongest next step is to run larger public benchmark evaluations and use the resulting metrics to test the project’s central hypothesis about hybrid retrieval and evidence quality.

## References

- Wadden et al. SciFact: Verifying Scientific Claims.
- Thorne et al. FEVER: a Large-scale Dataset for Fact Extraction and VERification.
- Schlichtkrull et al. AVeriTeC: A Dataset for Real-world Claim Verification with Evidence from the Web.
- Wang. LIAR, LIAR Pants on Fire: A New Benchmark Dataset for Fake News Detection.
- Robertson and Zaragoza. The Probabilistic Relevance Framework: BM25 and Beyond.
- Reimers and Gurevych. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.
- Pedregosa et al. Scikit-learn: Machine Learning in Python.
- Hugging Face Datasets documentation.
