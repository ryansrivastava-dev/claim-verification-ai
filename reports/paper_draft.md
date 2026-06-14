# Evidence-Grounded Claim Verification AI

## Abstract

Scientific misinformation can spread quickly, and AI systems that answer scientific questions need to show evidence rather than only produce confident-sounding responses. This project builds a lightweight scientific claim verification system that takes a claim, retrieves evidence passages from a scientific corpus, and predicts whether the claim is supported, refuted, or not enough information based on available labels. The research question is whether hybrid retrieval, which combines sparse keyword retrieval and dense sentence-embedding retrieval, improves evidence quality and end-to-end claim verification performance compared with either method alone. The system compares TF-IDF, BM25, dense retrieval with Sentence-Transformers, and a BM25 + dense hybrid retriever. It also trains lightweight classifiers using TF-IDF features and sentence embeddings. Evaluation includes Recall@k, Precision@k, MRR, accuracy, macro F1, per-class F1, and confusion matrices when the dataset format supports those metrics. Final results will be filled in after running the included experiment scripts; no metrics are reported here until they are computed from real data.

## 1. Introduction

Scientific claim verification is the task of checking whether a claim is supported, contradicted, or not proven by available scientific evidence. This matters because many AI systems can generate fluent answers without grounding them in reliable evidence. For scientific and medical topics, this can be especially risky because users may confuse a confident answer with a true answer.

This project focuses on evidence-grounded AI. Instead of only predicting a label, the system retrieves the passages that influenced the prediction. This makes the model easier to inspect and helps identify whether an error came from poor retrieval, poor classification, or ambiguous evidence.

## 2. Research Question and Hypothesis

**Research question:** How do sparse retrieval, dense retrieval, and hybrid retrieval affect the accuracy and explainability of an AI system for scientific claim verification?

**Hypothesis:** A hybrid retrieval system combining keyword-based retrieval with sentence-embedding retrieval will retrieve better evidence and improve end-to-end claim verification performance compared with either retrieval method alone.

## 3. Background

**Sparse retrieval** represents text using word-based features. TF-IDF and BM25 are sparse retrieval methods because they mainly depend on token overlap between a claim and evidence passage. These methods are strong baselines because they are fast, interpretable, and often effective when claims share important terms with the evidence.

**Dense retrieval** represents text using learned vector embeddings. A sentence embedding model can place semantically similar claims and evidence close together even when they use different wording. This can help with paraphrases, but dense retrieval can also retrieve semantically related passages that do not directly prove or disprove the claim.

**Hybrid retrieval** combines sparse and dense signals. The goal is to keep the precision and interpretability of keyword matching while adding semantic matching for claims that use different wording than the evidence.

**Claim verification** combines retrieval and classification. The pipeline in this project is:

```text
Claim → Evidence Retrieval → Claim/Evidence Classification → Prediction + Evidence
```

## 4. Dataset

The preferred dataset is SciFact, a scientific claim verification dataset. The included data loading script attempts to load SciFact through Hugging Face Datasets and saves processed claims and evidence passages into `data/processed/`.

Dataset size, split counts, label distribution, and evidence passage counts should be reported only after running:

```bash
python src/load_data.py
```

The script writes the actual computed statistics to:

```text
data/processed/dataset_summary.json
```

If the dataset fails to load, the script writes a failure report to:

```text
data/processed/LOAD_FAILED.md
```

No dataset statistics are invented in this draft.

## 5. Methodology

The system has three major stages.

First, the project loads and preprocesses claims and scientific documents. Documents are split into passages or sentence-like chunks so retrieval can return specific evidence rather than entire papers.

Second, the system retrieves evidence for each claim using several methods:

- TF-IDF cosine similarity
- BM25 keyword retrieval
- Dense retrieval with `all-MiniLM-L6-v2`
- Hybrid retrieval using normalized BM25 and dense scores

The hybrid score is:

```text
hybrid_score = alpha * bm25_score + (1 - alpha) * dense_score
```

Third, the system trains classifiers that take the claim plus retrieved or gold evidence text as input. The included classifiers are logistic regression with TF-IDF features and a sentence-embedding classifier.

## 6. Experiments

### Retrieval experiments

The retrieval experiments compare TF-IDF, BM25, dense retrieval, and hybrid retrieval. When gold evidence document IDs are available, the project computes Recall@1, Recall@3, Recall@5, Recall@10, Precision@k, and Mean Reciprocal Rank.

### Classification experiments

The classification experiments train lightweight models to predict `SUPPORTED`, `REFUTED`, or `NOT_ENOUGH_INFO` when labels are available. The project reports accuracy, macro F1, per-class F1, and confusion matrices.

### End-to-end experiments

The end-to-end experiments compare complete systems:

- TF-IDF retrieval + classifier
- BM25 retrieval + classifier
- Dense retrieval + classifier
- Hybrid retrieval + classifier

These experiments test whether better retrieval improves the final claim verification result.

## 7. Results

Results should be filled in only after running the experiment scripts.

### Retrieval results

Placeholder: Insert the table from `data/processed/retrieval_results.csv`.

Placeholder: Insert the chart from `reports/figures/retrieval_comparison.png`.

### Classifier results

Placeholder: Insert the table from `data/processed/classifier_results.csv`.

Placeholder: Insert confusion matrix figures from `reports/figures/`.

### End-to-end results

Placeholder: Insert the table from `data/processed/pipeline_results.csv`.

Placeholder: Insert the chart from `reports/figures/pipeline_comparison.png`.

No conclusions should claim that one method is best until these metrics are computed from real experiment runs.

## 8. Error Analysis

Mistakes are reviewed using `src/error_analysis.py`, which creates a CSV of incorrect predictions. The suggested categories are:

- Retrieval failure
- Classification failure
- Ambiguous claim
- Insufficient evidence
- Label confusion

Automatic error categories are only a starting point. Final error analysis should include manual review because the reason for an incorrect prediction may not be obvious from scores alone.

## 9. Discussion

The main purpose of this project is to understand how retrieval quality affects claim verification. If hybrid retrieval improves Recall@k and end-to-end macro F1, that would support the hypothesis that combining lexical and semantic signals improves evidence grounding. If it does not improve performance, the error analysis may show that classification errors, dataset ambiguity, or noisy evidence matching are limiting the system.

Explainability is also important. Sparse retrieval scores are easier to interpret because they are based on word overlap. Dense retrieval can find paraphrases but can be harder to explain. Hybrid retrieval may provide a practical balance between performance and interpretability.

## 10. Limitations

This project is not a medical or scientific truth engine. It is a research prototype. It depends on the dataset, the retrieved evidence, and the classifier. It can make incorrect predictions.

Other limitations include:

- Dataset coverage may not include all scientific claims.
- Some evidence may be ambiguous or require full-paper context.
- Evidence annotations may not perfectly match sentence-level passages.
- The project uses laptop-friendly models rather than large transformer fine-tuning.
- Confidence scores are not proof of truth.
- The system should support human judgment, not replace it.

## 11. Conclusion

This project builds a complete evidence-grounded scientific claim verification system. It compares sparse, dense, and hybrid retrieval methods; trains lightweight classifiers; evaluates retrieval and classification metrics; includes an end-to-end pipeline; and provides tools for error analysis and interactive app use. The project demonstrates how retrieval choices can affect trustworthy AI systems and why evidence inspection is important for scientific claims.

## 12. References

- Wadden, D., Lin, S., Lo, K., Wang, L. L., van Zuylen, M., Cohan, A., & Hajishirzi, H. (2020). Fact or Fiction: Verifying Scientific Claims. *Proceedings of EMNLP*.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *Proceedings of EMNLP-IJCNLP*.
- Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*.
- Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*.
- Lhoest, Q., del Moral, A. V., Jernite, Y., et al. (2021). Datasets: A Community Library for Natural Language Processing. *Proceedings of EMNLP System Demonstrations*.
- Wolf, T., Debut, L., Sanh, V., et al. (2020). Transformers: State-of-the-Art Natural Language Processing. *Proceedings of EMNLP System Demonstrations*.

## Streamlit App Extension

The updated hosted app extends the original scientific claim verification project into a broader evidence-grounded claim verification interface. Instead of requiring a preprocessed SciFact file before the app can start, the deployed app retrieves public evidence from general web search, Wikipedia through the MediaWiki API, and OpenAlex scholarly works. This makes the app functional immediately on Streamlit Community Cloud without paid APIs or private keys.

This extension does not mean the system can verify all real-world facts. It works best for public, well-documented claims and is weaker for private, local, very recent, or disputed claims. The app therefore reports an evidence signal and shows source links rather than presenting the prediction as absolute truth.

The live app uses TF-IDF, BM25, and hybrid keyword retrieval so it can run without a GPU. The full research repository also includes dense retrieval and SciFact experiment scripts for benchmark-style evaluation.

## Addendum: Modular Web-Grounded Fact Verification Upgrade

The deployed version of the project extends the original scientific claim verification pipeline into a broader web-grounded fact verification prototype. The app now follows a modular workflow: claim type detection, query planning, live evidence retrieval, source ranking, evidence summarization, evidence synthesis, freshness checking for time-sensitive claims, final evidence labeling, and report generation.

This upgrade improves explainability because users can inspect the search plan, retrieved sources, source-quality signals, evidence summaries, synthesis, and citations. The system also includes a benchmark evaluation script that can compute accuracy, macro F1, per-class metrics, and confusion matrices from a labeled CSV. These results are intentionally not filled in until the evaluation is actually run.

The system remains a research prototype. It does not guarantee truth, and it can fail when public evidence is incomplete, outdated, misleading, or ambiguous. Its purpose is to support human evidence review by making retrieval and reasoning steps visible.
