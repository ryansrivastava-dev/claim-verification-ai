# Evidence-Based Fact Verification AI

## Problem

People encounter factual claims online every day, but many claims are difficult to verify quickly. A trustworthy AI system should not simply guess. It should retrieve evidence, show sources, and clearly indicate when available evidence is insufficient.

## What I built

I built a Python-based evidence-grounded claim verification project. The research core focuses on scientific claim verification with SciFact-style data, while the deployed Streamlit app retrieves live public evidence from general web search, Wikipedia, and OpenAlex scholarly works.

The app takes a claim, searches public sources, extracts evidence passages, ranks those passages, and produces a transparent evidence label such as `Likely Supported`, `Possibly Refuted`, or `Not Enough Evidence`.

## Methods

The project compares multiple retrieval methods:

- TF-IDF keyword retrieval
- BM25 keyword retrieval
- Hybrid keyword retrieval
- Dense sentence-embedding retrieval in the full research environment

The deployed app uses no-GPU retrieval methods so it can run on Streamlit Community Cloud.

## Evaluation

The repository includes scripts for retrieval evaluation, classifier evaluation, end-to-end pipeline evaluation, and error analysis. Metrics include Recall@k, Precision@k, MRR, accuracy, macro F1, per-class F1, and confusion matrices when gold labels are available.

## Results placeholder

Final benchmark results should be filled in after running the experiment scripts. The project intentionally avoids fake metrics or fabricated dataset statistics.

## Skills demonstrated

- Python software engineering
- NLP preprocessing
- Information retrieval
- Machine learning baselines
- Streamlit deployment
- Reproducible research structure
- Error analysis
- Ethical AI design
- GitHub-ready project organization

## Future work

Future versions could add official-source filters, freshness ranking, cross-encoder reranking, larger trained NLI classifiers, better citation extraction, and evaluation on more fact-checking datasets.
