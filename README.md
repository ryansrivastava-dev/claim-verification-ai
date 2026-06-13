# Evidence-Based Fact Verification AI

A GitHub-ready Python project for building an evidence-grounded claim verification system. The app takes a claim, retrieves public evidence at runtime, ranks source passages, and returns an evidence label with citations.

The original research question focuses on **Hybrid Retrieval for Scientific Claim Verification**. The deployed version expands the project into a broader web-grounded verification system by using live public sources instead of a local sample corpus.

## Research question

How do sparse retrieval, dense retrieval, and hybrid retrieval affect the accuracy and explainability of an AI system for claim verification?

## Hypothesis

A hybrid retrieval system combining keyword-based retrieval with sentence-embedding retrieval will retrieve better evidence and improve end-to-end claim verification performance compared with either retrieval method alone.

## What the project does

- Accepts a user-entered claim.
- Searches public evidence sources at runtime.
- Extracts readable evidence passages from retrieved sources.
- Ranks passages using TF-IDF, BM25, or hybrid retrieval.
- Returns a source-grounded evidence label:
  - `Likely Supported`
  - `Possibly Refuted`
  - `Not Enough Evidence`
- Shows source titles, URLs, evidence passages, and retrieval scores.
- Includes a SciFact research pipeline for dataset loading, preprocessing, retrieval experiments, classification experiments, end-to-end evaluation, error analysis, notebooks, reports, and tests.

## Live app evidence sources

The Streamlit app does not rely on a fake sample dataset or a local sample-data fallback. It retrieves evidence live from public sources.

Included no-key sources:

- **General web search** through DuckDuckGo Search when the package is available.
- **Wikipedia** through the public MediaWiki API.
- **OpenAlex** for scholarly/scientific works.

The system can check many public, well-documented claims, but it cannot guarantee verification of every possible fact. Some facts are private, undocumented, very recent, ambiguous, or disputed. In those cases, the correct output is usually `Not Enough Evidence`.

## What it can check well

Examples that usually work better:

```text
The capital of France is Paris.
High blood pressure increases risk of stroke.
Antibiotics are used to treat bacterial infections.
NASA launched the James Webb Space Telescope in 2021.
The Eiffel Tower is located in Paris.
```

Examples that may require more specialized or current sources:

```text
A private person attended school yesterday.
A team won a game five minutes ago.
A local restaurant changed its hours today.
A disputed opinion is objectively true.
```

## Methods compared

### Sparse retrieval

Sparse retrieval uses keyword overlap between a claim and evidence passages.

- **TF-IDF:** Converts claims and evidence passages into weighted keyword vectors.
- **BM25:** A strong keyword-search baseline commonly used in information retrieval.

### Dense retrieval

Dense retrieval uses a lightweight Sentence-Transformers model such as `all-MiniLM-L6-v2` to encode claims and evidence passages into semantic vectors. The single `requirements.txt` file includes both the Streamlit app dependencies and the full research dependencies.

### Hybrid retrieval

Hybrid retrieval combines keyword retrieval signals:

```text
hybrid_score = alpha * bm25_score + (1 - alpha) * tfidf_score
```

In the SciFact research pipeline, the project also includes dense retrieval and hybrid dense/sparse comparison code.

## Evaluation metrics

Retrieval metrics:

- Recall@1, Recall@3, Recall@5, Recall@10
- Precision@k
- Mean Reciprocal Rank (MRR)

Classification metrics:

- Accuracy
- Macro F1
- Per-class F1
- Confusion matrix

End-to-end metrics:

- Accuracy
- Macro F1
- Evidence retrieval quality when gold evidence is available

The repository does not fabricate results. Metrics are produced only by running the experiment scripts.

## Installation

Clone the repo:

```bash
git clone <your-repo-url>
cd claim-verification-ai
```

Create and activate a virtual environment:

```bash
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

Install the full project dependencies:

```bash
pip install -r requirements.txt
```

## Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

Alternative local app path:

```bash
streamlit run app/streamlit_app.py
```

For Streamlit Community Cloud:

```text
Branch: main
Main file path: streamlit_app.py
```

## Run the web-grounded pipeline from Python

```python
from src.web_pipeline import run_web_fact_check

result = run_web_fact_check("The capital of France is Paris.")
print(result["predicted_label"])
for evidence in result["top_evidence"]:
    print(evidence["title"], evidence["url"])
```

## Run the SciFact research pipeline

Load and preprocess SciFact:

```bash
python src/load_data.py
python src/preprocess.py
```

Evaluate retrieval:

```bash
python src/evaluate_retrieval.py
```

Train classifiers:

```bash
python src/train_classifier.py
```

Evaluate classifiers:

```bash
python src/evaluate_classifier.py
```

Evaluate end-to-end systems:

```bash
python src/evaluate_pipeline.py
```

Run the scientific command-line pipeline:

```bash
python src/pipeline.py --claim "High blood pressure increases risk of stroke." --retriever hybrid --top_k 5
```

Run tests:

```bash
pytest
```

## Repository structure

```text
claim-verification-ai/
├── README.md
├── requirements.txt
├── streamlit_app.py
├── app/
│   └── streamlit_app.py
├── src/
│   ├── web_retriever.py
│   ├── web_pipeline.py
│   ├── streamlit_app_core.py
│   ├── load_data.py
│   ├── preprocess.py
│   ├── retrieve_tfidf.py
│   ├── retrieve_bm25.py
│   ├── retrieve_dense.py
│   ├── retrieve_hybrid.py
│   ├── train_classifier.py
│   ├── evaluate_retrieval.py
│   ├── evaluate_classifier.py
│   ├── evaluate_pipeline.py
│   ├── error_analysis.py
│   └── pipeline.py
├── notebooks/
├── reports/
├── tests/
└── data/
```

## Example output format

```json
{
  "claim": "The capital of France is Paris.",
  "predicted_label": "Likely Supported",
  "confidence": 0.83,
  "retrieval_method": "Hybrid",
  "source_mode": "Live public evidence retrieval",
  "top_evidence": [
    {
      "title": "Paris",
      "url": "https://en.wikipedia.org/wiki/Paris",
      "score": 0.81,
      "text": "..."
    }
  ]
}
```

This is an output format example, not a reported experiment result.

## Ethical considerations

- The system should support human judgment, not replace it.
- Predictions depend on retrieved evidence and can be wrong.
- The system may miss important sources or retrieve irrelevant evidence.
- Confidence is a ranking signal, not proof of truth.
- Time-sensitive claims require fresh sources.
- Medical, legal, financial, and safety-related claims should be checked with qualified experts or authoritative sources.

## Limitations

- No evidence retrieval system can verify every real-world fact.
- Some claims are private, local, undocumented, ambiguous, disputed, or too recent.
- DuckDuckGo/web retrieval can vary depending on network access and search availability.
- Wikipedia and OpenAlex coverage is strong for many public topics but not universal.
- The Streamlit app uses source-grounded retrieval and transparent evidence labeling, while the full research pipeline includes dense retrieval and classifier training scripts.
- The SciFact benchmark pipeline must be run separately to generate real experiment metrics.

## Future work

- Add optional search APIs such as Brave Search, Tavily, Bing Web Search, or Google Custom Search.
- Add official-source filters for `.gov`, `.edu`, and organization domains.
- Add freshness-aware source ranking for current claims.
- Add cross-encoder reranking.
- Add a trained natural language inference classifier.
- Add source comparison for conflicting evidence.
- Add quote highlighting and stronger citation extraction.
- Evaluate on additional datasets beyond SciFact.

## College admissions/project summary blurb

I built an evidence-grounded AI fact verification system that retrieves public sources, ranks evidence passages, and predicts whether a claim is likely supported, possibly refuted, or lacks enough evidence. The project compares TF-IDF, BM25, dense retrieval, and hybrid retrieval for trustworthy AI research, and includes a functional Streamlit app, SciFact research pipeline, modular Python code, evaluation scripts, tests, notebooks, error analysis, and a research paper draft. It demonstrates skills in Python, NLP, machine learning, information retrieval, software engineering, reproducible research, and ethical AI design.
