# Streamlit Deployment Notes

Use these settings on Streamlit Community Cloud:

```text
Repository: <your-github-username>/claim-verification-ai
Branch: master
Main file path: streamlit_app.py
```

The root `streamlit_app.py` file is the production entry point. It imports the shared UI from `src/streamlit_app_core.py`.

## Required files

Make sure these files are pushed to GitHub:

```text
streamlit_app.py
requirements.txt
src/streamlit_app_core.py
src/web_pipeline.py
src/web_retriever.py
src/evidence_extractor.py
src/source_ranker.py
src/claim_type_classifier.py
```

## What the app does on startup

The app starts without requiring a preprocessed local dataset. When the user submits a claim, it searches selected public evidence sources, extracts text, ranks passages, and displays citations.

## Main dependencies

```text
streamlit
pandas
numpy
scikit-learn
rank-bm25
requests
beautifulsoup4
duckduckgo-search
```

## Troubleshooting

If general web search fails, keep Wikipedia and OpenAlex enabled in the sidebar. If the app cannot retrieve sources, reboot it from Streamlit Cloud and check logs for dependency installation or network errors.
