"""Shared Streamlit UI for hosted and local app entry points."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from web_pipeline import run_web_fact_check
except ImportError:  # pragma: no cover
    from src.web_pipeline import run_web_fact_check


def _display_evidence(evidence: list[dict]) -> None:
    """Render retrieved evidence in clean cards."""
    if not evidence:
        st.write("No source-backed evidence passages were retrieved for this claim.")
        return

    for index, item in enumerate(evidence, start=1):
        with st.container(border=True):
            title = item.get("title", "Untitled source")
            url = item.get("url", "")
            source = item.get("source", "Source")
            st.markdown(f"### Evidence {index}")
            if url:
                st.markdown(f"**Source:** [{title}]({url})  ")
                st.caption(source)
            else:
                st.markdown(f"**Source:** {title}")
                st.caption(source)
            st.write(item.get("text", ""))

            score_cols = st.columns(4)
            score_cols[0].metric("Overall", f"{item.get('score', 0.0):.3f}")
            score_cols[1].metric("Relevance", f"{item.get('relevance_score', 0.0):.3f}")
            score_cols[2].metric("BM25", f"{item.get('bm25_score', 0.0):.3f}")
            score_cols[3].metric("TF-IDF", f"{item.get('tfidf_score', 0.0):.3f}")


def main(project_root: Path | None = None) -> None:
    """Run the Streamlit app."""
    st.set_page_config(
        page_title="Evidence-Based Fact Verification",
        page_icon="🔎",
        layout="wide",
    )

    st.title("Evidence-Based Fact Verification")
    st.write(
        "Enter a claim. The system searches public sources, ranks evidence passages, "
        "and returns a source-grounded evidence label with citations."
    )

    with st.sidebar:
        st.header("Retrieval settings")
        method = st.selectbox("Ranking method", ["Hybrid", "BM25", "TF-IDF"])
        top_k = st.slider("Evidence passages", min_value=1, max_value=10, value=5)
        alpha = st.slider(
            "Hybrid keyword weight",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.25,
            help="Higher values give BM25 more weight; lower values give TF-IDF more weight.",
        )
        max_pages = st.slider("Sources per search backend", min_value=2, max_value=10, value=6)

        st.header("Source settings")
        use_web = st.checkbox("General web search", value=True)
        use_wikipedia = st.checkbox("Wikipedia", value=True)
        use_openalex = st.checkbox("OpenAlex scholarly works", value=True)

        st.caption(
            "Use at least one source. General web search uses DuckDuckGo Search from requirements.txt."
        )

    claim = st.text_area(
        "Claim to verify",
        placeholder="Enter a factual claim to verify against public evidence sources.",
        height=110,
    )

    if not any([use_web, use_wikipedia, use_openalex]):
        st.write("Select at least one evidence source in the sidebar.")
        return

    run_button = st.button("Verify claim", type="primary")

    if run_button:
        if not claim.strip():
            st.write("Please enter a claim before running verification.")
            return

        with st.spinner("Searching sources, extracting passages, and ranking evidence..."):
            try:
                result = run_web_fact_check(
                    claim=claim,
                    method=method,
                    top_k=top_k,
                    alpha=alpha,
                    max_pages=max_pages,
                    use_web=use_web,
                    use_wikipedia=use_wikipedia,
                    use_openalex=use_openalex,
                )
            except Exception as exc:  # pragma: no cover - depends on network/service availability
                st.subheader("Evidence retrieval did not complete")
                st.write(
                    "The app is running, but one or more public evidence sources could not be reached. "
                    "Try a shorter claim, disable one source in the sidebar, or rerun the request."
                )
                st.code(f"{type(exc).__name__}: {exc}")
                return

        st.subheader("Result")
        col1, col2, col3 = st.columns(3)
        col1.metric("Evidence label", result["predicted_label"])
        col2.metric("Confidence signal", f"{result['confidence']:.2f}")
        col3.metric("Claim type", result["claim_profile"]["category"])

        st.write("**Explanation:**", result["explanation"])
        st.caption(result["claim_profile"]["reason"])

        st.subheader("Retrieved evidence")
        _display_evidence(result["top_evidence"])

        with st.expander("Method details"):
            st.write(
                "The app searches selected public sources, extracts readable text, splits it into passages, "
                "ranks those passages with keyword retrieval, and assigns an evidence label from the strongest sources. "
                "For fast-changing or private claims, the system may correctly return Not Enough Evidence."
            )
            detail_rows = [
                {
                    "passage_id": item.get("passage_id"),
                    "title": item.get("title"),
                    "source": item.get("source"),
                    "score": item.get("score"),
                    "url": item.get("url"),
                }
                for item in result["top_evidence"]
            ]
            if detail_rows:
                st.dataframe(pd.DataFrame(detail_rows), use_container_width=True)

    st.divider()
    st.caption(
        "Evidence-based verification means the label depends on retrieved public sources. "
        "The app should be used as a citation-first research tool, not as a replacement for expert review."
    )
