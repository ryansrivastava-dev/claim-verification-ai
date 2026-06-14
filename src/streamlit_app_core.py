"""Shared Streamlit UI for hosted and local app entry points."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from web_pipeline import run_web_fact_check
except ImportError:  # pragma: no cover
    from src.web_pipeline import run_web_fact_check


def _display_workflow(result: dict) -> None:
    """Render the step-by-step verification workflow."""
    steps = result.get("workflow_steps", [])
    if not steps:
        return

    st.subheader("Verification workflow")
    cols = st.columns(len(steps))
    for col, step in zip(cols, steps):
        col.metric(step.get("step", "Step"), "Done")
        col.caption(step.get("detail", ""))


def _display_search_plan(result: dict) -> None:
    plan = result.get("search_plan", {})
    st.markdown("### Search plan")
    st.write(f"**Primary query:** {plan.get('primary_query', result.get('retrieval_query', ''))}")

    queries = plan.get("generated_queries", [])
    if queries:
        st.write("**Generated queries:**")
        for query in queries:
            st.write(f"- {query}")

    notes = plan.get("strategy_notes", [])
    if notes:
        st.write("**Planning notes:**")
        for note in notes:
            st.write(f"- {note}")


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
            relevance_label = item.get("evidence_relevance_label", "Retrieved evidence")
            st.markdown(f"### Evidence {index}")
            if url:
                st.markdown(f"**Source:** [{title}]({url})")
                st.caption(f"{source} • {relevance_label}")
            else:
                st.markdown(f"**Source:** {title}")
                st.caption(f"{source} • {relevance_label}")

            summary = item.get("summary")
            if summary:
                st.write("**Evidence summary:**", summary)
                with st.expander("Full retrieved passage"):
                    st.write(item.get("text", ""))
            else:
                st.write(item.get("text", ""))

            score_cols = st.columns(5)
            score_cols[0].metric("Overall", f"{item.get('score', 0.0):.3f}")
            score_cols[1].metric("Relevance", f"{item.get('relevance_score', 0.0):.3f}")
            score_cols[2].metric("Trust", f"{item.get('trust_score', 0.0):.2f}")
            score_cols[3].metric("BM25", f"{item.get('bm25_score', 0.0):.3f}")
            score_cols[4].metric("TF-IDF", f"{item.get('tfidf_score', 0.0):.3f}")


def main(project_root: Path | None = None) -> None:
    """Run the Streamlit app."""
    st.set_page_config(
        page_title="Evidence-Based Fact Verification",
        page_icon="🔎",
        layout="wide",
    )

    st.title("Evidence-Based Fact Verification")
    st.write(
        "Enter a claim. The system plans search queries, retrieves public evidence, summarizes sources, "
        "synthesizes the evidence, and returns a citation-backed evidence label."
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

        progress = st.progress(0, text="Starting claim analysis...")
        try:
            progress.progress(15, text="Planning search queries...")
            progress.progress(35, text="Retrieving public evidence...")
            progress.progress(60, text="Summarizing evidence passages...")
            progress.progress(80, text="Synthesizing evidence and evaluating verdict...")
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
            progress.progress(100, text="Verification complete.")
        except Exception as exc:  # pragma: no cover - depends on network/service availability
            progress.empty()
            st.subheader("Evidence retrieval did not complete")
            st.write(
                "The app is running, but one or more public evidence sources could not be reached. "
                "Try a shorter claim, disable one source in the sidebar, or rerun the request."
            )
            st.code(f"{type(exc).__name__}: {exc}")
            return

        st.subheader("Result")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Evidence label", result["predicted_label"])
        col2.metric("Confidence signal", f"{result['confidence']:.2f}")
        col3.metric("Claim type", result["claim_profile"]["category"])
        col4.metric("Evidence strength", result.get("synthesis", {}).get("evidence_strength", "Unknown"))

        st.write("**Explanation:**", result["explanation"])
        st.caption(result["claim_profile"]["reason"])

        tabs = st.tabs(["Workflow", "Evidence", "Synthesis", "Report", "Technical details"])

        with tabs[0]:
            _display_workflow(result)
            _display_search_plan(result)

        with tabs[1]:
            _display_evidence(result["top_evidence"])

        with tabs[2]:
            synthesis = result.get("synthesis", {})
            st.markdown("### Evidence synthesis")
            st.write(synthesis.get("analysis", "No synthesis available."))
            synth_cols = st.columns(3)
            synth_cols[0].metric("Unique sources", synthesis.get("source_count", 0))
            synth_cols[1].metric("Strong sources", synthesis.get("strong_source_count", 0))
            synth_cols[2].metric("Needs more search", "Yes" if synthesis.get("needs_more_search") else "No")
            if synthesis.get("rereview_query"):
                st.write("**Re-retrieval query considered:**", synthesis["rereview_query"])

        with tabs[3]:
            report = result.get("fact_check_report", "")
            st.download_button(
                label="Download fact-check report",
                data=report,
                file_name="fact_check_report.md",
                mime="text/markdown",
            )
            st.markdown(report)

        with tabs[4]:
            detail_rows = [
                {
                    "passage_id": item.get("passage_id"),
                    "title": item.get("title"),
                    "source": item.get("source"),
                    "score": item.get("score"),
                    "trust_score": item.get("trust_score"),
                    "planned_query": item.get("planned_query"),
                    "url": item.get("url"),
                }
                for item in result["top_evidence"]
            ]
            if detail_rows:
                st.dataframe(pd.DataFrame(detail_rows), use_container_width=True)
            st.json(
                {
                    "retrieval_query": result.get("retrieval_query"),
                    "source_mode": result.get("source_mode"),
                    "claim_profile": result.get("claim_profile"),
                }
            )

    st.divider()
    st.caption(
        "Evidence-based verification means the label depends on retrieved public sources. "
        "The app should be used as a citation-first research tool, not as a replacement for expert review."
    )
