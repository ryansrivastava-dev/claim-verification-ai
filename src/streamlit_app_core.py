"""Shared Streamlit UI for hosted and local app entry points."""

from __future__ import annotations

from pathlib import Path
import html

import pandas as pd
import streamlit as st

try:
    from report_builder import build_citation_list, build_pdf_report_bytes
    from web_pipeline import run_web_fact_check
except ImportError:  # pragma: no cover
    from src.report_builder import build_citation_list, build_pdf_report_bytes
    from src.web_pipeline import run_web_fact_check


def _display_result_summary(result: dict) -> None:
    """Render final result fields without Streamlit metric truncation."""
    label = str(result.get("predicted_label", "Unknown"))
    confidence = result.get("confidence", 0.0)
    category = str(result.get("claim_profile", {}).get("category", "Unknown"))
    strength = str(result.get("synthesis", {}).get("evidence_strength", "Unknown"))
    freshness = "On" if result.get("claim_profile", {}).get("needs_current_source") else "Off"

    cards = [
        ("Evidence label", label),
        ("Confidence signal", f"{confidence:.2f}" if isinstance(confidence, (int, float)) else str(confidence)),
        ("Claim type", category),
        ("Evidence strength", strength),
        ("Freshness check", freshness),
    ]

    card_html_parts = []
    for name, value in cards:
        card_html_parts.append(
            "<div class='result-card'>"
            f"<div class='result-label'>{html.escape(name)}</div>"
            f"<div class='result-value'>{html.escape(value)}</div>"
            "</div>"
        )

    st.markdown(
        "<div class='result-grid'>" + "".join(card_html_parts) + "</div>",
        unsafe_allow_html=True,
    )


def _display_workflow(result: dict) -> None:
    """Render the step-by-step verification workflow."""
    steps = result.get("workflow_steps", [])
    if not steps:
        return

    st.subheader("Verification workflow")
    cols = st.columns(min(len(steps), 6))
    for col, step in zip(cols, steps[:6]):
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


def _display_freshness_guardrail(result: dict) -> None:
    profile = result.get("claim_profile", {})
    active = bool(profile.get("needs_current_source"))
    st.markdown("### Freshness guardrail")
    if active:
        st.success("Freshness check: enabled")
        st.write(
            "The claim contains time-sensitive wording. The system gives extra weight to current, official, "
            "or role-specific sources and avoids treating old biography matches as proof of current facts."
        )
        if profile.get("current_role_claim"):
            st.json(profile["current_role_claim"])
    else:
        st.write("Freshness check: not triggered for this claim.")


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
            category = item.get("source_category", "Unknown source type")
            trust_label = item.get("trust_label", "Unknown trust signal")
            strength_label = item.get("evidence_strength_label", "Evidence candidate")
            relevance_label = item.get("evidence_relevance_label", "Retrieved evidence")

            st.markdown(f"### Evidence {index}")
            if url:
                st.markdown(f"**Source:** [{title}]({url})")
                st.caption(f"{source} • {category} • {trust_label} • {relevance_label}")
            else:
                st.markdown(f"**Source:** {title}")
                st.caption(f"{source} • {category} • {trust_label} • {relevance_label}")

            st.write(f"**Evidence quality:** {strength_label}")
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


def _display_report_tab(result: dict) -> None:
    report = result.get("fact_check_report", "")
    st.markdown("### Downloadable report")
    col1, col2 = st.columns(2)
    col1.download_button(
        label="Download Markdown report",
        data=report,
        file_name="fact_check_report.md",
        mime="text/markdown",
    )
    try:
        pdf_bytes = build_pdf_report_bytes(result)
        col2.download_button(
            label="Download PDF report",
            data=pdf_bytes,
            file_name="fact_check_report.pdf",
            mime="application/pdf",
        )
    except Exception as exc:  # pragma: no cover - optional dependency/runtime edge case
        col2.caption(f"PDF export unavailable: {type(exc).__name__}")

    st.markdown("### Copyable citations")
    st.code(build_citation_list(result), language="markdown")
    st.markdown(report)


def _display_evaluation_tab(project_root: Path) -> None:
    st.header("Evaluation Results")
    st.write(
        "This tab is designed for real benchmark outputs. It does not invent accuracy, F1, or leaderboard numbers. "
        "After running the benchmark scripts, result files will appear here."
    )

    leaderboard_path = project_root / "reports" / "evaluation_leaderboard.csv"
    metrics_path = project_root / "reports" / "benchmark_metrics.json"
    confusion_path = project_root / "reports" / "figures" / "confusion_matrix.png"

    if leaderboard_path.exists():
        st.subheader("Leaderboard")
        st.dataframe(pd.read_csv(leaderboard_path), use_container_width=True)
    else:
        st.subheader("Leaderboard template")
        st.dataframe(
            pd.DataFrame(
                [
                    {"System": "Keyword baseline", "Evidence source": "Web/Wikipedia/OpenAlex", "Model": "Rule-based", "Accuracy": "Run benchmark", "Macro F1": "Run benchmark"},
                    {"System": "Hybrid retrieval", "Evidence source": "Web/Wikipedia/OpenAlex", "Model": "Rule-based", "Accuracy": "Run benchmark", "Macro F1": "Run benchmark"},
                    {"System": "Modular pipeline", "Evidence source": "Planned live retrieval", "Model": "Evidence synthesis baseline", "Accuracy": "Run benchmark", "Macro F1": "Run benchmark"},
                ]
            ),
            use_container_width=True,
        )

    if metrics_path.exists():
        st.subheader("Benchmark metrics")
        st.json(metrics_path.read_text(encoding="utf-8"))
    else:
        st.info("No benchmark metrics file found yet. Run `python src/evaluate_realworld_benchmark.py --help` for options.")

    if confusion_path.exists():
        st.subheader("Confusion matrix")
        st.image(str(confusion_path), use_container_width=True)

    st.subheader("How to generate real results")
    st.code(
        "python src/evaluate_realworld_benchmark.py --input data/raw/benchmark_claims.csv --limit 50\n"
        "python src/error_analysis.py",
        language="bash",
    )
    st.caption("The benchmark script expects a CSV with columns: claim,label. It saves metrics to reports/ without fabricating results.")


def _display_architecture_tab(project_root: Path) -> None:
    st.header("Project Architecture")
    diagram_path = project_root / "reports" / "figures" / "system_architecture.png"
    if diagram_path.exists():
        st.image(str(diagram_path), use_container_width=True)
    else:
        st.code(
            "Claim → Query Planning → Live Evidence Retrieval → Source Ranking → Evidence Summarization → "
            "Evidence Synthesis → Freshness Guardrail → Verdict + Citations + Report"
        )

    st.subheader("What makes the system different")
    st.write(
        "The app is built as a modular fact-checking workflow rather than a single keyword search. "
        "It plans queries, retrieves public evidence, ranks source quality, summarizes evidence, synthesizes findings, "
        "performs a current-fact guardrail, and exports a transparent report."
    )

    st.subheader("Source quality signals")
    st.write(
        "Evidence cards include source type, trust signal, relevance score, and overall evidence score. "
        "These signals help users inspect why a source was ranked highly."
    )

    st.subheader("Limitations")
    st.write(
        "The system can still be wrong if search results are incomplete, sources are misleading, or the claim requires private, "
        "hyper-local, or very recent information. The output should support human review, not replace it."
    )


def _live_fact_check_tab() -> None:
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
        _display_result_summary(result)

        st.write("**Explanation:**", result["explanation"])
        st.caption(result["claim_profile"]["reason"])

        tabs = st.tabs(["Workflow", "Evidence", "Synthesis", "Freshness", "Report", "Technical details"])

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
            _display_freshness_guardrail(result)

        with tabs[4]:
            _display_report_tab(result)

        with tabs[5]:
            detail_rows = [
                {
                    "passage_id": item.get("passage_id"),
                    "title": item.get("title"),
                    "source": item.get("source"),
                    "source_category": item.get("source_category"),
                    "trust_label": item.get("trust_label"),
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


def main(project_root: Path | None = None) -> None:
    """Run the Streamlit app."""
    project_root = project_root or Path(__file__).resolve().parents[1]
    st.set_page_config(
        page_title="Evidence-Based Fact Verification",
        page_icon="🔎",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .result-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 1rem;
            margin: 0.75rem 0 1.25rem 0;
        }
        .result-card {
            border: 1px solid rgba(128, 128, 128, 0.35);
            border-radius: 0.75rem;
            padding: 1rem;
            min-height: 120px;
            background: rgba(128, 128, 128, 0.06);
        }
        .result-label {
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 0.55rem;
            opacity: 0.92;
        }
        .result-value {
            font-size: 1.65rem;
            line-height: 1.18;
            font-weight: 500;
            white-space: normal;
            overflow-wrap: anywhere;
            word-break: break-word;
        }
        div[data-testid="stMetricValue"] {
            white-space: normal !important;
            overflow-wrap: anywhere !important;
            word-break: break-word !important;
            line-height: 1.15 !important;
        }
        div[data-testid="stMetricLabel"] {
            white-space: normal !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Evidence-Based Fact Verification")
    st.write(
        "A modular fact-checking system that plans search queries, retrieves public evidence, ranks source quality, "
        "summarizes findings, handles time-sensitive claims, and generates citation-backed reports."
    )

    app_tabs = st.tabs(["Live Fact Check", "Evaluation Results", "Project Architecture"])
    with app_tabs[0]:
        _live_fact_check_tab()
    with app_tabs[1]:
        _display_evaluation_tab(project_root)
    with app_tabs[2]:
        _display_architecture_tab(project_root)

    st.divider()
    st.caption(
        "Evidence-based verification means the label depends on retrieved public sources. "
        "The app should be used as a citation-first research tool, not as a replacement for expert review."
    )
