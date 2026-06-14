"""Markdown report generation for the Streamlit app and CLI."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _safe(value: Any) -> str:
    return str(value).replace("\n", " ").strip()


def build_markdown_report(result: dict[str, Any]) -> str:
    """Build a downloadable fact-checking report from a pipeline result."""
    created = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    profile = result.get("claim_profile", {})
    plan = result.get("search_plan", {})
    synthesis = result.get("synthesis", {})
    evidence = result.get("top_evidence", [])

    lines = [
        "# Evidence-Based Fact Verification Report",
        "",
        f"**Generated:** {created}",
        f"**Claim:** {_safe(result.get('claim', ''))}",
        f"**Evidence label:** {_safe(result.get('predicted_label', ''))}",
        f"**Confidence signal:** {float(result.get('confidence', 0.0)):.2f}",
        f"**Claim type:** {_safe(profile.get('category', 'unknown'))}",
        f"**Evidence strength:** {_safe(synthesis.get('evidence_strength', 'unknown'))}",
        "",
        "## Explanation",
        _safe(result.get("explanation", "")),
        "",
        "## Search Plan",
        f"**Primary query:** {_safe(plan.get('primary_query', result.get('retrieval_query', '')))}",
        "",
    ]

    queries = plan.get("generated_queries", [])
    if queries:
        lines.append("**Generated queries:**")
        for query in queries:
            lines.append(f"- {_safe(query)}")
        lines.append("")

    notes = plan.get("strategy_notes", [])
    if notes:
        lines.append("**Strategy notes:**")
        for note in notes:
            lines.append(f"- {_safe(note)}")
        lines.append("")

    lines.extend(
        [
            "## Evidence Synthesis",
            _safe(synthesis.get("analysis", "")),
            "",
            "## Retrieved Evidence",
        ]
    )

    if not evidence:
        lines.append("No retrieved evidence was available.")
    for index, item in enumerate(evidence, start=1):
        title = _safe(item.get("title", "Untitled source"))
        url = _safe(item.get("url", ""))
        source = _safe(item.get("source", "Source"))
        summary = _safe(item.get("summary", item.get("text", "")))
        score = float(item.get("score", 0.0))
        trust = float(item.get("trust_score", 0.0))
        lines.extend(
            [
                "",
                f"### Evidence {index}: {title}",
                f"- Source: {source}",
                f"- URL: {url}",
                f"- Score: {score:.3f}",
                f"- Source trust signal: {trust:.2f}",
                f"- Summary: {summary}",
            ]
        )

    lines.extend(
        [
            "",
            "## Important Note",
            "This report is evidence-grounded, not a guarantee of truth. It should support human review and source inspection.",
        ]
    )
    return "\n".join(lines).strip() + "\n"
