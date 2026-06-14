"""Report generation for the Streamlit app and CLI."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from textwrap import wrap
from typing import Any


def _safe(value: Any) -> str:
    return str(value).replace("\n", " ").strip()


def build_markdown_report(result: dict[str, Any]) -> str:
    """Build a downloadable Markdown fact-checking report from a pipeline result."""
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
        f"**Freshness check:** {'Enabled' if profile.get('needs_current_source') else 'Not triggered'}",
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
        category = _safe(item.get("source_category", "Unknown source type"))
        trust_display = _safe(item.get("trust_label", "Unknown trust signal"))
        strength_display = _safe(item.get("evidence_strength_label", "Evidence candidate"))
        lines.extend(
            [
                "",
                f"### Evidence {index}: {title}",
                f"- Source: {source}",
                f"- URL: {url}",
                f"- Source category: {category}",
                f"- Trust signal: {trust_display} ({trust:.2f})",
                f"- Evidence strength: {strength_display}",
                f"- Score: {score:.3f}",
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


def build_citation_list(result: dict[str, Any]) -> str:
    """Build a compact copyable citation list."""
    evidence = result.get("top_evidence", [])
    if not evidence:
        return "No citations available."
    lines = []
    seen = set()
    for item in evidence:
        url = _safe(item.get("url", ""))
        title = _safe(item.get("title", "Untitled source"))
        if not url or url in seen:
            continue
        seen.add(url)
        lines.append(f"- {title}: {url}")
    return "\n".join(lines) if lines else "No citations available."


def build_pdf_report_bytes(result: dict[str, Any]) -> bytes:
    """Build a simple PDF report.

    ReportLab is listed in requirements.txt. If it is unavailable, this function
    raises ImportError so the Streamlit app can fall back gracefully.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    markdown = build_markdown_report(result)
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    left = 54
    y = height - 54
    line_height = 12

    pdf.setTitle("Evidence-Based Fact Verification Report")
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(left, y, "Evidence-Based Fact Verification Report")
    y -= 24
    pdf.setFont("Helvetica", 9)

    for raw_line in markdown.splitlines():
        line = raw_line.replace("#", "").replace("**", "").strip()
        if not line:
            y -= line_height
            continue
        for wrapped in wrap(line, width=94):
            if y < 54:
                pdf.showPage()
                pdf.setFont("Helvetica", 9)
                y = height - 54
            pdf.drawString(left, y, wrapped)
            y -= line_height

    pdf.save()
    return buffer.getvalue()
