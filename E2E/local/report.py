from __future__ import annotations

import html
import textwrap
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from E2E.shared.io_utils import write_text


DEFAULT_RECOMMENDATIONS = [
    "Apply and manually review every accepted patch before deployment.",
    "Pin a supported Solidity compiler version and run compilation in CI.",
    "Run Slither and project tests after every security-sensitive change.",
    "Use checks-effects-interactions and established OpenZeppelin guards for external calls.",
    "Repeat the audit after dependency, privilege, or business-logic changes.",
]


def _phase(remote: dict[str, Any], name: str) -> dict[str, Any]:
    return remote.get("phases", {}).get(name, {}) if remote else {}


def _match(items: list[dict[str, Any]], finding: dict[str, Any]) -> dict[str, Any]:
    for item in items:
        if str(item.get("finding_id") or "") == str(finding.get("finding_id") or ""):
            return item
        if item.get("swc") == finding.get("swc") and (
            not item.get("unit_id") or item.get("unit_id") == finding.get("unit_id")
        ):
            return item
    return {}


def normalize_recommendations(value: Any) -> list[str]:
    if not isinstance(value, list):
        return list(DEFAULT_RECOMMENDATIONS)
    recommendations = []
    for item in value:
        if isinstance(item, str) and item.strip():
            recommendations.append(item.strip())
            continue
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        description = str(
            item.get("description")
            or item.get("recommendation")
            or item.get("summary")
            or ""
        ).strip()
        if title and description:
            recommendations.append(f"{title}: {description}")
        elif title or description:
            recommendations.append(title or description)
    return recommendations or list(DEFAULT_RECOMMENDATIONS)


def apply_verification_to_report(
    report_data: dict[str, Any],
    verification: dict[str, Any],
) -> None:
    verdict = str(verification.get("overall_verdict") or "Not complete")
    report_data["patch_verdict"] = verdict
    for row in report_data.get("phase_rows", []):
        if row.get("phase") == "5 Verifier":
            row["status"] = verification.get("status", "partial")

    if not verdict.startswith(("Patch rejected", "Patch inconclusive")):
        return

    report_data["model_executive_summary"] = report_data.get("executive_summary", "")
    finding_count = len(report_data.get("findings", []))
    behavioral = verification.get("behavioral_verification", {})
    regression_count = len(behavioral.get("findings", []))
    detail = (
        f" Deterministic behavioral checks found {regression_count} regression(s)."
        if regression_count
        else ""
    )
    outcome = "rejected" if verdict.startswith("Patch rejected") else "not accepted"
    report_data["executive_summary"] = (
        f"The audit identified {finding_count} finding(s). A CodeLlama patch was generated but {outcome} "
        f"by deterministic verification: {verdict}.{detail} The patch must not be deployed; "
        "the original finding remains open until a behavior-preserving patch passes redetection, "
        "compilation, Slither, and project tests."
    )
    for finding in report_data.get("findings", []):
        finding["verification_status"] = (
            "Rejected by deterministic verifier"
            if verdict.startswith("Patch rejected")
            else "Inconclusive due to detector disagreement"
        )
        finding["residual_risk"] = verdict
    rejection_recommendation = (
        "Do not deploy the generated patch; preserve the contract's intended transfers and behavior "
        "while addressing the root cause."
    )
    recommendations = report_data.get("recommendations", [])
    if rejection_recommendation not in recommendations:
        report_data["recommendations"] = [rejection_recommendation, *recommendations]


def build_report_data(
    request: dict[str, Any],
    fused: dict[str, Any],
    remote: dict[str, Any] | None,
    tool_verification: dict[str, Any],
) -> dict[str, Any]:
    remote = remote or {}
    phases = remote.get("phases", {})
    advisor_items = _phase(remote, "advisor").get("suggestions", [])
    assessor_items = _phase(remote, "assessor").get("assessments", [])
    verifier_items = _phase(remote, "verifier").get("verifications", [])
    fixer = _phase(remote, "fixer")
    reporter = _phase(remote, "reporter")
    findings = []

    for finding in fused.get("findings", []):
        advice = _match(advisor_items, finding)
        assessment = _match(assessor_items, finding)
        verification = _match(verifier_items, finding)
        findings.append(
            {
                **finding,
                "root_cause": advice.get("root_cause") or finding.get("description") or "",
                "impact": assessment.get("impact") or advice.get("impact") or "",
                "cvss_score": assessment.get("cvss_score"),
                "repair_priority": assessment.get("repair_priority"),
                "repair_steps": advice.get("repair_steps")
                or ([finding["remediation"]] if finding.get("remediation") else []),
                "verification_status": verification.get("status") or "Not verified by generative model",
                "residual_risk": verification.get("residual_risk") or "",
            }
        )

    distribution = Counter(item.get("severity", "Unknown") for item in findings)
    original_code = request["source"]["code"]
    fixed_code = fixer.get("fixed_code") or original_code
    compile_status = tool_verification.get("compile", {}).get("status", "not run")
    slither = tool_verification.get("slither", {})
    slither_count = len(slither.get("findings", []))
    function_names = ", ".join(unit["unit_id"] for unit in request.get("code_units", [])[:8])
    if len(request.get("code_units", [])) > 8:
        function_names += ", ..."
    executive_summary = reporter.get("executive_summary")
    if not executive_summary:
        executive_summary = (
            f"The audit result is {fused.get('verdict', 'Inconclusive')} with a fused confidence "
            f"score of {float(fused.get('score') or 0.0):.3f}. The pipeline identified "
            f"{len(findings)} confirmed finding(s). Compilation status: {compile_status}; "
            f"Slither reported {slither_count} detector finding(s)."
        )
    recommendations = normalize_recommendations(reporter.get("recommendations"))
    phase_rows = [
        {
            "phase": "1A Static detector",
            "location": "Local",
            "status": request["local_phase1"]["static"].get("status", "unknown"),
            "engine": request["local_phase1"]["static"].get("engine", ""),
        },
        {
            "phase": "1B RAG detector",
            "location": "Local",
            "status": request["local_phase1"]["rag"].get("status", "unknown"),
            "engine": request["local_phase1"]["rag"].get("engine", ""),
        },
        {
            "phase": "1C CodeLlama detector",
            "location": "Kaggle",
            "status": _phase(remote, "detector").get("status", "pending"),
            "engine": remote.get("models", {}).get("detector", ""),
        },
        {
            "phase": "2 Advisor",
            "location": "Kaggle",
            "status": _phase(remote, "advisor").get("status", "pending"),
            "engine": "CodeLlama Instruct",
        },
        {
            "phase": "3 Risk assessor",
            "location": "Kaggle",
            "status": _phase(remote, "assessor").get("status", "pending"),
            "engine": "CodeLlama Instruct",
        },
        {
            "phase": "4 Fixer",
            "location": "Kaggle",
            "status": fixer.get("status", "pending"),
            "engine": "CodeLlama Instruct + deterministic patch apply",
        },
        {
            "phase": "5 Verifier",
            "location": "Local + Kaggle",
            "status": _phase(remote, "verifier").get("status", tool_verification.get("status", "pending")),
            "engine": "redetection, compiler, Slither, CodeLlama review",
        },
        {
            "phase": "6 Reporter",
            "location": "Local + Kaggle",
            "status": reporter.get("status", "local fallback"),
            "engine": "deterministic renderer + CodeLlama narrative",
        },
    ]
    return {
        "title": "Smart Contract Audit Report",
        "run_id": request["run_id"],
        "audit_time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "file_name": request["source"]["file_name"],
        "source_sha256": request["source"]["sha256"],
        "analyzed_object": function_names or request["source"]["file_name"],
        "detection_result": fused.get("verdict", "Inconclusive"),
        "analysis_status": fused.get("status", "inconclusive"),
        "score": float(fused.get("score") or 0.0),
        "risk_level": fused.get("risk_level", "None"),
        "executive_summary": executive_summary,
        "findings": findings,
        "distribution": distribution,
        "phase_rows": phase_rows,
        "original_code": original_code,
        "fixed_code": fixed_code,
        "patch_outcomes": fixer.get("patch_outcomes", []),
        "patch_verdict": _phase(remote, "verifier").get("overall_verdict", "Not complete"),
        "tool_verification": tool_verification,
        "recommendations": recommendations,
        "limitations": [
            "SWC Registry source material is no longer actively maintained and is not exhaustive.",
            "The fine-tuned classifier is a binary detector; SWC attribution comes from local static/RAG evidence.",
            "A model-generated patch is not deployment approval. Manual review and project tests remain required.",
        ],
        "disclaimer": (
            "This report represents a best-effort automated security review. It cannot guarantee that all "
            "vulnerabilities, economic attacks, integration risks, or business-logic defects were identified."
        ),
    }


def render_markdown(data: dict[str, Any]) -> str:
    findings = data["findings"]
    dist = data["distribution"]
    lines = [
        "# Smart Contract Audit Report",
        "",
        "## 1. Contract Information",
        "",
        f"- **Run ID:** `{data['run_id']}`",
        f"- **Analyzed Object:** {data['analyzed_object']}",
        f"- **Source File:** `{data['file_name']}`",
        f"- **Source SHA-256:** `{data['source_sha256']}`",
        f"- **Detection Result:** **{data['detection_result']}**",
        f"- **Analysis Status:** {data['analysis_status']}",
        f"- **Fused Score:** {data['score']:.3f}",
        f"- **Audit Time:** {data['audit_time']}",
        "",
        "## 2. Executive Summary",
        "",
        data["executive_summary"],
        "",
        "## 3. Methodology",
        "",
        (
            "The audit combines restored context-aware static rules, local SWC TF-IDF retrieval, the "
            "fine-tuned CodeLlama sequence classifier, CodeLlama Instruct repair agents, compiler checks, "
            "Slither, fixed-code redetection, and deterministic report generation."
        ),
        "",
        "| Phase | Location | Status | Engine |",
        "| :---- | :------- | :----- | :----- |",
    ]
    lines.extend(
        f"| {row['phase']} | {row['location']} | {row['status']} | {row['engine']} |"
        for row in data["phase_rows"]
    )
    lines.extend(
        [
            "",
            "## 4. Findings",
            "",
            "### 4.1 Vulnerability Statistics",
            "",
            f"- **Detection Result:** {data['detection_result']}",
            f"- **Vulnerability Count:** {len(findings)} confirmed finding(s)",
            "",
            "### 4.2 Vulnerability Severity Distribution",
            "",
            "| Critical | High | Medium | Low |",
            "| :------: | :--: | :----: | :-: |",
            f"| {dist.get('Critical', 0)} | {dist.get('High', 0)} | {dist.get('Medium', 0)} | {dist.get('Low', 0)} |",
            "",
            "### 4.3 Vulnerability Reference Table",
            "",
            "| ID | Vulnerability | Unit | Severity | Confidence | Evidence |",
            "| :-- | :------------ | :--- | :------- | :--------: | :------- |",
        ]
    )
    for finding in findings:
        lines.append(
            f"| {finding['swc']} | {finding['title']} | {finding.get('unit_id') or 'contract'} | "
            f"{finding['severity']} | {float(finding.get('confidence') or 0):.3f} | "
            f"{', '.join(finding.get('sources', []))} |"
        )
    lines.extend(
        [
            "",
            "## 5. Detailed Analysis",
            "",
            "### 5.1 Contract Name",
            "",
            data["analyzed_object"],
            "",
            "### 5.2 Source Code",
            "",
            "```solidity",
            data["original_code"],
            "```",
            "",
            "### 5.3 Repair Suggestion",
            "",
        ]
    )
    if not findings:
        lines.append("No confirmed vulnerability was available for automated remediation.")
    for index, finding in enumerate(findings, 1):
        lines.extend(
            [
                f"#### {index}. {finding['title']} ({finding['swc']}) - {finding['severity']}",
                "",
                f"- **Unit:** `{finding.get('unit_id') or 'contract'}`",
                f"- **Root cause:** {finding.get('root_cause') or 'Not provided'}",
                f"- **Impact:** {finding.get('impact') or 'Requires manual impact analysis'}",
            ]
        )
        if finding.get("cvss_score") is not None:
            lines.append(f"- **CVSS:** {finding['cvss_score']}")
        lines.append("- **Repair steps:**")
        steps = finding.get("repair_steps") or ["Perform manual review and implement the cited remediation."]
        lines.extend(f"  {number}. {step}" for number, step in enumerate(steps, 1))
        lines.append(
            f"- **Patch verification:** {finding.get('verification_status')}"
            + (f"; residual risk: {finding['residual_risk']}" if finding.get("residual_risk") else "")
        )
        lines.append("")
    lines.extend(
        [
            "### 5.4 Fixed Code",
            "",
            f"_Patch verification verdict: **{data['patch_verdict']}**_",
            "",
            "```solidity",
            data["fixed_code"],
            "```",
            "",
            "### 5.5 Tool Verification",
            "",
            f"- **Compilation:** {data['tool_verification'].get('compile', {}).get('status', 'not run')}",
            f"- **Slither:** {data['tool_verification'].get('slither', {}).get('status', 'not run')}",
            f"- **Slither findings:** {len(data['tool_verification'].get('slither', {}).get('findings', []))}",
            f"- **Behavior preservation:** {data['tool_verification'].get('behavioral', {}).get('status', 'not run')}",
            "",
        ]
    )
    for finding in data["tool_verification"].get("behavioral", {}).get("findings", []):
        lines.append(f"- **Behavioral evidence:** {finding.get('evidence') or finding.get('check')}")
    lines.extend(["", "## 6. Summary and Recommendations", ""])
    lines.extend(f"- {item}" for item in data["recommendations"])
    lines.extend(["", "### Limitations", ""])
    lines.extend(f"- {item}" for item in data["limitations"])
    lines.extend(["", "## 7. Disclaimer", "", data["disclaimer"], ""])
    return "\n".join(lines)


def render_html(data: dict[str, Any], markdown_text: str) -> str:
    behavioral = data["tool_verification"].get("behavioral", {})
    behavioral_evidence = "".join(
        f"<li>{html.escape(str(item.get('evidence') or item.get('check') or ''))}</li>"
        for item in behavioral.get("findings", [])
    )
    findings_rows = "".join(
        "<tr>"
        f"<td>{html.escape(item['swc'])}</td>"
        f"<td>{html.escape(str(item['title']))}</td>"
        f"<td>{html.escape(str(item.get('unit_id') or 'contract'))}</td>"
        f"<td><span class='severity {item['severity'].lower()}'>{html.escape(item['severity'])}</span></td>"
        f"<td>{float(item.get('confidence') or 0):.3f}</td>"
        f"<td>{html.escape(', '.join(item.get('sources', [])))}</td>"
        "</tr>"
        for item in data["findings"]
    ) or "<tr><td colspan='6'>No confirmed findings</td></tr>"
    phase_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['phase'])}</td><td>{row['location']}</td>"
        f"<td>{html.escape(str(row['status']))}</td><td>{html.escape(str(row['engine']))}</td>"
        "</tr>"
        for row in data["phase_rows"]
    )
    details = ""
    for index, item in enumerate(data["findings"], 1):
        steps = "".join(f"<li>{html.escape(str(step))}</li>" for step in item.get("repair_steps", []))
        details += (
            f"<section><h4>{index}. {html.escape(item['title'])} ({item['swc']})</h4>"
            f"<p><b>Severity:</b> {item['severity']} &nbsp; <b>Unit:</b> "
            f"<code>{html.escape(str(item.get('unit_id') or 'contract'))}</code></p>"
            f"<p><b>Root cause:</b> {html.escape(str(item.get('root_cause') or 'Not provided'))}</p>"
            f"<p><b>Impact:</b> {html.escape(str(item.get('impact') or 'Requires manual analysis'))}</p>"
            f"<ol>{steps}</ol><p><b>Verification:</b> "
            f"{html.escape(str(item.get('verification_status') or 'Not verified'))}</p></section>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{data['title']}</title>
<style>
@page {{ size: A4; margin: 18mm; }}
body {{ font-family: Arial, sans-serif; color: #18202a; margin: 0 auto; max-width: 1050px; line-height: 1.48; }}
header {{ border-bottom: 4px solid #ba2d2d; padding: 30px 0 18px; margin-bottom: 26px; }}
h1 {{ font-size: 30px; margin: 0; }} h2 {{ color: #7b1f1f; border-bottom: 1px solid #d6dbe1; padding-bottom: 5px; }}
h3 {{ color: #26394d; }} h4 {{ font-size: 16px; }}
.meta {{ display: grid; grid-template-columns: 180px 1fr; gap: 5px 14px; }}
.verdict {{ font-weight: bold; color: #9e1d1d; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 22px; font-size: 13px; }}
th {{ background: #26394d; color: white; text-align: left; }} th, td {{ border: 1px solid #bcc5cf; padding: 7px; vertical-align: top; }}
.severity {{ font-weight: bold; }} .critical {{ color: #9e1d1d; }} .high {{ color: #bd4c13; }}
.medium {{ color: #8a6500; }} .low {{ color: #28633b; }}
pre {{ background: #f3f5f7; border-left: 4px solid #52687f; padding: 12px; overflow-wrap: anywhere; white-space: pre-wrap; font-size: 11px; }}
.notice {{ border-left: 4px solid #a17600; background: #fff8df; padding: 10px 14px; }}
footer {{ border-top: 1px solid #bcc5cf; margin-top: 30px; padding: 12px 0; color: #66717d; font-size: 12px; }}
</style></head><body>
<header><h1>Smart Contract Audit Report</h1><p>Run <code>{html.escape(data['run_id'])}</code></p></header>
<h2>1. Contract Information</h2><div class="meta">
<b>Analyzed Object</b><span>{html.escape(data['analyzed_object'])}</span>
<b>Source File</b><span>{html.escape(data['file_name'])}</span>
<b>Source SHA-256</b><code>{data['source_sha256']}</code>
<b>Detection Result</b><span class="verdict">{data['detection_result']}</span>
<b>Analysis Status</b><span>{data['analysis_status']}</span>
<b>Fused Score</b><span>{data['score']:.3f}</span>
<b>Audit Time</b><span>{data['audit_time']}</span></div>
<h2>2. Executive Summary</h2><p>{html.escape(data['executive_summary'])}</p>
<h2>3. Methodology</h2><p>The audit combines static analysis, local RAG, the fine-tuned CodeLlama detector,
CodeLlama Instruct repair agents, compilation, Slither, patch redetection, and deterministic reporting.</p>
<table><thead><tr><th>Phase</th><th>Location</th><th>Status</th><th>Engine</th></tr></thead><tbody>{phase_rows}</tbody></table>
<h2>4. Findings</h2><h3>4.1 Vulnerability Statistics</h3>
<p><b>{len(data['findings'])}</b> confirmed finding(s); highest risk: <b>{data['risk_level']}</b>.</p>
<h3>4.2 Vulnerability Severity Distribution</h3>
<table><tr><th>Critical</th><th>High</th><th>Medium</th><th>Low</th></tr><tr>
<td>{data['distribution'].get('Critical', 0)}</td><td>{data['distribution'].get('High', 0)}</td>
<td>{data['distribution'].get('Medium', 0)}</td><td>{data['distribution'].get('Low', 0)}</td></tr></table>
<h3>4.3 Vulnerability Reference Table</h3>
<table><thead><tr><th>ID</th><th>Vulnerability</th><th>Unit</th><th>Severity</th><th>Confidence</th><th>Evidence</th></tr></thead>
<tbody>{findings_rows}</tbody></table>
<h2>5. Detailed Analysis</h2><h3>5.1 Contract Name</h3><p>{html.escape(data['analyzed_object'])}</p>
<h3>5.2 Source Code</h3><pre>{html.escape(data['original_code'])}</pre>
<h3>5.3 Repair Suggestion</h3>{details or '<p>No confirmed vulnerability was available for remediation.</p>'}
<h3>5.4 Fixed Code</h3><p>Patch verification: <b>{html.escape(data['patch_verdict'])}</b></p>
<pre>{html.escape(data['fixed_code'])}</pre>
	<h3>5.5 Tool Verification</h3><p>Compilation: <b>{data['tool_verification'].get('compile', {}).get('status', 'not run')}</b>;
	Slither: <b>{data['tool_verification'].get('slither', {}).get('status', 'not run')}</b>;
	Behavior preservation: <b>{behavioral.get('status', 'not run')}</b>.</p>
	{f'<ul>{behavioral_evidence}</ul>' if behavioral_evidence else ''}
<h2>6. Summary and Recommendations</h2><ul>{''.join(f'<li>{html.escape(str(item))}</li>' for item in data['recommendations'])}</ul>
<div class="notice"><b>Limitations</b><ul>{''.join(f'<li>{html.escape(str(item))}</li>' for item in data['limitations'])}</ul></div>
<h2>7. Disclaimer</h2><p>{html.escape(data['disclaimer'])}</p>
<footer>Generated by the self-contained CodeLlama E2E audit pipeline.</footer>
</body></html>"""


def render_pdf(data: dict[str, Any], output_path: Path) -> tuple[bool, str]:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            PageBreak,
            Paragraph,
            Preformatted,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError:
        return False, "reportlab is not installed"

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], textColor=colors.HexColor("#7b1f1f")))
    styles.add(ParagraphStyle(name="CodeSmall", parent=styles["Code"], fontSize=5.8, leading=7.2))
    story: list[Any] = [Paragraph(data["title"], styles["ReportTitle"]), Spacer(1, 6 * mm)]

    def heading(text: str, level: int = 1) -> None:
        story.append(Paragraph(html.escape(text), styles[f"Heading{min(level, 3)}"]))

    def paragraph(text: Any) -> None:
        story.append(Paragraph(html.escape(str(text or "")), styles["BodyText"]))
        story.append(Spacer(1, 2 * mm))

    heading("1. Contract Information")
    metadata = [
        ["Analyzed Object", data["analyzed_object"]],
        ["Source File", data["file_name"]],
        ["Source SHA-256", data["source_sha256"]],
        ["Detection Result", data["detection_result"]],
        ["Analysis Status", data["analysis_status"]],
        ["Fused Score", f"{data['score']:.3f}"],
        ["Audit Time", data["audit_time"]],
    ]
    table = Table(metadata, colWidths=[40 * mm, 130 * mm])
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.extend([table, Spacer(1, 4 * mm)])
    heading("2. Executive Summary")
    paragraph(data["executive_summary"])
    heading("3. Methodology")
    paragraph(
        "Static analysis, local SWC retrieval, the fine-tuned CodeLlama detector, CodeLlama Instruct "
        "repair agents, compilation, Slither, patch redetection, and deterministic reporting."
    )
    phase_table = [["Phase", "Location", "Status"], *[[r["phase"], r["location"], r["status"]] for r in data["phase_rows"]]]
    table = Table(phase_table, repeatRows=1, colWidths=[92 * mm, 35 * mm, 43 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#26394d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.extend([table, Spacer(1, 4 * mm)])
    heading("4. Findings")
    heading("4.1 Vulnerability Statistics", 2)
    paragraph(f"{len(data['findings'])} confirmed finding(s); highest risk: {data['risk_level']}.")
    heading("4.2 Vulnerability Severity Distribution", 2)
    dist = data["distribution"]
    severity_table = [
        ["Critical", "High", "Medium", "Low"],
        [dist.get("Critical", 0), dist.get("High", 0), dist.get("Medium", 0), dist.get("Low", 0)],
    ]
    table = Table(severity_table, colWidths=[42.5 * mm] * 4)
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story.extend([table, Spacer(1, 4 * mm)])
    heading("4.3 Vulnerability Reference Table", 2)
    rows = [["ID", "Vulnerability", "Unit", "Severity"]]
    rows.extend(
        [item["swc"], item["title"], item.get("unit_id") or "contract", item["severity"]]
        for item in data["findings"]
    )
    if len(rows) == 1:
        rows.append(["-", "No confirmed findings", "-", "-"])
    table = Table(rows, repeatRows=1, colWidths=[22 * mm, 70 * mm, 53 * mm, 25 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#26394d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.extend([table, PageBreak()])
    heading("5. Detailed Analysis")
    heading("5.1 Contract Name", 2)
    paragraph(data["analyzed_object"])
    heading("5.2 Source Code", 2)
    wrapped_source = "\n".join(textwrap.fill(line, width=118, replace_whitespace=False) for line in data["original_code"].splitlines())
    story.extend([Preformatted(wrapped_source, styles["CodeSmall"]), Spacer(1, 3 * mm)])
    heading("5.3 Repair Suggestion", 2)
    if not data["findings"]:
        paragraph("No confirmed vulnerability was available for automated remediation.")
    for index, item in enumerate(data["findings"], 1):
        heading(f"{index}. {item['title']} ({item['swc']}) - {item['severity']}", 3)
        paragraph(f"Root cause: {item.get('root_cause') or 'Not provided'}")
        paragraph(f"Impact: {item.get('impact') or 'Requires manual analysis'}")
        for step_index, step in enumerate(item.get("repair_steps", []), 1):
            paragraph(f"{step_index}. {step}")
        paragraph(f"Patch verification: {item.get('verification_status') or 'Not verified'}")
    heading("5.4 Fixed Code", 2)
    paragraph(f"Patch verification verdict: {data['patch_verdict']}")
    wrapped_fixed = "\n".join(textwrap.fill(line, width=118, replace_whitespace=False) for line in data["fixed_code"].splitlines())
    story.extend([Preformatted(wrapped_fixed, styles["CodeSmall"]), Spacer(1, 3 * mm)])
    heading("5.5 Tool Verification", 2)
    paragraph(
        f"Compilation: {data['tool_verification'].get('compile', {}).get('status', 'not run')}; "
        f"Slither: {data['tool_verification'].get('slither', {}).get('status', 'not run')}; "
        f"Behavior preservation: "
        f"{data['tool_verification'].get('behavioral', {}).get('status', 'not run')}."
    )
    for finding in data["tool_verification"].get("behavioral", {}).get("findings", []):
        paragraph(f"- {finding.get('evidence') or finding.get('check')}")
    heading("6. Summary and Recommendations")
    for item in data["recommendations"]:
        paragraph(f"- {item}")
    heading("Limitations", 2)
    for item in data["limitations"]:
        paragraph(f"- {item}")
    heading("7. Disclaimer")
    paragraph(data["disclaimer"])

    def footer(canvas, document) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(20 * mm, 10 * mm, f"Run {data['run_id']}")
        canvas.drawRightString(190 * mm, 10 * mm, f"Page {document.page}")
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return True, ""


def write_reports(data: dict[str, Any], run_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(run_dir)
    markdown_text = render_markdown(data)
    markdown_path = write_text(output_dir / "phase6_audit_report.md", markdown_text)
    html_path = write_text(output_dir / "phase6_audit_report.html", render_html(data, markdown_text))
    pdf_path = output_dir / "phase6_audit_report.pdf"
    pdf_ok, pdf_error = render_pdf(data, pdf_path)
    return {
        "markdown": str(markdown_path),
        "html": str(html_path),
        "pdf": str(pdf_path) if pdf_ok else None,
        "pdf_error": pdf_error or None,
    }
