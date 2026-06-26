"""
reporter.py
──────────────────────────────────────────────────────────────────────────────
Report Generation Agent (Reporter) — agent #6, the final agent of the
LLM-BSCVM framework.

Integrates the outputs of every preceding agent into one complete smart-contract
audit report (paper §III-B Reporter, Fig. 2). The paper specifies seven key
sections; this implementation renders exactly the structure of the project's
reference audit report:

    1. Contract Information   (analyzed object, function overview, verdict, time)
    2. Executive Summary
    3. Methodology
    4. Findings               (4.1 statistics, 4.2 severity distribution,
                               4.3 vulnerability reference table)
    5. Detailed Analysis      (5.1 contract name, 5.2 source code,
                               5.3 repair suggestion, 5.4 fixed code)
    6. Summary and Recommendations
    7. Disclaimer

The narrative sections (function overview, executive summary, recommendations)
are produced by a single structured LLM call — the paper uses a more capable
model for this final audit step — and everything else is composed
deterministically from the upstream agent outputs, so the report renders even
when the LLM is unavailable.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

try:
    from .llm_client import build_llm_client
except ImportError:  # allow running as a script
    from llm_client import build_llm_client

RISK_LEVELS = ("Critical", "High", "Medium", "Low")
RISK_ORDER = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1, "None": 0, "Unknown": 0}

# ──────────────────────────────────────────────────────────────────────────────
# Section 4.3 — canonical vulnerability reference table (fixed catalogue).
# Mirrors the project's reference audit report; `keys` map detected SWC ids /
# static-analyzer vuln_types onto a row so found classes can be highlighted.
# ──────────────────────────────────────────────────────────────────────────────
VULNERABILITY_REFERENCE: List[Dict[str, Any]] = [
    {"name": "Reentrancy", "abbr": "RE", "severity": "Critical",
     "impact": "May lead to theft of funds, contract failure, or complete control of contract permissions.",
     "keys": ["SWC-107", "reentrancy"]},
    {"name": "Access Control Missing", "abbr": "AC", "severity": "Critical",
     "impact": "Contract permissions may be maliciously controlled, potentially leading to owner replacement and fund theft.",
     "keys": ["SWC-105", "SWC-106", "SWC-115", "SWC-100", "access_control", "unprotected", "tx_origin", "default_visibility"]},
    {"name": "Unchecked Low-level Call", "abbr": "ULC", "severity": "Critical",
     "impact": "Low-level calls cannot catch exceptions, potentially leading to failed contract calls or misoperations.",
     "keys": ["SWC-104", "unchecked_low_level_calls", "unchecked"]},
    {"name": "Integer Overflow/Underflow", "abbr": "IOU", "severity": "High",
     "impact": "May result in fund loss, severely impact core contract functionality, and cause calculation errors.",
     "keys": ["SWC-101", "integer_overflow_underflow", "overflow"]},
    {"name": "Denial of Service - DoS", "abbr": "DoS", "severity": "High",
     "impact": "Attackers can prevent normal contract operation through gas consumption or other resource exhaustion.",
     "keys": ["SWC-113", "dos_revert_griefing", "dos"]},
    {"name": "Flash Loan Vulnerability", "abbr": "FLV", "severity": "High",
     "impact": "Malicious users can manipulate market prices or contract states through flash loans, potentially leading to fund loss.",
     "keys": ["SWC-FLASHLOAN", "flash_loan_vulnerability", "flash"]},
    {"name": "Front Running", "abbr": "FR", "severity": "High",
     "impact": "Attackers can manipulate transaction order to execute certain transactions first, leading to profit loss.",
     "keys": ["SWC-114", "front_running"]},
    {"name": "Timestamp Dependence", "abbr": "TD", "severity": "Medium",
     "impact": "Contract behavior depends on block timestamps which can be manipulated by attackers.",
     "keys": ["SWC-116", "timestamp_dependence", "timestamp"]},
    {"name": "Block Info Dependence", "abbr": "BI", "severity": "Medium",
     "impact": "Contract relies on block information that can be manipulated by miners or predicted by attackers.",
     "keys": ["SWC-120", "block_info_dependence", "blockhash"]},
    {"name": "DoS with Gas Limit", "abbr": "DosGL", "severity": "Medium",
     "impact": "Gas limits during execution may cause contract suspension and prevent normal operation.",
     "keys": ["dos_gas_limit", "gas_limit"]},
    {"name": "Unsafe Type Casting", "abbr": "UR", "severity": "Medium",
     "impact": "Type casting errors can lead to arithmetic overflow and contract logic errors.",
     "keys": ["SWC-101-cast", "unsafe_type_casting", "casting"]},
    {"name": "Transaction Order Dependence", "abbr": "TOD", "severity": "Medium",
     "impact": "Attackers can manipulate transaction order affecting contract execution logic.",
     "keys": ["SWC-114-tod", "transaction_order", "tod"]},
    {"name": "Outdated Compiler Version", "abbr": "OCV", "severity": "Low",
     "impact": "Using outdated compiler versions may expose contract to known vulnerabilities and incompatibilities.",
     "keys": ["SWC-102", "SWC-103", "outdated_compiler", "pragma"]},
    {"name": "Naming Convention", "abbr": "NC", "severity": "Low",
     "impact": "Non-standard naming conventions may lead to poor code readability and maintenance difficulties.",
     "keys": ["naming_convention", "naming"]},
    {"name": "Redundant Code", "abbr": "RC", "severity": "Low",
     "impact": "Redundant code may increase gas costs and make contract more complex to maintain.",
     "keys": ["redundant_code", "redundant"]},
]

METHODOLOGY_TEXT = (
    "Our audit process combined four complementary techniques orchestrated by a "
    "multi-agent pipeline: (1) static analysis against a predefined vulnerability "
    "pattern library, (2) retrieval-augmented generation (RAG) over a smart-contract "
    "corpus and vulnerability knowledge base, (3) inference analysis by a fine-tuned "
    "detection model, and (4) generative repair with independent patch verification. "
    "Detection results from the three analysis dimensions are combined via calibrated "
    "weighted fusion; detected vulnerabilities are then analysed, risk-rated (CVSS + a "
    "four-level system), repaired, and the patch is re-verified before reporting."
)

DISCLAIMER_TEXT = (
    "This audit report represents our best effort in identifying potential security "
    "vulnerabilities using automated LLM-based analysis. However, we cannot guarantee "
    "that all possible vulnerabilities have been identified. It does not constitute "
    "financial or legal advice, and the contract should undergo independent human "
    "review before deployment."
)

DEFAULT_RECOMMENDATIONS = [
    "Regularly conduct security audits to identify vulnerabilities and ensure safety.",
    "Continuously update the code based on industry best practices for long-term security.",
    "Use the proxy pattern for upgradability and continuous monitoring.",
    "Implement error handling and fail-safes to manage failures safely.",
    "Minimize external calls and validate inputs to prevent reentrancy attacks.",
]

# ── narrative LLM schema ──────────────────────────────────────────────────────
REPORTER_SYSTEM_INSTRUCTION = (
    "You are a senior smart-contract security auditor writing the narrative sections "
    "of a formal audit report. Given a contract and the structured results of an "
    "automated detection / repair / verification pipeline, write concise, professional, "
    "factual prose. Do not invent vulnerabilities beyond those provided. Return ONLY "
    "valid JSON matching the requested schema."
)

NARRATIVE_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "analyzed_object": {"type": "string"},
        "contract_function": {"type": "string"},
        "executive_summary": {"type": "string"},
        "recommendations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["analyzed_object", "contract_function", "executive_summary", "recommendations"],
    "propertyOrdering": ["analyzed_object", "contract_function", "executive_summary", "recommendations"],
}


@dataclass
class ReportFinding:
    swc: str
    name: str
    severity: str
    risk_level: str
    cvss_score: float | None
    repair_priority: int | None
    root_cause: str
    impact: str
    repair_steps: List[str]
    fixed_snippet: str
    verification_status: str
    residual_risk: str


@dataclass
class AuditReport:
    agent: str = "Reporter (Report Generation)"
    llm_model: str = ""
    analyzed_object: str = ""
    contract_function: str = ""
    detection_result: str = "Safe"
    audit_time: str = ""
    is_vulnerable: bool = False
    executive_summary: str = ""
    methodology: str = METHODOLOGY_TEXT
    vulnerability_count: int = 0
    severity_distribution: Dict[str, int] = field(default_factory=lambda: {lvl: 0 for lvl in RISK_LEVELS})
    reference_table: List[Dict[str, Any]] = field(default_factory=list)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    original_code: str = ""
    fixed_code: str = ""
    patch_verdict: str = ""
    recommendations: List[str] = field(default_factory=list)
    disclaimer: str = DISCLAIMER_TEXT
    markdown: str = ""
    error: str | None = None


# ──────────────────────────────────────────────────────────────────────────────
# Input adaptation helpers (accept bare agent dicts or phaseN wrappers)
# ──────────────────────────────────────────────────────────────────────────────
def _unwrap(d: Dict[str, Any] | None, *keys: str) -> Dict[str, Any]:
    if not isinstance(d, dict):
        return {}
    for key in keys:
        if isinstance(d.get(key), dict):
            return d[key]
    return d


def _suggestions_of(repair: Dict[str, Any]) -> List[Dict[str, Any]]:
    repair = _unwrap(repair, "repair")
    items = repair.get("suggestions")
    return [s for s in items if isinstance(s, dict)] if isinstance(items, list) else []


def _assessments_of(risk: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    risk = _unwrap(risk, "risk_assessment")
    items = risk.get("assessments")
    return [a for a in items if isinstance(a, dict)] if isinstance(items, list) else []


def _verifications_of(verification: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    verification = _unwrap(verification, "verification")
    items = verification.get("verifications")
    return [v for v in items if isinstance(v, dict)] if isinstance(items, list) else []


def _norm(text: object) -> str:
    return "".join(ch for ch in str(text or "").lower() if ch.isalnum())


_CONTRACT_RE = re.compile(r"\b(?:contract|library|interface)\s+([A-Za-z_]\w*)")
_FUNCTION_RE = re.compile(r"\bfunction\s+([A-Za-z_]\w*)")


def extract_object_name(code: str) -> str:
    """Best-effort 'Analyzed Object' name: first contract, else first function."""
    match = _CONTRACT_RE.search(code or "")
    if match:
        return match.group(1)
    match = _FUNCTION_RE.search(code or "")
    return match.group(1) if match else "SmartContract"


# ──────────────────────────────────────────────────────────────────────────────
# Reporter agent
# ──────────────────────────────────────────────────────────────────────────────
class ReporterAgent:
    def __init__(self, llm: Any | None = None, backend: str = "auto") -> None:
        self.llm = llm or build_llm_client(backend)

    # ── findings assembly ────────────────────────────────────────────────────
    @staticmethod
    def _merge_findings(
        repair: Dict[str, Any],
        risk: Dict[str, Any] | None,
        fix: Dict[str, Any] | None,
        verification: Dict[str, Any] | None,
    ) -> List[ReportFinding]:
        risk_by_swc = {_norm(a.get("swc")): a for a in _assessments_of(risk)}
        fix = _unwrap(fix, "repair_patch")
        change_by_swc = {
            _norm(c.get("swc")): c
            for c in (fix.get("changes") or [])
            if isinstance(c, dict)
        }
        verify_by_swc = {_norm(v.get("swc")): v for v in _verifications_of(verification)}

        findings: List[ReportFinding] = []
        for suggestion in _suggestions_of(repair):
            swc = suggestion.get("swc") or "Unknown"
            nkey = _norm(swc)
            r = risk_by_swc.get(nkey, {})
            v = verify_by_swc.get(nkey, {})
            findings.append(
                ReportFinding(
                    swc=swc,
                    name=suggestion.get("vulnerability_name") or suggestion.get("title") or "Unknown vulnerability",
                    severity=suggestion.get("severity") or r.get("detector_severity") or "Unknown",
                    risk_level=r.get("risk_level") or suggestion.get("severity") or "Unknown",
                    cvss_score=r.get("cvss_score"),
                    repair_priority=r.get("repair_priority"),
                    root_cause=str(suggestion.get("root_cause") or suggestion.get("description") or "").strip(),
                    impact=str(suggestion.get("impact") or r.get("justification") or "").strip(),
                    repair_steps=[str(s) for s in (suggestion.get("repair_steps") or [])],
                    fixed_snippet=str(suggestion.get("fixed_code") or "").strip(),
                    verification_status=str(v.get("status") or "").strip(),
                    residual_risk=str(v.get("residual_risk") or "").strip(),
                )
            )
        findings.sort(key=lambda f: (f.repair_priority or 1_000, -RISK_ORDER.get(f.risk_level, 0)))
        return findings

    @staticmethod
    def _severity_distribution(
        findings: List[ReportFinding], risk: Dict[str, Any] | None
    ) -> Dict[str, int]:
        risk = _unwrap(risk, "risk_assessment")
        dist = risk.get("risk_distribution")
        if isinstance(dist, dict) and any(dist.values()):
            return {lvl: int(dist.get(lvl, 0)) for lvl in RISK_LEVELS}
        out = {lvl: 0 for lvl in RISK_LEVELS}
        for f in findings:
            if f.risk_level in out:
                out[f.risk_level] += 1
        return out

    @staticmethod
    def _reference_table(findings: List[ReportFinding]) -> List[Dict[str, Any]]:
        """The fixed catalogue, with a `detected` flag for classes found here."""
        detected_keys = {_norm(f.swc) for f in findings} | {_norm(f.name) for f in findings}
        table: List[Dict[str, Any]] = []
        for row in VULNERABILITY_REFERENCE:
            hit = any(
                _norm(k) in detected_keys or any(_norm(k) in dk or dk in _norm(k) for dk in detected_keys)
                for k in row["keys"]
            )
            table.append(
                {"name": row["name"], "abbr": row["abbr"], "severity": row["severity"],
                 "impact": row["impact"], "detected": hit}
            )
        return table

    # ── narrative (LLM) ──────────────────────────────────────────────────────
    def _generate_narrative(
        self, code: str, is_vulnerable: bool, findings: List[ReportFinding]
    ) -> Dict[str, Any]:
        finding_lines = "\n".join(
            f"- {f.swc} {f.name} (risk: {f.risk_level}, status: {f.verification_status or 'n/a'})"
            for f in findings
        ) or "(no vulnerabilities detected)"
        prompt = "\n".join(
            [
                "Write the narrative sections of a smart-contract audit report.",
                f"Overall detection verdict: {'Vulnerable' if is_vulnerable else 'Safe'}.",
                "",
                "<DetectedVulnerabilities>",
                finding_lines,
                "</DetectedVulnerabilities>",
                "",
                "<Contract>",
                "```solidity",
                (code or "").strip()[:8000],
                "```",
                "</Contract>",
                "",
                "Return ONLY a JSON object with this schema:",
                "{\n"
                '  "analyzed_object": string,     // the contract or primary function name\n'
                '  "contract_function": string,   // 1-3 sentences: what the contract/function does\n'
                '  "executive_summary": string,   // professional summary of the audit outcome\n'
                '  "recommendations": [string]    // 3-6 actionable security recommendations\n'
                "}",
                "Do not wrap the JSON in markdown fences.",
            ]
        )
        return self.llm.generate_json(
            prompt, REPORTER_SYSTEM_INSTRUCTION, response_schema=NARRATIVE_RESPONSE_SCHEMA
        )

    # ── orchestration ────────────────────────────────────────────────────────
    def report(
        self,
        code: str,
        detection_result: Dict[str, Any],
        repair_result: Dict[str, Any] | None = None,
        risk_result: Dict[str, Any] | None = None,
        fix_result: Dict[str, Any] | None = None,
        verification_result: Dict[str, Any] | None = None,
        audit_time: str = "",
    ) -> Dict[str, Any]:
        final = _unwrap(detection_result).get("final", {}) if isinstance(detection_result, dict) else {}
        is_vulnerable = bool(final.get("is_vulnerable"))
        detection_verdict = final.get("verdict") or ("Vulnerable" if is_vulnerable else "Safe")

        findings = self._merge_findings(repair_result or {}, risk_result, fix_result, verification_result)
        distribution = self._severity_distribution(findings, risk_result)
        reference = self._reference_table(findings)

        fix = _unwrap(fix_result, "repair_patch")
        fixed_code = str(fix.get("fixed_code") or "").strip()
        verification = _unwrap(verification_result, "verification")
        patch_verdict = str(verification.get("overall_verdict") or "").strip()

        # Narrative sections (graceful fallback if the LLM is unavailable).
        analyzed_object = extract_object_name(code)
        contract_function = ""
        executive_summary = ""
        recommendations: List[str] = list(DEFAULT_RECOMMENDATIONS)
        error: str | None = None
        try:
            narrative = self._generate_narrative(code, is_vulnerable, findings)
            analyzed_object = str(narrative.get("analyzed_object") or analyzed_object).strip()
            contract_function = str(narrative.get("contract_function") or "").strip()
            executive_summary = str(narrative.get("executive_summary") or "").strip()
            recs = [str(r).strip() for r in (narrative.get("recommendations") or []) if str(r).strip()]
            if recs:
                recommendations = recs
        except Exception as err:  # noqa: BLE001
            error = f"{type(err).__name__}: {err}"
            executive_summary = self._fallback_summary(is_vulnerable, findings, distribution, patch_verdict)

        report = AuditReport(
            llm_model=getattr(self.llm, "model_name", ""),
            analyzed_object=analyzed_object,
            contract_function=contract_function,
            detection_result=detection_verdict,
            audit_time=audit_time,
            is_vulnerable=is_vulnerable,
            executive_summary=executive_summary or self._fallback_summary(
                is_vulnerable, findings, distribution, patch_verdict
            ),
            vulnerability_count=len(findings),
            severity_distribution=distribution,
            reference_table=reference,
            findings=[asdict(f) for f in findings],
            original_code=code.strip(),
            fixed_code=fixed_code,
            patch_verdict=patch_verdict,
            recommendations=recommendations,
            error=error,
        )
        report.markdown = render_markdown(report, findings)
        return asdict(report)

    @staticmethod
    def _fallback_summary(
        is_vulnerable: bool,
        findings: List[ReportFinding],
        distribution: Dict[str, int],
        patch_verdict: str,
    ) -> str:
        if not is_vulnerable or not findings:
            return (
                "After auditing, we confirm that no critical or high-severity vulnerabilities "
                "were found in the contract. The contract, as currently implemented, does not "
                "present any obvious security risks. We recommend regular security audits and "
                "code updates to maintain long-term security."
            )
        parts = ", ".join(f"{distribution[l]} {l}" for l in RISK_LEVELS if distribution.get(l))
        verdict = f" The generated patch verification verdict is {patch_verdict}." if patch_verdict else ""
        return (
            f"The audit identified {len(findings)} vulnerability(ies) ({parts}). Each finding was "
            f"analysed for root cause and impact, risk-rated, and an automated repair was generated "
            f"and verified.{verdict} Review the detailed analysis and apply the fixed code before deployment."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Markdown rendering (mirrors the reference audit report layout)
# ──────────────────────────────────────────────────────────────────────────────
def render_markdown(report: AuditReport, findings: List[ReportFinding]) -> str:
    L: List[str] = []
    L.append("# Smart Contract Audit Report\n")

    # 1. Contract Information
    L.append("## 1. Contract Information\n")
    L.append(f"- **Analyzed Object:** {report.analyzed_object}")
    if report.contract_function:
        L.append(f"- **Contract Function:** {report.contract_function}")
    L.append(f"- **Detection Result:** {report.detection_result}")
    if report.audit_time:
        L.append(f"- **Audit Time:** {report.audit_time}")
    L.append("")

    # 2. Executive Summary
    L.append("## 2. Executive Summary\n")
    L.append(report.executive_summary + "\n")

    # 3. Methodology
    L.append("## 3. Methodology\n")
    L.append(report.methodology + "\n")

    # 4. Findings
    L.append("## 4. Findings\n")
    L.append("### 4.1 Vulnerability Statistics\n")
    found = report.vulnerability_count > 0
    L.append(f"- **Detection Result:** {'Found Vulnerabilities' if found else 'Not Found Vulnerabilities'}")
    L.append(f"- **Vulnerability Count:** Total of {report.vulnerability_count} vulnerabilities found\n")

    L.append("### 4.2 Vulnerability Severity Distribution\n")
    d = report.severity_distribution
    L.append("| Critical | High | Medium | Low |")
    L.append("| :------: | :--: | :----: | :-: |")
    L.append(f"| {d.get('Critical',0)} | {d.get('High',0)} | {d.get('Medium',0)} | {d.get('Low',0)} |\n")

    L.append("### 4.3 Vulnerability Reference Table\n")
    L.append("| Vulnerability Name | Severity | Detected | Impact Scope |")
    L.append("| :----------------- | :------- | :------: | :----------- |")
    for row in report.reference_table:
        mark = "✓" if row.get("detected") else ""
        L.append(f"| {row['name']} ({row['abbr']}) | {row['severity']} | {mark} | {row['impact']} |")
    L.append("")

    # 5. Detailed Analysis
    L.append("## 5. Detailed Analysis\n")
    L.append("### 5.1 Contract Name\n")
    L.append(report.analyzed_object + "\n")
    L.append("### 5.2 Source Code\n")
    L.append("```solidity")
    L.append(report.original_code or "(not provided)")
    L.append("```\n")

    L.append("### 5.3 Repair Suggestion\n")
    if not findings:
        L.append("No vulnerabilities were detected; therefore, no remediation actions are required.\n")
    else:
        for i, f in enumerate(findings, 1):
            cvss = f" · CVSS {f.cvss_score}" if f.cvss_score else ""
            prio = f" · Priority {f.repair_priority}" if f.repair_priority else ""
            L.append(f"#### {i}. {f.name} ({f.swc}) — {f.risk_level}{cvss}{prio}\n")
            if f.root_cause:
                L.append(f"- **Root cause:** {f.root_cause}")
            if f.impact:
                L.append(f"- **Impact:** {f.impact}")
            if f.repair_steps:
                L.append("- **Repair steps:**")
                L.extend(f"    {j}. {step}" for j, step in enumerate(f.repair_steps, 1))
            if f.verification_status:
                line = f"- **Patch verification:** {f.verification_status}"
                if f.residual_risk:
                    line += f" — residual risk: {f.residual_risk}"
                L.append(line)
            L.append("")

    L.append("### 5.4 Fixed Code\n")
    if report.fixed_code:
        if report.patch_verdict:
            L.append(f"_Patch verification verdict: **{report.patch_verdict}**_\n")
        L.append("```solidity")
        L.append(report.fixed_code)
        L.append("```\n")
    else:
        L.append("No vulnerabilities were detected, therefore no repair actions are required.\n")

    # 6. Summary and Recommendations
    L.append("## 6. Summary and Recommendations\n")
    L.append("Based on the audit results, we recommend:\n")
    L.extend(f"- {rec}" for rec in report.recommendations)
    L.append("")

    # 7. Disclaimer
    L.append("## 7. Disclaimer\n")
    L.append(report.disclaimer + "\n")

    return "\n".join(L)


def run_reporter(
    code: str,
    detection_result: Dict[str, Any],
    repair_result: Dict[str, Any] | None = None,
    risk_result: Dict[str, Any] | None = None,
    fix_result: Dict[str, Any] | None = None,
    verification_result: Dict[str, Any] | None = None,
    audit_time: str = "",
    backend: str = "auto",
) -> Dict[str, Any]:
    return ReporterAgent(backend=backend).report(
        code, detection_result, repair_result, risk_result, fix_result, verification_result, audit_time
    )
