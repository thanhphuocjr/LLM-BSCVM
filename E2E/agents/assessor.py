"""
assessor.py
──────────────────────────────────────────────────────────────────────────────
Risk Assessment Agent (Assessor) — agent #3 of the LLM-BSCVM framework.

Consumes the output of the Advisor (phase 2) and, for each vulnerability,
systematically evaluates its risk using:
    * the CVSS v3.1 scoring standard (base score + vector), and
    * a four-level risk system: Critical / High / Medium / Low.

It then produces, exactly as in the paper (Fig. 7):
    * a per-vulnerability risk level + CVSS assessment,
    * a statistical risk distribution (count per level), and
    * a repair-priority ordering to drive the subsequent Fixer agent.

Decompose -> one assessment per vulnerability
Retrieve  -> reuses the root-cause / impact context already produced upstream
Generate  -> CVSS-grounded risk judgement via a generative LLM (Gemini)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

try:
    from .llm_client import GeminiClient, build_llm_client
except ImportError:  # allow running as a script
    from llm_client import GeminiClient, build_llm_client

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RISK_LEVELS = ("Critical", "High", "Medium", "Low")
RISK_ORDER = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1, "None": 0, "Unknown": 0}

# Fallback mapping when the LLM returns a CVSS score but an off-vocabulary level.
def cvss_to_level(score: float) -> str:
    if score >= 9.0:
        return "Critical"
    if score >= 7.0:
        return "High"
    if score >= 4.0:
        return "Medium"
    return "Low"


def normalize_level(level: object, cvss_score: float | None = None) -> str:
    text = str(level or "").strip().title()
    if text in RISK_LEVELS:
        return text
    if cvss_score is not None:
        return cvss_to_level(cvss_score)
    return "Medium"


def compute_cvss_31_base_score(
    av: str = "N",
    ac: str = "L",
    pr: str = "N",
    ui: str = "N",
    s: str = "U",
    c: str = "N",
    i: str = "H",
    a: str = "H",
) -> tuple[float, str]:
    """Calculate CVSS v3.1 base score algorithmically from metrics according to FIRST spec."""
    import math

    av_weights = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.20}
    ac_weights = {"L": 0.77, "H": 0.44}
    pr_weights_u = {"N": 0.85, "L": 0.62, "H": 0.27}
    pr_weights_c = {"N": 0.85, "L": 0.68, "H": 0.50}
    ui_weights = {"N": 0.85, "R": 0.62}
    c_weights = {"N": 0.0, "L": 0.22, "H": 0.56}
    i_weights = {"N": 0.0, "L": 0.22, "H": 0.56}
    a_weights = {"N": 0.0, "L": 0.22, "H": 0.56}

    av_val = av_weights.get(av.upper(), 0.85)
    ac_val = ac_weights.get(ac.upper(), 0.77)
    scope_changed = s.upper() == "C"
    pr_val = (pr_weights_c if scope_changed else pr_weights_u).get(pr.upper(), 0.85)
    ui_val = ui_weights.get(ui.upper(), 0.85)

    c_val = c_weights.get(c.upper(), 0.0)
    i_val = i_weights.get(i.upper(), 0.56)
    a_val = a_weights.get(a.upper(), 0.56)

    iss = 1.0 - ((1.0 - c_val) * (1.0 - i_val) * (1.0 - a_val))
    if not scope_changed:
        impact = 6.42 * iss
    else:
        impact = 7.52 * (iss - 0.029) - 3.25 * ((iss - 0.02) ** 15)

    exploitability = 8.22 * av_val * ac_val * pr_val * ui_val

    if impact <= 0:
        base_score = 0.0
    else:
        if not scope_changed:
            raw_score = min(impact + exploitability, 10.0)
        else:
            raw_score = min(1.08 * (impact + exploitability), 10.0)
        base_score = math.ceil(raw_score * 10.0) / 10.0

    vector = f"CVSS:3.1/AV:{av.upper()}/AC:{ac.upper()}/PR:{pr.upper()}/UI:{ui.upper()}/S:{'C' if scope_changed else 'U'}/C:{c.upper()}/I:{i.upper()}/A:{a.upper()}"
    return round(base_score, 1), vector


ASSESSOR_SYSTEM_INSTRUCTION = (
    "You are a smart-contract security risk analyst. For a single confirmed "
    "vulnerability, select each CVSS v3.1 metric based on the specific smart-contract context: "
    "AV (Attack Vector), AC (Attack Complexity), PR (Privileges Required), UI (User Interaction), "
    "S (Scope), C (Confidentiality), I (Integrity), A (Availability). "
    "The final CVSS score will be computed algorithmically from your metric selections. "
    "Return ONLY valid JSON matching the requested schema."
)

# Structured-output schema
RISK_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "attack_vector": {"type": "string", "enum": ["N", "A", "L", "P"]},
        "attack_complexity": {"type": "string", "enum": ["L", "H"]},
        "privileges_required": {"type": "string", "enum": ["N", "L", "H"]},
        "user_interaction": {"type": "string", "enum": ["N", "R"]},
        "scope": {"type": "string", "enum": ["U", "C"]},
        "confidentiality": {"type": "string", "enum": ["N", "L", "H"]},
        "integrity": {"type": "string", "enum": ["N", "L", "H"]},
        "availability": {"type": "string", "enum": ["N", "L", "H"]},
        "funds_at_risk": {"type": "boolean"},
        "justification": {"type": "string"},
    },
    "required": [
        "attack_vector",
        "attack_complexity",
        "privileges_required",
        "user_interaction",
        "scope",
        "confidentiality",
        "integrity",
        "availability",
        "funds_at_risk",
        "justification",
    ],
}


@dataclass
class RiskAssessment:
    swc: str
    vulnerability_name: str
    detector_severity: str
    risk_level: str
    cvss_score: float
    cvss_vector: str
    likelihood: str
    impact_rating: str
    funds_at_risk: bool
    justification: str
    repair_priority: int = 0
    error: str | None = None


class AssessorAgent:
    def __init__(
        self,
        llm: GeminiClient | None = None,
        knowledge_base: Any | None = None,
        backend: str = "auto",
    ) -> None:
        self.llm = llm or build_llm_client(backend)
        if knowledge_base is not None:
            self.knowledge_base = knowledge_base
        else:
            try:
                from .advisor import RepairKnowledgeBase
                self.knowledge_base = RepairKnowledgeBase()
            except Exception:
                self.knowledge_base = None

    # ── prompt ──────────────────────────────────────────────────────────────
    @staticmethod
    def _build_prompt(
        code: str,
        item: Dict[str, Any],
        risk_guidance: Dict[str, Any] | None = None,
    ) -> str:
        swc = item.get("swc") or "Unknown"
        name = item.get("vulnerability_name") or item.get("title") or "Unknown vulnerability"
        detector_severity = item.get("severity") or "Unknown"
        root_cause = str(item.get("root_cause") or item.get("description") or "").strip()
        impact = str(item.get("impact") or "").strip()

        guidance_section: List[str] = []
        if risk_guidance:
            benchmark_sev = risk_guidance.get("benchmark_severity") or "Unknown"
            cwe_ids = risk_guidance.get("cwe_ids") or []
            cwe_str = f" (CWE: {', '.join(cwe_ids)})" if cwe_ids else ""
            remed = risk_guidance.get("remediation_summary") or ""
            checks = risk_guidance.get("checklist") or []
            guidance_section.extend([
                "<VulnerabilityKnowledgeGuidance>",
                f"SWC Benchmark Severity: {benchmark_sev}{cwe_str}",
            ])
            if remed:
                guidance_section.append(f"Remediation Standard: {remed[:300]}")
            if checks:
                guidance_section.append("Risk Evaluation Checklist:")
                guidance_section.extend(f"  - {c}" for c in checks[:4])
            guidance_section.extend([
                "</VulnerabilityKnowledgeGuidance>",
                "",
            ])

        schema = (
            "{\n"
            '  "attack_vector": "N|A|L|P",      // N: callable via any blockchain tx, A: adjacent, L: local, P: physical\n'
            '  "attack_complexity": "L|H",      // L: standard tx exploit, H: requires front-running/race condition/miner\n'
            '  "privileges_required": "N|L|H",  // N: public permissionless, L: regular caller, H: contract owner/admin\n'
            '  "user_interaction": "N|R",       // N: direct exploit, R: requires victim transaction / approve\n'
            '  "scope": "U|C",                  // U: this contract only, C: impacts external tokens / protocols\n'
            '  "confidentiality": "N|L|H",      // information disclosure\n'
            '  "integrity": "N|L|H",            // unauthorized token transfer, state corruption\n'
            '  "availability": "N|L|H",         // frozen ether, contract bricked, locked balances\n'
            '  "funds_at_risk": boolean,        // true if assets or token balances can be drained or frozen\n'
            '  "justification": string          // technical rationale for metric choices\n'
            "}"
        )

        sections = [
            "Assess the risk of the following smart-contract vulnerability.",
            "",
            "<Vulnerability>",
            f"SWC: {swc}",
            f"Name: {name}",
            f"Detector-assigned severity: {detector_severity}",
            f"Root cause: {root_cause[:1200] or '(not provided)'}",
            f"Impact (from repair analysis): {impact[:1200] or '(not provided)'}",
            "</Vulnerability>",
            "",
        ]
        if guidance_section:
            sections.extend(guidance_section)
        sections.extend([
            "<Contract>",
            "```solidity",
            code.strip()[:10000],
            "```",
            "</Contract>",
            "",
            "Return ONLY a JSON object with exactly this schema:",
            schema,
            "Carefully evaluate smart-contract exploitability (front-running, owner privileges) and impact. "
            "Do not wrap the JSON in markdown fences.",
        ])
        return "\n".join(sections)

    # ── single assessment ───────────────────────────────────────────────────
    def assess_one(self, code: str, item: Dict[str, Any]) -> RiskAssessment:
        swc = item.get("swc") or "Unknown"
        name = item.get("vulnerability_name") or item.get("title") or "Unknown vulnerability"
        detector_severity = item.get("severity") or "Unknown"

        if item.get("is_true_positive") is False:
            return RiskAssessment(
                swc=swc,
                vulnerability_name=name,
                detector_severity="Low",
                risk_level="Low",
                cvss_score=0.0,
                cvss_vector="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:N",
                likelihood="Low",
                impact_rating="Low",
                funds_at_risk=False,
                justification="Advisor filtered this finding as a false positive.",
            )

        risk_guidance = item.get("risk_guidance")
        if not risk_guidance and getattr(self, "knowledge_base", None):
            try:
                ret = self.knowledge_base.retrieve(swc)
                risk_guidance = ret.get("risk_guidance")
            except Exception:
                pass

        try:
            data = self.llm.generate_json(
                self._build_prompt(code, item, risk_guidance=risk_guidance),
                ASSESSOR_SYSTEM_INSTRUCTION,
                response_schema=RISK_RESPONSE_SCHEMA,
            )
        except Exception as error:  # noqa: BLE001 - reported per finding
            return RiskAssessment(
                swc=swc,
                vulnerability_name=name,
                detector_severity=detector_severity,
                risk_level=normalize_level(detector_severity),
                cvss_score=0.0,
                cvss_vector="",
                likelihood="",
                impact_rating="",
                funds_at_risk=False,
                justification="",
                error=f"{type(error).__name__}: {error}",
            )

        cvss_score, cvss_vector = compute_cvss_31_base_score(
            av=str(data.get("attack_vector") or "N"),
            ac=str(data.get("attack_complexity") or "L"),
            pr=str(data.get("privileges_required") or "N"),
            ui=str(data.get("user_interaction") or "N"),
            s=str(data.get("scope") or "U"),
            c=str(data.get("confidentiality") or "N"),
            i=str(data.get("integrity") or "H"),
            a=str(data.get("availability") or "H"),
        )
        risk_level = cvss_to_level(cvss_score)
        funds_at_risk = bool(data.get("funds_at_risk"))
        justification = str(data.get("justification") or "").strip()

        ac = str(data.get("attack_complexity") or "L").upper()
        pr = str(data.get("privileges_required") or "N").upper()
        likelihood = "High" if (ac == "L" and pr == "N") else ("Low" if (ac == "H" or pr == "H") else "Medium")

        i_metric = str(data.get("integrity") or "H").upper()
        a_metric = str(data.get("availability") or "H").upper()
        impact_rating = "High" if (i_metric == "H" or a_metric == "H" or funds_at_risk) else ("Low" if (i_metric == "N" and a_metric == "N") else "Medium")

        return RiskAssessment(
            swc=swc,
            vulnerability_name=str(data.get("vulnerability_name") or name).strip(),
            detector_severity=detector_severity,
            risk_level=risk_level,
            cvss_score=cvss_score,
            cvss_vector=cvss_vector,
            likelihood=likelihood,
            impact_rating=impact_rating,
            funds_at_risk=funds_at_risk,
            justification=justification,
        )

    # ── input adaptation ────────────────────────────────────────────────────
    @staticmethod
    def _extract_items(previous_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Accept either the Advisor output (phase 2) or a detection result (phase 1)."""
        if isinstance(previous_result.get("suggestions"), list):
            return previous_result["suggestions"]
        if isinstance(previous_result.get("repair"), dict):
            return previous_result["repair"].get("suggestions", []) or []
        if isinstance(previous_result.get("vulnerabilities"), list):
            return previous_result["vulnerabilities"]
        return []

    # ── batch + aggregation ─────────────────────────────────────────────────
    def assess(self, code: str, previous_result: Dict[str, Any]) -> Dict[str, Any]:
        items = self._extract_items(previous_result)

        if not items:
            return {
                "agent": "Assessor (Risk Assessment)",
                "overall_risk_level": "None",
                "assessment_count": 0,
                "risk_distribution": {level: 0 for level in RISK_LEVELS},
                "assessments": [],
                "note": "No vulnerabilities to assess.",
            }

        assessments = [self.assess_one(code, item) for item in items]

        # Repair priority: sort by (risk level, CVSS score) descending, 1-based rank.
        ranked = sorted(
            assessments,
            key=lambda a: (RISK_ORDER.get(a.risk_level, 0), a.cvss_score),
            reverse=True,
        )
        for priority, assessment in enumerate(ranked, start=1):
            assessment.repair_priority = priority

        distribution = {level: 0 for level in RISK_LEVELS}
        for assessment in assessments:
            if assessment.risk_level in distribution:
                distribution[assessment.risk_level] += 1

        overall = max(
            (a.risk_level for a in assessments),
            key=lambda level: RISK_ORDER.get(level, 0),
            default="None",
        )

        return {
            "agent": "Assessor (Risk Assessment)",
            "llm_model": self.llm.model_name,
            "overall_risk_level": overall,
            "assessment_count": len(assessments),
            "risk_distribution": distribution,
            # Ordered by repair priority so the Fixer can consume it directly.
            "assessments": [asdict(a) for a in ranked],
        }


def run_assessor(
    code: str,
    previous_result: Dict[str, Any],
    backend: str = "auto",
) -> Dict[str, Any]:
    return AssessorAgent(backend=backend).assess(code, previous_result)
