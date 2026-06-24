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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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


ASSESSOR_SYSTEM_INSTRUCTION = (
    "You are a smart-contract security risk analyst. For a single confirmed "
    "vulnerability, assess its risk using the CVSS v3.1 base-score standard and "
    "classify it into a four-level system: Critical, High, Medium, or Low. "
    "Base your judgement on exploitability (attack vector, complexity, privileges, "
    "user interaction) and impact (funds at risk, confidentiality, integrity, "
    "availability) for THIS specific contract. Return ONLY valid JSON matching the "
    "requested schema."
)

# Structured-output schema — guarantees a parseable, well-typed response.
RISK_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "risk_level": {"type": "string", "enum": list(RISK_LEVELS)},
        "cvss_score": {"type": "number"},
        "cvss_vector": {"type": "string"},
        "likelihood": {"type": "string", "enum": ["High", "Medium", "Low"]},
        "impact_rating": {"type": "string", "enum": ["High", "Medium", "Low"]},
        "justification": {"type": "string"},
        "funds_at_risk": {"type": "boolean"},
    },
    "required": [
        "risk_level",
        "cvss_score",
        "cvss_vector",
        "likelihood",
        "impact_rating",
        "justification",
        "funds_at_risk",
    ],
    "propertyOrdering": [
        "risk_level",
        "cvss_score",
        "cvss_vector",
        "likelihood",
        "impact_rating",
        "justification",
        "funds_at_risk",
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
    def __init__(self, llm: GeminiClient | None = None, backend: str = "auto") -> None:
        self.llm = llm or build_llm_client(backend)

    # ── prompt ──────────────────────────────────────────────────────────────
    @staticmethod
    def _build_prompt(code: str, item: Dict[str, Any]) -> str:
        swc = item.get("swc") or "Unknown"
        name = item.get("vulnerability_name") or item.get("title") or "Unknown vulnerability"
        detector_severity = item.get("severity") or "Unknown"
        root_cause = str(item.get("root_cause") or item.get("description") or "").strip()
        impact = str(item.get("impact") or "").strip()

        schema = (
            "{\n"
            '  "risk_level": "Critical|High|Medium|Low",\n'
            '  "cvss_score": number,            // CVSS v3.1 base score 0.0-10.0\n'
            '  "cvss_vector": string,           // e.g. CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:H/A:H\n'
            '  "likelihood": "High|Medium|Low", // exploitation likelihood\n'
            '  "impact_rating": "High|Medium|Low",\n'
            '  "justification": string,         // why this level, grounded in the contract\n'
            '  "funds_at_risk": boolean\n'
            "}"
        )

        return "\n".join(
            [
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
                "<Contract>",
                "```solidity",
                code.strip()[:10000],
                "```",
                "</Contract>",
                "",
                "Return ONLY a JSON object with exactly this schema:",
                schema,
                "Use the standard CVSS v3.1 base-score-to-severity bands "
                "(0.1-3.9 Low, 4.0-6.9 Medium, 7.0-8.9 High, 9.0-10.0 Critical). "
                "Do not wrap the JSON in markdown fences.",
            ]
        )

    # ── single assessment ───────────────────────────────────────────────────
    def assess_one(self, code: str, item: Dict[str, Any]) -> RiskAssessment:
        swc = item.get("swc") or "Unknown"
        name = item.get("vulnerability_name") or item.get("title") or "Unknown vulnerability"
        detector_severity = item.get("severity") or "Unknown"

        try:
            data = self.llm.generate_json(
                self._build_prompt(code, item),
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

        try:
            cvss_score = round(float(data.get("cvss_score") or 0.0), 1)
        except (TypeError, ValueError):
            cvss_score = 0.0
        cvss_score = max(0.0, min(10.0, cvss_score))

        return RiskAssessment(
            swc=swc,
            vulnerability_name=str(data.get("vulnerability_name") or name).strip(),
            detector_severity=detector_severity,
            risk_level=normalize_level(data.get("risk_level"), cvss_score),
            cvss_score=cvss_score,
            cvss_vector=str(data.get("cvss_vector") or "").strip(),
            likelihood=str(data.get("likelihood") or "").strip(),
            impact_rating=str(data.get("impact_rating") or "").strip(),
            funds_at_risk=bool(data.get("funds_at_risk")),
            justification=str(data.get("justification") or "").strip(),
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
