"""
fixer.py
──────────────────────────────────────────────────────────────────────────────
Vulnerability Repair Agent (Fixer) — agent #4 of the LLM-BSCVM framework.

Consumes the Advisor's repair suggestions (phase 2) and the Assessor's risk
assessment (phase 3), then, as described in the paper (§III-B Fixer, Fig. 8):

    1. sorts the vulnerabilities by repair priority,
    2. considers contextual information and dependencies between fixes, and
    3. generates the complete repaired contract that complies with programming
       standards while preserving the original functionality.

Unlike the Advisor (which proposes a per-vulnerability fix snippet), the Fixer
produces ONE coherent, fully-repaired contract that addresses every prioritised
finding at once — so overlapping findings on the same code collapse into a
single correct fix. It also returns a unified diff (Original vs. Fixed, Fig. 8).
"""

from __future__ import annotations

import difflib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

try:
    from .llm_client import build_llm_client
except ImportError:  # allow running as a script
    from llm_client import build_llm_client

RISK_ORDER = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1, "None": 0, "Unknown": 0}

FIXER_SYSTEM_INSTRUCTION = (
    "You are an expert Solidity engineer performing secure vulnerability repair. "
    "Given a contract and a prioritised list of confirmed vulnerabilities (each "
    "with a recommended fix), produce ONE complete, compilable, repaired version "
    "of the ENTIRE contract that fixes every listed vulnerability. Apply fixes in "
    "priority order, respect dependencies between them, follow the "
    "Checks-Effects-Interactions pattern and current Solidity best practices, and "
    "PRESERVE the original intended functionality and public interface. Do not "
    "introduce new vulnerabilities. Return ONLY valid JSON matching the schema."
)

# Structured-output schema (shared by Gemini + Ollama backends).
FIX_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "fixed_code": {"type": "string"},
        "changes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "swc": {"type": "string"},
                    "vulnerability_name": {"type": "string"},
                    "change_summary": {"type": "string"},
                    "addressed": {"type": "boolean"},
                },
                "required": ["swc", "vulnerability_name", "change_summary", "addressed"],
            },
        },
        "residual_risk_notes": {"type": "string"},
    },
    "required": ["fixed_code", "changes", "residual_risk_notes"],
    "propertyOrdering": ["fixed_code", "changes", "residual_risk_notes"],
}


@dataclass
class FixChange:
    swc: str
    vulnerability_name: str
    risk_level: str
    repair_priority: int
    change_summary: str
    addressed: bool


@dataclass
class RepairResult:
    agent: str = "Fixer (Vulnerability Repair)"
    llm_model: str = ""
    fixed_count: int = 0
    target_count: int = 0
    original_code: str = ""
    fixed_code: str = ""
    diff: str = ""
    changes: List[Dict[str, Any]] = field(default_factory=list)
    residual_risk_notes: str = ""
    error: str | None = None


# ──────────────────────────────────────────────────────────────────────────────
# Input merging: join Advisor suggestions with Assessor priorities
# ──────────────────────────────────────────────────────────────────────────────
def _suggestions_of(repair_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(repair_result.get("suggestions"), list):
        return repair_result["suggestions"]
    if isinstance(repair_result.get("repair"), dict):
        return repair_result["repair"].get("suggestions", []) or []
    return []


def _assessments_of(risk_result: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    if not risk_result:
        return []
    if isinstance(risk_result.get("assessments"), list):
        return risk_result["assessments"]
    if isinstance(risk_result.get("risk_assessment"), dict):
        return risk_result["risk_assessment"].get("assessments", []) or []
    return []


def merge_findings(
    repair_result: Dict[str, Any],
    risk_result: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Join repair suggestions with risk priorities and sort by repair priority."""
    suggestions = _suggestions_of(repair_result)
    risk_by_swc = {a.get("swc"): a for a in _assessments_of(risk_result)}

    merged: List[Dict[str, Any]] = []
    for suggestion in suggestions:
        swc = suggestion.get("swc") or "Unknown"
        risk = risk_by_swc.get(swc, {})
        merged.append(
            {
                "swc": swc,
                "vulnerability_name": suggestion.get("vulnerability_name")
                or suggestion.get("title")
                or "Unknown vulnerability",
                "severity": suggestion.get("severity") or "Unknown",
                "risk_level": risk.get("risk_level") or suggestion.get("severity") or "Unknown",
                "cvss_score": risk.get("cvss_score"),
                "repair_priority": risk.get("repair_priority"),
                "root_cause": suggestion.get("root_cause") or suggestion.get("description") or "",
                "repair_steps": suggestion.get("repair_steps") or [],
                "suggested_fixed_code": suggestion.get("fixed_code") or "",
            }
        )

    has_priority = any(item.get("repair_priority") for item in merged)
    if has_priority:
        merged.sort(key=lambda item: item.get("repair_priority") or 1_000)
    else:
        merged.sort(key=lambda item: RISK_ORDER.get(item.get("risk_level"), 0), reverse=True)

    # Assign a 1-based priority where the Assessor did not provide one.
    for index, item in enumerate(merged, start=1):
        if not item.get("repair_priority"):
            item["repair_priority"] = index
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Fixer agent
# ──────────────────────────────────────────────────────────────────────────────
class FixerAgent:
    def __init__(self, llm: Any | None = None, backend: str = "auto") -> None:
        self.llm = llm or build_llm_client(backend)

    @staticmethod
    def _build_findings_block(findings: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for item in findings:
            lines.append(
                f"[Priority {item['repair_priority']}] {item['swc']} — "
                f"{item['vulnerability_name']} (risk: {item['risk_level']})"
            )
            if item.get("root_cause"):
                lines.append(f"  Root cause: {str(item['root_cause'])[:500]}")
            steps = item.get("repair_steps") or []
            if steps:
                lines.append("  Recommended fix steps:")
                lines.extend(f"    - {str(step)[:300]}" for step in steps[:6])
            suggested = str(item.get("suggested_fixed_code") or "").strip()
            if suggested:
                lines.append("  Suggested fixed snippet (reference):")
                lines.append("  " + suggested[:800].replace("\n", "\n  "))
            lines.append("")
        return "\n".join(lines).strip()

    def _build_prompt(self, code: str, findings: List[Dict[str, Any]]) -> str:
        schema = (
            "{\n"
            '  "fixed_code": string,            // the COMPLETE repaired contract\n'
            '  "changes": [                     // one entry per vulnerability\n'
            '    { "swc": string, "vulnerability_name": string,\n'
            '      "change_summary": string, "addressed": boolean }\n'
            "  ],\n"
            '  "residual_risk_notes": string    // anything not fully fixable in code\n'
            "}"
        )
        return "\n".join(
            [
                "Repair ALL of the following vulnerabilities in the contract below, "
                "in the given priority order.",
                "",
                "<PrioritisedVulnerabilities>",
                self._build_findings_block(findings),
                "</PrioritisedVulnerabilities>",
                "",
                "<OriginalContract>",
                "```solidity",
                code.strip(),
                "```",
                "</OriginalContract>",
                "",
                "Return ONLY a JSON object with exactly this schema:",
                schema,
                "fixed_code MUST be the entire repaired contract (not a snippet), "
                "compilable, with the original functionality and public interface preserved. "
                "Do not wrap the JSON in markdown fences.",
            ]
        )

    def fix(
        self,
        code: str,
        repair_result: Dict[str, Any],
        risk_result: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        findings = merge_findings(repair_result, risk_result)

        if not findings:
            return asdict(
                RepairResult(
                    llm_model=getattr(self.llm, "model_name", ""),
                    original_code=code,
                    fixed_code=code,
                    residual_risk_notes="No vulnerabilities to repair.",
                )
            )

        try:
            data = self.llm.generate_json(
                self._build_prompt(code, findings),
                FIXER_SYSTEM_INSTRUCTION,
                response_schema=FIX_RESPONSE_SCHEMA,
            )
        except Exception as error:  # noqa: BLE001
            return asdict(
                RepairResult(
                    llm_model=getattr(self.llm, "model_name", ""),
                    target_count=len(findings),
                    original_code=code,
                    fixed_code=code,
                    error=f"{type(error).__name__}: {error}",
                )
            )

        fixed_code = str(data.get("fixed_code") or "").strip()

        # Map the model's free-form change list back onto the prioritised findings.
        # Match by SWC id first, then by normalised name, then positionally.
        raw_changes = [c for c in (data.get("changes") or []) if isinstance(c, dict)]

        def _norm(text: object) -> str:
            return "".join(ch for ch in str(text or "").lower() if ch.isalnum())

        by_swc = {_norm(c.get("swc")): c for c in raw_changes}
        by_name = {_norm(c.get("vulnerability_name")): c for c in raw_changes}

        changes: List[FixChange] = []
        for index, item in enumerate(findings):
            change = (
                by_swc.get(_norm(item["swc"]))
                or by_name.get(_norm(item["vulnerability_name"]))
                or (raw_changes[index] if index < len(raw_changes) else {})
            )
            changes.append(
                FixChange(
                    swc=item["swc"],
                    vulnerability_name=item["vulnerability_name"],
                    risk_level=item["risk_level"],
                    repair_priority=item["repair_priority"],
                    change_summary=str(change.get("change_summary") or "").strip(),
                    addressed=bool(change.get("addressed", bool(fixed_code))),
                )
            )

        diff = ""
        if fixed_code:
            diff = "\n".join(
                difflib.unified_diff(
                    code.strip().splitlines(),
                    fixed_code.splitlines(),
                    fromfile="original.sol",
                    tofile="fixed.sol",
                    lineterm="",
                )
            )

        return asdict(
            RepairResult(
                llm_model=getattr(self.llm, "model_name", ""),
                fixed_count=sum(1 for c in changes if c.addressed),
                target_count=len(findings),
                original_code=code,
                fixed_code=fixed_code or code,
                diff=diff,
                changes=[asdict(c) for c in changes],
                residual_risk_notes=str(data.get("residual_risk_notes") or "").strip(),
            )
        )


def run_fixer(
    code: str,
    repair_result: Dict[str, Any],
    risk_result: Dict[str, Any] | None = None,
    backend: str = "auto",
) -> Dict[str, Any]:
    return FixerAgent(backend=backend).fix(code, repair_result, risk_result)
