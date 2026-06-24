"""
advisor.py
──────────────────────────────────────────────────────────────────────────────
Repair Suggestion Agent (Advisor) — agent #2 of the LLM-BSCVM framework.

Consumes the output of the detection phase (phase1_integrated_detector.py) and,
for each detected vulnerability, produces a targeted repair suggestion using
Retrieval-Augmented Generation:

    Decompose  -> one suggestion task per detected vulnerability
    Retrieve   -> remediation knowledge from the vulnerability knowledge base
                  (keyed by the detected SWC id; offline, no model reload)
    Generate   -> structured repair suggestion via a generative LLM (Gemini)

Following the paper (Fig. 6), each suggestion covers five aspects:
    1. Vulnerability name
    2. Root cause analysis
    3. Potential impact assessment
    4. Repair steps + fixed code example
    5. Preventive measures

Returns structured data so the next agents (Assessor, Fixer, Reporter) can
consume it directly.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

try:
    from .llm_client import GeminiClient, build_llm_client
except ImportError:  # allow running as a script
    from llm_client import GeminiClient, build_llm_client

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KNOWLEDGE_DIR = PROJECT_ROOT / "rag" / "knowledge_store"

# Structured-output schema (OpenAPI subset) — guarantees Gemini returns valid,
# parseable JSON even when fixed_code contains quotes / newlines / special chars.
REPAIR_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "vulnerability_name": {"type": "string"},
        "root_cause": {"type": "string"},
        "impact": {"type": "string"},
        "repair_steps": {"type": "array", "items": {"type": "string"}},
        "fixed_code": {"type": "string"},
        "prevention": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "vulnerability_name",
        "root_cause",
        "impact",
        "repair_steps",
        "fixed_code",
        "prevention",
    ],
    "propertyOrdering": [
        "vulnerability_name",
        "root_cause",
        "impact",
        "repair_steps",
        "fixed_code",
        "prevention",
    ],
}

ADVISOR_SYSTEM_INSTRUCTION = (
    "You are a senior smart-contract security auditor specialising in Solidity. "
    "Your task is to produce a precise, actionable repair suggestion for a single "
    "confirmed vulnerability in the provided contract. Ground your answer in the "
    "supplied contract code and the retrieved security knowledge. Be concrete and "
    "specific to THIS contract — never give generic advice. Return ONLY valid JSON "
    "matching the requested schema."
)


# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class RepairSuggestion:
    swc: str
    vulnerability_name: str
    severity: str
    root_cause: str
    impact: str
    repair_steps: List[str]
    fixed_code: str
    prevention: List[str]
    knowledge_sources: List[str] = field(default_factory=list)
    error: str | None = None


# ──────────────────────────────────────────────────────────────────────────────
# Repair knowledge base (RAG retrieval by SWC id — offline, deterministic)
# ──────────────────────────────────────────────────────────────────────────────
def normalize_swc_id(value: object) -> str:
    match = re.search(r"SWC-\d{3}", str(value or "").upper())
    return match.group(0) if match else str(value or "").strip().upper()


class RepairKnowledgeBase:
    """Indexes the vulnerability knowledge store by SWC id for repair retrieval.

    This is the Advisor's RAG step: rather than re-running the BGE-M3 embedding
    model, it looks remediation knowledge up directly from the knowledge-store
    metadata by the SWC id the detector already identified.
    """

    def __init__(self, knowledge_dir: str | Path = DEFAULT_KNOWLEDGE_DIR) -> None:
        metadata_path = Path(knowledge_dir) / "sample_metadata.json"
        self._by_swc: Dict[str, List[Dict[str, Any]]] = {}
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as f:
                for record in json.load(f):
                    swc = normalize_swc_id(record.get("Swc"))
                    if swc:
                        self._by_swc.setdefault(swc, []).append(record)

    def retrieve(self, swc: str, limit: int = 3) -> Dict[str, Any]:
        """Aggregate remediation knowledge for a given SWC id."""
        records = self._by_swc.get(normalize_swc_id(swc), [])[:limit]
        remediations: List[str] = []
        checklist: List[str] = []
        code_examples: List[str] = []
        descriptions: List[str] = []
        titles: List[str] = []

        for record in records:
            for value, bucket in (
                (record.get("Remediation"), remediations),
                (record.get("Description"), descriptions),
                (record.get("Title"), titles),
            ):
                text = str(value or "").strip()
                if text and text not in bucket:
                    bucket.append(text)
            for item in record.get("BestPracticeChecklist", []) or []:
                text = str(item or "").strip()
                if text and text not in checklist:
                    checklist.append(text)
            for item in record.get("CodeExamples", []) or []:
                text = str(item or "").strip()
                if text and text not in code_examples:
                    code_examples.append(text)

        return {
            "titles": titles,
            "descriptions": descriptions,
            "remediations": remediations,
            "checklist": checklist,
            "code_examples": code_examples,
            "record_count": len(records),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Advisor agent
# ──────────────────────────────────────────────────────────────────────────────
class AdvisorAgent:
    def __init__(
        self,
        llm: GeminiClient | None = None,
        knowledge_base: RepairKnowledgeBase | None = None,
        knowledge_dir: str | Path = DEFAULT_KNOWLEDGE_DIR,
        backend: str = "auto",
    ) -> None:
        self.llm = llm or build_llm_client(backend)
        self.knowledge_base = knowledge_base or RepairKnowledgeBase(knowledge_dir)

    # ── knowledge formatting ────────────────────────────────────────────────
    @staticmethod
    def _truncate(items: List[str], max_items: int, max_chars: int) -> List[str]:
        return [item[:max_chars] for item in items[:max_items]]

    def _build_knowledge_block(self, vulnerability: Dict[str, Any], retrieved: Dict[str, Any]) -> str:
        # Knowledge already carried from the detection phase ...
        detection_practice = str(vulnerability.get("best_practice") or "").strip()
        detection_checklist = vulnerability.get("best_practice_checklist") or []
        detection_patterns = vulnerability.get("code_patterns_from_rag") or []

        # ... merged with a fresh lookup from the vulnerability knowledge base.
        remediations = self._truncate(retrieved.get("remediations", []), 3, 800)
        checklist = self._truncate(
            [str(x) for x in (list(detection_checklist) + retrieved.get("checklist", []))], 8, 200
        )
        code_examples = self._truncate(retrieved.get("code_examples", []), 2, 600)

        lines = ["<RetrievedSecurityKnowledge>"]
        if detection_practice:
            lines.append(f"Best practice (from detection): {detection_practice[:800]}")
        for idx, item in enumerate(remediations, 1):
            lines.append(f"Remediation {idx}: {item}")
        if checklist:
            lines.append("Checklist:")
            lines.extend(f"  - {item}" for item in checklist)
        if detection_patterns:
            lines.append("Vulnerable code patterns: " + "; ".join(str(p) for p in detection_patterns[:6]))
        for idx, item in enumerate(code_examples, 1):
            lines.append(f"Reference fix example {idx}: {item}")
        lines.append("</RetrievedSecurityKnowledge>")
        return "\n".join(lines)

    @staticmethod
    def _related_code_block(vulnerability: Dict[str, Any]) -> str:
        windows = []
        for related in vulnerability.get("related_code", []) or []:
            window = (related.get("window") or {}).get("code")
            if window:
                windows.append(window)
        return "\n---\n".join(windows[:4])

    def _build_prompt(self, code: str, vulnerability: Dict[str, Any], knowledge_block: str) -> str:
        swc = vulnerability.get("swc") or "Unknown"
        title = vulnerability.get("title") or "Unknown vulnerability"
        severity = vulnerability.get("severity") or "Unknown"
        description = str(vulnerability.get("description") or "").strip()
        related_code = self._related_code_block(vulnerability)

        schema = (
            "{\n"
            '  "vulnerability_name": string,\n'
            '  "root_cause": string,            // why this contract is vulnerable\n'
            '  "impact": string,                // what an attacker can achieve\n'
            '  "repair_steps": [string],        // concrete ordered steps\n'
            '  "fixed_code": string,            // corrected Solidity for the affected function(s)\n'
            '  "prevention": [string]           // preventive measures / best practices\n'
            "}"
        )

        sections = [
            "Analyze the detected vulnerability and provide a repair suggestion.",
            "",
            "<DetectedVulnerability>",
            f"SWC: {swc}",
            f"Name: {title}",
            f"Severity: {severity}",
            f"Detector note: {description[:600]}" if description else "Detector note: (none)",
            "</DetectedVulnerability>",
            "",
            knowledge_block,
            "",
            "<AffectedCode>",
            related_code if related_code else "(specific lines not isolated — use the full contract below)",
            "</AffectedCode>",
            "",
            "<FullContract>",
            "```solidity",
            code.strip()[:12000],
            "```",
            "</FullContract>",
            "",
            "Return ONLY a JSON object with exactly this schema:",
            schema,
            "Rules: repair_steps and prevention are arrays of short strings. "
            "fixed_code must be valid Solidity specific to this contract. "
            "Do not wrap the JSON in markdown fences.",
        ]
        return "\n".join(sections)

    # ── per-vulnerability advice ────────────────────────────────────────────
    def advise_one(self, code: str, vulnerability: Dict[str, Any]) -> RepairSuggestion:
        swc = vulnerability.get("swc") or "Unknown"
        title = vulnerability.get("title") or "Unknown vulnerability"
        severity = vulnerability.get("severity") or "Unknown"

        retrieved = self.knowledge_base.retrieve(swc)
        knowledge_block = self._build_knowledge_block(vulnerability, retrieved)
        prompt = self._build_prompt(code, vulnerability, knowledge_block)

        sources = list(dict.fromkeys(retrieved.get("titles", [])))[:3]

        try:
            data = self.llm.generate_json(
                prompt,
                ADVISOR_SYSTEM_INSTRUCTION,
                response_schema=REPAIR_RESPONSE_SCHEMA,
            )
        except Exception as error:  # noqa: BLE001 - reported per finding, never aborts the batch
            return RepairSuggestion(
                swc=swc,
                vulnerability_name=title,
                severity=severity,
                root_cause="",
                impact="",
                repair_steps=[],
                fixed_code="",
                prevention=[],
                knowledge_sources=sources,
                error=f"{type(error).__name__}: {error}",
            )

        def as_list(value: Any) -> List[str]:
            if isinstance(value, list):
                return [str(item).strip() for item in value if str(item).strip()]
            text = str(value or "").strip()
            return [text] if text else []

        return RepairSuggestion(
            swc=swc,
            vulnerability_name=str(data.get("vulnerability_name") or title).strip(),
            severity=severity,
            root_cause=str(data.get("root_cause") or "").strip(),
            impact=str(data.get("impact") or "").strip(),
            repair_steps=as_list(data.get("repair_steps")),
            fixed_code=str(data.get("fixed_code") or "").strip(),
            prevention=as_list(data.get("prevention")),
            knowledge_sources=sources,
        )

    # ── batch over a detection result ───────────────────────────────────────
    def advise(self, code: str, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        final = detection_result.get("final", {})
        vulnerabilities = detection_result.get("vulnerabilities", []) or []

        if not final.get("is_vulnerable") or not vulnerabilities:
            return {
                "agent": "Advisor (Repair Suggestion)",
                "input_verdict": final.get("verdict", "Unknown"),
                "input_risk_level": final.get("risk_level", "None"),
                "suggestion_count": 0,
                "suggestions": [],
                "note": "Detection phase reported no actionable vulnerabilities; nothing to repair.",
            }

        suggestions = [self.advise_one(code, vulnerability) for vulnerability in vulnerabilities]
        return {
            "agent": "Advisor (Repair Suggestion)",
            "input_verdict": final.get("verdict"),
            "input_risk_level": final.get("risk_level"),
            "llm_model": self.llm.model_name,
            "suggestion_count": len(suggestions),
            "suggestions": [asdict(item) for item in suggestions],
        }


def run_advisor(
    code: str,
    detection_result: Dict[str, Any],
    knowledge_dir: str | Path = DEFAULT_KNOWLEDGE_DIR,
    backend: str = "auto",
) -> Dict[str, Any]:
    agent = AdvisorAgent(knowledge_dir=knowledge_dir, backend=backend)
    return agent.advise(code, detection_result)
