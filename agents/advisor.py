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
DEFAULT_KNOWLEDGE_DIR = (
    PROJECT_ROOT / "E2E" / "data" / "runtime"
    if (PROJECT_ROOT / "E2E" / "data" / "runtime").exists()
    else PROJECT_ROOT / "rag" / "knowledge_store"
)

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

    This is the Advisor's RAG step: rather than re-running the heavy embedding
    model, it looks remediation knowledge up directly from the 11,869-case
    knowledge-store and 37-SWC registry by the SWC id the detector already identified.
    """

    def __init__(self, knowledge_dir: str | Path = DEFAULT_KNOWLEDGE_DIR) -> None:
        k_dir = Path(knowledge_dir)
        # Fall back to E2E/data/runtime if provided directory does not exist
        if not k_dir.exists():
            fallback = PROJECT_ROOT / "E2E" / "data" / "runtime"
            if fallback.exists():
                k_dir = fallback

        self._registry_by_swc: Dict[str, Dict[str, Any]] = {}
        self._items_by_swc: Dict[str, List[Dict[str, Any]]] = {}
        self._legacy_by_swc: Dict[str, List[Dict[str, Any]]] = {}

        registry_path = k_dir / "swc_registry.json"
        if registry_path.exists():
            try:
                with registry_path.open("r", encoding="utf-8") as f:
                    for record in json.load(f):
                        swc = normalize_swc_id(record.get("id"))
                        if swc:
                            self._registry_by_swc[swc] = record
            except Exception:
                pass

        items_path = k_dir / "knowledge_items.json"
        if items_path.exists():
            try:
                with items_path.open("r", encoding="utf-8") as f:
                    for record in json.load(f):
                        swc = normalize_swc_id(record.get("swc") or record.get("id"))
                        if swc:
                            self._items_by_swc.setdefault(swc, []).append(record)
            except Exception:
                pass

        metadata_path = k_dir / "sample_metadata.json"
        if metadata_path.exists():
            try:
                with metadata_path.open("r", encoding="utf-8") as f:
                    for record in json.load(f):
                        swc = normalize_swc_id(record.get("Swc") or record.get("swc"))
                        if swc:
                            self._legacy_by_swc.setdefault(swc, []).append(record)
            except Exception:
                pass

    def retrieve(self, swc: str, limit: int = 3, code: str | None = None) -> Dict[str, Any]:
        """Aggregate remediation knowledge for a given SWC id, prioritizing empirical cases closest to the input code."""
        norm_swc = normalize_swc_id(swc)
        remediations: List[str] = []
        checklist: List[str] = []
        code_examples: List[str] = []
        descriptions: List[str] = []
        titles: List[str] = []

        # 1. Canonical knowledge from swc_registry.json
        reg = self._registry_by_swc.get(norm_swc)
        if reg:
            title = str(reg.get("title") or norm_swc).strip()
            if title and title not in titles:
                titles.append(title)
            desc = str(reg.get("description") or reg.get("root_cause") or "").strip()
            if desc and desc not in descriptions:
                descriptions.append(desc)
            remed = str(reg.get("remediation") or "").strip()
            if remed and remed not in remediations:
                remediations.append(remed)
            for item in reg.get("checklist", []) or []:
                text = str(item or "").strip()
                if text and text not in checklist:
                    checklist.append(text)
            for s in reg.get("samples", []) or []:
                if isinstance(s, dict):
                    fixed = str(s.get("fixed") or "").strip()
                    if fixed and fixed not in code_examples:
                        code_examples.append(fixed)

        # 2. Granular empirical cases from knowledge_items.json (11,869 cases)
        items = list(self._items_by_swc.get(norm_swc, []))
        if code and items:
            from difflib import SequenceMatcher
            query_code = code.strip()

            def match_score(item: Dict[str, Any]) -> float:
                v_code = str(item.get("vulnerable_code") or "").strip()
                if not v_code:
                    return 0.0
                if v_code == query_code:
                    return 10.0
                if v_code in query_code or query_code in v_code:
                    return 5.0 + min(len(v_code), len(query_code)) / max(len(v_code), len(query_code))
                return SequenceMatcher(None, query_code[:500], v_code[:500]).ratio()

            items.sort(key=match_score, reverse=True)

        empirical_fixes: List[str] = []
        empirical_pairs: List[Dict[str, Any]] = []
        for item in items[:limit]:
            title = str(item.get("title") or "").strip()
            if title and title not in titles:
                titles.append(title)
            desc = str(item.get("description") or item.get("root_cause") or "").strip()
            if desc and desc not in descriptions:
                descriptions.append(desc)
            remed = str(item.get("remediation") or "").strip()
            if remed and remed not in remediations:
                remediations.append(remed)
            for c_item in item.get("checklist", []) or []:
                text = str(c_item or "").strip()
                if text and text not in checklist:
                    checklist.append(text)
            fixed = str(item.get("fixed_code") or "").strip()
            v_code = str(item.get("vulnerable_code") or "").strip()
            if fixed and fixed not in empirical_fixes:
                empirical_fixes.append(fixed)
            if fixed or v_code:
                empirical_pairs.append({
                    "id": item.get("id"),
                    "title": title or norm_swc,
                    "swc": norm_swc,
                    "vulnerable_code": v_code,
                    "fixed_code": fixed,
                    "root_cause": str(item.get("root_cause") or "").strip(),
                    "remediation": remed,
                    "checklist": item.get("checklist") or [],
                    "severity": str(item.get("severity") or (reg.get("severity") if reg else "Medium")),
                    "cwe_ids": item.get("cwe_ids") or [],
                })

        # Place the closest empirical fix at the top of code_examples
        code_examples = empirical_fixes + [x for x in code_examples if x not in empirical_fixes]

        # Fallback to registry samples if no empirical pairs were extracted
        if not empirical_pairs and reg and reg.get("samples"):
            for s in reg.get("samples", [])[:limit]:
                if isinstance(s, dict):
                    empirical_pairs.append({
                        "id": f"{norm_swc}_sample",
                        "title": titles[0] if titles else norm_swc,
                        "swc": norm_swc,
                        "vulnerable_code": str(s.get("vulnerable") or "").strip(),
                        "fixed_code": str(s.get("fixed") or "").strip(),
                        "root_cause": str(s.get("root_cause") or "").strip(),
                        "remediation": str(reg.get("remediation") or "").strip(),
                        "checklist": reg.get("checklist") or [],
                        "severity": str(reg.get("severity") or "Medium"),
                        "cwe_ids": [],
                    })

        best_pair = empirical_pairs[0] if empirical_pairs else {}

        # 3. Legacy sample_metadata.json records if any
        legacy_records = self._legacy_by_swc.get(norm_swc, [])
        for record in legacy_records[:limit]:
            for value, bucket in (
                (record.get("Remediation") or record.get("remediation"), remediations),
                (record.get("Description") or record.get("description"), descriptions),
                (record.get("Title") or record.get("title"), titles),
            ):
                text = str(value or "").strip()
                if text and text not in bucket:
                    bucket.append(text)
            for item in record.get("BestPracticeChecklist", []) or record.get("checklist", []) or []:
                text = str(item or "").strip()
                if text and text not in checklist:
                    checklist.append(text)
            for item in record.get("CodeExamples", []) or record.get("samples", []) or []:
                text = str(item or "").strip()
                if text and text not in code_examples:
                    code_examples.append(text)

        risk_guidance = {
            "swc": norm_swc,
            "title": titles[0] if titles else norm_swc,
            "benchmark_severity": reg.get("severity") if reg else (best_pair.get("severity") if best_pair else "Medium"),
            "cwe_ids": best_pair.get("cwe_ids", []) if best_pair else [],
            "remediation_summary": remediations[0] if remediations else "",
            "root_cause_summary": descriptions[0] if descriptions else "",
            "checklist": checklist[:6],
        }

        total_records = (1 if reg else 0) + len(items) + len(legacy_records)
        return {
            "titles": titles,
            "descriptions": descriptions,
            "remediations": remediations,
            "checklist": checklist,
            "code_examples": code_examples[:limit],
            "empirical_pairs": empirical_pairs,
            "best_pair": best_pair,
            "risk_guidance": risk_guidance,
            "record_count": total_records,
        }

    def enrich_finding(self, finding: Dict[str, Any], code: str = "") -> Dict[str, Any]:
        """Enrich a finding with the most relevant empirical code, fix patterns, and evaluation knowledge."""
        norm_swc = normalize_swc_id(finding.get("swc"))
        target_code = (
            finding.get("unit_code")
            or finding.get("vulnerable_code")
            or finding.get("matched_text")
            or code
        )
        retrieved = self.retrieve(norm_swc, limit=3, code=target_code)
        enriched = dict(finding)

        best_pair = retrieved.get("best_pair", {})
        if best_pair:
            if not enriched.get("vulnerable_code") and best_pair.get("vulnerable_code"):
                enriched["vulnerable_code"] = best_pair["vulnerable_code"]
            if not enriched.get("fixed_code") and best_pair.get("fixed_code"):
                enriched["fixed_code"] = best_pair["fixed_code"]
            if not enriched.get("remediation") and best_pair.get("remediation"):
                enriched["remediation"] = best_pair["remediation"]
            if not enriched.get("root_cause") and best_pair.get("root_cause"):
                enriched["root_cause"] = best_pair["root_cause"]

        if not enriched.get("remediation") and retrieved.get("remediations"):
            enriched["remediation"] = retrieved["remediations"][0]
        if not enriched.get("checklist") and retrieved.get("checklist"):
            enriched["checklist"] = retrieved["checklist"]
        if not enriched.get("root_cause") and retrieved.get("descriptions"):
            enriched["root_cause"] = retrieved["descriptions"][0]

        enriched["empirical_pairs"] = retrieved.get("empirical_pairs", [])
        enriched["risk_guidance"] = retrieved.get("risk_guidance", {})
        return enriched

    def enrich_findings(
        self,
        findings: List[Dict[str, Any]],
        code: str = "",
        code_units: List[Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """Batch-enrich all findings with empirical vulnerability knowledge from 11,869 cases."""
        unit_code_map = {}
        if code_units:
            for cu in code_units:
                uid = cu.get("unit_id") if isinstance(cu, dict) else getattr(cu, "unit_id", None)
                ucode = cu.get("code") if isinstance(cu, dict) else getattr(cu, "code", None)
                if uid and ucode:
                    unit_code_map[uid] = ucode

        enriched_list = []
        for f in findings:
            f_copy = dict(f)
            u_id = f_copy.get("unit_id")
            if u_id and u_id in unit_code_map and not f_copy.get("unit_code"):
                f_copy["unit_code"] = unit_code_map[u_id]
            enriched_list.append(self.enrich_finding(f_copy, code=code))
        return enriched_list


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

        # Empirical RAG case match from Phase 1B or knowledge retrieval
        best_pair = retrieved.get("best_pair", {})
        rag_fixed = str(vulnerability.get("fixed_code") or best_pair.get("fixed_code") or "").strip()
        rag_vuln = str(vulnerability.get("vulnerable_code") or best_pair.get("vulnerable_code") or "").strip()

        # ... merged with a fresh lookup from the vulnerability knowledge base.
        remediations = self._truncate(retrieved.get("remediations", []), 3, 800)
        checklist = self._truncate(
            [str(x) for x in (list(detection_checklist) + retrieved.get("checklist", []))], 8, 200
        )
        code_examples = self._truncate(retrieved.get("code_examples", []), 3, 800)
        if rag_fixed and rag_fixed not in code_examples:
            code_examples.insert(0, rag_fixed[:800])

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
        if rag_vuln and rag_fixed:
            lines.append(
                "Most Relevant Empirical Code Repair Pair (from 11,869-case knowledge base):\n"
                f"  Vulnerable pattern:\n```solidity\n{rag_vuln[:600]}\n```\n"
                f"  Validated fixed pattern:\n```solidity\n{rag_fixed[:600]}\n```"
            )
        elif code_examples:
            for idx, item in enumerate(code_examples[:2], 1):
                lines.append(f"Reference fix example {idx} (from vulnerability knowledge base):\n```solidity\n{item}\n```")
        lines.append("</RetrievedSecurityKnowledge>")
        return "\n".join(lines)

    @staticmethod
    def _related_code_block(vulnerability: Dict[str, Any]) -> str:
        parts: List[str] = []
        unit_id = vulnerability.get("unit_id")
        start_line = vulnerability.get("start_line") or vulnerability.get("line")
        end_line = vulnerability.get("end_line")
        vuln_code = vulnerability.get("vulnerable_code")
        matched_text = vulnerability.get("matched_text")

        if unit_id:
            parts.append(f"Target function unit: `{unit_id}`")
        if start_line:
            loc = f"Line {start_line}" + (f"-{end_line}" if end_line and end_line != start_line else "")
            parts.append(f"Source location: {loc}")
        if vuln_code:
            parts.append(f"Vulnerable function code:\n```solidity\n{vuln_code.strip()}\n```")
        elif matched_text:
            parts.append(f"Matched vulnerable snippet:\n```solidity\n{matched_text.strip()}\n```")

        for related in vulnerability.get("related_code", []) or []:
            window = (related.get("window") or {}).get("code")
            if window and window.strip() not in "".join(parts):
                parts.append(f"Related code context:\n```solidity\n{window.strip()}\n```")

        return "\n\n".join(parts).strip()

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

        target_code = (
            vulnerability.get("unit_code")
            or vulnerability.get("vulnerable_code")
            or vulnerability.get("matched_text")
            or code
        )
        retrieved = self.knowledge_base.retrieve(swc, code=target_code)
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
