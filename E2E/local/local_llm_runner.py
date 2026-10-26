from __future__ import annotations

import concurrent.futures
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

from E2E.config import DEFAULT_DETECTOR_THRESHOLD
from E2E.shared.function_parser import CodeUnit, apply_unit_patches, parse_code_units, validate_observable_behavior
from E2E.agents.advisor import AdvisorAgent, RepairKnowledgeBase
from E2E.agents.assessor import AssessorAgent, compute_cvss_31_base_score, cvss_to_level


OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "codellama:7b"
MAX_LLM_ADVISOR_CALLS = int(os.getenv("MAX_LLM_ADVISOR_CALLS", "5"))
MAX_LLM_ASSESSOR_CALLS = int(os.getenv("MAX_LLM_ASSESSOR_CALLS", "5"))


def _query_ollama(
    prompt: str,
    *,
    model: str = DEFAULT_OLLAMA_MODEL,
    temperature: float = 0.0,
    timeout: int = 60,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 1200,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_ENDPOINT,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            res_json = json.loads(response.read().decode("utf-8"))
            return str(res_json.get("response", "")).strip()
    except Exception as exc:
        raise ConnectionError(
            f"Failed to connect to local Ollama service at {OLLAMA_ENDPOINT} ({exc}). "
            "Please ensure Ollama is running (`ollama serve`)."
        ) from exc


def finding_key(finding: dict[str, Any]) -> str:
    """Stable identifier shared by fusion, Advisor, Assessor and Verifier outputs."""
    return str(
        finding.get("finding_id")
        or f"{finding.get('swc') or 'SWC-100'}:{finding.get('unit_id') or 'contract'}"
    )


def _derive_findings(request: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    local_p1 = request.get("local_phase1", {})
    for src in ("static", "rag", "provisional_fusion"):
        for f in local_p1.get(src, {}).get("findings", []):
            findings.append(dict(f))
    # Deduplicate findings by (swc, unit_id) - bỏ line để tránh RAG và Static nhân đôi finding
    seen = set()
    unique_findings = []
    for f in findings:
        key = (f.get("swc"), f.get("unit_id"))
        if key not in seen:
            seen.add(key)
            unique_findings.append(f)
    return unique_findings


SWC_CVSS_MAP: dict[str, dict[str, Any]] = {
    "SWC-107": {
        "severity": "Critical",
        "cvss_score": 8.6,
        "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:H/A:H",
        "likelihood": "High",
        "impact_rating": "Critical",
        "funds_at_risk": True,
        "justification": "Reentrant call before state deduction allows malicious contracts to drain funds recursively.",
        "repair_priority": 1,
    },
    "SWC-105": {
        "severity": "Critical",
        "cvss_score": 8.6,
        "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:H/A:H",
        "likelihood": "High",
        "impact_rating": "Critical",
        "funds_at_risk": True,
        "justification": "Unprotected state-changing function allows any external account to drain contract funds or change owner.",
        "repair_priority": 1,
    },
    "SWC-115": {
        "severity": "High",
        "cvss_score": 7.5,
        "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:U/C:N/I:H/A:H",
        "likelihood": "Medium",
        "impact_rating": "High",
        "funds_at_risk": True,
        "justification": "Authorization check using tx.origin can be bypassed via phishing attacks with intermediate contracts.",
        "repair_priority": 1,
    },
    "SWC-104": {
        "severity": "High",
        "cvss_score": 7.5,
        "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:H/A:N",
        "likelihood": "Medium",
        "impact_rating": "High",
        "funds_at_risk": True,
        "justification": "Failure to check return values of low-level calls allows execution to proceed after failed transactions.",
        "repair_priority": 2,
    },
    "SWC-101": {
        "severity": "High",
        "cvss_score": 7.5,
        "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:H/A:N",
        "likelihood": "Medium",
        "impact_rating": "High",
        "funds_at_risk": True,
        "justification": "Integer overflow/underflow in arithmetic calculations causes incorrect token or balance accounting.",
        "repair_priority": 2,
    },
    "SWC-114": {
        "severity": "High",
        "cvss_score": 7.0,
        "cvss_vector": "CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:U/C:N/I:H/A:N",
        "likelihood": "Medium",
        "impact_rating": "High",
        "funds_at_risk": True,
        "justification": "Transaction ordering dependence allows miners or front-running bots to extract value.",
        "repair_priority": 2,
    },
    "SWC-116": {
        "severity": "Medium",
        "cvss_score": 5.3,
        "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:L/A:N",
        "likelihood": "Medium",
        "impact_rating": "Low",
        "funds_at_risk": False,
        "justification": "Relying on block.timestamp allows miners to manipulate timestamps within small drift windows.",
        "repair_priority": 3,
    },
    "SWC-120": {
        "severity": "Medium",
        "cvss_score": 5.3,
        "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:L/A:N",
        "likelihood": "Medium",
        "impact_rating": "Low",
        "funds_at_risk": False,
        "justification": "Blockhash dependence for randomness can be predicted or biased by miners.",
        "repair_priority": 3,
    },
    "SWC-102": {
        "severity": "Low",
        "cvss_score": 3.1,
        "cvss_vector": "CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:U/C:N/I:L/A:N",
        "likelihood": "Low",
        "impact_rating": "Low",
        "funds_at_risk": False,
        "justification": "Outdated compiler pragma version may contain unpatched compiler bugs.",
        "repair_priority": 4,
    },
}


def _with_computed_score(meta: dict[str, Any]) -> dict[str, Any]:
    """Derive the CVSS v3.1 base score and its qualitative rating from the vector itself."""
    metrics = dict(part.split(":", 1) for part in meta["cvss_vector"].split("/")[1:])
    score, vector = compute_cvss_31_base_score(**{key.lower(): value for key, value in metrics.items()})
    return {**meta, "cvss_score": score, "cvss_vector": vector, "severity": cvss_to_level(score)}


def get_cvss_defaults(swc: str, severity: str = "Medium") -> dict[str, Any]:
    return _with_computed_score(_cvss_template(swc, severity))


def _cvss_template(swc: str, severity: str = "Medium") -> dict[str, Any]:
    if swc in SWC_CVSS_MAP:
        return SWC_CVSS_MAP[swc]
    sev = str(severity or "Medium").capitalize()
    if sev in {"Critical"}:
        return {
            "severity": "Critical",
            "cvss_score": 8.6,
            "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:H/A:H",
            "likelihood": "High",
            "impact_rating": "Critical",
            "funds_at_risk": True,
            "justification": "Critical vulnerability posing direct threat to contract integrity or funds.",
            "repair_priority": 1,
        }
    if sev in {"High"}:
        return {
            "severity": "High",
            "cvss_score": 7.5,
            "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:H/A:N",
            "likelihood": "Medium",
            "impact_rating": "High",
            "funds_at_risk": True,
            "justification": "High severity flaw that could lead to significant unintended behavior or loss.",
            "repair_priority": 2,
        }
    if sev in {"Low"}:
        return {
            "severity": "Low",
            "cvss_score": 3.1,
            "cvss_vector": "CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:U/C:N/I:L/A:N",
            "likelihood": "Low",
            "impact_rating": "Low",
            "funds_at_risk": False,
            "justification": "Low severity code quality or best-practice violation.",
            "repair_priority": 4,
        }
    return {
        "severity": "Medium",
        "cvss_score": 5.3,
        "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:L/A:N",
        "likelihood": "Medium",
        "impact_rating": "Medium",
        "funds_at_risk": False,
        "justification": "Medium severity vulnerability requiring code remediation.",
        "repair_priority": 3,
    }


def _generate_advisor_section(
    findings: list[dict[str, Any]],
    code: str = "",
    *,
    backend: str | None = None,
    use_llm: bool = False,
) -> dict[str, Any]:
    suggestions = []
    recommendations = [
        "Enforce strict access control modifiers (e.g. onlyOwner) on state-changing administrative functions.",
        "Adhere to the Checks-Effects-Interactions pattern and consider nonReentrant guards for external calls.",
        "Validate all cryptographic signatures against replay attacks by including a unique nonce and chainId.",
    ]
    kb = None
    try:
        kb = RepairKnowledgeBase()
    except Exception:
        pass

    agent = None
    if use_llm and backend:
        try:
            agent = AdvisorAgent(knowledge_base=kb, backend=backend)
        except Exception:
            pass

    # 1. Query Advisor LLM for findings sorted by severity
    llm_advisor_results = {}
    false_positives: list[str] = []
    if agent is not None and findings:
        risk_weights = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1, "Info": 0}
        findings_to_query = sorted(
            findings,
            key=lambda item: risk_weights.get(item.get("severity", "Medium"), 0),
            reverse=True,
        )[:MAX_LLM_ADVISOR_CALLS]

        def _call_advisor(finding_dict):
            f_key = finding_key(finding_dict)
            try:
                res = agent.advise_one(code, finding_dict)
                return f_key, res
            except Exception:
                return f_key, None

        for finding_dict in findings_to_query:
            f_key, res = _call_advisor(finding_dict)
            if res and not res.error:
                llm_advisor_results[f_key] = res

    for f in findings:
        swc = f.get("swc", "SWC-100")
        unit_id = f.get("unit_id") or "contract"
        title = f.get("title") or f"Vulnerability {swc}"
        remediation = f.get("remediation") or f"Implement secure defense pattern for {title}."
        description = f.get("description") or f"Identified security vulnerability matching {swc}."
        fixed_code = f.get("fixed_code") or ""
        vulnerable_code = f.get("vulnerable_code") or ""
        checklist = list(f.get("checklist") or ([remediation] if remediation else []))
        fid = finding_key(f)

        # 1. Check if LLM result succeeded
        item_sug = llm_advisor_results.get(fid)
        if item_sug:
            if not item_sug.is_true_positive:
                # Advisor triage: the caller must drop this finding from repair and reporting.
                false_positives.append(fid)
                continue
            suggestions.append({
                "finding_id": fid,
                "swc": swc,
                "unit_id": unit_id,
                "title": item_sug.vulnerability_name or title,
                "severity": item_sug.severity or f.get("severity", "Medium"),
                "root_cause": item_sug.root_cause,
                "impact": item_sug.impact,
                "repair_steps": item_sug.repair_steps or [remediation],
                "fixed_code": item_sug.fixed_code or fixed_code,
                "vulnerable_code": vulnerable_code,
                "checklist": item_sug.prevention or checklist,
                "prevention": item_sug.prevention or checklist,
                "recommendations": [remediation],
                "fix_strategy": item_sug.fix_strategy,
                "is_true_positive": True,
            })
            continue

        # 2. Grounded generation via RepairKnowledgeBase
        retrieved_rems = []
        retrieved_checks = []
        if kb is not None:
            try:
                ret = kb.retrieve(swc, code=code)
                retrieved_rems = ret.get("remediations", [])
                retrieved_checks = ret.get("checklist", [])
                if not fixed_code and ret.get("code_examples"):
                    fixed_code = ret["code_examples"][0]
            except Exception:
                pass

        repair_steps = [
            f"Identify the vulnerable state transition or external interaction in `{unit_id}`.",
            f"Apply guard conditions or proper authorization checks before modifying state.",
            remediation,
        ]
        if retrieved_rems:
            repair_steps.extend(retrieved_rems[:2])

        prevention_list = list(checklist)
        if retrieved_checks:
            prevention_list.extend(retrieved_checks[:3])
        if not prevention_list:
            prevention_list = [
                f"Adhere to established OpenZeppelin security patterns for {swc}.",
                "Perform automated symbolic execution and fuzz testing before mainnet deployment.",
            ]

        suggestions.append({
            "finding_id": fid,
            "swc": swc,
            "unit_id": unit_id,
            "title": title,
            "severity": f.get("severity", "Medium"),
            "root_cause": f.get("root_cause") or description or f"Vulnerable code structure matching pattern for {swc}.",
            "impact": f.get("impact") or f"High risk of unauthorized state modification or asset drain via {title}.",
            "repair_steps": repair_steps,
            "fixed_code": fixed_code,
            "vulnerable_code": vulnerable_code,
            "checklist": prevention_list,
            "prevention": prevention_list,
            "recommendations": [remediation],
        })

    return {
        "status": "ok",
        "suggestions": suggestions,
        "false_positives": false_positives,
        "recommendations": recommendations,
    }


def _generate_assessor_section(
    findings: list[dict[str, Any]],
    code: str = "",
    *,
    backend: str | None = None,
    use_llm: bool = False,
    advisor_suggestions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    assessments = []
    agent = None
    if use_llm and backend:
        try:
            agent = AssessorAgent(backend=backend)
        except Exception:
            pass

    # Map suggestions by finding_id and SWC to carry forward Phase 2 insights
    suggestions_by_id = {s.get("finding_id"): s for s in (advisor_suggestions or []) if s.get("finding_id")}
    suggestions_by_swc = {s.get("swc"): s for s in (advisor_suggestions or []) if s.get("swc")}

    # 1. Parallelize LLM risk assessments for top findings
    llm_assessor_results = {}
    if agent is not None and findings:
        findings_to_query = findings[:MAX_LLM_ASSESSOR_CALLS]
        def _call_assessor(finding_dict):
            f_key = finding_key(finding_dict)
            f_enr = dict(finding_dict)
            s = suggestions_by_id.get(finding_dict.get("finding_id")) or suggestions_by_swc.get(finding_dict.get("swc")) or {}
            if s:
                if s.get("root_cause") and not f_enr.get("root_cause"):
                    f_enr["root_cause"] = s["root_cause"]
                if s.get("impact") and not f_enr.get("impact"):
                    f_enr["impact"] = s["impact"]
            try:
                res = agent.assess_one(code, f_enr)
                return f_key, res
            except Exception:
                return f_key, None

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(findings_to_query), 4)) as executor:
            for f_key, res in executor.map(_call_assessor, findings_to_query):
                if res and getattr(res, "cvss_score", 0) > 0 and not getattr(res, "error", None):
                    llm_assessor_results[f_key] = res

    for f in findings:
        swc = f.get("swc", "SWC-100")
        unit_id = f.get("unit_id") or "contract"
        title = f.get("title") or f"Vulnerability {swc}"
        severity = f.get("severity") or ("High" if swc in {"SWC-107", "SWC-105", "SWC-115"} else "Medium")
        fid = finding_key(f)

        # 1. Check if parallel LLM risk assessment succeeded
        res = llm_assessor_results.get(fid)
        if res:
            assessments.append({
                "finding_id": fid,
                "swc": swc,
                "unit_id": unit_id,
                "title": title,
                "severity": res.risk_level or severity,
                "risk_level": res.risk_level or severity,
                "impact": res.justification or f.get("impact") or f.get("description") or f"Risk evaluated with CVSS score {res.cvss_score} ({res.risk_level}).",
                "cvss_score": res.cvss_score,
                "cvss_vector": res.cvss_vector,
                "likelihood": res.likelihood,
                "impact_rating": res.impact_rating,
                "funds_at_risk": res.funds_at_risk,
                "justification": res.justification,
                "repair_priority": res.repair_priority or 1,
            })
            continue

        # 2. Grounded CVSS calculation based on calibrated SWC standard
        cvss_meta = get_cvss_defaults(swc, severity)
        assessments.append({
            "finding_id": fid,
            "swc": swc,
            "unit_id": unit_id,
            "title": title,
            "severity": cvss_meta["severity"],
            "risk_level": cvss_meta["severity"],
            "impact": f"High risk of unauthorized state modification or asset drain via {title}.",
            "cvss_score": cvss_meta["cvss_score"],
            "cvss_vector": cvss_meta["cvss_vector"],
            "likelihood": cvss_meta["likelihood"],
            "impact_rating": cvss_meta["impact_rating"],
            "funds_at_risk": cvss_meta["funds_at_risk"],
            "justification": cvss_meta["justification"],
            "repair_priority": cvss_meta["repair_priority"],
        })

    assessments.sort(key=lambda item: item.get("repair_priority", 100))
    for idx, item in enumerate(assessments, 1):
        item["repair_priority"] = idx

    return {
        "status": "ok",
        "assessments": assessments,
    }


def _to_code_unit(unit: Any) -> CodeUnit | None:
    if isinstance(unit, CodeUnit):
        return unit
    if not isinstance(unit, dict):
        return None
    code = str(unit.get("code") or "")
    return CodeUnit(
        unit_id=unit.get("unit_id", "unknown"),
        kind=unit.get("kind", "function"),
        name=unit.get("name", "func"),
        contract_name=unit.get("contract_name", "Contract"),
        start_offset=unit.get("start_offset", 0),
        end_offset=unit.get("end_offset", len(code)),
        start_line=unit.get("start_line", 1),
        end_line=unit.get("end_line", 1),
        code=code,
        has_body=unit.get("has_body", True),
        is_view_or_pure=unit.get("is_view_or_pure", False),
        visibility=unit.get("visibility", "public"),
    )


def _changed_unit_outcomes(original: str, candidate: str) -> list[dict[str, Any]]:
    """Describe a whole-contract rewrite as per-unit outcomes so Phase 5 can verify each unit."""
    before = {unit.unit_id: unit.code for unit in parse_code_units(original)}
    after = {unit.unit_id: unit.code for unit in parse_code_units(candidate)}
    outcomes = []
    for unit_id, code in before.items():
        if unit_id not in after:
            outcomes.append({"finding_id": f"LLM:{unit_id}", "unit_id": unit_id, "status": "applied",
                             "engine": "llm_whole_contract", "reason": f"Model rewrite removed {unit_id}."})
        elif re.sub(r"\s+", "", after[unit_id]) != re.sub(r"\s+", "", code):
            outcomes.append({"finding_id": f"LLM:{unit_id}", "unit_id": unit_id, "status": "applied",
                             "engine": "llm_whole_contract", "reason": f"Model rewrite changed {unit_id}."})
    return outcomes


def _normalize_ws(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


# A knowledge-base snippet shorter than this is too generic to be trusted as an exact match.
MIN_EMPIRICAL_SNIPPET_CHARS = 40


def _apply_heuristic_security_patch(
    code: str,
    findings: list[dict[str, Any]],
    code_units: list[Any] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Deterministic fallback used when the generative fixer produced no applicable patch.

    1. Empirical RAG patch: only when the knowledge-base vulnerable snippet matches the
       contract text exactly, or the whole unit; a replacement function must keep the
       unit's name and pass the observable-behaviour guard.
    2. Advisor fixed code for a unit (same guard).
    3. Rule-based fallback for two textbook patterns. These outcomes are tagged
       `engine=rule_based_fallback` so reports never present them as model output.
    """
    units = [unit for unit in (_to_code_unit(item) for item in code_units or []) if unit is not None]
    units_by_id = {unit.unit_id: unit for unit in units}
    fixed_code = code
    patch_outcomes: list[dict[str, Any]] = []
    patched_units: set[str] = set()

    def splice(unit: CodeUnit, replacement: str, summary: str) -> bool:
        nonlocal fixed_code
        if unit.unit_id in patched_units or unit.name not in replacement:
            return False
        if re.sub(r"\s+", "", replacement) == re.sub(r"\s+", "", unit.code):
            return False
        try:
            validate_observable_behavior(unit.code, replacement)
        except ValueError:
            return False
        updated, outcomes = apply_unit_patches(
            fixed_code,
            [unit],
            [{"unit_id": unit.unit_id, "replacement": replacement, "summary": summary}],
        )
        if not any(item.get("status") == "applied" for item in outcomes):
            return False
        fixed_code = updated
        patched_units.add(unit.unit_id)
        return True

    for f in findings:
        v_code = str(f.get("vulnerable_code") or "").strip()
        f_code = str(f.get("fixed_code") or "").strip()
        uid = f.get("unit_id")
        unit = units_by_id.get(uid) if uid else None
        if not f_code or v_code == f_code:
            continue
        if v_code:
            if len(v_code) >= MIN_EMPIRICAL_SNIPPET_CHARS and v_code in fixed_code:
                fixed_code = fixed_code.replace(v_code, f_code, 1)
                patch_outcomes.append({
                    "finding_id": finding_key(f),
                    "unit_id": uid,
                    "status": "applied",
                    "engine": "rag_empirical_patch",
                    "reason": f"Applied verified RAG patch from knowledge item {f.get('best_match_id', '')} ({f.get('swc', '')}).",
                })
            elif unit is not None and _normalize_ws(v_code) == _normalize_ws(unit.code):
                if splice(unit, f_code, f"RAG patch for {uid}"):
                    patch_outcomes.append({
                        "finding_id": finding_key(f),
                        "unit_id": uid,
                        "status": "applied",
                        "engine": "rag_empirical_patch",
                        "reason": f"Applied unit-level RAG patch for {uid}.",
                    })
        elif unit is not None and splice(unit, f_code, f"Advisor patch for {uid}"):
            patch_outcomes.append({
                "finding_id": finding_key(f),
                "unit_id": uid,
                "status": "applied",
                "engine": "advisor_fixed_code",
                "reason": f"Applied Advisor remediation patch for {uid}.",
            })

    if patch_outcomes:
        return fixed_code, patch_outcomes

    def unit_named(name: str) -> CodeUnit | None:
        matches = [unit for unit in units if unit.name == name]
        return matches[0] if len(matches) == 1 else None

    # Rule-based fallback 1: missing onlyOwner on setOwner.
    if any(f.get("swc") in {"SWC-105", "SWC-100"} for f in findings):
        if "onlyOwner" in code and "address public owner;" in code:
            pattern = r"function\s+setOwner\s*\(([^)]*)\)\s*public\s*\{"
            if re.search(pattern, fixed_code):
                fixed_code = re.sub(pattern, r"function setOwner(\1) public onlyOwner {", fixed_code)
                unit = unit_named("setOwner")
                patch_outcomes.append({
                    "finding_id": f"SWC-105:{unit.unit_id if unit else 'setOwner'}",
                    "unit_id": unit.unit_id if unit else None,
                    "status": "applied",
                    "engine": "rule_based_fallback",
                    "reason": "Rule-based fallback: added onlyOwner modifier to setOwner.",
                })

    # Rule-based fallback 2: checks-effects-interactions for a value call followed by a balance update.
    if any(f.get("swc") == "SWC-107" for f in findings):
        pattern = re.compile(
            r"(\((?:bool\s+[A-Za-z_]\w*|)\s*,\s*(?:bytes\s+memory\s+[A-Za-z_]\w*|)\)\s*=\s*msg\.sender\.call\{value:\s*[^}]+\}\([^)]*\);\s*require\([^;]+\);)(\s*)(balances\[msg\.sender\]\s*(?:-=\s*[^;]+|=\s*0);)",
            re.MULTILINE,
        )
        match = pattern.search(fixed_code)
        if match:
            fixed_code = pattern.sub(r"\3\2\1", fixed_code)
            unit = next(
                (u for u in units if u.start_offset <= match.start() < u.end_offset),
                None,
            )
            patch_outcomes.append({
                "finding_id": f"SWC-107:{unit.unit_id if unit else 'contract'}",
                "unit_id": unit.unit_id if unit else None,
                "status": "applied",
                "engine": "rule_based_fallback",
                "reason": "Rule-based fallback: moved the balance update before the external call (CEI).",
            })

    return fixed_code, patch_outcomes


def run_local_llm_pipeline(
    request: dict[str, Any],
    *,
    backend: str = "local",
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
) -> dict[str, Any]:
    """Runs the complete CodeLlama / Local LLM pipeline on Mac M2."""
    started_at = time.perf_counter()
    run_id = request.get("run_id", "local-run")
    source_code = request.get("source", {}).get("code", "")
    code_units = request.get("code_units", [])
    findings = _derive_findings(request)
    detector_threshold = float(
        request.get("execution", {}).get("detector_threshold") or DEFAULT_DETECTOR_THRESHOLD
    )

    # 1. Detector results on Code Units - Local Ollama / heuristic backend does not run neural detector
    unit_results = []
    for unit in code_units:
        unit_id = unit.get("unit_id", "unknown")
        unit_results.append({
            "unit_id": unit_id,
            "label": "NotRun",
            "status": "not_run",
            "vulnerability_probability": None,
            "threshold": detector_threshold,
            "start_line": unit.get("start_line"),
            "end_line": unit.get("end_line"),
        })

    is_contract_vuln = len(findings) > 0
    print(f"[{time.strftime('%H:%M:%S')}] [PHASE 1C] Neural detector not run on local heuristic backend (detector: not_run).", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] [PHASE 1 FUSED] Synchronizing multi-modal detection signals...", flush=True)

    # Truy xuất cơ sở tri thức lỗ hổng (11,869 cases + 37 SWC registry) theo LLM-BSCVM
    try:
        from E2E.agents.advisor import RepairKnowledgeBase
        kb = RepairKnowledgeBase()
        findings = kb.enrich_findings(findings, code=source_code, code_units=code_units)
    except Exception:
        pass

    print(f"[{time.strftime('%H:%M:%S')}] [PHASE 1 FUSED] Completed: Multi-modal signals synchronized ({len(findings)} finding(s)).", flush=True)

    # 2. Advisor, Assessor, and Fixer
    print(f"[{time.strftime('%H:%M:%S')}] [PHASE 2] Running Security Advisor agent (root cause & remediation analysis)...", flush=True)
    advisor = _generate_advisor_section(findings, source_code, backend=backend, use_llm=(backend == "ollama"))
    print(f"[{time.strftime('%H:%M:%S')}] [PHASE 2] Completed: Security recommendations generated ({len(advisor.get('suggestions', []))} suggestion(s)).", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] [PHASE 3] Running Risk Assessor agent (CVSS v3.1 scoring & priority ranking)...", flush=True)
    assessor = _generate_assessor_section(
        findings,
        source_code,
        backend=backend,
        use_llm=(backend == "ollama"),
        advisor_suggestions=advisor.get("suggestions", []),
    )
    print(f"[{time.strftime('%H:%M:%S')}] [PHASE 3] Completed: Risk assessment finished ({len(assessor.get('assessments', []))} item(s) assessed).", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] [PHASE 4] Running Fixer agent (synthesizing secure Solidity patch)...", flush=True)
    patch_candidates = list(findings) + advisor.get("suggestions", [])
    fixed_code, patch_outcomes = _apply_heuristic_security_patch(source_code, patch_candidates, code_units=code_units)
    print(f"[{time.strftime('%H:%M:%S')}] [PHASE 4] Completed: Security patch synthesized ({len(patch_outcomes)} outcome(s) generated).", flush=True)

    # If backend is ollama and server is reachable, optionally enrich the patch
    if backend == "ollama":
        try:
            ref_patterns = []
            for f in findings:
                if f.get("fixed_code") and f.get("vulnerable_code"):
                    ref_patterns.append(
                        f"Vulnerable pattern:\n```solidity\n{f['vulnerable_code']}\n```\n"
                        f"Fixed pattern:\n```solidity\n{f['fixed_code']}\n```"
                    )
            ref_str = ("\n\nReference Fix Patterns from 11,869-case knowledge base:\n" + "\n\n".join(ref_patterns)) if ref_patterns else ""
            rag_ctx = request.get("local_phase1", {}).get("rag", {}).get("retrieved_context", [])
            rag_section = ""
            if rag_ctx:
                rag_items = []
                for ctx in rag_ctx[:3]:
                    if float(ctx.get("similarity") or 0.0) >= 0.08:
                        c_swc = ctx.get("swc") or "Security"
                        c_title = ctx.get("title") or c_swc
                        c_desc = (ctx.get("root_cause") or ctx.get("description") or "")[:200]
                        rag_items.append(f"- [{c_swc}] {c_title}: {c_desc}")
                if rag_items:
                    rag_section = "\n\nRAG Retrieved Security Context:\n" + "\n".join(rag_items)

            prompt = (
                f"You are a smart contract security expert. The following Solidity code has vulnerabilities:\n"
                f"{source_code}\n\n"
                f"Identified findings: {json.dumps(findings, indent=2)}\n"
                f"{ref_str}\n"
                f"{rag_section}\n\n"
                f"Please produce the complete secure fixed code for the contract inside ```solidity ... ``` fence."
            )
            llm_response = _query_ollama(prompt, model=ollama_model)
            if "contract " in llm_response and ("{" in llm_response and "}" in llm_response):
                # Extract code block if formatted with markdown
                match = re.search(r"```solidity\s*([\s\S]*?)\s*```", llm_response)
                if match and "contract " in match.group(1):
                    candidate = match.group(1).strip()
                    llm_outcomes = _changed_unit_outcomes(source_code, candidate)
                    if llm_outcomes:
                        fixed_code = candidate
                        patch_outcomes = llm_outcomes
        except Exception:
            pass  # Fall back gracefully to verified AST patch

    # The local backend has no detector re-scan: verification is left to Phase 5
    # (compiler, security tools, behaviour guard) instead of being asserted here.
    verifier_items = [
        {
            "finding_id": finding_key(f),
            "status": "NOT_VERIFIED",
            "residual_risk": "Unknown: no neural re-scan on the local backend",
            "verification_notes": "Deferred to Phase 5 deterministic verification.",
            "checklist": (f.get("checklist") or [])[:3],
        }
        for f in findings
    ]

    phases = {
        "detector": {
            "status": "not_run",
            "original": {
                "verdict": "Inconclusive",
                "vulnerability_probability": None,
                "threshold": detector_threshold,
                "unit_results": unit_results,
            },
            "fixed": {
                "verdict": "Inconclusive",
                "vulnerability_probability": None,
                "threshold": detector_threshold,
                "unit_results": unit_results,
            },
        },
        "advisor": advisor,
        "assessor": assessor,
        "fixer": {
            "status": "ok",
            "fixed_code": fixed_code,
            "patch_outcomes": patch_outcomes,
            "patches": [
                {
                    "finding_id": p["finding_id"],
                    "description": p["reason"],
                }
                for p in patch_outcomes
            ],
        },
        "verifier": {
            "status": "ok",
            "verifications": verifier_items,
            "verdict": "Not verified",
            "overall_verdict": "NOT_VERIFIED",
        },
        "reporter": {
            "status": "ok",
            "executive_summary": (
                f"Local audit completed for run {run_id}. Identified {len(findings)} findings. "
                f"Applied {len(patch_outcomes)} security patch(es)."
            ),
            "recommendations": advisor["recommendations"],
        },
    }

    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
    return {
        "schema_version": "1.0.0",
        "pipeline_version": "1.0.0",
        "status": "ok",
        "run_id": run_id,
        "backend": f"local_m2_{backend}",
        "source_sha256": request.get("source", {}).get("sha256", ""),
        "phases": phases,
        "timings": {
            "total_ms": elapsed_ms,
            "detector_ms": 0.0,
            "fixer_ms": elapsed_ms,
        },
    }
