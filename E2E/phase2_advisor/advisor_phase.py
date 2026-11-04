"""
Phase 2 - run the Advisor over every confirmed finding of a contract.

For each finding the LLM Advisor (advisor.py) decides true / false positive and gives
root cause + fix strategy. A false positive is returned in `false_positives` and the
pipeline drops it from repair. When the LLM call fails, the suggestion is built from
the repair knowledge base for the finding's SWC instead (no triage in that case).
"""

from __future__ import annotations

import os
from typing import Any

from E2E.common.findings import finding_key
from E2E.phase2_advisor.advisor import AdvisorAgent, RepairKnowledgeBase


# Findings per contract in the 2374-contract batch: median 7, p99 32, max 54; 0 = no limit.
MAX_LLM_ADVISOR_CALLS = int(os.getenv("MAX_LLM_ADVISOR_CALLS", "40"))


def _generate_advisor_section(
    findings: list[dict[str, Any]],
    code: str = "",
    *,
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
    if use_llm:
        try:
            agent = AdvisorAgent(knowledge_base=kb)
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
        )[:MAX_LLM_ADVISOR_CALLS or None]

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
            "Apply guard conditions or proper authorization checks before modifying state.",
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
