"""
Phase 2 - run the Advisor over every confirmed finding of a contract.

For each finding the LLM Advisor (advisor.py) decides true / false positive and gives
root cause + fix strategy. A false positive is returned in `false_positives` and the
pipeline drops it from repair. When the LLM call fails (or the finding is beyond
MAX_LLM_ADVISOR_CALLS), the suggestion is built from the repair knowledge base for the
finding's SWC instead (no triage in that case).
"""

from __future__ import annotations

from typing import Any

from E2E.common.findings import finding_key
from E2E.config import MAX_LLM_ADVISOR_CALLS, RISK_ORDER
from E2E.phase2_advisor.advisor import AdvisorAgent, RepairKnowledgeBase


# [HÀM CHÍNH PHASE 2] Chạy Advisor cho các finding của contract.
#   1) Gọi LLM cho tối đa MAX_LLM_ADVISOR_CALLS (40, config.py) finding, ưu tiên mức nghiêm trọng cao.
#   2) Finding LLM kết luận is_true_positive=false -> đưa vào false_positives (pipeline loại khỏi bước sửa).
#   3) Finding LLM trả lời được -> suggestion có root cause, impact, fix_strategy.
#   4) Finding LLM lỗi/không gọi -> suggestion dựng từ RepairKnowledgeBase theo SWC (không triage).
# knowledge_base: KB dùng chung của gpu_pipeline (không truyền thì tự nạp).
# Trả về {suggestions, false_positives, false_positive_reasons} (lý do = root_cause của LLM, đưa vào mục 4.5 báo cáo).
# Gọi bởi: gpu_pipeline.run_server_gpu_pipeline (use_llm=True); test dùng use_llm=False (không gọi LLM).
def _generate_advisor_section(
    findings: list[dict[str, Any]],
    code: str = "",
    *,
    use_llm: bool = False,
    knowledge_base: RepairKnowledgeBase | None = None,
) -> dict[str, Any]:
    kb = knowledge_base or RepairKnowledgeBase()
    agent = AdvisorAgent(knowledge_base=kb) if use_llm else None

    # 1. Query the Advisor LLM, highest severity first
    llm_advisor_results = {}
    false_positives: list[str] = []
    false_positive_reasons: dict[str, str] = {}
    if agent is not None and findings:
        findings_to_query = sorted(
            findings,
            key=lambda item: RISK_ORDER.get(item.get("severity", "Medium"), 0),
            reverse=True,
        )[:MAX_LLM_ADVISOR_CALLS or None]
        for finding in findings_to_query:
            try:
                result = agent.advise_one(code, finding)
            except Exception:  # noqa: BLE001 - e.g. the model returned a JSON list; this finding gets the KB suggestion
                continue
            if not result.error:
                llm_advisor_results[finding_key(finding)] = result

    # Phía sau chỉ đọc: finding_id, swc, unit_id (gpu_pipeline / Assessor), root_cause, impact, repair_steps (Assessor,
    # report), fix_strategy + is_true_positive (Fixer: chỉ dùng suggestion đã được LLM xác nhận).
    suggestions = []
    for f in findings:
        swc = f.get("swc", "SWC-100")
        unit_id = f.get("unit_id") or "contract"
        title = f.get("title") or f"Vulnerability {swc}"
        remediation = f.get("remediation") or f"Implement secure defense pattern for {title}."
        description = f.get("description") or f"Identified security vulnerability matching {swc}."
        fid = finding_key(f)

        # 1. LLM verdict available
        item_sug = llm_advisor_results.get(fid)
        if item_sug:
            if not item_sug.is_true_positive:
                # Advisor triage: the caller must drop this finding from repair and reporting.
                false_positives.append(fid)
                false_positive_reasons[fid] = item_sug.root_cause
                continue
            suggestions.append({
                "finding_id": fid,
                "swc": swc,
                "unit_id": unit_id,
                "root_cause": item_sug.root_cause,
                "impact": item_sug.impact,
                "repair_steps": [item_sug.fix_strategy] if item_sug.fix_strategy else [remediation],
                "fix_strategy": item_sug.fix_strategy,
                "is_true_positive": True,
            })
            continue

        # 2. No LLM verdict: repair steps from the knowledge base for this SWC (no triage)
        repair_steps = [
            f"Identify the vulnerable state transition or external interaction in `{unit_id}`.",
            "Apply guard conditions or proper authorization checks before modifying state.",
            remediation,
        ]
        repair_steps.extend(kb.retrieve(swc, code=code).get("remediations", [])[:2])

        suggestions.append({
            "finding_id": fid,
            "swc": swc,
            "unit_id": unit_id,
            "root_cause": f.get("root_cause") or description or f"Vulnerable code structure matching pattern for {swc}.",
            "impact": f.get("impact") or f"High risk of unauthorized state modification or asset drain via {title}.",
            "repair_steps": repair_steps,
        })

    return {
        "status": "ok",
        "suggestions": suggestions,
        "false_positives": false_positives,
        "false_positive_reasons": false_positive_reasons,
    }
