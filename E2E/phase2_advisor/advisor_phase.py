"""
Phase 2 - run the Advisor over every confirmed finding of a contract (recall-first triage).

Each finding gets one suggestion, unless it is dismissed:
  * LLM verdict confirmed / uncertain -> suggestion with the LLM root cause + fix strategy;
  * LLM verdict false_positive        -> dismissed (listed in `false_positives`) only when the check
    passes: a quoted guard must exist in the code and the finding must not be corroborated by
    static + RAG. Otherwise it is kept as "disputed" with a knowledge-base suggestion;
  * no LLM verdict (call failed or beyond MAX_LLM_ADVISOR_CALLS) -> knowledge-base suggestion.
An UNMAPPED (detector-only) finding also carries the class the Advisor named (`advisor_swc`).
"""

from __future__ import annotations

from typing import Any

from E2E.common.findings import finding_key
from E2E.common.regex_patterns import RE_QUOTE_ELLIPSIS, RE_WHITESPACE
from E2E.config import MAX_LLM_ADVISOR_CALLS, RISK_ORDER
from E2E.phase2_advisor.advisor import (
    VERDICT_FALSE_POSITIVE,
    AdvisorAgent,
    RepairKnowledgeBase,
    RepairSuggestion,
)


# Nhiệm vụ: kiểm tra đoạn guard mà LLM trích (guard_quote) có thật trong mã: bỏ dấu ` và mọi khoảng trắng, tách theo
# dấu lược "..." rồi yêu cầu MỌI mảnh (>= 3 ký tự) đều nằm trong mã đã bỏ khoảng trắng.
# Mục đích: chặn lý do bác bỏ bịa ra ("protected by an authorization modifier" khi hàm không có modifier nào).
def _quote_in_code(quote: str, code: str) -> bool:
    compact_code = RE_WHITESPACE.sub("", code)
    pieces = [RE_WHITESPACE.sub("", piece.replace("`", "")).strip(";") for piece in RE_QUOTE_ELLIPSIS.split(quote)]
    pieces = [piece for piece in pieces if len(piece) >= 3]
    return bool(pieces) and all(piece in compact_code for piece in pieces)


# Nhiệm vụ: quyết định một kết luận false_positive của LLM có được phép loại finding không. Trả về lý do GIỮ LẠI
# finding (-> "disputed"), hoặc None nếu chấp nhận loại:
#   1) static và RAG cùng báo (corroborated): một ý kiến LLM không bác được hai kênh độc lập -> giữ;
#   2) LLM trích guard_quote nhưng đoạn đó không có trong mã của hàm / contract -> giữ.
# guard_quote rỗng = LLM nói lỗi không có trong hàm (không dựa vào guard) -> chấp nhận.
def _dismissal_rejection(finding: dict[str, Any], result: RepairSuggestion, code: str) -> str | None:
    if finding.get("corroborated"):
        return "static rules and RAG both reported this finding; one Advisor verdict does not overrule two channels"
    if result.guard_quote and not _quote_in_code(result.guard_quote, f"{finding.get('unit_code') or ''}\n{code}"):
        return "the guard quoted by the Advisor does not appear in the code"
    return None


# Nhiệm vụ: dựng suggestion từ RepairKnowledgeBase theo SWC của finding (không có phân tích của LLM).
# Dùng khi không có kết luận LLM, và cho finding "disputed" (lý do bác bỏ của LLM không mô tả lỗi).
def _knowledge_base_suggestion(finding: dict[str, Any], kb: RepairKnowledgeBase, code: str) -> dict[str, Any]:
    swc = finding.get("swc", "SWC-100")
    unit_id = finding.get("unit_id") or "contract"
    title = finding.get("title") or f"Vulnerability {swc}"
    remediation = finding.get("remediation") or f"Implement secure defense pattern for {title}."
    description = finding.get("description") or f"Identified security vulnerability matching {swc}."
    return {
        "finding_id": finding_key(finding),
        "swc": swc,
        "unit_id": unit_id,
        "root_cause": finding.get("root_cause") or description,
        "impact": finding.get("impact") or f"High risk of unauthorized state modification or asset drain via {title}.",
        "repair_steps": [
            f"Identify the vulnerable state transition or external interaction in `{unit_id}`.",
            "Apply guard conditions or proper authorization checks before modifying state.",
            remediation,
            *kb.retrieve(swc, code=code).get("remediations", [])[:2],
        ],
    }


# Nhiệm vụ: dựng suggestion từ phân tích của LLM (verdict confirmed / uncertain): root cause, impact, fix_strategy
# (Fixer dùng), advisor_verdict (report hiển thị). Không có fix_strategy -> bước sửa = remediation của finding.
def _llm_suggestion(finding: dict[str, Any], result: RepairSuggestion) -> dict[str, Any]:
    swc = finding.get("swc", "SWC-100")
    title = finding.get("title") or f"Vulnerability {swc}"
    return {
        "finding_id": finding_key(finding),
        "swc": swc,
        "unit_id": finding.get("unit_id") or "contract",
        "root_cause": result.root_cause,
        "impact": result.impact,
        "repair_steps": [result.fix_strategy or finding.get("remediation") or f"Implement secure defense pattern for {title}."],
        "fix_strategy": result.fix_strategy,
        "advisor_verdict": result.verdict,
    }


# [HÀM CHÍNH PHASE 2] Chạy Advisor cho các finding của contract.
#   1) use_llm: gọi LLM cho tối đa MAX_LLM_ADVISOR_CALLS (40, config.py) finding, ưu tiên mức nghiêm trọng cao;
#      lời gọi lỗi -> finding đó không có kết luận LLM.
#   2) Với từng finding: false_positive qua _dismissal_rejection -> false_positives (+ lý do = root_cause của LLM, đưa
#      vào mục 4.5 báo cáo); false_positive không qua -> suggestion KB có advisor_verdict "disputed" + kept_because;
#      confirmed / uncertain -> _llm_suggestion; không có kết luận LLM -> suggestion KB.
#   3) Finding UNMAPPED được LLM gán lớp lỗi -> suggestion có advisor_swc (gpu_pipeline đổi tiêu đề finding).
# knowledge_base: KB dùng chung của gpu_pipeline (không truyền thì tự nạp).
# rag_context: top ca tương tự của Phase 1B (gợi ý phân loại cho finding UNMAPPED).
# Trả về {status, suggestions, false_positives, false_positive_reasons}.
# Gọi bởi: gpu_pipeline.run_server_gpu_pipeline (use_llm=True); test dùng use_llm=False (không gọi LLM).
def _generate_advisor_section(
    findings: list[dict[str, Any]],
    code: str = "",
    *,
    use_llm: bool = False,
    knowledge_base: RepairKnowledgeBase | None = None,
    rag_context: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    kb = knowledge_base or RepairKnowledgeBase()

    llm_results: dict[str, RepairSuggestion] = {}
    if use_llm and findings:
        agent = AdvisorAgent(knowledge_base=kb)
        by_severity = sorted(findings, key=lambda item: RISK_ORDER.get(item.get("severity", "Medium"), 0), reverse=True)
        for finding in by_severity[:MAX_LLM_ADVISOR_CALLS or None]:
            result = agent.advise_one(code, finding, rag_context=rag_context)
            if not result.error:
                llm_results[finding_key(finding)] = result

    suggestions: list[dict[str, Any]] = []
    false_positives: list[str] = []
    false_positive_reasons: dict[str, str] = {}
    for finding in findings:
        fid = finding_key(finding)
        result = llm_results.get(fid)
        if result is None:
            suggestions.append(_knowledge_base_suggestion(finding, kb, code))
            continue

        if result.verdict == VERDICT_FALSE_POSITIVE:
            rejection = _dismissal_rejection(finding, result, code)
            if rejection is None:
                # Advisor triage: the caller drops this finding from repair and lists it as dismissed.
                false_positives.append(fid)
                false_positive_reasons[fid] = result.root_cause
                continue
            suggestion = {
                **_knowledge_base_suggestion(finding, kb, code),
                "advisor_verdict": "disputed",
                "advisor_note": result.root_cause,
                "kept_because": rejection,
            }
        else:
            suggestion = _llm_suggestion(finding, result)
        if result.vulnerability_class:
            suggestion["advisor_swc"] = result.vulnerability_class
        suggestions.append(suggestion)

    return {
        "status": "ok",
        "suggestions": suggestions,
        "false_positives": false_positives,
        "false_positive_reasons": false_positive_reasons,
    }
