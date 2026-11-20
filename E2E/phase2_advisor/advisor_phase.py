"""Phase 2 - chạy Advisor cho mọi finding (recall-first: chỉ loại finding khi bác bỏ có bằng chứng kiểm chứng được)."""

from E2E.common.findings import finding_key
from E2E.common.regex_patterns import RE_QUOTE_ELLIPSIS, RE_WHITESPACE
from E2E.config import MAX_LLM_ADVISOR_CALLS, RISK_ORDER
from E2E.phase2_advisor.advisor import VERDICT_FALSE_POSITIVE, AdvisorAgent, RepairKnowledgeBase, RepairSuggestion


# Đoạn guard LLM trích có thật trong mã: bỏ ` và khoảng trắng, tách theo "...", MỌI mảnh (>= 3 ký tự) phải có trong mã.
def _quote_in_code(quote: str, code: str) -> bool:
    compact_code = RE_WHITESPACE.sub("", code)
    pieces = [RE_WHITESPACE.sub("", piece.replace("`", "")).strip(";") for piece in RE_QUOTE_ELLIPSIS.split(quote)]
    pieces = [piece for piece in pieces if len(piece) >= 3]
    return bool(pieces) and all(piece in compact_code for piece in pieces)


# Lý do GIỮ LẠI finding LLM cho là false positive (static + RAG cùng báo / guard trích không có trong mã); None = loại.
def _dismissal_rejection(finding: dict, result: RepairSuggestion, code: str) -> str | None:
    if finding.get("corroborated"):
        return "static rules and RAG both reported this finding; one Advisor verdict does not overrule two channels"
    if result.guard_quote and not _quote_in_code(result.guard_quote, f"{finding.get('unit_code') or ''}\n{code}"):
        return "the guard quoted by the Advisor does not appear in the code"
    return None


# Suggestion từ KB theo SWC (không có phân tích LLM): dùng khi không có kết luận LLM và cho finding "disputed".
def _knowledge_base_suggestion(finding: dict, kb: RepairKnowledgeBase, code: str) -> dict:
    swc, unit_id = finding.get("swc", "SWC-100"), finding.get("unit_id") or "contract"
    title = finding.get("title") or f"Vulnerability {swc}"
    return {
        "finding_id": finding_key(finding), "swc": swc, "unit_id": unit_id,
        "root_cause": finding.get("root_cause") or finding.get("description")
        or f"Identified security vulnerability matching {swc}.",
        "impact": finding.get("impact") or f"High risk of unauthorized state modification or asset drain via {title}.",
        "repair_steps": [
            f"Identify the vulnerable state transition or external interaction in `{unit_id}`.",
            "Apply guard conditions or proper authorization checks before modifying state.",
            finding.get("remediation") or f"Implement secure defense pattern for {title}.",
            *kb.retrieve(swc, code=code).get("remediations", [])[:2],
        ],
    }


# Suggestion từ phân tích LLM (confirmed / uncertain): root cause, impact, fix_strategy (cho Fixer), advisor_verdict.
def _llm_suggestion(finding: dict, result: RepairSuggestion) -> dict:
    swc = finding.get("swc", "SWC-100")
    title = finding.get("title") or f"Vulnerability {swc}"
    step = result.fix_strategy or finding.get("remediation") or f"Implement secure defense pattern for {title}."
    return {
        "finding_id": finding_key(finding), "swc": swc, "unit_id": finding.get("unit_id") or "contract",
        "root_cause": result.root_cause, "impact": result.impact, "repair_steps": [step],
        "fix_strategy": result.fix_strategy, "advisor_verdict": result.verdict,
    }


# [HÀM CHÍNH PHASE 2] LLM cho <= MAX_LLM_ADVISOR_CALLS finding (nặng trước) -> loại false positive đã kiểm chứng.
def _generate_advisor_section(findings: list[dict], code: str = "", *, use_llm: bool = False,
                              knowledge_base: RepairKnowledgeBase | None = None,
                              rag_context: list[dict] | None = None) -> dict:
    kb = knowledge_base or RepairKnowledgeBase()
    llm_results = {}
    if use_llm and findings:
        agent = AdvisorAgent(knowledge_base=kb)
        by_severity = sorted(findings, key=lambda item: RISK_ORDER.get(item.get("severity", "Medium"), 0), reverse=True)
        for finding in by_severity[:MAX_LLM_ADVISOR_CALLS or None]:
            result = agent.advise_one(code, finding, rag_context=rag_context)
            if not result.error:
                llm_results[finding_key(finding)] = result
    suggestions, false_positives, reasons = [], [], {}
    for finding in findings:
        fid = finding_key(finding)
        result = llm_results.get(fid)
        if result is None:
            suggestions.append(_knowledge_base_suggestion(finding, kb, code))
            continue
        if result.verdict != VERDICT_FALSE_POSITIVE:
            suggestion = _llm_suggestion(finding, result)
        elif rejection := _dismissal_rejection(finding, result, code):
            suggestion = {**_knowledge_base_suggestion(finding, kb, code), "advisor_verdict": "disputed",
                          "advisor_note": result.root_cause, "kept_because": rejection}
        else:
            false_positives.append(fid)
            reasons[fid] = result.root_cause
            continue
        if result.vulnerability_class:
            suggestion["advisor_swc"] = result.vulnerability_class
        suggestions.append(suggestion)
    return {"status": "ok", "suggestions": suggestions, "false_positives": false_positives,
            "false_positive_reasons": reasons}
