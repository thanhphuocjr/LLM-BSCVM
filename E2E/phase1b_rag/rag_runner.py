"""Phase 1B - chạy RAG cho mọi code unit của một contract (chỉ phát hiện + phân loại, không chuẩn bị tri thức sửa lỗi)."""

from E2E.common.knowledge_base import default_knowledge_base
from E2E.config import RISK_ORDER
from E2E.phase0_parsing.code_units import CodeUnit, parse_code_units
from E2E.phase1b_rag.retriever import analyze, nearest_class


# [HÀM CHÍNH PHASE 1B] Mỗi unit: SWC của các ca lỗi giống nó (finding) + lớp của ca giống nhất (unit có thân).
def run_rag(code: str, units: list[CodeUnit] | None = None) -> dict:
    kb = default_knowledge_base()
    by_key, classifications = {}, []
    for unit in units or parse_code_units(code):
        for finding in analyze(unit.code, kb):
            by_key[(str(finding.get("swc")), unit.unit_id)] = {**finding, "source": "rag", "unit_id": unit.unit_id,
                                                                "start_line": unit.start_line,
                                                                "end_line": unit.end_line}
        if unit.has_body:
            classifications.append({"unit_id": unit.unit_id, **nearest_class(unit.code, kb)})
    findings = sorted(by_key.values(), key=lambda item: (RISK_ORDER.get(item["severity"], 0), item["confidence"]),
                      reverse=True)
    return {
        "component": "rag", "status": "ok", "engine": "tfidf_nearest_known_cases",
        "score": round(float(max((item["confidence"] for item in findings), default=0.0)), 6),
        "verdict": "Vulnerable" if findings else "Safe",
        "risk_level": findings[0]["severity"] if findings else "None",
        "findings": findings, "classifications": classifications,
    }
