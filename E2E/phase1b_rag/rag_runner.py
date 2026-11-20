"""Phase 1B - chạy RAG retriever cho mọi code unit của một contract."""

from E2E.config import RISK_ORDER
from E2E.phase0_parsing.code_units import CodeUnit, parse_code_units
from E2E.phase1b_rag.retriever import default_retriever


# [HÀM CHÍNH PHASE 1B] retriever.analyze từng unit (mỗi (SWC, unit) một finding) + top-8 ca tương tự cho Fixer.
def run_rag(code: str, units: list[CodeUnit] | None = None) -> dict:
    retriever = default_retriever()
    by_key = {}
    for unit in units or parse_code_units(code):
        for finding in retriever.analyze(unit.code):
            by_key[(str(finding.get("swc")), unit.unit_id)] = {**finding, "source": "rag", "unit_id": unit.unit_id,
                                                                "start_line": unit.start_line,
                                                                "end_line": unit.end_line}
    findings = sorted(by_key.values(), key=lambda item: (RISK_ORDER.get(item["severity"], 0), item["confidence"]),
                      reverse=True)
    return {
        "component": "rag", "status": "ok", "engine": "local_swc_tfidf_plus_signal_gating",
        "score": round(float(max((item["confidence"] for item in findings), default=0.0)), 6),
        "verdict": "Vulnerable" if findings else "Safe",
        "risk_level": findings[0]["severity"] if findings else "None",
        "findings": findings,
        "retrieved_context": retriever.retrieve_context(code, top_k=8),
    }
