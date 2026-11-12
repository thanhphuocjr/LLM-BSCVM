"""Phase 1B - run the RAG retriever on every code unit of a contract (run_rag)."""

from __future__ import annotations

from typing import Any

from E2E.config import RISK_ORDER
from E2E.phase0_parsing.code_units import CodeUnit, parse_code_units
from E2E.phase1b_rag.retriever import default_retriever


# [HÀM CHÍNH PHASE 1B] Chạy RAG cho từng code unit của contract.
# Với mỗi unit: retriever.analyze(unit.code) -> các finding theo SWC (confidence >= 0.60); gắn unit_id + dòng;
# khoá = (SWC, unit) nên mỗi lỗ hổng chỉ có một finding.
# Trả về: score (confidence cao nhất), verdict, findings, retrieved_context (top-8 ca tương tự
# cho cả contract - Fixer dùng làm ví dụ sửa lỗi).
# Gọi bởi: orchestrator.run_pipeline (song song với run_static), candidate_check/final_verification (chạy lại trên bản vá).
def run_rag(code: str, units: list[CodeUnit] | None = None) -> dict[str, Any]:
    parsed_units = units or parse_code_units(code)
    retriever = default_retriever()
    findings_by_key: dict[tuple[str, str], dict[str, Any]] = {}

    for unit in parsed_units:
        for finding in retriever.analyze(unit.code):
            findings_by_key[(str(finding.get("swc")), unit.unit_id)] = {
                **finding,
                "source": "rag",
                "unit_id": unit.unit_id,
                "start_line": unit.start_line,
                "end_line": unit.end_line,
            }

    findings = sorted(
        findings_by_key.values(),
        key=lambda item: (RISK_ORDER.get(item["severity"], 0), item["confidence"]),
        reverse=True,
    )
    score = max((item["confidence"] for item in findings), default=0.0)
    risk_level = findings[0]["severity"] if findings else "None"
    return {
        "component": "rag",
        "status": "ok",
        "engine": "local_swc_tfidf_plus_signal_gating",
        "score": round(float(score), 6),
        "verdict": "Vulnerable" if findings else "Safe",
        "risk_level": risk_level,
        "findings": findings,
        "retrieved_context": retriever.retrieve_context(code, top_k=8),
    }
