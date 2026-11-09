from __future__ import annotations

from typing import Any

from E2E.config import KNOWLEDGE_BASE_DIR, RISK_ORDER
from E2E.phase0_parsing.code_units import CodeUnit, parse_code_units
from E2E.phase1b_rag.retriever import LocalRAGRetriever


# [HÀM CHÍNH PHASE 1B] Chạy RAG cho từng code unit của contract.
# Với mỗi unit: LocalRAGRetriever.analyze(unit.code) -> các finding theo SWC; gắn unit_id + dòng;
# chỉ finding "vulnerable" (confidence >= 0.60) được đưa vào danh sách findings của contract (khoá = (SWC, unit)).
# Trả về: score (confidence cao nhất), verdict, findings, unit_results, retrieved_context (top-8 ca tương tự
# cho cả contract - Fixer/Advisor dùng làm ví dụ sửa lỗi).
# Gọi bởi: orchestrator.run_pipeline (song song với run_static), candidate_check/final_verification (chạy lại trên bản vá).
def run_rag(
    code: str,
    units: list[CodeUnit] | None = None,
    store_dir=KNOWLEDGE_BASE_DIR,
) -> dict[str, Any]:
    parsed_units = units or parse_code_units(code)
    retriever = LocalRAGRetriever(store_dir)
    unit_results: list[dict[str, Any]] = []
    findings_by_key: dict[tuple[str, str], dict[str, Any]] = {}

    for unit in parsed_units:
        result = retriever.analyze(unit.code)
        unit_findings = []
        for finding in result["top_findings"]:
            enriched = {
                **finding,
                "source": "rag",
                "unit_id": unit.unit_id,
                "start_line": unit.start_line,
                "end_line": unit.end_line,
            }
            unit_findings.append(enriched)
            if finding.get("vulnerable"):
                key = (str(finding.get("swc")), unit.unit_id)
                findings_by_key[key] = enriched
        unit_results.append(
            {
                "unit_id": unit.unit_id,
                "start_line": unit.start_line,
                "end_line": unit.end_line,
                "score": result["final_score"],
                "verdict": result["final_verdict"],
                "risk_level": result["risk_level"],
                "findings": unit_findings,
            }
        )

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
        "final_score": round(float(score), 6),
        "verdict": "Vulnerable" if findings else "Safe",
        "final_verdict": "Vulnerable" if findings else "Safe",
        "risk_level": risk_level,
        "findings": findings,
        "top_findings": findings,
        "unit_results": unit_results,
        "retrieved_context": retriever.retrieve_context(code, top_k=8),
    }
