"""
Phase 1A - static evidence: the rule engine (static_analyzer.py) on the whole contract and on
every code unit, plus native Slither mapped to SWC ids. Error-prone rule types are confirmed only
when the matching SWC agent of Phase 1B also sees its code signal (SIGNAL_GATED_TYPES).
"""

from __future__ import annotations

import logging
from typing import Any

from E2E.config import RISK_ORDER
from E2E.common.io_utils import jsonable
from E2E.common.security_tools import _run_native_slither
from E2E.common.solc import select_best_solc
from E2E.phase0_parsing.code_units import CodeUnit, parse_code_units
from E2E.phase1a_static.static_analyzer import extract_guard_modifiers, run_static_analysis
from E2E.phase1b_rag.swc_agents import find_custom_signal_hits

logger = logging.getLogger(__name__)

SIGNAL_GATED_TYPES = {
    "reentrancy": "SWC-107",
    "unprotected_ether_withdrawal": "SWC-105",
    "unprotected_selfdestruct": "SWC-106",
    "function_default_visibility": "SWC-100",
    "unchecked_low_level_calls": "SWC-104",
    "integer_overflow_underflow": "SWC-101",
    "dos_gas_limit": "SWC-113",
    "dos_revert_griefing": "SWC-128",
}

SLITHER_CHECK_TO_SWC: dict[str, tuple[str, str, str]] = {
    "tx-origin": ("SWC-115", "Tx Origin Authentication", "Critical"),
    "reentrancy-eth": ("SWC-107", "Reentrancy", "Critical"),
    "reentrancy-no-eth": ("SWC-107", "Reentrancy", "Medium"),
    "arbitrary-send-eth": ("SWC-105", "Unprotected Ether Withdrawal", "Critical"),
    "suicidal": ("SWC-106", "Unprotected Selfdestruct", "Critical"),
    "unprotected-upgrade": ("SWC-105", "Unprotected Upgrade (Access Control)", "Critical"),
    "unchecked-transfer": ("SWC-104", "Unchecked Low Level Calls", "Medium"),
    "unchecked-lowlevel": ("SWC-104", "Unchecked Low Level Calls", "Medium"),
    "calls-loop": ("SWC-113", "Dos Gas Limit", "Medium"),
    "timestamp": ("SWC-116", "Timestamp Dependence", "Low"),
    "solc-version": ("SWC-102", "Outdated Compiler Version", "Low"),
}


# Nhiệm vụ: chuyển kết quả thô của StaticAnalyzer thành các "dòng finding" chuẩn của pipeline
# {source, swc, title, severity, confidence, confirmed, line, unit_id, evidence, notes}.
# Điểm quan trọng: với các loại lỗi dễ báo nhầm (SIGNAL_GATED_TYPES: reentrancy, overflow, ...),
# finding chỉ được "confirmed" khi SWC agent tương ứng (phase1b_rag/swc_agents.py) cũng thấy tín hiệu.
# Dòng (line) được đổi từ dòng trong hàm sang dòng trong cả file (unit.start_line + line - 1).
def _finding_rows(
    result: dict[str, Any],
    analysis_code: str,
    unit: CodeUnit | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for finding in result.get("findings", {}).values():
        finding_dict = jsonable(finding)
        vuln_type = str(finding_dict.get("vuln_type") or "")
        confirmed = bool(finding_dict.get("confirmed"))
        gate_swc = SIGNAL_GATED_TYPES.get(vuln_type)
        if gate_swc:
            confirmed = bool(find_custom_signal_hits(gate_swc, analysis_code))
        match_weight = sum(float(item.get("pattern_weight") or 0.0) for item in finding_dict.get("matches", []))
        confidence = min(1.0, match_weight)
        matches = finding_dict.get("matches") or [{}]
        local_line = matches[0].get("line")
        line = local_line
        if unit is not None and isinstance(local_line, int):
            line = unit.start_line + local_line - 1
        rows.append(
            {
                "source": "static",
                "swc": finding_dict.get("vuln_id"),
                "title": str(finding_dict.get("vuln_type") or "").replace("_", " ").title(),
                "severity": finding_dict.get("risk_level", "Unknown"),
                "confidence": round(confidence, 6),
                "confirmed": confirmed,
                "line": line,
                "unit_id": unit.unit_id if unit else None,
                "matched_text": matches[0].get("matched_text"),
                "evidence": [
                    {
                        "line": (
                            unit.start_line + match.get("line") - 1
                            if unit is not None and isinstance(match.get("line"), int)
                            else match.get("line")
                        ),
                        "matched_text": match.get("matched_text"),
                        "context": match.get("context"),
                    }
                    for match in matches
                ],
                "notes": finding_dict.get("notes", []),
            }
        )
    return rows


# [HÀM CHÍNH PHASE 1A] Phân tích tĩnh toàn contract.
# Các bước: (1) tìm guard modifier (onlyOwner...), (2) chạy bộ luật trên cả contract và trên từng unit,
# (3) chạy Slither thật (nếu đã cài) và ánh xạ check -> SWC bằng SLITHER_CHECK_TO_SWC,
# (4) khử trùng lặp theo (SWC, dòng) - ưu tiên Slither/finding có unit_id,
# (5) chỉ giữ finding confirmed; finding chưa xác nhận để ở "candidates".
# Trả về: score (= confidence cao nhất), verdict, risk_level, findings (confirmed), candidates (chưa xác nhận).
# Gọi bởi: orchestrator.run_pipeline (song song với run_rag), candidate_check/final_verification (chạy lại trên bản vá).
def run_static(code: str, units: list[CodeUnit] | None = None) -> dict[str, Any]:
    parsed_units = units or parse_code_units(code)
    guard_modifiers = extract_guard_modifiers(code)
    contract_result = run_static_analysis(code, guard_modifiers=guard_modifiers)
    all_rows = _finding_rows(contract_result, code)
    for unit in parsed_units:
        result = run_static_analysis(unit.code, guard_modifiers=guard_modifiers)
        all_rows.extend(_finding_rows(result, unit.code, unit))

    engine_name = "Regex rules (27 rules + 25 semantic checks)"
    # Slither thật (chỉ khi có solc phù hợp; chưa cài slither -> run_status "unavailable", chỉ dùng bộ luật)
    solc_path, selection_reason, _ = select_best_solc(code)
    if solc_path is not None:
        try:
            slither_res = _run_native_slither(code, "Contract.sol", None, solc_path, selection_reason)
            if slither_res.get("run_status") == "completed":
                engine_name = "Slither + Regex rules"
                for item in slither_res.get("findings", []):
                    check = str(item.get("check") or "")
                    lines = item.get("line") or []
                    first_line = lines[0] if lines else None
                    swc_info = SLITHER_CHECK_TO_SWC.get(check)
                    if swc_info:
                        swc_id, title, severity = swc_info
                        matched_unit = None
                        if first_line:
                            matched_unit = next(
                                (u for u in parsed_units if u.start_line <= first_line <= u.end_line),
                                None,
                            )
                        all_rows.append(
                            {
                                "source": "slither",
                                "swc": swc_id,
                                "title": title,
                                "severity": severity,
                                "confidence": 0.95,
                                "confirmed": True,
                                "line": first_line,
                                "unit_id": matched_unit.unit_id if matched_unit else None,
                                "matched_text": check,
                                "evidence": [
                                    {
                                        "line": l,
                                        "matched_text": check,
                                        "context": item.get("message"),
                                    }
                                    for l in lines
                                ],
                                "notes": [f"Slither detector '{check}': {item.get('message')}"],
                            }
                        )
        except Exception as e:
            logger.debug(f"[static_runner] Slither execution failed: {e}")

    deduplicated: dict[tuple, dict[str, Any]] = {}
    for row in all_rows:
        key = (row.get("swc"), row.get("line"))
        current = deduplicated.get(key)
        # Ưu tiên finding có unit_id hoặc từ slither
        if current is None or row.get("source") == "slither" or row.get("unit_id"):
            deduplicated[key] = row

    candidates = sorted(
        deduplicated.values(),
        key=lambda item: (RISK_ORDER.get(item["severity"], 0), item.get("confidence", 0.0)),
        reverse=True,
    )
    findings = [item for item in candidates if item["confirmed"]]
    score = max((float(item["confidence"]) for item in findings), default=0.0)
    risk_level = max(
        (item.get("severity", "None") for item in findings),
        key=lambda level: RISK_ORDER.get(level, 0),
        default="None",
    )
    return {
        "component": "static",
        "status": "ok",
        "engine": engine_name,
        "score": round(score, 6),
        "verdict": "Vulnerable" if findings else "Safe",
        "risk_level": risk_level,
        "findings": findings,
        "candidates": [item for item in candidates if not item["confirmed"]],
    }
