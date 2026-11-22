"""Phase 1A - bằng chứng tĩnh: bộ luật trên cả contract + từng unit, Slither native ánh xạ sang SWC."""

import contextlib

from E2E.common.io_utils import jsonable
from E2E.common.security_tools import _run_native_slither
from E2E.common.solc import select_best_solc
from E2E.config import RISK_ORDER
from E2E.phase0_parsing.code_units import CodeUnit, parse_code_units
from E2E.phase1a_static.static_analyzer import extract_guard_modifiers, run_static_analysis
from E2E.phase1a_static.swc_signals import signal_gate_hits

# Loại lỗi dễ báo nhầm: chỉ confirmed khi cổng tín hiệu SWC (swc_signals.py) thấy tín hiệu đúng ngữ cảnh trong mã.
SIGNAL_GATED_TYPES = {
    "reentrancy": "SWC-107", "unprotected_ether_withdrawal": "SWC-105", "unprotected_selfdestruct": "SWC-106",
    "function_default_visibility": "SWC-100", "unchecked_low_level_calls": "SWC-104",
    "integer_overflow_underflow": "SWC-101", "dos_gas_limit": "SWC-113", "dos_revert_griefing": "SWC-128",
}
# Check của Slither -> (SWC, tiêu đề, mức độ); check ngoài bảng bị bỏ qua.
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


# Đổi dòng trong hàm thành dòng trong cả file (unit.start_line + line - 1).
def _file_line(unit: CodeUnit | None, line):
    return unit.start_line + line - 1 if unit is not None and isinstance(line, int) else line


# Kết quả bộ luật -> các "dòng finding" chuẩn của pipeline (loại dễ báo nhầm phải qua cổng tín hiệu SWC).
def _finding_rows(result: dict, analysis_code: str, unit: CodeUnit | None = None) -> list[dict]:
    rows = []
    for finding in map(jsonable, result.get("findings", {}).values()):
        vuln_type = str(finding.get("vuln_type") or "")
        gate_swc = SIGNAL_GATED_TYPES.get(vuln_type)
        matches = finding.get("matches") or [{}]
        rows.append({
            "source": "static", "swc": finding.get("vuln_id"), "title": vuln_type.replace("_", " ").title(),
            "severity": finding.get("risk_level", "Unknown"),
            "confidence": round(min(1.0, sum(float(m.get("pattern_weight") or 0.0)
                                             for m in finding.get("matches", []))), 6),
            "confirmed": bool(signal_gate_hits(gate_swc, analysis_code)) if gate_swc
            else bool(finding.get("confirmed")),
            "line": _file_line(unit, matches[0].get("line")), "unit_id": unit.unit_id if unit else None,
            "matched_text": matches[0].get("matched_text"),
            "evidence": [{"line": _file_line(unit, m.get("line")), "matched_text": m.get("matched_text"),
                          "context": m.get("context")} for m in matches],
            "notes": finding.get("notes", []),
        })
    return rows


# Chạy Slither (khi có solc hợp pragma), đổi check trong SLITHER_CHECK_TO_SWC thành dòng finding -> (rows, chạy xong?).
def _slither_rows(code: str, units: list[CodeUnit]) -> tuple[list[dict], bool]:
    rows, completed = [], False
    solc_path, reason, _ = select_best_solc(code)
    with contextlib.suppress(Exception):
        result = _run_native_slither(code, "Contract.sol", None, solc_path, reason) if solc_path else {}
        completed = result.get("run_status") == "completed"
        for item in result.get("findings", []) if completed else []:
            check = str(item.get("check") or "")
            if check not in SLITHER_CHECK_TO_SWC:
                continue
            swc_id, title, severity = SLITHER_CHECK_TO_SWC[check]
            lines = item.get("line") or []
            first = lines[0] if lines else None
            unit = next((u for u in units if u.start_line <= first <= u.end_line), None) if first else None
            rows.append({
                "source": "slither", "swc": swc_id, "title": title, "severity": severity, "confidence": 0.95,
                "confirmed": True, "line": first, "unit_id": unit.unit_id if unit else None, "matched_text": check,
                "evidence": [{"line": line, "matched_text": check, "context": item.get("message")} for line in lines],
                "notes": [f"Slither detector '{check}': {item.get('message')}"],
            })
    return rows, completed


# [HÀM CHÍNH PHASE 1A] Bộ luật (contract + từng unit) + Slither -> khử trùng (SWC, dòng) -> findings / candidates.
def run_static(code: str, units: list[CodeUnit] | None = None) -> dict:
    units = units or parse_code_units(code)
    guards = extract_guard_modifiers(code)
    rows = _finding_rows(run_static_analysis(code, guard_modifiers=guards), code)
    for unit in units:
        rows += _finding_rows(run_static_analysis(unit.code, guard_modifiers=guards), unit.code, unit)
    slither_rows, slither_completed = _slither_rows(code, units)
    best: dict[tuple, dict] = {}
    for row in rows + slither_rows:
        key = (row.get("swc"), row.get("line"))
        if key not in best or row.get("source") == "slither" or row.get("unit_id"):
            best[key] = row
    candidates = sorted(best.values(), key=lambda r: (RISK_ORDER.get(r["severity"], 0), r.get("confidence", 0.0)),
                        reverse=True)
    findings = [row for row in candidates if row["confirmed"]]
    return {
        "component": "static", "status": "ok",
        "engine": "Slither + Regex rules" if slither_completed else "Regex rules (27 rules + 25 semantic checks)",
        "score": round(max((float(row["confidence"]) for row in findings), default=0.0), 6),
        "verdict": "Vulnerable" if findings else "Safe",
        "risk_level": max((row.get("severity", "None") for row in findings), key=lambda level: RISK_ORDER.get(level, 0),
                          default="None"),
        "findings": findings,
        "candidates": [row for row in candidates if not row["confirmed"]],
    }
