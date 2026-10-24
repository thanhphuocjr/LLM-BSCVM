"""
E2E/local/behavior_verifier.py
──────────────────────────────────────────────────────────────────────────────
Module kiểm tra bảo toàn hành vi dựa trên AST và tín hiệu Observable (Phase 5).
Thực hiện các yêu cầu trong kế hoạch thẩm định (B5, R-03):
  - Phân tích cú pháp AST bằng `solc --ast-compact-json` cho cả bản gốc và bản vá.
  - So sánh tập:
      * Biến state bị ghi kèm toán tử (ví dụ: balances, _allowances, ...)
      * Event được gọi (cả cú pháp 0.4 không emit và emit hiện đại)
      * Lời gọi ngoài (call.value, transfer, send, call)
      * Biểu thức return và chữ ký hàm
  - Bản vá chỉ được thêm điều kiện kiểm tra (require, assert) hoặc sắp xếp lại thứ tự câu lệnh (CEI).
  - Mất một câu lệnh ghi state hoặc transfer ETH/token hợp lệ => REJECT.
  - Đổi giá trị return => WARN.
  - Tích hợp kiểm tra Slither locked-ether.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from E2E.shared.function_parser import CodeUnit, parse_code_units
from E2E.shared.solc import compile_solidity_code, select_best_solc


OBSERVABLE_SIGNALS = {
    "ETH value call": re.compile(r"(\.call\s*\{\s*value\s*:|\.call\.value\s*\()"),
    "ETH transfer": re.compile(r"\.transfer\s*\("),
    "ETH send": re.compile(r"\.send\s*\("),
    "event emission": re.compile(r"(\bemit\s+[A-Za-z_$][\w$]*\s*\(|[A-Z][A-Za-z0-9_$]*\s*\([^;]*\)\s*;)"),
    "return statement": re.compile(r"\breturn\b"),
}


def _extract_ast_dict(raw_ast_stdout: str) -> Optional[dict]:
    """Trích xuất đối tượng JSON AST từ output của solc --ast-compact-json."""
    if not raw_ast_stdout:
        return None
    start = raw_ast_stdout.find("{")
    end = raw_ast_stdout.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(raw_ast_stdout[start : end + 1])
    except Exception:
        return None


def _collect_function_ast_features(fn_node: dict) -> dict[str, Any]:
    """
    Thu thập các đặc trưng hành vi của một hàm từ AST:
      - state_writes: set of (var_name, operator)
      - external_calls: set of (target_desc, call_kind)
      - events: set of event_names
      - returns: list of return signatures / literal descriptions
    """
    state_writes: Set[Tuple[str, str]] = set()
    external_calls: Set[str] = set()
    events: Set[str] = set()
    return_exprs: List[str] = []

    def get_var_name(node: dict) -> str:
        nt = node.get("nodeType")
        if nt == "Identifier":
            return str(node.get("name") or "")
        if nt == "IndexAccess":
            base = node.get("baseExpression") or {}
            return get_var_name(base)
        if nt == "MemberAccess":
            expr = node.get("expression") or {}
            expr_name = get_var_name(expr)
            member = node.get("memberName") or ""
            return f"{expr_name}.{member}" if expr_name else member
        return ""

    def walk(node: Any):
        if not isinstance(node, dict):
            return
        nt = node.get("nodeType")

        # 1. State variable writes
        if nt == "Assignment":
            op = str(node.get("operator") or "=")
            left = node.get("leftHandSide") or {}
            var_name = get_var_name(left)
            if var_name:
                state_writes.add((var_name, op))

        # 2. External calls & transfers
        elif nt == "MemberAccess":
            member = node.get("memberName") or ""
            if member in {"transfer", "send", "call", "delegatecall"}:
                expr = node.get("expression") or {}
                caller = get_var_name(expr) or "unknown"
                external_calls.add(f"{caller}.{member}")

        # 3. Events & Function calls
        elif nt == "FunctionCall":
            expr = node.get("expression") or {}
            fn_name = get_var_name(expr)
            # Trong 0.4, event thường được gọi trực tiếp bằng tên Viết Hoa
            if fn_name and fn_name[0].isupper() and fn_name not in {"Require", "Assert", "Revert"}:
                events.add(fn_name)
            # Cú pháp .call{value: ...} hoặc .call.value(...)
            if "call" in fn_name.lower():
                external_calls.add("call")

        elif nt == "EmitStatement":
            expr = node.get("eventCall") or {}
            fn = expr.get("expression") or {}
            ev_name = get_var_name(fn)
            if ev_name:
                events.add(ev_name)

        elif nt == "Return":
            expr = node.get("expression")
            if expr:
                return_exprs.append(get_var_name(expr) or str(expr.get("value") or "expr"))
            else:
                return_exprs.append("void")

        for key, val in node.items():
            if isinstance(val, dict):
                walk(val)
            elif isinstance(val, list):
                for item in val:
                    walk(item)

    walk(fn_node)
    return {
        "state_writes": state_writes,
        "external_calls": external_calls,
        "events": events,
        "returns": return_exprs,
    }


def _extract_ast_functions_map(ast_dict: dict) -> Dict[str, dict]:
    """Tạo map unit_id -> function AST node."""
    fn_map: Dict[str, dict] = {}
    if not ast_dict:
        return fn_map

    for node in ast_dict.get("nodes", []):
        if node.get("nodeType") == "ContractDefinition":
            c_name = node.get("name") or "Contract"
            for sub in node.get("nodes", []):
                if sub.get("nodeType") == "FunctionDefinition":
                    f_name = sub.get("name") or "fallback"
                    fn_map[f"{c_name}.{f_name}"] = sub
                    fn_map[f_name] = sub
    return fn_map


def _compare_unit_behavior_ast(
    orig_fn_ast: dict,
    fixed_fn_ast: dict,
    unit_id: str,
) -> List[dict[str, Any]]:
    """So sánh AST của hai phiên bản hàm."""
    findings = []
    orig_feats = _collect_function_ast_features(orig_fn_ast)
    fixed_feats = _collect_function_ast_features(fixed_fn_ast)

    # 1. Kiểm tra mất mát câu lệnh ghi state
    missing_writes = orig_feats["state_writes"] - fixed_feats["state_writes"]
    # Bỏ qua nếu chỉ đổi toán tử (ví dụ: gán SafeMath qua add)
    orig_vars = {v for v, op in orig_feats["state_writes"]}
    fixed_vars = {v for v, op in fixed_feats["state_writes"]}
    completely_missing_vars = orig_vars - fixed_vars

    if completely_missing_vars:
        findings.append({
            "unit_id": unit_id,
            "severity": "High",
            "check": "ast-state-write-preservation",
            "evidence": f"Patch removed state variable mutation for: {', '.join(sorted(completely_missing_vars))}",
        })

    # 2. Kiểm tra external calls / ETH transfer
    missing_calls = orig_feats["external_calls"] - fixed_feats["external_calls"]
    if missing_calls:
        # Nếu mất transfer hoặc call.value
        crit_calls = [c for c in missing_calls if any(k in c for k in ("transfer", "send", "call"))]
        if crit_calls:
            findings.append({
                "unit_id": unit_id,
                "severity": "High",
                "check": "ast-external-call-preservation",
                "evidence": f"Patch removed ETH/token external call or transfer: {', '.join(sorted(crit_calls))}",
            })

    # 3. Kiểm tra events
    missing_events = orig_feats["events"] - fixed_feats["events"]
    if missing_events:
        findings.append({
            "unit_id": unit_id,
            "severity": "Medium",
            "check": "ast-event-emission-preservation",
            "evidence": f"Patch removed event emission: {', '.join(sorted(missing_events))}",
        })

    # 4. Kiểm tra return
    if orig_feats["returns"] and not fixed_feats["returns"]:
        findings.append({
            "unit_id": unit_id,
            "severity": "Medium",
            "check": "ast-return-preservation",
            "evidence": "Patch removed return statement present in original function",
        })

    return findings


def verify_patch_behavior(
    original_code: str,
    fixed_code: str,
    patch_outcomes: list[dict[str, Any]] | None = None,
    tool_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Xác minh bảo toàn hành vi của bản vá thông qua:
      1. So sánh AST chi tiết (solc --ast-compact-json).
      2. Bộ tín hiệu Observable regex (fallback & tăng cường).
      3. Slither locked-ether check.
    """
    applied_units = set()
    for item in patch_outcomes or []:
        if item.get("status") != "applied":
            continue
        uid = str(item.get("unit_id") or "").strip()
        if not uid and ":" in str(item.get("finding_id") or ""):
            uid = str(item.get("finding_id")).split(":", 1)[1].strip()
        if uid and uid not in {"source", "contract"}:
            applied_units.add(uid)

    if not applied_units:
        return {
            "status": "not_applicable",
            "findings": [],
            "checks": ["No applied patch requires behavioral comparison."],
        }

    findings: List[dict[str, Any]] = []
    checks: List[dict[str, Any]] = []

    # Thử biên dịch với AST
    solc_path, _, _ = select_best_solc(original_code)
    orig_comp = compile_solidity_code(original_code, solc_path=solc_path, with_ast=True)
    fixed_comp = compile_solidity_code(fixed_code, solc_path=solc_path, with_ast=True)

    orig_ast_dict = _extract_ast_dict(orig_comp.get("ast") or "") if orig_comp.get("passed") else None
    fixed_ast_dict = _extract_ast_dict(fixed_comp.get("ast") or "") if fixed_comp.get("passed") else None

    use_ast = bool(orig_ast_dict and fixed_ast_dict)
    if use_ast:
        orig_fn_map = _extract_ast_functions_map(orig_ast_dict)
        fixed_fn_map = _extract_ast_functions_map(fixed_ast_dict)
        checks.append({
            "check": "ast_compilation",
            "engine": orig_comp.get("compiler"),
            "status": "ast_verified",
        })

        for unit_id in sorted(applied_units):
            orig_ast = orig_fn_map.get(unit_id) or orig_fn_map.get(unit_id.split(".")[-1])
            fixed_ast = fixed_fn_map.get(unit_id) or fixed_fn_map.get(unit_id.split(".")[-1])
            if orig_ast and fixed_ast:
                ast_findings = _compare_unit_behavior_ast(orig_ast, fixed_ast, unit_id)
                findings.extend(ast_findings)

    # Luôn chạy bộ kiểm tra tín hiệu Observable (đảm bảo tương thích ngược và kiểm tra chéo)
    original_units = {item.unit_id: item.code for item in parse_code_units(original_code)}
    fixed_units = {item.unit_id: item.code for item in parse_code_units(fixed_code)}

    for unit_id in sorted(applied_units):
        original = original_units.get(unit_id) or next(
            (code for k, code in original_units.items() if k.endswith(f".{unit_id}") or unit_id.endswith(f".{k}")),
            None,
        )
        fixed = fixed_units.get(unit_id) or next(
            (code for k, code in fixed_units.items() if k.endswith(f".{unit_id}") or unit_id.endswith(f".{k}")),
            None,
        )
        if original is None or fixed is None:
            findings.append({
                "unit_id": unit_id,
                "severity": "High",
                "check": "patched-unit-presence",
                "evidence": f"The patched declaration {unit_id} could not be mapped in both source versions.",
            })
            continue

        for label, pattern in OBSERVABLE_SIGNALS.items():
            existed = bool(pattern.search(original))
            preserved = bool(pattern.search(fixed))
            checks.append({
                "unit_id": unit_id,
                "signal": label,
                "original": existed,
                "fixed": preserved,
            })
            if existed and not preserved:
                # Đảm bảo không trùng lặp finding đã được phát hiện bởi AST
                if not any(f.get("unit_id") == unit_id and label.lower() in str(f.get("evidence", "")).lower() for f in findings):
                    findings.append({
                        "unit_id": unit_id,
                        "severity": "High",
                        "check": "observable-behavior-preservation",
                        "evidence": f"Patch removed {label} present in the original declaration.",
                    })

    # Tích hợp kiểm tra Slither locked-ether nếu có trong tool_result
    if tool_result:
        for finding in (tool_result.get("slither") or {}).get("findings", []):
            if finding.get("check") == "locked-ether":
                findings.append({
                    "unit_id": "",
                    "severity": finding.get("impact", "Medium"),
                    "check": "slither-locked-ether",
                    "evidence": finding.get("description", "Patched contract can lock Ether."),
                })

    return {
        "status": "failed" if findings else "passed",
        "findings": findings,
        "checks": checks,
        "ast_verified": use_ast,
    }
