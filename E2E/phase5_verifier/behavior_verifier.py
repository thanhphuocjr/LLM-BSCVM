"""
Phase 5 - behaviour preservation of the patched units (behavior_verifier.py).

1. Compile original and patch with `solc --ast-compact-json` and compare every patched
   function on the AST: losing a written state variable or a value-moving call is High,
   losing an event or the return statement is Medium. Safe rewrites (SafeMath, transfer ->
   call{value}, safe wrappers) are accepted.
2. Functions the AST could not cover fall back to the syntactic guard of Phase 4
   (behavior_guard.validate_observable_behavior) - e.g. every function on solc < 0.4.12,
   which has no compact JSON AST.
3. A Slither `locked-ether` finding the patch introduced (absent from the original) is a regression.
Any finding -> status "failed" -> "Patch rejected: behavioral regression detected".
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set, Tuple

from E2E.common.solc import compile_solidity_code, select_best_solc
from E2E.phase0_parsing.code_units import parse_code_units
from E2E.phase4_fixer.behavior_guard import validate_observable_behavior


# Member calls that move value; safe wrappers count as the same observable behaviour.
TRANSFER_MEMBERS = {
    "transfer": "transfer",
    "safeTransfer": "transfer",
    "send": "transfer",
    "sendValue": "transfer",
    "safeTransferETH": "transfer",
    "call": "call",
    "delegatecall": "delegatecall",
    "transferFrom": "transferFrom",
    "safeTransferFrom": "transferFrom",
}
CALL_LABELS = {
    "call": "ETH value call / low-level call",
    "transfer": "ETH or token transfer",
    "transferFrom": "token transferFrom",
    "delegatecall": "delegatecall",
}


# Lấy đối tượng JSON AST từ output của solc --ast-compact-json.
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


# Duyệt AST của MỘT hàm, thu đặc trưng hành vi: biến state bị ghi (kèm toán tử), lời gọi ngoài
# (call/transfer/send/delegatecall), event được phát, các lệnh return.
def _collect_function_ast_features(fn_node: dict, state_vars: Set[str] | None = None) -> dict[str, Any]:
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

    # Lấy tên biến gốc từ một nút AST (Identifier / IndexAccess / MemberAccess) - vd balances[msg.sender] -> balances.
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

    # Duyệt đệ quy cây AST và ghi nhận các đặc trưng ở trên.
    def walk(node: Any):
        if not isinstance(node, dict):
            return
        nt = node.get("nodeType")

        # 1. State variable writes (locals and parameters are not observable state)
        if nt == "Assignment":
            op = str(node.get("operator") or "=")
            left = node.get("leftHandSide") or {}
            var_name = get_var_name(left)
            root = var_name.split(".")[0]
            if var_name and (state_vars is None or root in state_vars or var_name in state_vars):
                state_writes.add((var_name, op))

        # 2. External calls & transfers, normalised so safe wrappers match the raw call
        elif nt == "MemberAccess":
            member = node.get("memberName") or ""
            if member in TRANSFER_MEMBERS:
                external_calls.add(TRANSFER_MEMBERS[member])

        # 3. Legacy events: a plain call to an upper-case name that is neither a type
        # conversion (IERC20(x)) nor a struct constructor.
        elif nt == "FunctionCall":
            expr = node.get("expression") or {}
            fn_name = get_var_name(expr)
            if (
                node.get("kind") == "functionCall"
                and expr.get("nodeType") == "Identifier"
                and fn_name
                and fn_name[0].isupper()
            ):
                events.add(fn_name)

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

        for val in node.values():
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


# Tạo map unit_id -> nút AST của hàm (hàm trùng tên được thêm hậu tố "#n" giống Phase 0).
def _extract_ast_functions_map(ast_dict: dict) -> Dict[str, dict]:
    """Tạo map unit_id -> function AST node (overloads get the same `#n` suffix as the parser)."""
    fn_map: Dict[str, dict] = {}
    if not ast_dict:
        return fn_map

    for node in ast_dict.get("nodes", []):
        if node.get("nodeType") == "ContractDefinition":
            c_name = node.get("name") or "Contract"
            seen: Dict[str, int] = {}
            for sub in node.get("nodes", []):
                if sub.get("nodeType") == "FunctionDefinition":
                    f_name = sub.get("name") or sub.get("kind") or "fallback"
                    seen[f_name] = seen.get(f_name, 0) + 1
                    suffix = f"#{seen[f_name]}" if seen[f_name] > 1 else ""
                    fn_map[f"{c_name}.{f_name}{suffix}"] = sub
                    fn_map.setdefault(f_name, sub)
    return fn_map


# Tập tên biến trạng thái khai báo trong các contract (từ AST).
def _state_variable_names(ast_dict: dict) -> Set[str]:
    names: Set[str] = set()
    for node in (ast_dict or {}).get("nodes", []):
        if node.get("nodeType") == "ContractDefinition":
            for sub in node.get("nodes", []):
                if sub.get("nodeType") == "VariableDeclaration" and sub.get("stateVariable"):
                    names.add(str(sub.get("name") or ""))
    return names


# So sánh đặc trưng AST của hàm gốc và hàm đã vá: mất hẳn lệnh ghi một biến state / mất lời gọi chuyển giá trị
# (High), mất event / mất return (Medium). Trả danh sách vi phạm.
def _compare_unit_behavior_ast(
    orig_fn_ast: dict,
    fixed_fn_ast: dict,
    unit_id: str,
    state_vars: Set[str] | None = None,
) -> List[dict[str, Any]]:
    """So sánh AST của hai phiên bản hàm."""
    findings = []
    orig_feats = _collect_function_ast_features(orig_fn_ast, state_vars)
    fixed_feats = _collect_function_ast_features(fixed_fn_ast, state_vars)

    # 1. Kiểm tra mất mát câu lệnh ghi state
    # Bỏ qua nếu chỉ đổi toán tử (ví dụ: gán SafeMath qua add)
    orig_vars = {v for v, _ in orig_feats["state_writes"]}
    fixed_vars = {v for v, _ in fixed_feats["state_writes"]}
    completely_missing_vars = orig_vars - fixed_vars

    if completely_missing_vars:
        findings.append({
            "unit_id": unit_id,
            "severity": "High",
            "check": "ast-state-write-preservation",
            "evidence": f"Patch removed state variable mutation for: {', '.join(sorted(completely_missing_vars))}",
        })

    # 2. Kiểm tra external calls / ETH transfer (a raw transfer may become a value call or a
    # safe wrapper: both still move value)
    missing_calls = orig_feats["external_calls"] - fixed_feats["external_calls"]
    value_moves = {"transfer", "call"}
    if missing_calls & value_moves and fixed_feats["external_calls"] & value_moves:
        missing_calls -= value_moves
    if missing_calls:
        findings.append({
            "unit_id": unit_id,
            "severity": "High",
            "check": "ast-external-call-preservation",
            "evidence": "Patch removed observable behavior: "
            + ", ".join(CALL_LABELS.get(call, call) for call in sorted(missing_calls)),
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


# [KIỂM TRA HỒI QUY HÀNH VI - Phase 5] Với các hàm đã vá:
#   1) compile cả 2 bản lấy AST và so sánh từng hàm (_compare_unit_behavior_ast);
#   2) hàm không so được bằng AST -> dùng guard regex validate_observable_behavior (của Phase 4);
#   3) thêm cảnh báo Slither "locked-ether" nếu bản vá sinh ra nó (bản gốc không có).
# status = failed nếu có bất kỳ vi phạm nào -> decide_overall_verdict sẽ ra "Patch rejected: behavioral regression".
def verify_patch_behavior(
    original_code: str,
    fixed_code: str,
    patch_outcomes: list[dict[str, Any]] | None = None,
    tool_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Xác minh bảo toàn hành vi của bản vá thông qua:
      1. So sánh AST chi tiết (solc --ast-compact-json).
      2. Guard regex của Phase 4 cho hàm AST không so được.
      3. Slither locked-ether do bản vá sinh ra.
    """
    applied_units = {
        str(item.get("unit_id") or "").strip()
        for item in patch_outcomes or []
        if item.get("status") == "applied"
    } - {"", "source", "contract"}

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
    ast_checked: Set[str] = set()
    if use_ast:
        orig_fn_map = _extract_ast_functions_map(orig_ast_dict)
        fixed_fn_map = _extract_ast_functions_map(fixed_ast_dict)
        state_vars = _state_variable_names(orig_ast_dict) | _state_variable_names(fixed_ast_dict)
        checks.append({
            "check": "ast_compilation",
            "engine": orig_comp.get("compiler"),
            "status": "ast_verified",
        })

        for unit_id in sorted(applied_units):
            orig_ast = orig_fn_map.get(unit_id) or orig_fn_map.get(unit_id.split(".")[-1])
            fixed_ast = fixed_fn_map.get(unit_id) or fixed_fn_map.get(unit_id.split(".")[-1])
            if orig_ast and fixed_ast:
                ast_findings = _compare_unit_behavior_ast(orig_ast, fixed_ast, unit_id, state_vars)
                findings.extend(ast_findings)
                ast_checked.add(unit_id)

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

        # Syntactic guard shared with Phase 4; only consulted when the AST comparison could
        # not run, so the two checks do not double-report the same regression.
        if unit_id in ast_checked:
            continue
        try:
            validate_observable_behavior(original, fixed)
            checks.append({"unit_id": unit_id, "check": "observable-behavior-preservation", "status": "passed"})
        except ValueError as error:
            findings.append({
                "unit_id": unit_id,
                "severity": "High",
                "check": "observable-behavior-preservation",
                "evidence": str(error),
            })

    # Slither locked-ether trên bản vá: chỉ là hồi quy khi bản gốc chưa có cảnh báo này (cảnh báo sẵn có của
    # contract gốc không phải lỗi của bản vá).
    tool_result = tool_result or {}
    original_slither = (tool_result.get("tools") or {}).get("slither", {}).get("original_findings", [])
    if not any((item.get("check") or item.get("name")) == "locked-ether" for item in original_slither):
        for finding in (tool_result.get("slither") or {}).get("findings", []):
            if (finding.get("check") or finding.get("name")) == "locked-ether":
                findings.append({
                    "unit_id": "",
                    "severity": finding.get("severity") or finding.get("impact") or "Medium",
                    "check": "slither-locked-ether",
                    "evidence": finding.get("message") or finding.get("description") or "Patched contract can lock Ether.",
                })

    return {
        "status": "failed" if findings else "passed",
        "findings": findings,
        "checks": checks,
        "ast_verified": use_ast,
    }
