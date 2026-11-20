"""Phase 5 - bản vá có giữ hành vi không: so AST từng hàm đã vá; không so được bằng AST -> guard cú pháp của Phase 4."""

import json

from E2E.common.solc import compile_solidity_code, select_best_solc
from E2E.phase0_parsing.code_units import parse_code_units
from E2E.phase4_fixer.behavior_guard import validate_observable_behavior

# Lời gọi chuyển giá trị; các wrapper an toàn được coi là cùng hành vi.
TRANSFER_MEMBERS = {"transfer": "transfer", "safeTransfer": "transfer", "send": "transfer", "sendValue": "transfer",
                    "safeTransferETH": "transfer", "call": "call", "delegatecall": "delegatecall",
                    "transferFrom": "transferFrom", "safeTransferFrom": "transferFrom"}
CALL_LABELS = {"call": "ETH value call / low-level call", "transfer": "ETH or token transfer",
               "transferFrom": "token transferFrom", "delegatecall": "delegatecall"}
_VALUE_MOVES = {"transfer", "call"}


# JSON AST từ output của solc --ast-compact-json (cắt từ "{" đầu tới "}" cuối); không đọc được -> None.
def _extract_ast_dict(raw_ast_stdout: str) -> dict | None:
    start, end = raw_ast_stdout.find("{"), raw_ast_stdout.rfind("}")
    if not raw_ast_stdout or start == -1 or end <= start:
        return None
    try:
        return json.loads(raw_ast_stdout[start:end + 1])
    except Exception:
        return None


# Duyệt mọi nút dict của cây AST theo thứ tự trước (pre-order).
def _iter_nodes(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            for item in value if isinstance(value, list) else [value]:
                yield from _iter_nodes(item)


# Tên biến gốc của một nút AST (Identifier / IndexAccess / MemberAccess): balances[msg.sender] -> balances.
def _var_name(node: dict) -> str:
    kind = node.get("nodeType")
    if kind == "Identifier":
        return str(node.get("name") or "")
    if kind == "IndexAccess":
        return _var_name(node.get("baseExpression") or {})
    if kind == "MemberAccess":
        base, member = _var_name(node.get("expression") or {}), node.get("memberName") or ""
        return f"{base}.{member}" if base else member
    return ""


# Đặc trưng hành vi của MỘT hàm từ AST: biến state bị ghi (kèm toán tử), lời gọi chuyển giá trị, event, các lệnh return.
def _collect_function_ast_features(fn_node: dict, state_vars: set[str] | None = None) -> dict:
    writes, calls, events, returns = set(), set(), set(), []
    for node in _iter_nodes(fn_node):
        kind = node.get("nodeType")
        if kind == "Assignment":
            name = _var_name(node.get("leftHandSide") or {})
            if name and (state_vars is None or name.split(".")[0] in state_vars or name in state_vars):
                writes.add((name, str(node.get("operator") or "=")))
        elif kind == "MemberAccess" and node.get("memberName") in TRANSFER_MEMBERS:
            calls.add(TRANSFER_MEMBERS[node["memberName"]])
        elif kind == "FunctionCall":
            expression = node.get("expression") or {}
            name = _var_name(expression)
            if node.get("kind") == "functionCall" and expression.get("nodeType") == "Identifier" \
                    and name and name[0].isupper():
                events.add(name)
        elif kind == "EmitStatement":
            if name := _var_name((node.get("eventCall") or {}).get("expression") or {}):
                events.add(name)
        elif kind == "Return":
            expression = node.get("expression")
            returns.append((_var_name(expression) or str(expression.get("value") or "expr")) if expression else "void")
    return {"state_writes": writes, "external_calls": calls, "events": events, "returns": returns}


# unit_id -> nút AST của hàm (hàm trùng tên thêm hậu tố "#n" giống Phase 0; tên trần cũng được map).
def _extract_ast_functions_map(ast_dict: dict) -> dict[str, dict]:
    functions = {}
    for contract in (ast_dict or {}).get("nodes", []):
        if contract.get("nodeType") != "ContractDefinition":
            continue
        seen = {}
        for node in contract.get("nodes", []):
            if node.get("nodeType") == "FunctionDefinition":
                name = node.get("name") or node.get("kind") or "fallback"
                seen[name] = seen.get(name, 0) + 1
                suffix = f"#{seen[name]}" if seen[name] > 1 else ""
                functions[f"{contract.get('name') or 'Contract'}.{name}{suffix}"] = node
                functions.setdefault(name, node)
    return functions


# Tên các biến trạng thái khai báo trong các contract (từ AST).
def _state_variable_names(ast_dict: dict) -> set[str]:
    return {str(node.get("name") or "") for contract in (ast_dict or {}).get("nodes", [])
            if contract.get("nodeType") == "ContractDefinition" for node in contract.get("nodes", [])
            if node.get("nodeType") == "VariableDeclaration" and node.get("stateVariable")}


# So AST hàm gốc / hàm vá: mất lệnh ghi một biến state hoặc lời gọi chuyển giá trị (High), mất event / return (Medium).
def _compare_unit_behavior_ast(orig_fn_ast: dict, fixed_fn_ast: dict, unit_id: str,
                               state_vars: set[str] | None = None) -> list[dict]:
    old = _collect_function_ast_features(orig_fn_ast, state_vars)
    new = _collect_function_ast_features(fixed_fn_ast, state_vars)
    finding = lambda severity, check, evidence: {"unit_id": unit_id, "severity": severity, "check": check,
                                                 "evidence": evidence}
    findings = []
    if missing := {name for name, _ in old["state_writes"]} - {name for name, _ in new["state_writes"]}:
        findings.append(finding("High", "ast-state-write-preservation",
                                f"Patch removed state variable mutation for: {', '.join(sorted(missing))}"))
    missing_calls = old["external_calls"] - new["external_calls"]
    if missing_calls & _VALUE_MOVES and new["external_calls"] & _VALUE_MOVES:
        missing_calls -= _VALUE_MOVES
    if missing_calls:
        findings.append(finding("High", "ast-external-call-preservation", "Patch removed observable behavior: "
                                + ", ".join(CALL_LABELS.get(call, call) for call in sorted(missing_calls))))
    if missing := old["events"] - new["events"]:
        findings.append(finding("Medium", "ast-event-emission-preservation",
                                f"Patch removed event emission: {', '.join(sorted(missing))}"))
    if old["returns"] and not new["returns"]:
        findings.append(finding("Medium", "ast-return-preservation",
                                "Patch removed return statement present in original function"))
    return findings


# Mã của một unit theo id (khớp chính xác, hoặc khớp phần đuôi "Contract.f" <-> "f").
def _unit_code(units: dict[str, str], unit_id: str) -> str | None:
    return units.get(unit_id) or next((code for key, code in units.items()
                                       if key.endswith(f".{unit_id}") or unit_id.endswith(f".{key}")), None)


# Cảnh báo Slither "locked-ether" do bản vá sinh ra (bản gốc chưa có) -> hồi quy hành vi.
def _locked_ether_regressions(tool_result: dict) -> list[dict]:
    original = (tool_result.get("tools") or {}).get("slither", {}).get("original_findings", [])
    if any((item.get("check") or item.get("name")) == "locked-ether" for item in original):
        return []
    patched = (tool_result.get("slither") or {}).get("findings", [])
    return [{"unit_id": "", "severity": item.get("severity") or item.get("impact") or "Medium",
             "check": "slither-locked-ether",
             "evidence": item.get("message") or item.get("description") or "Patched contract can lock Ether."}
            for item in patched if (item.get("check") or item.get("name")) == "locked-ether"]


# [KIỂM TRA HỒI QUY HÀNH VI] So AST các hàm đã vá; không so được -> validate_observable_behavior; + locked-ether mới.
def verify_patch_behavior(original_code: str, fixed_code: str, patch_outcomes: list[dict] | None = None,
                          tool_result: dict | None = None) -> dict:
    applied = {str(item.get("unit_id") or "").strip() for item in patch_outcomes or []
               if item.get("status") == "applied"} - {"", "source", "contract"}
    if not applied:
        return {"status": "not_applicable", "findings": [],
                "checks": ["No applied patch requires behavioral comparison."]}
    findings, checks, ast_checked = [], [], set()
    solc_path = select_best_solc(original_code)[0]
    orig_comp = compile_solidity_code(original_code, solc_path=solc_path, with_ast=True)
    fixed_comp = compile_solidity_code(fixed_code, solc_path=solc_path, with_ast=True)
    orig_ast = _extract_ast_dict(orig_comp.get("ast") or "") if orig_comp.get("passed") else None
    fixed_ast = _extract_ast_dict(fixed_comp.get("ast") or "") if fixed_comp.get("passed") else None
    use_ast = bool(orig_ast and fixed_ast)
    if use_ast:
        orig_map, fixed_map = _extract_ast_functions_map(orig_ast), _extract_ast_functions_map(fixed_ast)
        state_vars = _state_variable_names(orig_ast) | _state_variable_names(fixed_ast)
        checks.append({"check": "ast_compilation", "engine": orig_comp.get("compiler"), "status": "ast_verified"})
        for unit_id in sorted(applied):
            old = orig_map.get(unit_id) or orig_map.get(unit_id.split(".")[-1])
            new = fixed_map.get(unit_id) or fixed_map.get(unit_id.split(".")[-1])
            if old and new:
                findings += _compare_unit_behavior_ast(old, new, unit_id, state_vars)
                ast_checked.add(unit_id)
    original_units = {unit.unit_id: unit.code for unit in parse_code_units(original_code)}
    fixed_units = {unit.unit_id: unit.code for unit in parse_code_units(fixed_code)}
    for unit_id in sorted(applied):
        original, fixed = _unit_code(original_units, unit_id), _unit_code(fixed_units, unit_id)
        if original is None or fixed is None:
            findings.append({"unit_id": unit_id, "severity": "High", "check": "patched-unit-presence",
                             "evidence": f"The patched declaration {unit_id} could not be mapped "
                                         "in both source versions."})
        elif unit_id not in ast_checked:
            try:
                validate_observable_behavior(original, fixed)
                checks.append({"unit_id": unit_id, "check": "observable-behavior-preservation", "status": "passed"})
            except ValueError as error:
                findings.append({"unit_id": unit_id, "severity": "High", "check": "observable-behavior-preservation",
                                 "evidence": str(error)})
    findings += _locked_ether_regressions(tool_result or {})
    return {"status": "failed" if findings else "passed", "findings": findings, "checks": checks,
            "ast_verified": use_ast}
