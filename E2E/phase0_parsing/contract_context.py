"""
Contract context around one code unit, used in the Advisor and Fixer prompts.

Instead of pasting the whole contract (which overflows the 4096-token server limit),
the prompt gets the state variables, the modifiers with their bodies, the events and
the invariants of the target function that a patch must keep.
"""

from __future__ import annotations

import re

from E2E.phase0_parsing.code_units import CodeUnit


def extract_function_invariants(func_code: str) -> dict[str, list[str]]:
    """Tự động trích xuất các bất biến logic từ mã nguồn hàm gốc (Task B3, B4, D3).
    
    Bao gồm:
    - Biến trạng thái bị ghi (state mutations)
    - Sự kiện được gọi (events, cả 0.4 lẫn 0.5+)
    - Các luồng chuyển ETH / calls ra ngoài (external value calls)
    - Các biểu thức trả về (return statements)
    """
    invariants: dict[str, list[str]] = {
        "state_writes": [],
        "events": [],
        "external_calls": [],
        "returns": [],
    }

    # Bỏ comment để tránh nhận diện nhầm
    clean = re.sub(r"//[^\n]*", "", func_code)
    clean = re.sub(r"/\*.*?\*/", "", clean, flags=re.DOTALL)

    # 1. State writes: biến bị ghi với =, +=, -=, *=, /=, %=
    write_re = re.compile(r"\b([a-zA-Z0-9_]+(?:\s*\[[^\]]+\])*(?:\.[a-zA-Z0-9_]+)*)\s*(\+=|-=|\*=|\/=|%=|=)(?!=)")
    for match in write_re.finditer(clean):
        lhs = match.group(1).strip()
        op = match.group(2).strip()
        # Bỏ qua khai báo biến cục bộ như uint x = 0, address a = msg.sender
        if not re.match(r"^(?:uint\d*|int\d*|address|bool|bytes\d*|string|var)\s+", lhs):
            invariants["state_writes"].append(f"{lhs} {op}")

    # 2. Events: emit Event(...) hoặc Event(...) trong 0.4
    event_re = re.compile(r"(?:emit\s+([A-Za-z_$][\w$]*)|(?<!function\s)(?<!return\s)(?<!require\()(?<!assert\()(?<!revert\()([A-Z][A-Za-z0-9_$]*))\s*\(")
    for match in event_re.finditer(clean):
        ev_name = match.group(1) or match.group(2)
        if ev_name and ev_name not in {"SafeMath", "ERC20", "StandardToken", "Token"}:
            invariants["events"].append(ev_name)

    # 3. External calls / ETH transfers
    ext_re = re.compile(r"\b([a-zA-Z0-9_\[\]\.]+\s*\.\s*(?:transfer|send|call\.value|call\{value:[^}]+\}|call))\s*\(")
    for match in ext_re.finditer(clean):
        invariants["external_calls"].append(match.group(1).strip())

    # 4. Returns
    ret_re = re.compile(r"\breturn\s+([^;]+);")
    for match in ret_re.finditer(clean):
        invariants["returns"].append(match.group(1).strip())

    # Deduplicate preserving order
    for k in invariants:
        invariants[k] = list(dict.fromkeys(invariants[k]))

    return invariants


def build_contract_context(
    unit: CodeUnit | dict,
    source_code: str,
    budget_tokens: int = 1500,
) -> str:
    """Xây dựng ngữ cảnh hợp đồng bao quanh cho Advisor, Assessor và Fixer (Task B3).
    
    Thay thế việc nhồi 12,000 ký tự hợp đồng nguyên vẹn (dẫn tới vượt max_tokens 4096 và cắt prompt).
    Trích xuất:
    - Biến trạng thái hợp đồng (mapping, address, uint, balance, v.v.)
    - Modifiers kèm toàn bộ thân hàm (ví dụ: onlyOwner { require(msg.sender == owner); _; })
    - Khai báo Events
    - Hàm gọi tới / được gọi bởi unit
    - Bất biến bắt buộc bảo toàn
    """
    if isinstance(unit, dict):
        unit_name = str(unit.get("name") or "")
        contract_name = str(unit.get("contract_name") or "")
        unit_code = str(unit.get("code") or "")
    else:
        unit_name = unit.name
        contract_name = unit.contract_name
        unit_code = unit.code

    sections: list[str] = []

    # 1. Tìm hợp đồng bao quanh
    contract_match = re.search(
        rf"\b(?:contract|library|interface)\s+{re.escape(contract_name)}\b[^{{]*\{{",
        source_code,
    )
    contract_body = ""
    if contract_match:
        c_start = contract_match.end() - 1
        depth = 0
        c_end = len(source_code)
        for i in range(c_start, len(source_code)):
            if source_code[i] == "{":
                depth += 1
            elif source_code[i] == "}":
                depth -= 1
                if depth == 0:
                    c_end = i + 1
                    break
        contract_body = source_code[c_start:c_end]
    else:
        contract_body = source_code

    # 2. Trích xuất biến trạng thái
    state_vars: list[str] = []
    state_re = re.compile(
        r"^[ \t]*(?:(?:mapping\s*\(.*?\)|address(?:\s+payable)?|uint\d*|int\d*|bool|bytes\d*|string|[A-Z][a-zA-Z0-9_]*)\b[^\n;{}]*;)",
        re.MULTILINE,
    )
    for match in state_re.finditer(contract_body):
        sv = match.group(0).strip()
        if not sv.startswith(("function ", "event ", "return ", "require(", "revert(")):
            state_vars.append(sv)

    if state_vars:
        sections.append("State Variables:\n" + "\n".join(state_vars[:30]))

    # 3. Trích xuất modifiers kèm toàn bộ thân hàm
    mod_matches = list(re.finditer(r"\bmodifier\s+([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*\{", contract_body))
    modifiers: list[str] = []
    for mm in mod_matches:
        start_idx = mm.start()
        open_brace = contract_body.find("{", start_idx)
        if open_brace != -1:
            depth = 0
            for k in range(open_brace, len(contract_body)):
                if contract_body[k] == "{":
                    depth += 1
                elif contract_body[k] == "}":
                    depth -= 1
                    if depth == 0:
                        modifiers.append(contract_body[start_idx : k + 1].strip())
                        break
    if modifiers:
        sections.append("Modifiers with Bodies (DO NOT reinvent these):\n" + "\n\n".join(modifiers))

    # 4. Trích xuất events
    event_re = re.compile(r"^[ \t]*event\s+[A-Za-z_$][\w$]*\s*\([^;{}]*\)\s*;", re.MULTILINE)
    events = [m.group(0).strip() for m in event_re.finditer(contract_body)]
    if events:
        sections.append("Events:\n" + "\n".join(events[:15]))

    # 5. Bất biến tự động từ hàm mục tiêu
    if unit_code:
        invariants = extract_function_invariants(unit_code)
        inv_lines = []
        if invariants["state_writes"]:
            inv_lines.append(f"- State writes to preserve: {', '.join(invariants['state_writes'])}")
        if invariants["events"]:
            inv_lines.append(f"- Events emitted: {', '.join(invariants['events'])}")
        if invariants["external_calls"]:
            inv_lines.append(f"- External value transfers: {', '.join(invariants['external_calls'])}")
        if invariants["returns"]:
            inv_lines.append(f"- Return values: {', '.join(invariants['returns'])}")
        if inv_lines:
            inv_lines.append("CRITICAL: ALL original state write statements and event emissions MUST remain intact.")
            sections.append("Observable Invariants:\n" + "\n".join(inv_lines))

    full_context = "\n\n".join(sections)

    # Đảm bảo ngân sách token (~4 chars / token)
    max_chars = budget_tokens * 4
    if len(full_context) > max_chars:
        full_context = full_context[:max_chars].rsplit("\n", 1)[0] + "\n// ... [truncated to preserve token budget]"

    return full_context
