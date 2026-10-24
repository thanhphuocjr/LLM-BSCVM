from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Iterable


DECLARATION_RE = re.compile(
    r"\b(function\s+(?P<function>[A-Za-z_$][\w$]*)\s*\(|"
    r"function\s*(?P<unnamed_fallback>\()\s*|"
    r"(?P<special>constructor|receive|fallback)\s*\()",
    re.MULTILINE,
)
CONTRACT_RE = re.compile(r"\b(?:abstract\s+)?(contract|library|interface)\s+([A-Za-z_$][\w$]*)")


@dataclass(frozen=True)
class CodeUnit:
    unit_id: str
    kind: str
    name: str
    contract_name: str
    start_offset: int
    end_offset: int
    start_line: int
    end_line: int
    code: str
    has_body: bool = True
    is_view_or_pure: bool = False
    visibility: str = "public"

    def to_dict(self) -> dict:
        return asdict(self)


def _line_number(code: str, offset: int) -> int:
    return code.count("\n", 0, max(0, offset)) + 1


def _skip_comment_or_string(code: str, index: int) -> int | None:
    if code.startswith("//", index):
        newline = code.find("\n", index + 2)
        return len(code) if newline < 0 else newline + 1
    if code.startswith("/*", index):
        end = code.find("*/", index + 2)
        return len(code) if end < 0 else end + 2
    if code[index] in {'"', "'"}:
        quote = code[index]
        cursor = index + 1
        while cursor < len(code):
            if code[cursor] == "\\":
                cursor += 2
                continue
            if code[cursor] == quote:
                return cursor + 1
            cursor += 1
        return len(code)
    return None


def _mask_noncode(code: str) -> str:
    masked = list(code)
    cursor = 0
    while cursor < len(code):
        end = _skip_comment_or_string(code, cursor)
        if end is None:
            cursor += 1
            continue
        for index in range(cursor, end):
            if masked[index] != "\n":
                masked[index] = " "
        cursor = end
    return "".join(masked)


def _declaration_end(code: str, declaration_start: int, search_start: int) -> tuple[int, bool]:
    """
    Tìm vị trí kết thúc của khai báo hàm và xác định hàm có thân (has_body) hay không.
    Trả về: (end_offset, has_body)
    """
    cursor = search_start
    while cursor < len(code):
        skipped = _skip_comment_or_string(code, cursor)
        if skipped is not None:
            cursor = skipped
            continue
        if code[cursor] == ";":
            # Hàm interface / abstract không có body
            return cursor + 1, False
        if code[cursor] == "{":
            depth = 1
            cursor += 1
            while cursor < len(code) and depth:
                skipped = _skip_comment_or_string(code, cursor)
                if skipped is not None:
                    cursor = skipped
                    continue
                if code[cursor] == "{":
                    depth += 1
                elif code[cursor] == "}":
                    depth -= 1
                cursor += 1
            return cursor, True
        cursor += 1
    return len(code), False


def _contract_at(code: str, offset: int) -> tuple[str, str]:
    """Trả về (contract_name, contract_kind) tại vị trí offset."""
    matches = list(CONTRACT_RE.finditer(_mask_noncode(code), 0, offset))
    if matches:
        return matches[-1].group(2), matches[-1].group(1)
    return "UnknownContract", "contract"


def parse_code_units(code: str) -> list[CodeUnit]:
    units: list[CodeUnit] = []
    names: dict[str, int] = {}
    masked_code = _mask_noncode(code)
    for match in DECLARATION_RE.finditer(masked_code):
        if match.group("unnamed_fallback"):
            name = "fallback"
            kind = "fallback"
        else:
            name = match.group("function") or match.group("special") or "unknown"
            kind = "function" if match.group("function") else name
        end, has_body = _declaration_end(code, match.start(), match.end())
        contract_name, contract_kind = _contract_at(code, match.start())
        if contract_kind == "interface":
            has_body = False

        base_id = f"{contract_name}.{name}"
        names[base_id] = names.get(base_id, 0) + 1
        suffix = f"#{names[base_id]}" if names[base_id] > 1 else ""
        snippet = code[match.start() : end].strip()

        # Kiểm tra view/pure và visibility
        header_part = snippet[: snippet.find("{")] if "{" in snippet else snippet
        is_view_or_pure = bool(re.search(r"\b(view|pure)\b", header_part))
        vis_match = re.search(r"\b(public|external|internal|private)\b", header_part)
        visibility = vis_match.group(1) if vis_match else "public"

        units.append(
            CodeUnit(
                unit_id=f"{base_id}{suffix}",
                kind=kind,
                name=name,
                contract_name=contract_name,
                start_offset=match.start(),
                end_offset=end,
                start_line=_line_number(code, match.start()),
                end_line=_line_number(code, end),
                code=snippet,
                has_body=has_body,
                is_view_or_pure=is_view_or_pure,
                visibility=visibility,
            )
        )
    if units:
        return units
    stripped = code.strip()
    return [
        CodeUnit(
            unit_id="source",
            kind="source",
            name="source",
            contract_name=_contract_at(code, len(code))[0],
            start_offset=code.find(stripped) if stripped else 0,
            end_offset=len(code),
            start_line=1,
            end_line=_line_number(code, len(code)),
            code=stripped,
            has_body=True,
            is_view_or_pure=False,
            visibility="public",
        )
    ]


def format_solidity_declaration_for_insertion(decl: str, base_indent: str = "    ") -> str:
    """Format and adjust replacement function declaration to match contract base indentation and Solidity Style Guide."""
    raw_lines = [l.rstrip() for l in decl.strip().split("\n") if l.strip()]
    if not raw_lines:
        return decl.strip()
    formatted = []
    level = 0
    for i, line in enumerate(raw_lines):
        trimmed = line.strip()
        leading_closes = 0
        for ch in trimmed:
            if ch == "}":
                leading_closes += 1
            else:
                break
        current_level = max(0, level - leading_closes)
        if i == 0:
            formatted.append(trimmed)
        else:
            indent = base_indent + ("    " * current_level)
            formatted.append(f"{indent}{trimmed}")
        net = trimmed.count("{") - trimmed.count("}")
        level = max(0, level + net)
    return "\n".join(formatted)


def apply_unit_patches(code: str, units: Iterable[CodeUnit], patches: Iterable[dict]) -> tuple[str, list[dict]]:
    by_id = {unit.unit_id: unit for unit in units}
    accepted: list[tuple[CodeUnit, str, dict]] = []
    outcomes: list[dict] = []
    seen: set[str] = set()

    for patch in patches:
        unit_id = str(patch.get("unit_id") or "")
        replacement = str(patch.get("replacement") or "").strip()
        if unit_id in seen:
            outcomes.append({"unit_id": unit_id, "status": "rejected", "reason": "duplicate patch"})
            continue
        unit = by_id.get(unit_id)
        if unit is None:
            outcomes.append({"unit_id": unit_id, "status": "rejected", "reason": "unknown unit_id"})
            continue
        if not replacement:
            outcomes.append({"unit_id": unit_id, "status": "rejected", "reason": "empty replacement"})
            continue
        if re.sub(r"\s+", "", replacement) == re.sub(r"\s+", "", unit.code):
            outcomes.append({"unit_id": unit_id, "status": "rejected", "reason": "replacement is identical to original (no changes)"})
            continue
        seen.add(unit_id)
        accepted.append((unit, replacement, patch))

    updated = code
    for unit, replacement, patch in sorted(accepted, key=lambda item: item[0].start_offset, reverse=True):
        line_start = code.rfind("\n", 0, unit.start_offset)
        line_start = 0 if line_start == -1 else line_start + 1
        leading_space = code[line_start:unit.start_offset]
        base_indent = leading_space if leading_space.strip() == "" and len(leading_space) > 0 else "    "
        formatted_replacement = format_solidity_declaration_for_insertion(replacement, base_indent)
        updated = updated[: unit.start_offset] + formatted_replacement + updated[unit.end_offset :]
        outcomes.append(
            {
                "unit_id": unit.unit_id,
                "status": "applied",
                "summary": str(patch.get("summary") or ""),
            }
        )
    return updated, outcomes


def validate_observable_behavior(original: str, replacement: str) -> None:
    """
    Guard bảo toàn hành vi (Phase 4 / Phase 5).
    Kiểm tra các tín hiệu observable:
      - ETH transfers: call.value, transfer, send
      - Event emissions: emit EventName(...) hoặc EventName(...) trong Solidity 0.4
      - State variable writes: phát hiện phép gán state (ví dụ: balances[...] += ..., owner = ...)
      - Return statements
    """
    # 1. External ETH calls & transfers
    eth_call_original = bool(re.search(r"(\.call\s*\{\s*value\s*:|\.call\.value\s*\()", original))
    eth_call_replacement = bool(re.search(r"(\.call\s*\{\s*value\s*:|\.call\.value\s*\()", replacement))
    if eth_call_original and not eth_call_replacement:
        raise ValueError("Generated patch removed observable behavior: ETH value call")

    eth_transfer_original = bool(re.search(r"\.transfer\s*\(", original))
    eth_transfer_replacement = bool(re.search(r"\.transfer\s*\(", replacement))
    if eth_transfer_original and not eth_transfer_replacement:
        raise ValueError("Generated patch removed observable behavior: ETH transfer")

    eth_send_original = bool(re.search(r"\.send\s*\(", original))
    eth_send_replacement = bool(re.search(r"\.send\s*\(", replacement))
    if eth_send_original and not eth_send_replacement:
        raise ValueError("Generated patch removed observable behavior: ETH send")

    # 2. Return statements
    if re.search(r"\breturn\b", original) and not re.search(r"\breturn\b", replacement):
        raise ValueError("Generated patch removed observable behavior: return statement")

    # 3. Events: bao gồm cả cú pháp emit hiện đại và gọi trực tiếp trong 0.4
    orig_events = set(re.findall(r"(?:emit\s+)?([A-Z][A-Za-z0-9_$]*)\s*\(", original))
    # Bỏ qua các hàm built-in/keywords có thể viết hoa
    orig_events.discard("Assert")
    orig_events.discard("Require")
    if orig_events:
        repl_events = set(re.findall(r"(?:emit\s+)?([A-Z][A-Za-z0-9_$]*)\s*\(", replacement))
        missing_events = orig_events - repl_events
        if missing_events:
            raise ValueError(f"Generated patch removed observable event emission: {', '.join(missing_events)}")

    # 4. State variable writes: phát hiện các câu lệnh gán biến state như balances[to] += amount, etc.
    state_write_pattern = re.compile(r"([A-Za-z_$][\w$]*(?:\[[^\]]+\])?)\s*(\+=|-=|\*=|/=|%=|=)(?!=)")
    orig_writes = set()
    for m in state_write_pattern.finditer(original):
        var_name = m.group(1).split("[")[0].strip()
        # Loại trừ biến cục bộ thường gặp
        if var_name not in {"uint", "uint256", "bool", "address", "bytes32", "var", "int"}:
            orig_writes.add(var_name)

    if orig_writes:
        repl_writes = set()
        for m in state_write_pattern.finditer(replacement):
            var_name = m.group(1).split("[")[0].strip()
            repl_writes.add(var_name)
        missing_writes = orig_writes - repl_writes
        if missing_writes:
            raise ValueError(f"Generated patch removed state variable mutation: {', '.join(missing_writes)}")


def extract_declaration_body(text: str, target_name: str | None = None) -> str:
    """
    Trích xuất một khai báo hàm/constructor duy nhất từ phản hồi của model.
    Ưu tiên parse JSON {"replacement": "..."} trước.
    Sau đó kiểm tra code fence ```solidity.
    Fallback đếm ngoặc chính xác, bỏ qua comment và chuỗi ký tự.
    """
    if not text or not text.strip():
        return ""

    raw_text = text.strip()

    # 1. Parse JSON format {"replacement": "..."}
    try:
        # Tìm khối JSON nếu có
        json_match = re.search(r"\{[\s\S]*\"replacement\"[\s\S]*\}", raw_text)
        if json_match:
            data = json.loads(json_match.group(0))
            if isinstance(data, dict) and "replacement" in data and data["replacement"]:
                return extract_declaration_body(str(data["replacement"]), target_name=target_name)
    except Exception:
        pass

    # 2. Tìm khối code fence ```solidity ... ```
    fence_matches = list(re.finditer(r"```(?:solidity)?\s*([\s\S]*?)```", raw_text))
    search_space = raw_text
    if fence_matches:
        # Nếu có fence, dùng nội dung fence đầu tiên có chứa target_name
        for fm in fence_matches:
            f_code = fm.group(1).strip()
            if not target_name or (target_name in f_code):
                search_space = f_code
                break

    # 3. Xây dựng pattern tìm vị trí bắt đầu khai báo
    if target_name in ("fallback", "receive"):
        pattern = rf"\b((?:function\s+{re.escape(target_name)}\s*\(|function\s*\(|(?:constructor|receive|fallback)\s*\())"
    elif target_name:
        pattern = rf"\b(function\s+{re.escape(target_name)}\s*\(|(?:constructor|receive|fallback)\s*\()"
    else:
        pattern = r"\b(function\s+[A-Za-z_$][\w$]*\s*\(|function\s*\(|(?:constructor|receive|fallback)\s*\()"

    match = re.search(pattern, search_space)
    if not match:
        if target_name:
            return ""
        return search_space.strip()

    start_idx = match.start()
    brace_idx = search_space.find("{", start_idx)
    if brace_idx < 0:
        return search_space[start_idx:].strip()

    # 4. Đếm ngoặc chính xác từ brace_idx, bỏ qua string & comment
    depth = 0
    quote = ""
    line_comment = False
    block_comment = False
    cursor = brace_idx

    while cursor < len(search_space):
        char = search_space[cursor]
        pair = search_space[cursor : cursor + 2]
        if line_comment:
            if char == "\n":
                line_comment = False
            cursor += 1
            continue
        if block_comment:
            if pair == "*/":
                block_comment = False
                cursor += 2
                continue
            cursor += 1
            continue
        if pair == "//":
            line_comment = True
            cursor += 2
            continue
        if pair == "/*":
            block_comment = True
            cursor += 2
            continue
        if char in {'"', "'"}:
            if not quote:
                quote = char
            elif quote == char:
                quote = ""
        elif not quote:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    cursor += 1
                    break
        cursor += 1

    extracted = search_space[start_idx:cursor].strip()

    # Kiểm tra đảm bảo extracted chỉ chứa MỘT khai báo hàm duy nhất
    inner_declarations = list(re.finditer(r"\bfunction\s+[A-Za-z_$][\w$]*\s*\(", extracted[10:]))
    if inner_declarations:
        first_inner = inner_declarations[0].start() + 10
        last_close = extracted.rfind("}", 0, first_inner)
        if last_close > 0:
            extracted = extracted[: last_close + 1].strip()

    return extracted


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

