"""
Contract context around one code unit, used in the Advisor and Fixer prompts.

Instead of pasting the whole contract (which overflows the 4096-token server limit),
the prompt gets the state variables, the modifiers with their bodies, the events and
the invariants of the target function that a patch must keep.
"""

from __future__ import annotations

import re

from E2E.phase0_parsing.code_units import CodeUnit, mask_noncode


# Khai báo tham số / biến cục bộ: "<kiểu>[[]] [memory|storage|calldata] <tên>" đứng trước "=", ";", "," hoặc ")".
_LOCAL_DECL_RE = re.compile(
    r"\b(?:u?int\d*|address(?:\s+payable)?|bool|bytes\d*|string|var|[A-Z][\w$]*)"
    r"(?:\s*\[[^\]]*\])*\s+(?:(memory|storage|calldata)\s+)?([A-Za-z_$][\w$]*)\s*(?=[=;,)])"
)

# Câu lệnh cấp contract KHÔNG phải biến trạng thái (khai báo hàm không thân, event, using, ...).
_NOT_STATE_VARIABLE = (
    "function ", "modifier ", "event ", "error ", "using ", "struct ", "enum ",
    "import ", "pragma ", "constructor", "fallback", "receive",
)


# Nhiệm vụ: tên tham số và biến cục bộ KHÔNG phải storage của một hàm (gán lại chúng không làm đổi state;
# biến `storage` là tham chiếu tới state nên không tính là cục bộ).
# Mục đích: một định nghĩa duy nhất cho cả bất biến trong prompt (extract_function_invariants) và guard hành vi
# của Fixer (phase4_fixer/behavior_guard.py).
def local_variable_names(code: str) -> set[str]:
    """Parameters and non-storage local variables: rewriting them is not a state change."""
    names = set()
    for match in _LOCAL_DECL_RE.finditer(code):
        if match.group(1) != "storage":
            names.add(match.group(2))
    for match in re.finditer(r"\(\s*bool\s+([A-Za-z_$][\w$]*)", code):
        names.add(match.group(1))
    return names


# Nhiệm vụ: trích "bất biến" của hàm gốc bằng regex: biến TRẠNG THÁI bị ghi (=, +=, ...; bỏ tham số và biến
# cục bộ), event được phát, lời gọi chuyển ETH / call ra ngoài, biểu thức return.
# Mục đích: đưa vào prompt Fixer ("không được làm mất những thứ này"); việc kiểm tra bản vá thật sự do
# behavior_guard (Phase 4) và behavior_verifier (Phase 5) làm.
def extract_function_invariants(func_code: str) -> dict[str, list[str]]:
    """Tự động trích xuất các bất biến logic từ mã nguồn hàm gốc.

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

    # Tìm trên `clean` (đã che comment + chuỗi) để không khớp nhầm vào nội dung chuỗi/comment,
    # nhưng lấy chữ hiển thị từ `readable` (chỉ che comment). Hai bản cùng độ dài nên cắt theo offset là khớp.
    clean = mask_noncode(func_code)
    readable = mask_noncode(func_code, mask_strings=False)

    # Đoạn mã gốc (giữ chuỗi, bỏ comment) tương ứng với group của một match trên `clean`.
    def text_of(match: re.Match[str], group: int = 1) -> str:
        return readable[match.start(group) : match.end(group)].strip()

    # 1. State writes: biến bị ghi với =, +=, -=, *=, /=, %= (bỏ tham số và biến cục bộ như `uint x = 0`)
    local_names = local_variable_names(clean)
    write_re = re.compile(r"\b([a-zA-Z0-9_]+(?:\s*\[[^\]]+\])*(?:\.[a-zA-Z0-9_]+)*)\s*(\+=|-=|\*=|\/=|%=|=)(?!=)")
    for match in write_re.finditer(clean):
        lhs = text_of(match)
        if re.match(r"[A-Za-z0-9_]+", lhs).group() in local_names:
            continue
        invariants["state_writes"].append(f"{lhs} {match.group(2).strip()}")

    # 2. Events: emit Event(...) hoặc Event(...) trong 0.4
    event_re = re.compile(r"(?:emit\s+([A-Za-z_$][\w$]*)|(?<!function\s)(?<!return\s)(?<!require\()(?<!assert\()(?<!revert\()([A-Z][A-Za-z0-9_$]*))\s*\(")
    for match in event_re.finditer(clean):
        ev_name = match.group(1) or match.group(2)
        if ev_name and ev_name not in {"SafeMath", "ERC20", "StandardToken", "Token"}:
            invariants["events"].append(ev_name)

    # 3. External calls / ETH transfers
    ext_re = re.compile(r"\b([a-zA-Z0-9_\[\]\.]+\s*\.\s*(?:transfer|send|call\.value|call\{value:[^}]+\}|call))\s*\(")
    for match in ext_re.finditer(clean):
        invariants["external_calls"].append(text_of(match))

    # 4. Returns
    # Group lấy cả khoảng trắng sau `return`: chuỗi đã bị che thành dấu cách, `\s+` sẽ nuốt mất nó.
    ret_re = re.compile(r"\breturn(?=\s)([^;]+);")
    for match in ret_re.finditer(clean):
        expression = text_of(match)
        if expression:
            invariants["returns"].append(expression)

    # Deduplicate preserving order
    for k in invariants:
        invariants[k] = list(dict.fromkeys(invariants[k]))

    return invariants


# Vị trí của dấu "}" khớp với dấu "{" tại open_pos (trên mã đã che comment/chuỗi); không khớp -> cuối mã.
def _matching_brace(masked: str, open_pos: int) -> int:
    depth = 0
    for position in range(open_pos, len(masked)):
        if masked[position] == "{":
            depth += 1
        elif masked[position] == "}":
            depth -= 1
            if depth == 0:
                return position
    return len(masked)


# Các câu lệnh kết thúc bằng ";" nằm ĐÚNG ở độ sâu ngoặc `depth` trong đoạn [start, end) của mã đã che.
# Thân hàm/modifier/struct sâu hơn nên bị bỏ qua -> biến cục bộ không bị nhầm thành biến trạng thái.
def _statements_at_depth(masked: str, start: int, end: int, depth: int) -> list[tuple[int, int]]:
    statements: list[tuple[int, int]] = []
    level = 0
    statement_start = start
    for position in range(start, end):
        char = masked[position]
        if char in "{}":
            level += 1 if char == "{" else -1
            if level == depth:
                statement_start = position + 1
        elif char == ";" and level == depth:
            statements.append((statement_start, position + 1))
            statement_start = position + 1
    return statements


# Nhiệm vụ: dựng đoạn ngữ cảnh ngắn quanh hàm đang xét: biến trạng thái của contract chứa hàm, modifier kèm thân,
# khai báo event, và bất biến của hàm (extract_function_invariants); cắt theo ngân sách ~1500 token.
# Cấu trúc (ngoặc, ";") được tìm trên mã đã che comment/chuỗi; chữ hiển thị lấy từ mã chỉ che comment.
# Mục đích: thay cho việc dán cả contract vào prompt (vượt giới hạn 4096 token của server)
# mà Advisor/Fixer vẫn biết hàm dùng biến/modifier nào.
def build_contract_context(
    unit: CodeUnit | dict,
    source_code: str,
    budget_tokens: int = 1500,
) -> str:
    """Xây dựng ngữ cảnh hợp đồng bao quanh cho Advisor và Fixer.

    Thay thế việc nhồi 12,000 ký tự hợp đồng nguyên vẹn (dẫn tới vượt max_tokens 4096 và cắt prompt).
    Trích xuất:
    - Biến trạng thái hợp đồng (mapping, address, uint, balance, v.v.)
    - Modifiers kèm toàn bộ thân hàm (ví dụ: onlyOwner { require(msg.sender == owner); _; })
    - Khai báo Events
    - Bất biến bắt buộc bảo toàn
    """
    if isinstance(unit, dict):
        contract_name = str(unit.get("contract_name") or "")
        unit_code = str(unit.get("code") or "")
    else:
        contract_name = unit.contract_name
        unit_code = unit.code

    masked = mask_noncode(source_code)
    readable = mask_noncode(source_code, mask_strings=False)
    sections: list[str] = []

    # 1. Hợp đồng bao quanh: khai báo cấp contract nằm ở độ sâu 0 bên trong thân của nó.
    #    Không tìm thấy (finding cấp contract) -> xét mọi contract trong file (độ sâu 1 tính từ đầu file).
    contract_match = re.search(
        rf"\b(?:contract|library|interface)\s+{re.escape(contract_name)}\b[^{{]*\{{",
        masked,
    )
    if contract_match:
        body_start, body_end, depth = contract_match.end(), _matching_brace(masked, contract_match.end() - 1), 0
    else:
        body_start, body_end, depth = 0, len(masked), 1
    declarations = [
        re.sub(r"\s+", " ", readable[start:end]).strip()
        for start, end in _statements_at_depth(masked, body_start, body_end, depth)
    ]

    # 2. Biến trạng thái
    state_vars = [item for item in declarations if item and not item.startswith(_NOT_STATE_VARIABLE)]
    if state_vars:
        sections.append("State Variables:\n" + "\n".join(state_vars[:30]))

    # 3. Modifiers kèm toàn bộ thân hàm (tham số không bắt buộc: `modifier onlyOwner { ... }`): modifier của
    #    contract bao quanh + modifier mà hàm đích dùng trong header dù khai báo ở contract cha (vd Ownable).
    unit_header = mask_noncode(unit_code).split("{", 1)[0]
    modifiers: list[str] = []
    for match in re.finditer(r"\bmodifier\s+([A-Za-z_$][\w$]*)\s*(?:\([^)]*\))?[^{;]*\{", masked):
        in_contract = body_start <= match.start() < body_end
        if not in_contract and not re.search(rf"\b{re.escape(match.group(1))}\b", unit_header):
            continue
        close = _matching_brace(masked, match.end() - 1)
        modifiers.append(readable[match.start() : close + 1].strip())
    if modifiers:
        sections.append("Modifiers with Bodies (DO NOT reinvent these):\n" + "\n\n".join(modifiers))

    # 4. Events
    events = [item for item in declarations if item.startswith("event ")]
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
