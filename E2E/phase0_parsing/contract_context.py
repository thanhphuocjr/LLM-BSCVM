"""Phase 0 - ngữ cảnh ngắn quanh một hàm cho prompt Advisor/Fixer: biến trạng thái, modifier, event, bất biến."""

from E2E.common.regex_patterns import (
    RE_BOOL_TUPLE_LOCAL, RE_INVARIANT_EVENT, RE_INVARIANT_EXTERNAL_CALL, RE_INVARIANT_RETURN, RE_INVARIANT_STATE_WRITE,
    RE_LEADING_NAME, RE_LOCAL_DECLARATION, RE_MODIFIER_OPENING, RE_WHITESPACE, re_contract_opening, re_word,
)
from E2E.phase0_parsing.code_units import CodeUnit, mask_noncode, matching_close

# Câu lệnh cấp contract KHÔNG phải biến trạng thái.
_NOT_STATE_VARIABLE = ("function ", "modifier ", "event ", "error ", "using ", "struct ", "enum ",
                       "import ", "pragma ", "constructor", "fallback", "receive")
# Tên viết hoa trông như event nhưng là thư viện/contract hay gặp.
_NOT_EVENTS = {"SafeMath", "ERC20", "StandardToken", "Token"}


# Tên tham số và biến cục bộ không phải storage (gán lại chúng không làm đổi state).
def local_variable_names(code: str) -> set[str]:
    names = {match.group(2) for match in RE_LOCAL_DECLARATION.finditer(code) if match.group(1) != "storage"}
    return names | set(RE_BOOL_TUPLE_LOCAL.findall(code))


# Bất biến của hàm gốc cho prompt Fixer: biến state bị ghi, event phát ra, lời gọi chuyển ETH/call, biểu thức return.
def extract_function_invariants(func_code: str) -> dict[str, list[str]]:
    clean, readable = mask_noncode(func_code), mask_noncode(func_code, mask_strings=False)
    text = lambda match: readable[match.start(1):match.end(1)].strip()
    local = local_variable_names(clean)
    events = (match.group(1) or match.group(2) for match in RE_INVARIANT_EVENT.finditer(clean))
    invariants = {
        "state_writes": [f"{text(m)} {m.group(2).strip()}" for m in RE_INVARIANT_STATE_WRITE.finditer(clean)
                         if RE_LEADING_NAME.match(text(m)).group() not in local],
        "events": [event for event in events if event and event not in _NOT_EVENTS],
        "external_calls": [text(m) for m in RE_INVARIANT_EXTERNAL_CALL.finditer(clean)],
        "returns": [text(m) for m in RE_INVARIANT_RETURN.finditer(clean) if text(m)],
    }
    return {key: list(dict.fromkeys(values)) for key, values in invariants.items()}


# Các câu lệnh kết thúc bằng ";" nằm đúng ở độ sâu ngoặc `depth` trong [start, end) (thân hàm sâu hơn bị bỏ qua).
def statements_at_depth(masked: str, start: int, end: int, depth: int) -> list[tuple[int, int]]:
    statements, level, statement_start = [], 0, start
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


# Dựng ngữ cảnh (~1500 token) quanh hàm: biến trạng thái, modifier kèm thân, event, bất biến của hàm.
def build_contract_context(unit: CodeUnit | dict, source_code: str, budget_tokens: int = 1500) -> str:
    contract_name = str(unit.get("contract_name") or "") if isinstance(unit, dict) else unit.contract_name
    unit_code = str(unit.get("code") or "") if isinstance(unit, dict) else unit.code
    masked, readable = mask_noncode(source_code), mask_noncode(source_code, mask_strings=False)
    opening = re_contract_opening(contract_name).search(masked)
    if opening:
        close = matching_close(masked, opening.end() - 1)
        body_start, body_end, depth = opening.end(), len(masked) if close is None else close, 0
    else:
        body_start, body_end, depth = 0, len(masked), 1
    declarations = [RE_WHITESPACE.sub(" ", readable[start:end]).strip()
                    for start, end in statements_at_depth(masked, body_start, body_end, depth)]
    sections = []
    state_vars = [item for item in declarations if item and not item.startswith(_NOT_STATE_VARIABLE)]
    if state_vars:
        sections.append("State Variables:\n" + "\n".join(state_vars[:30]))
    unit_header = mask_noncode(unit_code).split("{", 1)[0]
    modifiers = []
    for match in RE_MODIFIER_OPENING.finditer(masked):
        if body_start <= match.start() < body_end or re_word(match.group(1)).search(unit_header):
            close = matching_close(masked, match.end() - 1)
            modifiers.append(readable[match.start():(len(masked) if close is None else close) + 1].strip())
    if modifiers:
        sections.append("Modifiers with Bodies (DO NOT reinvent these):\n" + "\n\n".join(modifiers))
    events = [item for item in declarations if item.startswith("event ")]
    if events:
        sections.append("Events:\n" + "\n".join(events[:15]))
    invariants = extract_function_invariants(unit_code) if unit_code else {}
    labels = (("state_writes", "State writes to preserve"), ("events", "Events emitted"),
              ("external_calls", "External value transfers"), ("returns", "Return values"))
    inv_lines = [f"- {label}: {', '.join(invariants[key])}" for key, label in labels if invariants.get(key)]
    if inv_lines:
        inv_lines.append("CRITICAL: ALL original state write statements and event emissions MUST remain intact.")
        sections.append("Observable Invariants:\n" + "\n".join(inv_lines))
    context = "\n\n".join(sections)
    if len(context) > budget_tokens * 4:
        context = context[:budget_tokens * 4].rsplit("\n", 1)[0] + "\n// ... [truncated to preserve token budget]"
    return context
