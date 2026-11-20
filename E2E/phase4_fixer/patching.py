"""Phase 4 - lấy hàm thay thế từ câu trả lời của model (chịu được JSON hỏng) và ghép vào contract theo unit id."""

import json
from typing import Iterable

from E2E.common.regex_patterns import (
    RE_JSON_WITH_REPLACEMENT, RE_NAMED_FUNCTION_START, RE_SOLIDITY_FENCE, RE_WHITESPACE, re_declaration_start,
    re_json_string_field_start,
)
from E2E.phase0_parsing.code_units import CodeUnit, mask_noncode, matching_close

_JSON_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/"}


# Giá trị chuỗi của một trường JSON kể cả khi model quên escape dấu " bên trong (kết thúc ở " đứng trước "," hoặc "}").
def lenient_json_string_field(text: str, field: str) -> str | None:
    match = re_json_string_field_start(field).search(text)
    if not match:
        return None
    chars, cursor = [], match.end()
    while cursor < len(text):
        char = text[cursor]
        if char == "\\" and cursor + 1 < len(text):
            chars.append(_JSON_ESCAPES.get(text[cursor + 1], "\\" + text[cursor + 1]))
            cursor += 2
            continue
        if char == '"' and text[cursor + 1:].lstrip()[:1] in {",", "}"}:
            return "".join(chars)
        chars.append(char)
        cursor += 1
    return None


# Lấy ĐÚNG MỘT khai báo hàm từ câu trả lời: JSON {"replacement"} -> code fence có tên hàm -> "function <tên>(" + ngoặc.
def extract_declaration_body(text: str, target_name: str | None = None) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    if '"replacement"' in raw:
        try:
            data = json.loads(match.group(0)) if (match := RE_JSON_WITH_REPLACEMENT.search(raw)) else None
            replacement = str(data["replacement"]) if isinstance(data, dict) and data.get("replacement") else None
        except Exception:
            replacement = lenient_json_string_field(raw, "replacement")
        if replacement and replacement.strip() != raw:
            return extract_declaration_body(replacement, target_name=target_name)
    search_space = next((fence.group(1).strip() for fence in RE_SOLIDITY_FENCE.finditer(raw)
                         if not target_name or target_name in fence.group(1).strip()), raw)
    start = re_declaration_start(target_name).search(search_space)
    if not start:
        return "" if target_name else search_space.strip()
    declaration = search_space[start.start():]
    masked = mask_noncode(declaration)
    if (brace := masked.find("{")) < 0:
        return declaration.strip()
    close = matching_close(masked, brace)
    extracted = declaration[:len(masked) if close is None else close + 1].strip()
    masked_extracted = mask_noncode(extracted)
    if inner := RE_NAMED_FUNCTION_START.search(masked_extracted[10:]):
        last_close = masked_extracted.rfind("}", 0, inner.start() + 10)
        if last_close > 0:
            extracted = extracted[:last_close + 1].strip()
    return extracted


# Căn lề lại hàm thay thế theo thụt lề của contract gốc (mỗi cấp { } thêm 4 dấu cách).
def format_solidity_declaration_for_insertion(decl: str, base_indent: str = "    ") -> str:
    lines = [line.strip() for line in decl.strip().split("\n") if line.strip()]
    if not lines:
        return decl.strip()
    formatted, level = [], 0
    for index, line in enumerate(lines):
        closes = len(line) - len(line.lstrip("}"))
        formatted.append(line if index == 0 else base_indent + "    " * max(0, level - closes) + line)
        level = max(0, level + line.count("{") - line.count("}"))
    return "\n".join(formatted)


# Ghép các hàm thay thế vào mã theo unit_id (từ cuối file lên để offset không lệch); từ chối patch trùng / rỗng / y hệt.
def apply_unit_patches(code: str, units: Iterable[CodeUnit], patches: Iterable[dict]) -> tuple[str, list[dict]]:
    by_id = {unit.unit_id: unit for unit in units}
    accepted, outcomes, seen = [], [], set()
    for patch in patches:
        unit_id, replacement = str(patch.get("unit_id") or ""), str(patch.get("replacement") or "").strip()
        unit = by_id.get(unit_id)
        if unit_id in seen:
            reason = "duplicate patch"
        elif unit is None:
            reason = "unknown unit_id"
        elif not replacement:
            reason = "empty replacement"
        elif RE_WHITESPACE.sub("", replacement) == RE_WHITESPACE.sub("", unit.code):
            reason = "replacement is identical to original (no changes)"
        else:
            seen.add(unit_id)
            accepted.append((unit, replacement, patch))
            continue
        outcomes.append({"unit_id": unit_id, "status": "rejected", "reason": reason})
    updated = code
    for unit, replacement, patch in sorted(accepted, key=lambda item: item[0].start_offset, reverse=True):
        leading = code[code.rfind("\n", 0, unit.start_offset) + 1:unit.start_offset]
        indent = leading if leading and not leading.strip() else "    "
        formatted = format_solidity_declaration_for_insertion(replacement, indent)
        updated = updated[:unit.start_offset] + formatted + updated[unit.end_offset:]
        outcomes.append({"unit_id": unit.unit_id, "status": "applied", "summary": str(patch.get("summary") or "")})
    return updated, outcomes
