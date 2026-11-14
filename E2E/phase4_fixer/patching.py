"""
Phase 4 - turn the Fixer's reply into a patched contract.

* `extract_declaration_body` / `lenient_json_string_field`: read the replacement
  function out of the model reply (tolerates broken JSON escaping).
* `apply_unit_patches`: splice replacement functions into the source by unit id,
  re-indented to the contract's style.
"""

from __future__ import annotations

import json
from typing import Iterable

from E2E.common.regex_patterns import (
    RE_JSON_WITH_REPLACEMENT,
    RE_NAMED_FUNCTION_START,
    RE_SOLIDITY_FENCE,
    RE_WHITESPACE,
    re_declaration_start,
    re_json_string_field_start,
)
from E2E.phase0_parsing.code_units import CodeUnit, mask_noncode


# Căn lề lại hàm thay thế theo thụt lề của contract gốc (mỗi cấp { } thêm 4 dấu cách).
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


# Nhiệm vụ: ghép các hàm thay thế vào mã nguồn theo unit_id. Từ chối patch trùng, unit không tồn tại, rỗng
# hoặc giống hệt bản gốc. Ghép từ cuối file lên đầu để offset của các unit phía trước không bị lệch.
# Trả về (mã đã vá, danh sách outcome applied/rejected).
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
        if RE_WHITESPACE.sub("", replacement) == RE_WHITESPACE.sub("", unit.code):
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


_JSON_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/"}


# Đọc giá trị chuỗi của một trường JSON ngay cả khi model quên escape dấu " bên trong
# (giá trị kết thúc ở dấu " đứng trước "," hoặc "}").
def lenient_json_string_field(text: str, field: str) -> str | None:
    """Read a JSON string value even when the model left inner quotes unescaped.

    The value ends at the first quote that is followed by `,` or `}` (end of the field).
    """
    match = re_json_string_field_start(field).search(text)
    if not match:
        return None
    chars: list[str] = []
    cursor = match.end()
    while cursor < len(text):
        char = text[cursor]
        if char == "\\" and cursor + 1 < len(text):
            following = text[cursor + 1]
            chars.append(_JSON_ESCAPES.get(following, "\\" + following))
            cursor += 2
            continue
        if char == '"' and text[cursor + 1 :].lstrip()[:1] in {",", "}"}:
            return "".join(chars)
        chars.append(char)
        cursor += 1
    return None


# Nhiệm vụ: lấy ĐÚNG MỘT khai báo hàm từ câu trả lời của model, theo thứ tự:
# (1) JSON {"replacement": ...} (kể cả JSON hỏng), (2) code fence ```solidity có chứa tên hàm,
# (3) tìm "function <tên>(" rồi đếm ngoặc trên mã đã che comment/chuỗi (mask_noncode, giữ nguyên độ dài);
# nếu dính thêm hàm khác thì cắt bỏ.
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
    if '"replacement"' in raw_text:
        replacement = None
        try:
            json_match = RE_JSON_WITH_REPLACEMENT.search(raw_text)
            if json_match:
                data = json.loads(json_match.group(0))
                if isinstance(data, dict) and data.get("replacement"):
                    replacement = str(data["replacement"])
        except Exception:
            # Models often emit invalid JSON (e.g. an unescaped `("")` inside the code).
            replacement = lenient_json_string_field(raw_text, "replacement")
        if replacement and replacement.strip() != raw_text:
            return extract_declaration_body(replacement, target_name=target_name)

    # 2. Tìm khối code fence ```solidity ... ```
    fence_matches = list(RE_SOLIDITY_FENCE.finditer(raw_text))
    search_space = raw_text
    if fence_matches:
        # Nếu có fence, dùng nội dung fence đầu tiên có chứa target_name
        for fm in fence_matches:
            f_code = fm.group(1).strip()
            if not target_name or (target_name in f_code):
                search_space = f_code
                break

    # 3. Tìm vị trí bắt đầu khai báo của hàm cần lấy (regex tuỳ theo target_name)
    match = re_declaration_start(target_name).search(search_space)
    if not match:
        if target_name:
            return ""
        return search_space.strip()

    start_idx = match.start()
    declaration = search_space[start_idx:]
    # Comments and strings blanked (same length): a brace or `//` inside them is not structure.
    masked = mask_noncode(declaration)
    brace_idx = masked.find("{")
    if brace_idx < 0:
        return declaration.strip()

    # 4. Đếm ngoặc từ dấu "{" đầu tiên của thân hàm
    end = len(masked)
    depth = 0
    for position in range(brace_idx, len(masked)):
        if masked[position] == "{":
            depth += 1
        elif masked[position] == "}":
            depth -= 1
            if depth == 0:
                end = position + 1
                break
    extracted = declaration[:end].strip()

    # Kiểm tra đảm bảo extracted chỉ chứa MỘT khai báo hàm duy nhất
    masked_extracted = mask_noncode(extracted)
    inner_declarations = list(RE_NAMED_FUNCTION_START.finditer(masked_extracted[10:]))
    if inner_declarations:
        first_inner = inner_declarations[0].start() + 10
        last_close = masked_extracted.rfind("}", 0, first_inner)
        if last_close > 0:
            extracted = extracted[: last_close + 1].strip()

    return extracted
