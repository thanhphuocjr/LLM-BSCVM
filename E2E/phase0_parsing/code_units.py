"""Phase 0 - tách mã Solidity thành code unit (hàm, constructor, fallback/receive); mọi phase sau làm việc theo unit."""

from dataclasses import asdict, dataclass

from E2E.common.regex_patterns import (
    RE_CONTRACT_DECLARATION, RE_NONCODE, RE_NOT_NEWLINE, RE_UNIT_DECLARATION, RE_VIEW_PURE_CONSTANT, RE_VISIBILITY,
)


# Một code unit: id "Contract.func" (trùng tên thêm "#2"), loại, vị trí, mã, có thân, view/pure, visibility.
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

    # Đổi unit thành dict để ghi JSON (phase0_input.json, request.json).
    def to_dict(self) -> dict:
        return asdict(self)


# Che comment (và chuỗi nếu mask_strings=True) bằng dấu cách, giữ "\n" và độ dài -> offset/số dòng vẫn khớp mã gốc.
def mask_noncode(code: str, *, mask_strings: bool = True) -> str:
    return RE_NONCODE.sub(
        lambda m: m.group() if not mask_strings and m.group()[0] in "\"'" else RE_NOT_NEWLINE.sub(" ", m.group()), code)


# Vị trí ký tự đóng khớp với ký tự mở tại open_pos (đếm ngoặc lồng nhau); không khớp -> None.
def matching_close(text: str, open_pos: int, opening: str = "{", closing: str = "}") -> int | None:
    if not 0 <= open_pos < len(text) or text[open_pos] != opening:
        return None
    depth = 0
    for pos in range(open_pos, len(text)):
        if text[pos] == opening:
            depth += 1
        elif text[pos] == closing:
            depth -= 1
            if depth == 0:
                return pos
    return None


# Đổi vị trí ký tự (offset) thành số dòng, bắt đầu từ 1.
def line_number(code: str, offset: int) -> int:
    return code.count("\n", 0, max(0, offset)) + 1


# Điểm kết thúc của khai báo: gặp ";" trước -> không có thân; gặp "{" -> tới "}" khớp. Trả (end_offset, has_body).
def _declaration_end(masked: str, start: int) -> tuple[int, bool]:
    semicolon, brace = masked.find(";", start), masked.find("{", start)
    if brace < 0 or 0 <= semicolon < brace:
        return (semicolon + 1 if semicolon >= 0 else len(masked)), False
    close = matching_close(masked, brace)
    return (len(masked) if close is None else close + 1), True


# Contract/library/interface gần nhất phía trước offset -> (tên, loại).
def _contract_at(masked: str, offset: int) -> tuple[str, str]:
    matches = list(RE_CONTRACT_DECLARATION.finditer(masked, 0, offset))
    return (matches[-1].group(2), matches[-1].group(1)) if matches else ("UnknownContract", "contract")


# Dựng lại CodeUnit từ dict của request.json (đã là CodeUnit thì trả nguyên).
def as_code_unit(unit: CodeUnit | dict) -> CodeUnit:
    if isinstance(unit, CodeUnit):
        return unit
    code = str(unit.get("code") or "")
    defaults = {"unit_id": "unknown", "kind": "function", "name": "func", "contract_name": "Contract",
                "start_offset": 0, "end_offset": len(code), "start_line": 1, "end_line": 1, "has_body": True,
                "is_view_or_pure": False, "visibility": "public"}
    return CodeUnit(code=code, **{key: unit.get(key, value) for key, value in defaults.items()})


# [HÀM CHÍNH PHASE 0] Che comment/chuỗi -> tìm từng khai báo -> điểm kết thúc, contract chứa nó, visibility, view/pure.
def parse_code_units(code: str) -> list[CodeUnit]:
    masked = mask_noncode(code)
    units, seen = [], {}
    for match in RE_UNIT_DECLARATION.finditer(masked):
        name = "fallback" if match.group("unnamed_fallback") else \
            match.group("function") or match.group("special") or "unknown"
        end, has_body = _declaration_end(masked, match.end())
        contract_name, contract_kind = _contract_at(masked, match.start())
        base_id = f"{contract_name}.{name}"
        seen[base_id] = seen.get(base_id, 0) + 1
        snippet = code[match.start():end].strip()
        header = snippet.split("{", 1)[0]
        visibility = RE_VISIBILITY.search(header)
        units.append(CodeUnit(
            unit_id=base_id + (f"#{seen[base_id]}" if seen[base_id] > 1 else ""),
            kind="function" if match.group("function") else name, name=name, contract_name=contract_name,
            start_offset=match.start(), end_offset=end, start_line=line_number(code, match.start()),
            end_line=line_number(code, end), code=snippet, has_body=has_body and contract_kind != "interface",
            is_view_or_pure=bool(RE_VIEW_PURE_CONSTANT.search(header)),
            visibility=visibility.group(1) if visibility else "public",
        ))
    if units:
        return units
    stripped = code.strip()
    return [CodeUnit(unit_id="source", kind="source", name="source", contract_name=_contract_at(masked, len(code))[0],
                     start_offset=code.find(stripped) if stripped else 0, end_offset=len(code), start_line=1,
                     end_line=line_number(code, len(code)), code=stripped)]
