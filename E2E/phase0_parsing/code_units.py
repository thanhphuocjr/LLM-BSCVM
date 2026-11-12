"""
Phase 0 - split a Solidity source into code units (functions, constructors,
fallback/receive). A source without any declaration becomes one "source" unit.

Every later phase works per unit: Phase 1 scores units, Phase 4 patches units and
Phase 5 re-checks the patched units. Comments and strings are masked before the
brace matching, so a `{` inside a string never ends a function early.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


DECLARATION_RE = re.compile(
    r"\b(function\s+(?P<function>[A-Za-z_$][\w$]*)\s*\(|"
    r"function\s*(?P<unnamed_fallback>\()\s*|"
    r"(?P<special>constructor|receive|fallback)\s*\()",
    re.MULTILINE,
)


CONTRACT_RE = re.compile(r"\b(?:abstract\s+)?(contract|library|interface)\s+([A-Za-z_$][\w$]*)")


# [Phase 0] Cấu trúc dữ liệu cho MỘT đơn vị mã (hàm / constructor / fallback / receive).
# Lưu id dạng "Contract.func" (trùng tên thì thêm "#2"), loại, vị trí (offset + dòng), mã nguồn,
# có thân hay không, có phải view/pure, visibility.
# Mục đích: mọi phase sau làm việc theo unit (detector chấm từng unit, Fixer vá từng unit,
# Phase 5 kiểm tra lại từng unit đã vá).
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

    # Chuyển unit thành dict để ghi JSON (phase0_input.json, request.json).
    def to_dict(self) -> dict:
        return asdict(self)


# Đổi vị trí ký tự (offset) thành số dòng (bắt đầu từ 1).
def _line_number(code: str, offset: int) -> int:
    return code.count("\n", 0, max(0, offset)) + 1


# Comment hoặc chuỗi, theo đúng thứ tự xuất hiện. Regex luôn khớp token SỚM NHẤT trước nên `//` nằm trong
# "http://..." thuộc về chuỗi, còn dấu ' trong `// don't` thuộc về comment. Comment / chuỗi chưa đóng kéo tới
# hết mã.
_NONCODE_RE = re.compile(
    r"//[^\n]*"                               # comment một dòng
    r"|/\*[\s\S]*?(?:\*/|\Z)"                 # comment khối
    r"|\"(?:\\[\s\S]?|[^\"\\])*(?:\"|\Z)"     # "chuỗi" (có ký tự thoát)
    r"|'(?:\\[\s\S]?|[^'\\])*(?:'|\Z)"        # 'chuỗi'
)
_NOT_NEWLINE_RE = re.compile(r"[^\n]")


# [NƠI DUY NHẤT BỎ COMMENT CỦA CẢ PIPELINE] Thay mọi ký tự trong comment (và chuỗi, nếu mask_strings=True)
# bằng dấu cách, giữ nguyên xuống dòng và độ dài -> offset / số dòng vẫn khớp với mã gốc.
#   mask_strings=True : che cả comment lẫn chuỗi (Phase 0 tìm hàm, Phase 1A chạy luật, tìm bất biến của hàm).
#   mask_strings=False: chỉ che comment, giữ chuỗi (chữ hiển thị trong prompt, behaviour guard của Fixer).
# Dùng bởi: code_units (Phase 0), contract_context, static_analyzer._clean_code (Phase 1A), behavior_guard (Phase 4).
def mask_noncode(code: str, *, mask_strings: bool = True) -> str:
    # Thay một comment/chuỗi bằng dấu cách (giữ "\n"); chuỗi được giữ nguyên khi mask_strings=False.
    def blank(match: re.Match[str]) -> str:
        text = match.group()
        if not mask_strings and text[0] in "\"'":
            return text
        return _NOT_NEWLINE_RE.sub(" ", text)

    return _NONCODE_RE.sub(blank, code)


# Từ chỗ bắt đầu khai báo hàm, tìm vị trí kết thúc trên mã ĐÃ CHE comment/chuỗi (mask_noncode giữ nguyên độ dài,
# nên offset khớp với mã gốc): gặp ";" trước -> hàm không thân (interface/abstract); gặp "{" -> đếm ngoặc tới "}" khớp.
# Trả về (end_offset, has_body).
def _declaration_end(masked_code: str, search_start: int) -> tuple[int, bool]:
    """
    Tìm vị trí kết thúc của khai báo hàm và xác định hàm có thân (has_body) hay không.
    Trả về: (end_offset, has_body)
    """
    cursor = search_start
    while cursor < len(masked_code):
        char = masked_code[cursor]
        if char == ";":
            # Hàm interface / abstract không có body
            return cursor + 1, False
        if char == "{":
            depth = 0
            for position in range(cursor, len(masked_code)):
                if masked_code[position] == "{":
                    depth += 1
                elif masked_code[position] == "}":
                    depth -= 1
                    if depth == 0:
                        return position + 1, True
            return len(masked_code), True
        cursor += 1
    return len(masked_code), False


# Tìm contract/library/interface gần nhất phía trước offset (trên mã đã che comment/chuỗi) -> (tên, loại).
# Mục đích: gán unit vào đúng contract để tạo unit_id "Contract.func".
def _contract_at(masked_code: str, offset: int) -> tuple[str, str]:
    """Trả về (contract_name, contract_kind) tại vị trí offset."""
    matches = list(CONTRACT_RE.finditer(masked_code, 0, offset))
    if matches:
        return matches[-1].group(2), matches[-1].group(1)
    return "UnknownContract", "contract"


# [HÀM CHÍNH PHASE 0] Tách mã nguồn Solidity thành danh sách CodeUnit.
# Cách làm: che comment/chuỗi -> regex tìm "function X(", "function(" (fallback cũ), constructor/receive/fallback
# -> tìm điểm kết thúc -> xác định contract chứa nó, visibility, view/pure. Nếu không có hàm nào thì
# trả về 1 unit "source" chứa toàn bộ file.
# Gọi bởi: orchestrator.run_pipeline, các bước kiểm chứng Phase 5.
def parse_code_units(code: str) -> list[CodeUnit]:
    units: list[CodeUnit] = []
    names: dict[str, int] = {}
    masked_code = mask_noncode(code)
    for match in DECLARATION_RE.finditer(masked_code):
        if match.group("unnamed_fallback"):
            name = "fallback"
            kind = "fallback"
        else:
            name = match.group("function") or match.group("special") or "unknown"
            kind = "function" if match.group("function") else name
        end, has_body = _declaration_end(masked_code, match.end())
        contract_name, contract_kind = _contract_at(masked_code, match.start())
        if contract_kind == "interface":
            has_body = False

        base_id = f"{contract_name}.{name}"
        names[base_id] = names.get(base_id, 0) + 1
        suffix = f"#{names[base_id]}" if names[base_id] > 1 else ""
        snippet = code[match.start() : end].strip()

        # Kiểm tra view/pure và visibility
        header_part = snippet[: snippet.find("{")] if "{" in snippet else snippet
        is_view_or_pure = bool(re.search(r"\b(view|pure|constant)\b", header_part))  # `constant` = pre-0.5 view
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
            contract_name=_contract_at(masked_code, len(code))[0],
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


# Dựng lại CodeUnit từ dict (request.json lưu unit ở dạng dict). Nếu đã là CodeUnit thì trả nguyên.
# Mục đích: GPU pipeline / Fixer nhận unit từ JSON nhưng cần đối tượng CodeUnit.
def as_code_unit(unit: CodeUnit | dict[str, Any]) -> CodeUnit:
    """Rebuild a CodeUnit from its JSON form (request.json stores units as dicts)."""
    if isinstance(unit, CodeUnit):
        return unit
    code = str(unit.get("code") or "")
    return CodeUnit(
        unit_id=unit.get("unit_id", "unknown"),
        kind=unit.get("kind", "function"),
        name=unit.get("name", "func"),
        contract_name=unit.get("contract_name", "Contract"),
        start_offset=unit.get("start_offset", 0),
        end_offset=unit.get("end_offset", len(code)),
        start_line=unit.get("start_line", 1),
        end_line=unit.get("end_line", 1),
        code=code,
        has_body=unit.get("has_body", True),
        is_view_or_pure=unit.get("is_view_or_pure", False),
        visibility=unit.get("visibility", "public"),
    )
