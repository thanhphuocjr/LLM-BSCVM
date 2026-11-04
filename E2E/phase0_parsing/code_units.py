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
