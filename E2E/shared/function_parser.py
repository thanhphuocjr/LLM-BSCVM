from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable


DECLARATION_RE = re.compile(
    r"\b(function\s+(?P<function>[A-Za-z_$][\w$]*)\s*\(|"
    r"(?P<special>constructor|receive|fallback)\s*\()",
    re.MULTILINE,
)
CONTRACT_RE = re.compile(r"\b(?:abstract\s+)?(?:contract|library|interface)\s+([A-Za-z_$][\w$]*)")


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


def _declaration_end(code: str, declaration_start: int, search_start: int) -> int:
    cursor = search_start
    while cursor < len(code):
        skipped = _skip_comment_or_string(code, cursor)
        if skipped is not None:
            cursor = skipped
            continue
        if code[cursor] == ";":
            return cursor + 1
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
            return cursor
        cursor += 1
    return len(code)


def _contract_at(code: str, offset: int) -> str:
    matches = list(CONTRACT_RE.finditer(_mask_noncode(code), 0, offset))
    return matches[-1].group(1) if matches else "UnknownContract"


def parse_code_units(code: str) -> list[CodeUnit]:
    units: list[CodeUnit] = []
    names: dict[str, int] = {}
    masked_code = _mask_noncode(code)
    for match in DECLARATION_RE.finditer(masked_code):
        name = match.group("function") or match.group("special") or "unknown"
        kind = "function" if match.group("function") else name
        end = _declaration_end(code, match.start(), match.end())
        base_id = f"{_contract_at(code, match.start())}.{name}"
        names[base_id] = names.get(base_id, 0) + 1
        suffix = f"#{names[base_id]}" if names[base_id] > 1 else ""
        snippet = code[match.start() : end].strip()
        units.append(
            CodeUnit(
                unit_id=f"{base_id}{suffix}",
                kind=kind,
                name=name,
                contract_name=_contract_at(code, match.start()),
                start_offset=match.start(),
                end_offset=end,
                start_line=_line_number(code, match.start()),
                end_line=_line_number(code, end),
                code=snippet,
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
            contract_name=_contract_at(code, len(code)),
            start_offset=code.find(stripped) if stripped else 0,
            end_offset=len(code),
            start_line=1,
            end_line=_line_number(code, len(code)),
            code=stripped,
        )
    ]


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
        seen.add(unit_id)
        accepted.append((unit, replacement, patch))

    updated = code
    for unit, replacement, patch in sorted(accepted, key=lambda item: item[0].start_offset, reverse=True):
        updated = updated[: unit.start_offset] + replacement + updated[unit.end_offset :]
        outcomes.append(
            {
                "unit_id": unit.unit_id,
                "status": "applied",
                "summary": str(patch.get("summary") or ""),
            }
        )
    return updated, outcomes
