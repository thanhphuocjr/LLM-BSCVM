"""
Phase 4 - deterministic fallback when the generative Fixer produced no applicable patch.

Order: (1) a knowledge-base fix whose vulnerable snippet matches this contract exactly,
(2) the Advisor's fixed code for the unit, (3) two textbook rules (onlyOwner on setOwner,
checks-effects-interactions for a value call followed by a balance update). Every
outcome is tagged with its `engine`, so a report never presents it as model output.
"""

from __future__ import annotations

import re
from typing import Any

from E2E.common.findings import finding_key
from E2E.phase0_parsing.code_units import CodeUnit, as_code_unit
from E2E.phase4_fixer.behavior_guard import validate_observable_behavior
from E2E.phase4_fixer.patching import apply_unit_patches


def _normalize_ws(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


# A knowledge-base snippet shorter than this is too generic to be trusted as an exact match.
MIN_EMPIRICAL_SNIPPET_CHARS = 40


def _apply_heuristic_security_patch(
    code: str,
    findings: list[dict[str, Any]],
    code_units: list[Any] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Deterministic fallback used when the generative fixer produced no applicable patch.

    1. Empirical RAG patch: only when the knowledge-base vulnerable snippet matches the
       contract text exactly, or the whole unit; a replacement function must keep the
       unit's name and pass the observable-behaviour guard.
    2. Advisor fixed code for a unit (same guard).
    3. Rule-based fallback for two textbook patterns. These outcomes are tagged
       `engine=rule_based_fallback` so reports never present them as model output.
    """
    units = [as_code_unit(item) for item in code_units or [] if isinstance(item, (CodeUnit, dict))]
    units_by_id = {unit.unit_id: unit for unit in units}
    fixed_code = code
    patch_outcomes: list[dict[str, Any]] = []
    patched_units: set[str] = set()

    def splice(unit: CodeUnit, replacement: str, summary: str) -> bool:
        nonlocal fixed_code
        if unit.unit_id in patched_units or unit.name not in replacement:
            return False
        if re.sub(r"\s+", "", replacement) == re.sub(r"\s+", "", unit.code):
            return False
        try:
            validate_observable_behavior(unit.code, replacement)
        except ValueError:
            return False
        updated, outcomes = apply_unit_patches(
            fixed_code,
            [unit],
            [{"unit_id": unit.unit_id, "replacement": replacement, "summary": summary}],
        )
        if not any(item.get("status") == "applied" for item in outcomes):
            return False
        fixed_code = updated
        patched_units.add(unit.unit_id)
        return True

    for f in findings:
        v_code = str(f.get("vulnerable_code") or "").strip()
        f_code = str(f.get("fixed_code") or "").strip()
        uid = f.get("unit_id")
        unit = units_by_id.get(uid) if uid else None
        if not f_code or v_code == f_code:
            continue
        if v_code:
            if len(v_code) >= MIN_EMPIRICAL_SNIPPET_CHARS and v_code in fixed_code:
                fixed_code = fixed_code.replace(v_code, f_code, 1)
                patch_outcomes.append({
                    "finding_id": finding_key(f),
                    "unit_id": uid,
                    "status": "applied",
                    "engine": "rag_empirical_patch",
                    "reason": f"Applied verified RAG patch from knowledge item {f.get('best_match_id', '')} ({f.get('swc', '')}).",
                })
            elif unit is not None and _normalize_ws(v_code) == _normalize_ws(unit.code):
                if splice(unit, f_code, f"RAG patch for {uid}"):
                    patch_outcomes.append({
                        "finding_id": finding_key(f),
                        "unit_id": uid,
                        "status": "applied",
                        "engine": "rag_empirical_patch",
                        "reason": f"Applied unit-level RAG patch for {uid}.",
                    })
        elif unit is not None and splice(unit, f_code, f"Advisor patch for {uid}"):
            patch_outcomes.append({
                "finding_id": finding_key(f),
                "unit_id": uid,
                "status": "applied",
                "engine": "advisor_fixed_code",
                "reason": f"Applied Advisor remediation patch for {uid}.",
            })

    if patch_outcomes:
        return fixed_code, patch_outcomes

    def unit_named(name: str) -> CodeUnit | None:
        matches = [unit for unit in units if unit.name == name]
        return matches[0] if len(matches) == 1 else None

    # Rule-based fallback 1: missing onlyOwner on setOwner.
    if any(f.get("swc") in {"SWC-105", "SWC-100"} for f in findings):
        if "onlyOwner" in code and "address public owner;" in code:
            pattern = r"function\s+setOwner\s*\(([^)]*)\)\s*public\s*\{"
            if re.search(pattern, fixed_code):
                fixed_code = re.sub(pattern, r"function setOwner(\1) public onlyOwner {", fixed_code)
                unit = unit_named("setOwner")
                patch_outcomes.append({
                    "finding_id": f"SWC-105:{unit.unit_id if unit else 'setOwner'}",
                    "unit_id": unit.unit_id if unit else None,
                    "status": "applied",
                    "engine": "rule_based_fallback",
                    "reason": "Rule-based fallback: added onlyOwner modifier to setOwner.",
                })

    # Rule-based fallback 2: checks-effects-interactions for a value call followed by a balance update.
    if any(f.get("swc") == "SWC-107" for f in findings):
        pattern = re.compile(
            r"(\((?:bool\s+[A-Za-z_]\w*|)\s*,\s*(?:bytes\s+memory\s+[A-Za-z_]\w*|)\)\s*=\s*msg\.sender\.call\{value:\s*[^}]+\}\([^)]*\);\s*require\([^;]+\);)(\s*)(balances\[msg\.sender\]\s*(?:-=\s*[^;]+|=\s*0);)",
            re.MULTILINE,
        )
        match = pattern.search(fixed_code)
        if match:
            fixed_code = pattern.sub(r"\3\2\1", fixed_code)
            unit = next(
                (u for u in units if u.start_offset <= match.start() < u.end_offset),
                None,
            )
            patch_outcomes.append({
                "finding_id": f"SWC-107:{unit.unit_id if unit else 'contract'}",
                "unit_id": unit.unit_id if unit else None,
                "status": "applied",
                "engine": "rule_based_fallback",
                "reason": "Rule-based fallback: moved the balance update before the external call (CEI).",
            })

    return fixed_code, patch_outcomes
