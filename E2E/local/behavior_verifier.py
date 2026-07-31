from __future__ import annotations

import re
from typing import Any

from E2E.shared.function_parser import parse_code_units


OBSERVABLE_SIGNALS = {
    "ETH value call": re.compile(r"\.call\s*\{\s*value\s*:"),
    "ETH transfer": re.compile(r"\.transfer\s*\("),
    "ETH send": re.compile(r"\.send\s*\("),
    "event emission": re.compile(r"\bemit\s+"),
    "return statement": re.compile(r"\breturn\b"),
}


def verify_patch_behavior(
    original_code: str,
    fixed_code: str,
    patch_outcomes: list[dict[str, Any]],
    tool_result: dict[str, Any],
) -> dict[str, Any]:
    applied_units = {
        str(item.get("unit_id") or "")
        for item in patch_outcomes
        if item.get("status") == "applied"
    }
    if not applied_units:
        return {
            "status": "not_applicable",
            "findings": [],
            "checks": ["No applied patch requires behavioral comparison."],
        }

    original_units = {item.unit_id: item.code for item in parse_code_units(original_code)}
    fixed_units = {item.unit_id: item.code for item in parse_code_units(fixed_code)}
    findings = []
    checks = []

    for unit_id in sorted(applied_units):
        original = original_units.get(unit_id)
        fixed = fixed_units.get(unit_id)
        if original is None or fixed is None:
            findings.append(
                {
                    "unit_id": unit_id,
                    "severity": "High",
                    "check": "patched-unit-presence",
                    "evidence": "The patched declaration could not be mapped in both source versions.",
                }
            )
            continue
        for label, pattern in OBSERVABLE_SIGNALS.items():
            existed = bool(pattern.search(original))
            preserved = bool(pattern.search(fixed))
            checks.append(
                {
                    "unit_id": unit_id,
                    "signal": label,
                    "original": existed,
                    "fixed": preserved,
                }
            )
            if existed and not preserved:
                findings.append(
                    {
                        "unit_id": unit_id,
                        "severity": "High",
                        "check": "observable-behavior-preservation",
                        "evidence": f"Patch removed {label} present in the original declaration.",
                    }
                )

    for finding in tool_result.get("slither", {}).get("findings", []):
        if finding.get("check") == "locked-ether":
            findings.append(
                {
                    "unit_id": "",
                    "severity": finding.get("impact", "Medium"),
                    "check": "slither-locked-ether",
                    "evidence": finding.get("description", "Patched contract can lock Ether."),
                }
            )

    return {
        "status": "failed" if findings else "passed",
        "findings": findings,
        "checks": checks,
    }
