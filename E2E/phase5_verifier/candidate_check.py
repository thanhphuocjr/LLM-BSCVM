"""
Phase 5 - check one candidate patch inside the repair loop (gpu_pipeline.py).

Three channels, all on the patched source:
  1. compiler: solc picked by pragma, compared against the original as a baseline
     (a contract that never compiled standalone does not blame the patch);
  2. neural detector: the patched units are scored again on the GPU server;
  3. static + RAG: Phase 1A/1B re-run; a targeted finding that is still there, or a
     new High/Critical finding in a patched unit, fails the candidate.
A failing candidate produces feedback text for the next Fixer attempt.
"""

from __future__ import annotations

import re
from typing import Any

from E2E.common.solc import check_compilation_with_baseline
from E2E.config import RISK_ORDER
from E2E.phase0_parsing.code_units import parse_code_units
from E2E.phase1a_static.static_runner import run_static
from E2E.phase1b_rag.rag_runner import run_rag
from E2E.phase1c_detector.detector import score_units_via_server


_ERROR_LOCATION_RE = re.compile(r"(-->\s*\S+:\d+:\d+:?|/\S+\.sol:\d+:\d+:?)")


def _compiler_error_signatures(text: str) -> set[str]:
    signatures = set()
    for line in (text or "").splitlines():
        if "Error" not in line:
            continue
        signatures.add(_ERROR_LOCATION_RE.sub("", line).strip())
    return signatures


def evaluate_compilation(original_code: str, fixed_code: str) -> dict[str, Any]:
    """Compile against the original as a baseline.

    - skipped:        no compiler available (not a failure of the patch)
    - not_applicable: the original does not compile standalone (e.g. unresolved imports)
                      and the patch introduced no new compiler error
    """
    result = check_compilation_with_baseline(original_code, fixed_code)
    fixed = result.get("fixed") or {}
    baseline = result.get("baseline") or {}
    if fixed.get("passed") is None:
        state = "skipped"
    elif fixed.get("passed"):
        state = "passed"
    elif result.get("baseline_failed"):
        new_errors = _compiler_error_signatures(fixed.get("error", "")) - _compiler_error_signatures(
            baseline.get("error", "")
        )
        state = "failed" if new_errors else "not_applicable"
    else:
        state = "failed"
    return {
        "state": state,
        "failed": state == "failed",
        "compiler": result.get("compiler"),
        "selection_reason": result.get("selection_reason"),
        "baseline_failed": result.get("baseline_failed", False),
        "error": fixed.get("error", "") if state == "failed" else "",
        "passed": {"passed": True, "failed": False}.get(state),
        "status": state,
    }


_ERROR_LINE_RE = re.compile(r"\S+?\.sol:(\d+):\d+")  # `--> a.sol:12:5` (>=0.5) and `a.sol:12:5: Error` (0.4)


def units_with_compiler_errors(candidate_code: str, error_text: str) -> set[str]:
    """Map solc error locations (`--> file:LINE:COL`) to the units that contain them."""
    lines = {int(value) for value in _ERROR_LINE_RE.findall(error_text or "")}
    return {
        unit.unit_id
        for unit in parse_code_units(candidate_code)
        if any(unit.start_line <= line <= unit.end_line for line in lines)
    }


def _finding_pairs(findings: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {
        (str(item.get("swc")), str(item.get("unit_id")))
        for item in findings
        if item.get("swc") and item.get("unit_id")
    }


def verify_candidate(
    source_code: str,
    candidate_code: str,
    *,
    patched_unit_ids: set[str],
    target_unit_ids: set[str],
    original_unit_results: list[dict[str, Any]],
    target_pairs: set[tuple[str, str]],
    known_pairs: set[tuple[str, str]],
    threshold: float,
    server_url: str | None,
) -> dict[str, Any]:
    """Three-channel verification of a candidate: compiler, neural detector, static + RAG re-run."""
    compile_info = evaluate_compilation(source_code, candidate_code)
    fixed_units = parse_code_units(candidate_code)
    fixed_by_id = {unit.unit_id: unit for unit in fixed_units}

    # Neural detector: re-score patched units only; other units keep their original score.
    rescored: list[dict[str, Any]] = []
    for original in original_unit_results:
        unit_id = original["unit_id"]
        if unit_id not in patched_unit_ids:
            rescored.append(dict(original))
            continue
        fixed_unit = fixed_by_id.get(unit_id)
        if fixed_unit is None:
            rescored.append({**original, "status": "error", "label": "Inconclusive",
                             "vulnerability_probability": None,
                             "error": "patched unit not found in candidate source"})
            continue
        rescored.extend(score_units_via_server([fixed_unit], threshold, server_url))
    detector_failing = sorted(
        item["unit_id"] for item in rescored
        if item["unit_id"] in target_unit_ids and item.get("label") == "Vulnerable"
    )
    unverified = sorted(
        item["unit_id"] for item in rescored
        if item["unit_id"] in target_unit_ids and item.get("vulnerability_probability") is None
    )

    # Static + RAG re-run on the candidate source.
    channel_errors: dict[str, str] = {}
    fixed_local: list[dict[str, Any]] = []
    for name, runner in (("static", run_static), ("rag", run_rag)):
        try:
            fixed_local.extend(runner(candidate_code, fixed_units).get("findings", []))
        except Exception as error:
            channel_errors[name] = f"{type(error).__name__}: {error}"
    residual = [item for item in fixed_local if (str(item.get("swc")), str(item.get("unit_id"))) in target_pairs]
    introduced = [
        item for item in fixed_local
        if str(item.get("unit_id")) in patched_unit_ids
        and (str(item.get("swc")), str(item.get("unit_id"))) not in known_pairs
        and RISK_ORDER.get(str(item.get("severity")), 0) >= RISK_ORDER["High"]
    ]

    compile_culprits = (
        units_with_compiler_errors(candidate_code, compile_info.get("error", ""))
        if compile_info["failed"]
        else set()
    )
    failing_units = set(detector_failing) | set(unverified) | (compile_culprits & patched_unit_ids)
    failing_units.update(str(item.get("unit_id")) for item in residual + introduced if item.get("unit_id"))
    passed = (
        not compile_info["failed"]
        and not failing_units
        and not channel_errors
    )
    scored_targets = [
        float(item["vulnerability_probability"]) for item in rescored
        if item["unit_id"] in target_unit_ids and item.get("vulnerability_probability") is not None
    ]
    return {
        "passed": passed,
        "compilation": compile_info,
        "unit_results": rescored,
        "detector_failing_units": detector_failing,
        "unverified_units": unverified,
        "residual_findings": residual,
        "introduced_findings": introduced,
        "channel_errors": channel_errors,
        "failing_units": sorted(failing_units),
        "compile_culprit_units": sorted(compile_culprits),
        "max_vulnerability_probability": round(max(scored_targets), 4) if scored_targets else None,
    }


def _verification_feedback(attempt: int, max_attempts: int, verification: dict[str, Any]) -> str:
    parts = [f"\n\n[CRITICAL RESIDUAL ERROR FEEDBACK - ATTEMPT {attempt}/{max_attempts}]"]
    compile_info = verification["compilation"]
    if compile_info["failed"]:
        error_lines = [
            line for line in compile_info.get("error", "").splitlines()
            if "Error" in line
        ]
        clean = "\n".join(error_lines[:8]) or compile_info.get("error", "")[:400]
        parts.append(
            f"Solidity compiler ({compile_info.get('compiler')}) failed with error:\n{clean}\n"
            "Your patch broke compilation. You MUST fix this compiler error."
        )
    if verification["detector_failing_units"]:
        parts.append(
            "The neural detector re-evaluated your patch and STILL FLAGGED it as Vulnerable: "
            f"{', '.join(verification['detector_failing_units'])}."
        )
    for label, key in (("still present", "residual_findings"), ("newly introduced", "introduced_findings")):
        for item in verification[key]:
            parts.append(
                f"Static/RAG finding {label}: [{item.get('swc')}] {item.get('title', '')} "
                f"in {item.get('unit_id')} (line {item.get('line')})."
            )
    return "\n".join(parts)
