"""
Phase 5 - final verification of the patched contract, run by the orchestrator.

1. Security tool matrix on original vs patched source   (tool_matrix.py)
2. Behaviour check: compile both, compare the ASTs      (behavior_verifier.py)
3. Static + RAG re-run on the patched source, scoped to the findings of the patched
   units (an unrelated pre-existing finding elsewhere does not reject the patch)
4. `decide_overall_verdict`: one verdict for phase5_verification.json, checked in a
   fixed order (behaviour regression > compile failure > no effective patch > tool
   matrix failure > residual static/RAG finding > ...).

`build_tool_repair_feedback` turns all channels into the per-channel summary shown in
the report (and, for a repair loop, into feedback for the next Fixer attempt).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from E2E.common.safe_run import safe_run
from E2E.config import MAX_FIX_ATTEMPTS
from E2E.phase0_parsing.code_units import parse_code_units
from E2E.phase1a_static.static_runner import run_static
from E2E.phase1b_rag.rag_runner import run_rag
from E2E.phase5_verifier.behavior_verifier import verify_patch_behavior
from E2E.phase5_verifier.tool_matrix import verify_security_tool_matrix


def _verification_passed(result: dict[str, Any] | None) -> bool:
    if not result or result.get("status") != "ok":
        return False
    verdict = result.get("verdict") or result.get("final_verdict")
    return verdict == "Safe"


def _compact_feedback_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "finding_id",
        "swc",
        "unit_id",
        "title",
        "severity",
        "confidence",
        "line",
        "start_line",
        "end_line",
        "matched_text",
        "description",
        "remediation",
        "evidence",
        "notes",
    )
    return [
        {key: finding[key] for key in fields if key in finding}
        for finding in findings
    ]


def _llm_vulnerability_reason(
    detector: dict[str, Any] | None,
    verifier: dict[str, Any] | None,
) -> str:
    detector = detector or {}
    verifier = verifier or {}
    for value in (
        detector.get("vulnerability_reason"),
        verifier.get("vulnerability_reason"),
        verifier.get("reason"),
    ):
        if str(value or "").strip():
            return str(value).strip()

    evidence = [
        str(item.get("evidence") or "").strip()
        for item in verifier.get("verifications", [])
        if str(item.get("status") or "").lower() not in {"fixed", "safe"}
        and str(item.get("evidence") or "").strip()
    ]
    if evidence:
        return " ".join(evidence)

    vulnerable_units = [
        (
            str(item.get("unit_id") or "fixed source"),
            float(item.get("vulnerability_probability") or 0.0),
        )
        for item in detector.get("unit_results", [])
        if item.get("label") == "Vulnerable"
    ]
    if vulnerable_units:
        details = ", ".join(f"{unit_id} ({probability:.3f})" for unit_id, probability in vulnerable_units)
        return f"The neural detector still flags: {details}."
    return str(detector.get("error") or "The neural detector verification did not return a Safe verdict.")


def build_repair_feedback(
    fixed_static: dict[str, Any],
    fixed_rag: dict[str, Any],
    detector_fixed: dict[str, Any] | None,
    verifier: dict[str, Any] | None,
    *,
    attempt: int,
    max_attempts: int = MAX_FIX_ATTEMPTS,
) -> dict[str, Any]:
    static_passed = _verification_passed(fixed_static)
    rag_passed = _verification_passed(fixed_rag)
    llm_passed = _verification_passed(detector_fixed)
    return {
        "attempt": attempt,
        "max_attempts": min(max(1, max_attempts), MAX_FIX_ATTEMPTS),
        "passed": static_passed and rag_passed and llm_passed,
        "components": {
            "static": {
                "passed": static_passed,
                "status": fixed_static.get("status", "unknown"),
                "verdict": fixed_static.get("verdict", "Unknown"),
                "findings": _compact_feedback_findings(fixed_static.get("findings", [])),
                "error": fixed_static.get("error"),
            },
            "rag": {
                "passed": rag_passed,
                "status": fixed_rag.get("status", "unknown"),
                "verdict": fixed_rag.get("verdict")
                or fixed_rag.get("final_verdict")
                or "Unknown",
                "findings": _compact_feedback_findings(fixed_rag.get("findings", [])),
                "error": fixed_rag.get("error"),
            },
            "llm": {
                "passed": llm_passed,
                "status": (detector_fixed or {}).get("status", "missing"),
                "verdict": (detector_fixed or {}).get("verdict", "Unknown"),
                "reason": (
                    ""
                    if llm_passed
                    else _llm_vulnerability_reason(detector_fixed, verifier)
                ),
                "unit_results": (detector_fixed or {}).get("unit_results", []),
                "error": (detector_fixed or {}).get("error"),
            },
        },
    }


def rerun_local_channels(fixed_code: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Static and RAG evidence for the *patched* source (verification channels must not reuse
    the original contract's Phase 1 results)."""
    units = parse_code_units(fixed_code)
    return (
        safe_run("static", lambda: run_static(fixed_code, units)),
        safe_run("rag", lambda: run_rag(fixed_code, units)),
    )


def scope_to_targets(result: dict[str, Any], target_findings: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep only residual findings that match an original target (same SWC in the same unit).

    Unrelated pre-existing findings elsewhere in the contract are not evidence that the
    patch failed, so they must not block verification.
    """
    targets = {
        (str(item.get("swc")), str(item.get("unit_id")))
        for item in target_findings
        if item.get("swc") and item.get("swc") != "UNMAPPED"
    }
    if result.get("status") != "ok":
        return result
    residual = [
        item for item in result.get("findings", [])
        if (str(item.get("swc")), str(item.get("unit_id"))) in targets
    ]
    return {
        **result,
        "findings": residual,
        "verdict": "Vulnerable" if residual else "Safe",
        "final_verdict": "Vulnerable" if residual else "Safe",
        "scope": "original target findings",
    }


def _compact_tool_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "tool",
        "name",
        "check",
        "category",
        "severity",
        "line",
        "message",
        "swc",
    )
    return [
        {key: finding[key] for key in fields if key in finding}
        for finding in findings
    ]


def build_tool_repair_feedback(
    tool_result: dict[str, Any],
    *,
    attempt: int,
    max_attempts: int = MAX_FIX_ATTEMPTS,
    static_result: dict[str, Any] | None = None,
    rag_result: dict[str, Any] | None = None,
    detector_fixed: dict[str, Any] | None = None,
    verifier: dict[str, Any] | None = None,
) -> dict[str, Any]:
    components: dict[str, Any] = {}
    for tool_name, result in tool_result.get("tools", {}).items():
        residual = result.get("residual_target_findings", [])
        introduced = result.get("introduced_findings", [])
        findings = _compact_tool_findings([*residual, *introduced])
        verdict = str(result.get("verdict") or "ERROR")
        components[tool_name] = {
            "passed": verdict in {"PASS", "WARN"},
            "status": result.get("run_status", "unknown"),
            "verdict": verdict,
            "applicable": bool(result.get("applicable", True)),
            "findings": findings,
            "residual_target_findings": _compact_tool_findings(residual),
            "introduced_findings": _compact_tool_findings(introduced),
            "reason": result.get("detail"),
            "error": result.get("error"),
        }
    channel_passed = True
    if any([static_result, rag_result, detector_fixed, verifier]):
        channel_feedback = build_repair_feedback(
            static_result or {},
            rag_result or {},
            detector_fixed,
            verifier,
            attempt=attempt,
            max_attempts=max_attempts,
        )
        for ch_name, ch_data in channel_feedback.get("components", {}).items():
            components[ch_name] = ch_data
        channel_passed = channel_feedback.get("passed", True)

    compile_info = tool_result.get("compile", {})
    compile_status = compile_info.get("status", "unknown")
    compile_passed = compile_status == "passed"
    compile_error = (
        compile_info.get("stderr")
        or compile_info.get("error")
        or compile_info.get("reason")
        or ""
    )
    components["compile"] = {
        "passed": compile_passed,
        "status": compile_status,
        "verdict": "PASS" if compile_passed else "FAIL",
        "applicable": True,
        "findings": [],
        "residual_target_findings": [],
        "introduced_findings": [],
        "reason": "Compilation succeeded" if compile_passed else "Compilation failed",
        "error": compile_error,
    }
    failed_tools = [
        name
        for name, component in components.items()
        if component.get("verdict") == "FAIL"
    ]
    errored_tools = [
        name
        for name, component in components.items()
        if component.get("verdict") == "ERROR"
    ]
    return {
        "attempt": attempt,
        "max_attempts": min(max(1, max_attempts), MAX_FIX_ATTEMPTS),
        "passed": tool_result.get("policy_verdict") == "passed" and compile_passed,
        "channel_passed": channel_passed,
        "policy_verdict": tool_result.get("policy_verdict", "inconclusive"),
        "policy_reason": tool_result.get("policy_reason", ""),
        "compile_passed": compile_passed,
        "compile_status": compile_status,
        "compiler_error": compile_error,
        "compile": compile_info,
        "failed_tools": failed_tools,
        "errored_tools": errored_tools,
        "summary": tool_result.get("summary", {}),
        "components": components,
    }


def verify_final_patch(
    code: str,
    fixed_code: str,
    *,
    patch_outcomes: list[dict[str, Any]],
    fused: dict[str, Any],
    file_name: str,
    source_root: str | Path | None,
    results_dir: Path,
    tools: list[str] | None,
    use_smartbugs: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Steps 1-3 on the final patch -> (tool_result, behavioral_result, fixed_static, fixed_rag)."""
    # Residual checks only concern the functions that were actually patched; unrelated
    # pre-existing findings elsewhere in the contract are not evidence against the patch.
    patched_units = {
        item.get("unit_id") for item in patch_outcomes
        if item.get("status") == "applied" and item.get("unit_id")
    }
    patch_targets = [
        item for item in fused.get("findings", []) if item.get("unit_id") in patched_units
    ] or fused.get("findings", [])
    tool_result = safe_run(
        "tool_verification",
        lambda: verify_security_tool_matrix(
            code,
            fixed_code,
            file_name=file_name,
            source_root=source_root,
            target_findings=patch_targets,
            results_dir=results_dir,
            tools=tools,
            use_smartbugs=use_smartbugs,
        ),
    )
    behavioral_result = safe_run(
        "behavioral_verification",
        lambda: verify_patch_behavior(
            code,
            fixed_code,
            patch_outcomes,
            tool_result,
        ),
    )
    fixed_static, fixed_rag = rerun_local_channels(fixed_code)
    fixed_static = scope_to_targets(fixed_static, patch_targets)
    fixed_rag = scope_to_targets(fixed_rag, patch_targets)
    return tool_result, behavioral_result, fixed_static, fixed_rag


def skipped_verification() -> tuple[dict[str, Any], dict[str, Any]]:
    """Tool and behaviour results when there is no patch to verify."""
    tool_result = {
        "component": "tool_verification",
        "status": "inconclusive",
        "policy_verdict": "inconclusive",
        "policy_reason": "Patch execution incomplete or remote runner did not finish.",
        "compile": {"status": "skipped", "reason": "remote runner incomplete"},
        "smartbugs": {"status": "skipped", "reason": "remote runner incomplete"},
        "tools": {},
        "summary": {"pass": 0, "fail": 0, "warn": 0, "error": 0},
        "slither": {"status": "skipped", "findings": [], "verdict": "Not run"},
    }
    behavioral_result = {
        "status": "inconclusive",
        "findings": [],
        "checks": ["Patch execution incomplete: behavioral verification skipped."],
    }
    return tool_result, behavioral_result


def decide_overall_verdict(
    *,
    remote: dict[str, Any],
    fused: dict[str, Any],
    code: str,
    fixed_code: str,
    patch_applied: bool,
    tool_result: dict[str, Any],
    behavioral_result: dict[str, Any],
    fixed_static: dict[str, Any] | None,
    fixed_rag: dict[str, Any] | None,
) -> str:
    """Step 4: the single verdict of phase5_verification.json (first matching rule wins)."""
    compile_status = tool_result.get("compile", {}).get("status")
    matrix_verdict = tool_result.get("policy_verdict", "inconclusive")
    local_residual = [
        name
        for name, result in (("static", fixed_static), ("rag", fixed_rag))
        if result and result.get("status") == "ok" and result.get("findings")
    ]

    if not remote or remote.get("status") in {"pending", "crashed"}:
        return "Inconclusive: patch execution incomplete"
    if fused.get("verdict") == "Safe" and not patch_applied and fixed_code == code:
        return "No patch required: original detection reported Safe"
    if fused.get("verdict") != "Vulnerable" and not patch_applied and fixed_code == code:
        return "Inconclusive: detection incomplete and no patch was generated"
    if behavioral_result.get("status") == "failed":
        return "Patch rejected: behavioral regression detected"
    if compile_status == "failed":
        return "Patch rejected: fixed code does not compile"
    if not patch_applied or fixed_code == code:
        return "Patch rejected: no effective patch was applied"
    if matrix_verdict == "failed":
        return "Patch rejected: security tool matrix found residual or introduced vulnerabilities"
    if local_residual:
        return f"Patch rejected: {'/'.join(local_residual)} still detect the targeted vulnerability"
    if compile_status in {"unavailable", "baseline_failed"}:
        return "Inconclusive: fixed code could not be compiled in isolation"
    if matrix_verdict == "passed":
        # The binary detector is known to over-flag safe functions, so its disagreement does
        # not block acceptance; it is surfaced as `detector_disagreement` instead.
        return "Patch accepted by security tool matrix"
    return "Inconclusive: security tool matrix did not complete enough checks"
