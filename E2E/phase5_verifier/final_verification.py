"""
Phase 5 - final verification of the patched contract, run by the orchestrator.

1. Compile + Slither on original vs patched source      (tool_matrix.py)
2. Behaviour check: compile both, compare the ASTs      (behavior_verifier.py)
3. Static + RAG re-run on the patched source, scoped to the findings of the patched
   units (an unrelated pre-existing finding elsewhere does not reject the patch)
4. `decide_overall_verdict`: one verdict for phase5_verification.json, checked in a
   fixed order (behaviour regression > compile failure > no effective patch > Slither
   failure > residual static/RAG finding > ...).

`build_verification_channels` turns all channels into the per-channel table of the
report (section 5.5 "Verification Results").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from E2E.common.safe_run import safe_run
from E2E.phase0_parsing.code_units import parse_code_units
from E2E.phase1a_static.static_runner import run_static
from E2E.phase1b_rag.rag_runner import run_rag
from E2E.phase5_verifier.behavior_verifier import verify_patch_behavior
from E2E.phase5_verifier.tool_matrix import verify_security_tool_matrix


# Một kênh (static/RAG/detector) được coi là "đạt" khi status=ok và verdict=Safe.
def _verification_passed(result: dict[str, Any] | None) -> bool:
    return bool(result) and result.get("status") == "ok" and result.get("verdict") == "Safe"


# Rút gọn finding chỉ còn các trường cần thiết để đưa vào feedback/báo cáo.
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


# Lý do detector vẫn coi bản vá là lỗi: liệt kê các hàm đã vá còn bị gắn Vulnerable kèm xác suất.
def _detector_reason(detector: dict[str, Any] | None) -> str:
    detector = detector or {}
    vulnerable_units = [
        f"{item.get('unit_id') or 'fixed source'} ({float(item.get('vulnerability_probability') or 0.0):.3f})"
        for item in detector.get("unit_results", [])
        if item.get("label") == "Vulnerable"
    ]
    if vulnerable_units:
        return f"The neural detector still flags: {', '.join(vulnerable_units)}."
    return str(detector.get("error") or "The neural detector verification did not return a Safe verdict.")


# Ba kênh static / RAG / detector trên bản vá -> {static, rag, llm}: passed, status, verdict, findings rút gọn, lý do.
def _local_channel_components(
    fixed_static: dict[str, Any],
    fixed_rag: dict[str, Any],
    detector_fixed: dict[str, Any] | None,
) -> dict[str, Any]:
    detector = detector_fixed or {}
    llm_passed = _verification_passed(detector_fixed)
    components = {
        name: {
            "passed": _verification_passed(result),
            "status": result.get("status", "unknown"),
            "verdict": result.get("verdict", "Unknown"),
            "findings": _compact_feedback_findings(result.get("findings", [])),
            "error": result.get("error"),
        }
        for name, result in (("static", fixed_static), ("rag", fixed_rag))
    }
    components["llm"] = {
        "passed": llm_passed,
        "status": detector.get("status", "missing"),
        "verdict": detector.get("verdict", "Unknown"),
        "reason": "" if llm_passed else _detector_reason(detector_fixed),
        "unit_results": detector.get("unit_results", []),
        "error": detector.get("error"),
    }
    return components


# Chạy lại static (Phase 1A) và RAG (Phase 1B) trên MÃ ĐÃ VÁ (không dùng lại kết quả Phase 1 của bản gốc).
def rerun_local_channels(fixed_code: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Static and RAG evidence for the *patched* source (verification channels must not reuse
    the original contract's Phase 1 results)."""
    units = parse_code_units(fixed_code)
    return (
        safe_run("static", lambda: run_static(fixed_code, units)),
        safe_run("rag", lambda: run_rag(fixed_code, units)),
    )


# Chỉ giữ finding còn sót trùng (SWC, unit) với lỗi đã nhắm sửa. Finding cũ ở chỗ khác của contract
# không phải bằng chứng bản vá thất bại nên không được chặn kết quả.
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
        "scope": "original target findings",
    }


# Rút gọn finding của tool (tool, name, check, category, severity, line, message, swc).
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


# Nhiệm vụ: gộp mọi kênh kiểm chứng (Slither, compile, static, RAG, detector) thành tóm tắt theo kênh
# {tên kênh: {passed, verdict, findings, reason, ...}}.
# Mục đích: bảng "Verification Results" (mục 5.5) của báo cáo Phase 6 (report_data["verification_channels"]).
def build_verification_channels(
    tool_result: dict[str, Any],
    *,
    static_result: dict[str, Any] | None = None,
    rag_result: dict[str, Any] | None = None,
    detector_fixed: dict[str, Any] | None = None,
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
    if any([static_result, rag_result, detector_fixed]):
        components.update(_local_channel_components(static_result or {}, rag_result or {}, detector_fixed))

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
    return components


# [KIỂM CHỨNG CUỐI - orchestrator gọi ở lượt 2] Trên bản vá cuối cùng:
#   1) compile + Slither trên bản gốc và bản vá - verify_security_tool_matrix;
#   2) kiểm tra hành vi bằng so sánh AST - verify_patch_behavior;
#   3) chạy lại static + RAG, chỉ xét các finding của hàm đã vá (scope_to_targets).
# Mỗi bước bọc trong safe_run. Trả (tool_result, behavioral_result, fixed_static, fixed_rag).
def verify_final_patch(
    code: str,
    fixed_code: str,
    *,
    patch_outcomes: list[dict[str, Any]],
    fused: dict[str, Any],
    file_name: str,
    source_root: str | Path | None,
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


# Kết quả tool/hành vi "inconclusive" khi không có bản vá nào để kiểm chứng.
def skipped_verification() -> tuple[dict[str, Any], dict[str, Any]]:
    """Tool and behaviour results when there is no patch to verify."""
    tool_result = {
        "component": "tool_verification",
        "status": "inconclusive",
        "policy_verdict": "inconclusive",
        "policy_reason": "Patch execution incomplete or remote runner did not finish.",
        "compile": {"status": "skipped", "reason": "remote runner incomplete"},
        "tools": {},
        "slither": {"status": "skipped", "findings": [], "verdict": "Not run"},
    }
    behavioral_result = {
        "status": "inconclusive",
        "findings": [],
        "checks": ["Patch execution incomplete: behavioral verification skipped."],
    }
    return tool_result, behavioral_result


# [KẾT LUẬN DUY NHẤT của phase5_verification.json] Xét các luật theo thứ tự, luật khớp đầu tiên thắng:
# chưa có kết quả GPU -> Inconclusive; gốc Safe & không vá -> No patch required;
# Advisor loại hết finding -> No patch required; còn finding nhưng không có hàm nào để vá (chỉ finding cấp contract
# hoặc hàm view/pure, vòng sửa 0 attempt) -> Inconclusive; hồi quy hành vi -> rejected;
# không compile -> rejected; không có bản vá hiệu lực -> rejected; Slither FAIL -> rejected;
# static/RAG vẫn thấy lỗi mục tiêu -> rejected; không compile riêng được -> Inconclusive;
# Slither passed -> "Patch accepted by Slither"; còn lại -> Inconclusive.
# (Detector vẫn báo lỗi KHÔNG chặn chấp nhận - chỉ ghi detector_disagreement, vì detector hay báo nhầm hàm an toàn.)
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
    slither_verdict = tool_result.get("policy_verdict", "inconclusive")
    local_residual = [
        name
        for name, result in (("static", fixed_static), ("rag", fixed_rag))
        if result and result.get("status") == "ok" and result.get("findings")
    ]

    if not remote:
        return "Inconclusive: patch execution incomplete"
    if fused.get("verdict") == "Safe" and not patch_applied and fixed_code == code:
        return "No patch required: original detection reported Safe"
    if fused.get("verdict") != "Vulnerable" and not patch_applied and fixed_code == code:
        return "Inconclusive: detection incomplete and no patch was generated"
    if not patch_applied and fixed_code == code:
        phases = remote.get("phases", {})
        detector = phases.get("detector", {})
        if not detector.get("confirmed_findings") and detector.get("dismissed_findings"):
            return "No patch required: the Advisor dismissed every finding as a false positive"
        if phases.get("verifier", {}).get("attempts_count") == 0:
            return "Inconclusive: no function-level repair target for the confirmed findings"
    if behavioral_result.get("status") == "failed":
        return "Patch rejected: behavioral regression detected"
    if compile_status == "failed":
        return "Patch rejected: fixed code does not compile"
    if not patch_applied or fixed_code == code:
        return "Patch rejected: no effective patch was applied"
    if slither_verdict == "failed":
        return "Patch rejected: Slither found residual or introduced vulnerabilities"
    if local_residual:
        return f"Patch rejected: {'/'.join(local_residual)} still detect the targeted vulnerability"
    if compile_status in {"unavailable", "baseline_failed"}:
        return "Inconclusive: fixed code could not be compiled in isolation"
    if slither_verdict == "passed":
        # The binary detector is known to over-flag safe functions, so its disagreement does
        # not block acceptance; it is surfaced as `detector_disagreement` instead.
        return "Patch accepted by Slither"
    return "Inconclusive: Slither did not complete enough checks"
