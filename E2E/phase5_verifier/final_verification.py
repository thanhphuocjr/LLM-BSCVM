"""Phase 5 - kiểm chứng cuối bản vá (orchestrator): Slither, hành vi (AST), static/RAG chạy lại -> overall_verdict."""

from pathlib import Path

from E2E.common.safe_run import safe_run
from E2E.phase0_parsing.code_units import parse_code_units
from E2E.phase1a_static.static_runner import run_static
from E2E.phase1b_rag.rag_runner import run_rag
from E2E.phase5_verifier.behavior_verifier import verify_patch_behavior
from E2E.phase5_verifier.tool_matrix import verify_security_tool_matrix

_FEEDBACK_FIELDS = ("finding_id", "swc", "unit_id", "title", "severity", "confidence", "line", "start_line", "end_line",
                    "matched_text", "description", "remediation", "evidence", "notes")
_TOOL_FIELDS = ("tool", "name", "check", "category", "severity", "line", "message", "swc")


# Chỉ giữ các trường `fields` của mỗi finding (rút gọn cho báo cáo).
def _compact(findings: list[dict], fields: tuple[str, ...]) -> list[dict]:
    return [{key: finding[key] for key in fields if key in finding} for finding in findings]


# Một kênh (static / RAG / detector) "đạt" khi status=ok và verdict=Safe.
def _verification_passed(result: dict | None) -> bool:
    return bool(result) and result.get("status") == "ok" and result.get("verdict") == "Safe"


# Lý do detector vẫn coi bản vá là lỗi: các hàm còn bị gắn Vulnerable kèm xác suất.
def _detector_reason(detector: dict | None) -> str:
    detector = detector or {}
    flagged = [f"{item.get('unit_id') or 'fixed source'} ({float(item.get('vulnerability_probability') or 0.0):.3f})"
               for item in detector.get("unit_results", []) if item.get("label") == "Vulnerable"]
    if flagged:
        return f"The neural detector still flags: {', '.join(flagged)}."
    return str(detector.get("error") or "The neural detector verification did not return a Safe verdict.")


# Gộp mọi kênh kiểm chứng (Slither, compile, static, RAG, detector) thành bảng "Verification Results" (mục 5.5 báo cáo).
def build_verification_channels(tool_result: dict, *, static_result: dict | None = None, rag_result: dict | None = None,
                                detector_fixed: dict | None = None) -> dict:
    components = {}
    for tool_name, result in tool_result.get("tools", {}).items():
        residual, introduced = result.get("residual_target_findings", []), result.get("introduced_findings", [])
        verdict = str(result.get("verdict") or "ERROR")
        components[tool_name] = {
            "passed": verdict in {"PASS", "WARN"}, "status": result.get("run_status", "unknown"), "verdict": verdict,
            "applicable": bool(result.get("applicable", True)),
            "findings": _compact([*residual, *introduced], _TOOL_FIELDS),
            "residual_target_findings": _compact(residual, _TOOL_FIELDS),
            "introduced_findings": _compact(introduced, _TOOL_FIELDS),
            "reason": result.get("detail"), "error": result.get("error"),
        }
    if any([static_result, rag_result, detector_fixed]):
        for name, result in (("static", static_result or {}), ("rag", rag_result or {})):
            components[name] = {"passed": _verification_passed(result), "status": result.get("status", "unknown"),
                                "verdict": result.get("verdict", "Unknown"),
                                "findings": _compact(result.get("findings", []), _FEEDBACK_FIELDS),
                                "error": result.get("error")}
        detector, llm_passed = detector_fixed or {}, _verification_passed(detector_fixed)
        components["llm"] = {"passed": llm_passed, "status": detector.get("status", "missing"),
                             "verdict": detector.get("verdict", "Unknown"),
                             "reason": "" if llm_passed else _detector_reason(detector_fixed),
                             "unit_results": detector.get("unit_results", []), "error": detector.get("error")}
    compile_info = tool_result.get("compile", {})
    compile_status = compile_info.get("status", "unknown")
    compile_passed = compile_status == "passed"
    components["compile"] = {
        "passed": compile_passed, "status": compile_status, "verdict": "PASS" if compile_passed else "FAIL",
        "applicable": True,
        "findings": [], "residual_target_findings": [], "introduced_findings": [],
        "reason": "Compilation succeeded" if compile_passed else "Compilation failed",
        "error": compile_info.get("stderr") or compile_info.get("error") or compile_info.get("reason") or "",
    }
    return components


# Chỉ giữ finding còn sót trùng (SWC, unit) với lỗi đã nhắm sửa (finding cũ ở chỗ khác không chặn bản vá).
def scope_to_targets(result: dict, target_findings: list[dict]) -> dict:
    if result.get("status") != "ok":
        return result
    targets = {(str(item.get("swc")), str(item.get("unit_id"))) for item in target_findings
               if item.get("swc") and item.get("swc") != "UNMAPPED"}
    residual = [item for item in result.get("findings", [])
                if (str(item.get("swc")), str(item.get("unit_id"))) in targets]
    return {**result, "findings": residual, "verdict": "Vulnerable" if residual else "Safe",
            "scope": "original target findings"}


# Kết quả tool / hành vi "inconclusive" khi không có bản vá nào để kiểm chứng.
def skipped_verification() -> tuple[dict, dict]:
    tool_result = {
        "component": "tool_verification", "status": "inconclusive", "policy_verdict": "inconclusive",
        "policy_reason": "Patch execution incomplete or remote runner did not finish.",
        "compile": {"status": "skipped", "reason": "remote runner incomplete"}, "tools": {},
        "slither": {"status": "skipped", "findings": [], "verdict": "Not run"},
    }
    return tool_result, {"status": "inconclusive", "findings": [],
                         "checks": ["Patch execution incomplete: behavioral verification skipped."]}


# [KIỂM CHỨNG CUỐI] compile + Slither (bản gốc vs bản vá), hành vi qua AST, static + RAG chạy lại trên các hàm đã vá.
def verify_final_patch(code: str, fixed_code: str, *, patch_outcomes: list[dict], fused: dict, file_name: str,
                       source_root: str | Path | None) -> tuple[dict, dict, dict, dict]:
    patched = {item.get("unit_id") for item in patch_outcomes
               if item.get("status") == "applied" and item.get("unit_id")}
    targets = [item for item in fused.get("findings", []) if item.get("unit_id") in patched] \
        or fused.get("findings", [])
    tool_result = safe_run("tool_verification", lambda: verify_security_tool_matrix(
        code, fixed_code, file_name=file_name, source_root=source_root, target_findings=targets))
    behavioral_result = safe_run("behavioral_verification",
                                 lambda: verify_patch_behavior(code, fixed_code, patch_outcomes, tool_result))
    units = parse_code_units(fixed_code)
    fixed_static = safe_run("static", lambda: run_static(fixed_code, units))
    fixed_rag = safe_run("rag", lambda: run_rag(fixed_code, units))
    return tool_result, behavioral_result, scope_to_targets(fixed_static, targets), scope_to_targets(fixed_rag, targets)


# [KẾT LUẬN DUY NHẤT của phase5_verification.json] Xét các luật theo thứ tự, luật khớp đầu tiên thắng.
def decide_overall_verdict(*, remote: dict, fused: dict, code: str, fixed_code: str, patch_applied: bool,
                           tool_result: dict, behavioral_result: dict, fixed_static: dict | None,
                           fixed_rag: dict | None) -> str:
    compile_status = tool_result.get("compile", {}).get("status")
    slither_verdict = tool_result.get("policy_verdict", "inconclusive")
    residual = [name for name, result in (("static", fixed_static), ("rag", fixed_rag))
                if result and result.get("status") == "ok" and result.get("findings")]
    unchanged = not patch_applied and fixed_code == code
    if not remote:
        return "Inconclusive: patch execution incomplete"
    if unchanged and fused.get("verdict") == "Safe":
        return "No patch required: original detection reported Safe"
    if unchanged and fused.get("verdict") != "Vulnerable":
        return "Inconclusive: detection incomplete and no patch was generated"
    detector = remote.get("phases", {}).get("detector", {})
    if unchanged and not detector.get("confirmed_findings") and detector.get("dismissed_findings"):
        return "No patch required: the Advisor dismissed every finding as a false positive"
    if unchanged and remote.get("phases", {}).get("verifier", {}).get("attempts_count") == 0:
        return "Inconclusive: no function-level repair target for the confirmed findings"
    if behavioral_result.get("status") == "failed":
        return "Patch rejected: behavioral regression detected"
    if compile_status == "failed":
        return "Patch rejected: fixed code does not compile"
    if not patch_applied or fixed_code == code:
        return "Patch rejected: no effective patch was applied"
    if slither_verdict == "failed":
        return "Patch rejected: Slither found residual or introduced vulnerabilities"
    if residual:
        return f"Patch rejected: {'/'.join(residual)} still detect the targeted vulnerability"
    if compile_status in {"unavailable", "baseline_failed"}:
        return "Inconclusive: fixed code could not be compiled in isolation"
    if slither_verdict == "passed":
        return "Patch accepted by Slither"
    return "Inconclusive: Slither did not complete enough checks"
