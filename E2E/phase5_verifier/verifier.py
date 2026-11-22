"""Phase 5 - Verifier DUY NHẤT: kiểm chứng mỗi bản vá (compile, hành vi, Slither, static, detector) trong vòng sửa."""

from pathlib import Path

from E2E.common.regex_patterns import RE_SOL_FILE_PATH, RE_SOLC_ERROR_LINE
from E2E.common.solc import extract_solc_errors
from E2E.config import RISK_ORDER
from E2E.phase0_parsing.code_units import CodeUnit, parse_code_units
from E2E.phase1a_static.static_runner import run_static
from E2E.phase1c_detector.detector import score_units_via_server
from E2E.phase5_verifier.behavior_verifier import verify_patch_behavior
from E2E.phase5_verifier.tool_matrix import verify_security_tool_matrix

# Trạng thái compile của tool_matrix -> trạng thái cho vòng sửa (baseline_failed: bản gốc cũng không compile được).
_COMPILE_STATE = {"passed": "passed", "failed": "failed", "error": "failed", "baseline_failed": "not_applicable",
                  "unavailable": "skipped", "skipped": "skipped"}
_FEEDBACK_FIELDS = ("finding_id", "swc", "unit_id", "title", "severity", "confidence", "line", "start_line", "end_line",
                    "matched_text", "description", "remediation", "evidence", "notes")
_TOOL_FIELDS = ("tool", "name", "check", "category", "severity", "line", "message", "swc")


# Các unit chứa dòng bị lỗi compile ("--> file:DÒNG:CỘT") -> biết hàm vá nào làm hỏng compile.
def units_with_compiler_errors(candidate_code: str, error_text: str) -> set[str]:
    lines = {int(value) for value in RE_SOLC_ERROR_LINE.findall(error_text or "")}
    return {unit.unit_id for unit in parse_code_units(candidate_code)
            if any(unit.start_line <= line <= unit.end_line for line in lines)}


# Tập cặp (SWC, unit_id) của các finding (so lỗi còn sót / lỗi mới).
def finding_pairs(findings: list[dict]) -> set[tuple[str, str]]:
    return {(str(item.get("swc")), str(item.get("unit_id"))) for item in findings
            if item.get("swc") and item.get("unit_id")}


# Chỉ giữ các trường `fields` của mỗi finding (rút gọn cho báo cáo).
def _compact(findings: list[dict], fields: tuple[str, ...]) -> list[dict]:
    return [{key: finding[key] for key in fields if key in finding} for finding in findings]


# Lỗi compile cho feedback: <= 3 khối lỗi solc, mỗi khối <= 6 dòng (Error, vị trí, mã, dấu ^); đường dẫn tạm -> Contract.sol.
def _compiler_feedback(error_text: str) -> str:
    blocks = [block.splitlines()[:6] for block in (error_text or "").split("\n\n") if block.strip()][:3]
    return RE_SOL_FILE_PATH.sub("Contract.sol", "\n".join(line for block in blocks for line in block))


# Detector chấm lại CHỈ các hàm đã vá (hàm khác giữ điểm cũ; hàm vá không còn trong mã -> error).
def _rescore_patched_units(original_unit_results: list[dict], fixed_units: list[CodeUnit], patched_unit_ids: set[str],
                           threshold: float, server_url: str | None) -> list[dict]:
    fixed_by_id, rescored = {unit.unit_id: unit for unit in fixed_units}, []
    for original in original_unit_results:
        if original["unit_id"] not in patched_unit_ids:
            rescored.append(dict(original))
        elif original["unit_id"] in fixed_by_id:
            rescored.extend(score_units_via_server([fixed_by_id[original["unit_id"]]], threshold, server_url))
        else:
            rescored.append({**original, "status": "error", "label": "Inconclusive", "vulnerability_probability": None,
                             "error": "patched unit not found in candidate source"})
    return rescored


# Hàm đã vá chứa các dòng của một finding Slither (không định vị được -> mọi hàm đã vá).
def _units_of(finding: dict, fixed_units: list[CodeUnit], patched: set[str]) -> set[str]:
    value = finding.get("line") if finding.get("line") is not None else finding.get("lines")
    lines = [value] if isinstance(value, int) else [item for item in value if isinstance(item, int)] \
        if isinstance(value, (list, tuple, set)) else []
    found = {unit.unit_id for unit in fixed_units if any(unit.start_line <= line <= unit.end_line for line in lines)}
    return (found & patched) or set(patched)


# Hàm đã vá bị kiểm tra hành vi đánh rớt (finding không gắn hàm, vd locked-ether -> mọi hàm đã vá).
def _behavior_units(behavioral: dict, patched: set[str]) -> set[str]:
    units = set()
    for item in behavioral.get("findings", []):
        units |= ({str(item.get("unit_id"))} & patched) or set(patched)
    return units


# Kết quả "không chạy" khi không có bản vá nào để kiểm chứng (compile / Slither / hành vi đều skipped).
def skipped_verification(reason: str = "No applied patch to verify.") -> tuple[dict, dict]:
    tool_result = {
        "component": "tool_verification", "status": "inconclusive", "policy_verdict": "inconclusive",
        "policy_reason": reason, "compile": {"status": "skipped", "reason": reason}, "tools": {},
        "slither": {"status": "skipped", "findings": [], "verdict": "Not run"},
    }
    return tool_result, {"status": "not_applicable", "findings": [], "checks": [reason]}


# [KIỂM CHỨNG MỘT BẢN VÁ] compile + Slither (bản gốc vs bản vá), hành vi AST, static chạy lại, detector chấm lại.
def verify_patch(source_code: str, candidate_code: str, *, patch_outcomes: list[dict], target_findings: list[dict],
                 original_unit_results: list[dict], target_unit_ids: set[str], known_pairs: set[tuple[str, str]],
                 threshold: float, server_url: str | None, file_name: str = "Contract.sol",
                 source_root: str | Path | None = None) -> dict:
    patched = {str(item.get("unit_id")) for item in patch_outcomes
               if item.get("status") == "applied" and item.get("unit_id")}
    fixed_units = parse_code_units(candidate_code)
    targets = [item for item in target_findings if item.get("unit_id") in patched] or target_findings
    if patched:
        tool_result = verify_security_tool_matrix(source_code, candidate_code, file_name=file_name,
                                                  source_root=source_root, target_findings=targets)
        behavioral = verify_patch_behavior(source_code, candidate_code, patch_outcomes, tool_result)
    else:
        tool_result, behavioral = skipped_verification()
    compile_info = tool_result.get("compile", {})
    state = _COMPILE_STATE.get(str(compile_info.get("status")), "failed")
    compile_error = extract_solc_errors(compile_info.get("stderr") or compile_info.get("error") or "") \
        if state == "failed" else ""
    rescored = _rescore_patched_units(original_unit_results, fixed_units, patched, threshold, server_url)
    scored_targets = [item for item in rescored if item["unit_id"] in target_unit_ids]
    detector_failing = sorted(item["unit_id"] for item in scored_targets if item.get("label") == "Vulnerable")
    unverified = sorted(item["unit_id"] for item in scored_targets if item.get("vulnerability_probability") is None)
    channel_errors, fixed_static = {}, {"status": "skipped", "findings": [], "verdict": "Not run"}
    if patched:
        try:
            fixed_static = run_static(candidate_code, fixed_units)
        except Exception as error:
            channel_errors["static"] = f"{type(error).__name__}: {error}"
            fixed_static = {"status": "error", "findings": [], "verdict": "Error", "error": channel_errors["static"]}
    target_pairs = {pair for pair in finding_pairs(target_findings) if pair[1] in target_unit_ids}
    pair = lambda item: (str(item.get("swc")), str(item.get("unit_id")))
    residual = [item for item in fixed_static.get("findings", []) if pair(item) in target_pairs]
    introduced = [item for item in fixed_static.get("findings", []) if str(item.get("unit_id")) in patched
                  and pair(item) not in known_pairs and RISK_ORDER.get(str(item.get("severity")), 0) >= RISK_ORDER["High"]]
    culprits = units_with_compiler_errors(candidate_code, compile_error) if state == "failed" else set()
    slither = (tool_result.get("tools") or {}).get("slither", {})
    slither_units = set().union(*(_units_of(item, fixed_units, patched) for item in
                                  [*slither.get("residual_target_findings", []), *slither.get("introduced_findings", [])]))
    behavior_units = _behavior_units(behavioral, patched)
    failing = set(detector_failing) | set(unverified) | (culprits & patched) | slither_units | behavior_units
    failing.update(str(item.get("unit_id")) for item in residual + introduced if item.get("unit_id"))
    scored = [float(item["vulnerability_probability"]) for item in scored_targets
              if item.get("vulnerability_probability") is not None]
    passed = bool(patched) and state != "failed" and not failing and not channel_errors \
        and behavioral.get("status") != "failed" and tool_result.get("policy_verdict") != "failed"
    scoped_static = {**fixed_static, "findings": residual, "introduced_findings": introduced,
                     "verdict": "Vulnerable" if residual or introduced else "Safe",
                     "scope": "original target findings + new High/Critical findings in patched units"} \
        if fixed_static.get("status") == "ok" else fixed_static
    return {
        "passed": passed,
        "compilation": {"state": state, "failed": state == "failed", "compiler": compile_info.get("compiler"),
                        "selection_reason": compile_info.get("selection_reason"),
                        "baseline_failed": state == "not_applicable", "error": compile_error},
        "tool_verification": tool_result, "behavioral_verification": behavioral, "fixed_static": scoped_static,
        "unit_results": rescored, "detector_failing_units": detector_failing, "unverified_units": unverified,
        "residual_findings": residual, "introduced_findings": introduced, "channel_errors": channel_errors,
        "slither_failing_units": sorted(slither_units), "behavior_failing_units": sorted(behavior_units),
        "failing_units": sorted(failing), "compile_culprit_units": sorted(culprits),
        "max_vulnerability_probability": round(max(scored), 4) if scored else None,
    }


# Feedback cho lần Fixer tiếp theo: MỌI kiểm tra không đạt (compile, hành vi, Slither, static, detector).
def verification_feedback(attempt: int, max_attempts: int, verification: dict) -> str:
    parts = [f"\n\n[CRITICAL RESIDUAL ERROR FEEDBACK - ATTEMPT {attempt}/{max_attempts}]"]
    compile_info = verification["compilation"]
    if compile_info["failed"]:
        clean = _compiler_feedback(compile_info.get("error", "")) or compile_info.get("error", "")[:400]
        parts.append(f"Solidity compiler ({compile_info.get('compiler')}) failed with error:\n{clean}\n"
                     "Your patch broke compilation. You MUST fix this compiler error.")
    for item in verification["behavioral_verification"].get("findings", []):
        parts.append(f"Behavior check failed in {item.get('unit_id') or 'the patched contract'}: "
                     f"{item.get('evidence') or item.get('check')}. Keep every value transfer, state write, "
                     "event and return of the original function.")
    slither = (verification["tool_verification"].get("tools") or {}).get("slither", {})
    for label, key in (("still reports", "residual_target_findings"), ("reports a NEW high-impact", "introduced_findings")):
        parts += [f"Slither {label} `{item.get('check') or item.get('name')}` at line(s) {item.get('line')}: "
                  f"{str(item.get('message') or '')[:200]}" for item in slither.get(key, [])]
    if verification["detector_failing_units"]:
        parts.append("The neural detector re-evaluated your patch and STILL FLAGGED it as Vulnerable: "
                     f"{', '.join(verification['detector_failing_units'])}.")
    for label, key in (("still present", "residual_findings"), ("newly introduced", "introduced_findings")):
        parts += [f"Static finding {label}: [{item.get('swc')}] {item.get('title', '')} "
                  f"in {item.get('unit_id')} (line {item.get('line')})." for item in verification[key]]
    return "\n".join(parts)


# Gộp các kênh kiểm chứng (Slither, compile, static, detector) thành bảng "Verification Results" (mục 5.5 báo cáo).
def build_verification_channels(tool_result: dict, *, static_result: dict | None = None,
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
    if static_result or detector_fixed:
        static = static_result or {}
        components["static"] = {"passed": static.get("status") == "ok" and static.get("verdict") == "Safe",
                                "status": static.get("status", "unknown"), "verdict": static.get("verdict", "Unknown"),
                                "findings": _compact([*static.get("findings", []),
                                                      *static.get("introduced_findings", [])], _FEEDBACK_FIELDS),
                                "error": static.get("error")}
        detector = detector_fixed or {}
        llm_passed = detector.get("status") == "ok" and detector.get("verdict") == "Safe"
        flagged = [f"{item.get('unit_id') or 'fixed source'} ({float(item.get('vulnerability_probability') or 0.0):.3f})"
                   for item in detector.get("unit_results", []) if item.get("label") == "Vulnerable"]
        reason = "" if llm_passed else f"The neural detector still flags: {', '.join(flagged)}." if flagged \
            else str(detector.get("error") or "The neural detector verification did not return a Safe verdict.")
        components["llm"] = {"passed": llm_passed, "status": detector.get("status", "missing"),
                             "verdict": detector.get("verdict", "Unknown"), "reason": reason,
                             "unit_results": detector.get("unit_results", []), "error": detector.get("error")}
    compile_info = tool_result.get("compile", {})
    compile_status = compile_info.get("status", "unknown")
    compile_passed = compile_status == "passed"
    components["compile"] = {
        "passed": compile_passed, "status": compile_status, "verdict": "PASS" if compile_passed else "FAIL",
        "applicable": True, "findings": [], "residual_target_findings": [], "introduced_findings": [],
        "reason": "Compilation succeeded" if compile_passed else "Compilation failed",
        "error": compile_info.get("stderr") or compile_info.get("error") or compile_info.get("reason") or "",
    }
    return components


# [KẾT LUẬN DUY NHẤT của phase5_verification.json] Xét các luật theo thứ tự trên bản vá tốt nhất, luật khớp đầu tiên thắng.
def decide_overall_verdict(*, remote: dict, fused: dict, code: str, fixed_code: str, patch_applied: bool,
                           tool_result: dict, behavioral_result: dict, fixed_static: dict | None) -> str:
    compile_status = tool_result.get("compile", {}).get("status")
    slither_verdict = tool_result.get("policy_verdict", "inconclusive")
    static_ok = bool(fixed_static) and fixed_static.get("status") == "ok"
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
    if static_ok and fixed_static.get("findings"):
        return "Patch rejected: static rules still detect the targeted vulnerability"
    if static_ok and fixed_static.get("introduced_findings"):
        return "Patch rejected: static rules report a new High/Critical vulnerability in the patched code"
    if compile_status in {"unavailable", "baseline_failed"}:
        return "Inconclusive: fixed code could not be compiled in isolation"
    if slither_verdict == "passed":
        return "Patch accepted by Slither"
    return "Inconclusive: Slither did not complete enough checks"
