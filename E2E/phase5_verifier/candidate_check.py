"""Phase 5 - kiểm tra bản vá ứng viên trong vòng sửa: compiler (so với bản gốc), detector chấm lại, static/RAG chạy lại."""

from E2E.common.regex_patterns import RE_SOL_FILE_PATH, RE_SOLC_ERROR_LINE
from E2E.common.solc import check_compilation_with_baseline, compiler_error_signatures
from E2E.config import RISK_ORDER
from E2E.phase0_parsing.code_units import parse_code_units
from E2E.phase1a_static.static_runner import run_static
from E2E.phase1b_rag.rag_runner import run_rag
from E2E.phase1c_detector.detector import score_units_via_server


# Compile bản vá so với bản gốc -> passed / failed / skipped (không có solc) / not_applicable (bản gốc vốn lỗi).
def evaluate_compilation(original_code: str, fixed_code: str) -> dict:
    result = check_compilation_with_baseline(original_code, fixed_code)
    fixed, baseline = result.get("fixed") or {}, result.get("baseline") or {}
    if fixed.get("passed") is None:
        state = "skipped"
    elif fixed.get("passed"):
        state = "passed"
    elif result.get("baseline_failed"):
        new_errors = compiler_error_signatures(fixed.get("error", "")) \
            - compiler_error_signatures(baseline.get("error", ""))
        state = "failed" if new_errors else "not_applicable"
    else:
        state = "failed"
    return {"state": state, "failed": state == "failed", "compiler": result.get("compiler"),
            "selection_reason": result.get("selection_reason"), "baseline_failed": result.get("baseline_failed", False),
            "error": fixed.get("error", "") if state == "failed" else ""}


# Các unit chứa dòng bị lỗi compile ("--> file:DÒNG:CỘT") -> biết hàm vá nào làm hỏng compile.
def units_with_compiler_errors(candidate_code: str, error_text: str) -> set[str]:
    lines = {int(value) for value in RE_SOLC_ERROR_LINE.findall(error_text or "")}
    return {unit.unit_id for unit in parse_code_units(candidate_code)
            if any(unit.start_line <= line <= unit.end_line for line in lines)}


# Tập cặp (SWC, unit_id) của các finding (so lỗi còn sót / lỗi mới).
def _finding_pairs(findings: list[dict]) -> set[tuple[str, str]]:
    return {(str(item.get("swc")), str(item.get("unit_id"))) for item in findings
            if item.get("swc") and item.get("unit_id")}


# Lỗi compile cho feedback: <= 3 khối lỗi solc, mỗi khối <= 6 dòng (Error, vị trí, mã, dấu ^); đường dẫn tạm -> Contract.sol.
def _compiler_feedback(error_text: str) -> str:
    blocks = [block.splitlines()[:6] for block in (error_text or "").split("\n\n") if block.strip()][:3]
    return RE_SOL_FILE_PATH.sub("Contract.sol", "\n".join(line for block in blocks for line in block))


# Feedback cho lần Fixer tiếp theo: lỗi compile, hàm detector vẫn báo Vulnerable, finding static/RAG còn sót / mới.
def _verification_feedback(attempt: int, max_attempts: int, verification: dict) -> str:
    parts = [f"\n\n[CRITICAL RESIDUAL ERROR FEEDBACK - ATTEMPT {attempt}/{max_attempts}]"]
    compile_info = verification["compilation"]
    if compile_info["failed"]:
        clean = _compiler_feedback(compile_info.get("error", "")) or compile_info.get("error", "")[:400]
        parts.append(f"Solidity compiler ({compile_info.get('compiler')}) failed with error:\n{clean}\n"
                     "Your patch broke compilation. You MUST fix this compiler error.")
    if verification["detector_failing_units"]:
        parts.append("The neural detector re-evaluated your patch and STILL FLAGGED it as Vulnerable: "
                     f"{', '.join(verification['detector_failing_units'])}.")
    for label, key in (("still present", "residual_findings"), ("newly introduced", "introduced_findings")):
        parts += [f"Static/RAG finding {label}: [{item.get('swc')}] {item.get('title', '')} "
                  f"in {item.get('unit_id')} (line {item.get('line')})." for item in verification[key]]
    return "\n".join(parts)


# Detector chấm lại CHỈ các hàm đã vá (hàm khác giữ điểm cũ; hàm vá không còn trong mã -> error).
def _rescore_patched_units(original_unit_results: list[dict], fixed_units: list, patched_unit_ids: set[str],
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


# [KIỂM CHỨNG TRONG VÒNG SỬA] compile + detector (hàm mục tiêu còn Vulnerable) + static/RAG (lỗi cũ còn / lỗi nặng mới).
def verify_candidate(source_code: str, candidate_code: str, *, patched_unit_ids: set[str], target_unit_ids: set[str],
                     original_unit_results: list[dict], target_pairs: set[tuple[str, str]],
                     known_pairs: set[tuple[str, str]], threshold: float, server_url: str | None) -> dict:
    compile_info = evaluate_compilation(source_code, candidate_code)
    fixed_units = parse_code_units(candidate_code)
    rescored = _rescore_patched_units(original_unit_results, fixed_units, patched_unit_ids, threshold, server_url)
    targets = [item for item in rescored if item["unit_id"] in target_unit_ids]
    detector_failing = sorted(item["unit_id"] for item in targets if item.get("label") == "Vulnerable")
    unverified = sorted(item["unit_id"] for item in targets if item.get("vulnerability_probability") is None)
    channel_errors, fixed_local = {}, []
    for name, runner in (("static", run_static), ("rag", run_rag)):
        try:
            fixed_local.extend(runner(candidate_code, fixed_units).get("findings", []))
        except Exception as error:
            channel_errors[name] = f"{type(error).__name__}: {error}"
    pair = lambda item: (str(item.get("swc")), str(item.get("unit_id")))
    residual = [item for item in fixed_local if pair(item) in target_pairs]
    introduced = [item for item in fixed_local if str(item.get("unit_id")) in patched_unit_ids
                  and pair(item) not in known_pairs and RISK_ORDER.get(str(item.get("severity")), 0) >= RISK_ORDER["High"]]
    culprits = units_with_compiler_errors(candidate_code, compile_info.get("error", "")) \
        if compile_info["failed"] else set()
    failing = set(detector_failing) | set(unverified) | (culprits & patched_unit_ids)
    failing.update(str(item.get("unit_id")) for item in residual + introduced if item.get("unit_id"))
    scored = [float(item["vulnerability_probability"]) for item in targets
              if item.get("vulnerability_probability") is not None]
    return {
        "passed": not compile_info["failed"] and not failing and not channel_errors, "compilation": compile_info,
        "unit_results": rescored, "detector_failing_units": detector_failing, "unverified_units": unverified,
        "residual_findings": residual, "introduced_findings": introduced, "channel_errors": channel_errors,
        "failing_units": sorted(failing), "compile_culprit_units": sorted(culprits),
        "max_vulnerability_probability": round(max(scored), 4) if scored else None,
    }
