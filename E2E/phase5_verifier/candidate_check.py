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

from typing import Any

from E2E.common.regex_patterns import RE_SOL_FILE_PATH, RE_SOLC_ERROR_LINE
from E2E.common.solc import check_compilation_with_baseline, compiler_error_signatures
from E2E.config import RISK_ORDER
from E2E.phase0_parsing.code_units import parse_code_units
from E2E.phase1a_static.static_runner import run_static
from E2E.phase1b_rag.rag_runner import run_rag
from E2E.phase1c_detector.detector import score_units_via_server


# Nhiệm vụ: biên dịch bản vá so với bản gốc (baseline) -> state:
# passed / failed / skipped (không có solc - không phải lỗi bản vá) /
# not_applicable (bản gốc vốn không compile riêng được, vd thiếu import, và bản vá không thêm lỗi mới).
# Mục đích: không đổ lỗi cho Fixer với những contract vốn đã không compile được.
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
        new_errors = compiler_error_signatures(fixed.get("error", "")) - compiler_error_signatures(
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
    }


# Ánh xạ vị trí lỗi solc ("--> file:DÒNG:CỘT") sang các unit chứa dòng đó -> biết hàm vá nào làm hỏng compile.
def units_with_compiler_errors(candidate_code: str, error_text: str) -> set[str]:
    """Map solc error locations (`--> file:LINE:COL`) to the units that contain them."""
    lines = {int(value) for value in RE_SOLC_ERROR_LINE.findall(error_text or "")}
    return {
        unit.unit_id
        for unit in parse_code_units(candidate_code)
        if any(unit.start_line <= line <= unit.end_line for line in lines)
    }


# Tập cặp (SWC, unit_id) từ danh sách finding - dùng để so lỗi còn sót / lỗi mới.
def _finding_pairs(findings: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {
        (str(item.get("swc")), str(item.get("unit_id")))
        for item in findings
        if item.get("swc") and item.get("unit_id")
    }


# [KIỂM CHỨNG TRONG VÒNG SỬA - 3 kênh, chạy trên mã đã vá]
#   1) compiler: evaluate_compilation (so với baseline);
#   2) neural detector: CHỈ chấm lại các hàm đã vá, hàm khác giữ điểm cũ; hàm mục tiêu vẫn Vulnerable -> fail;
#   3) static + RAG chạy lại: finding mục tiêu (SWC, unit) còn sót -> fail; finding High/Critical MỚI trong hàm đã vá -> fail.
# passed = compile được + không hàm nào fail + không kênh nào lỗi.
# Trả về chi tiết từng kênh và failing_units (để lần sửa sau chỉ vá lại các hàm này).
# Gọi bởi: gpu_pipeline.run_server_gpu_pipeline (mỗi attempt).
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


# Lỗi compile đưa vào feedback: tối đa 3 khối lỗi của solc (extract_solc_errors), mỗi khối giữ dòng "Error",
# vị trí, dòng mã và dấu ^ chỉ chỗ sai (tối đa 6 dòng); đường dẫn file tạm được thay bằng Contract.sol.
# Mục đích: Fixer biết chính xác định danh / dòng nào làm hỏng compile (chỉ dòng "Error" thì không đủ để sửa).
def _compiler_feedback(error_text: str) -> str:
    blocks = [block.splitlines()[:6] for block in (error_text or "").split("\n\n") if block.strip()][:3]
    return RE_SOL_FILE_PATH.sub("Contract.sol", "\n".join(line for block in blocks for line in block))


# Viết feedback cho lần Fixer tiếp theo: lỗi compile (kèm dòng mã bị lỗi), hàm detector vẫn báo Vulnerable,
# finding static/RAG còn sót hoặc mới phát sinh.
def _verification_feedback(attempt: int, max_attempts: int, verification: dict[str, Any]) -> str:
    parts = [f"\n\n[CRITICAL RESIDUAL ERROR FEEDBACK - ATTEMPT {attempt}/{max_attempts}]"]
    compile_info = verification["compilation"]
    if compile_info["failed"]:
        clean = _compiler_feedback(compile_info.get("error", "")) or compile_info.get("error", "")[:400]
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
