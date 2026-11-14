"""
Phase 5 - Slither on the original and the patched contract.

The patch must first compile (compared with the original as a baseline, so a contract
that never compiled standalone does not blame the patch). Slither then runs natively on
both versions:
  - a finding of the targeted vulnerability class still on the patched lines -> FAIL;
  - a new High/Critical finding the original did not have                  -> FAIL;
  - a clean run when Slither does not cover the targeted class              -> inconclusive.
`_policy_verdict` turns the Slither result into passed / failed / inconclusive.
(Phase 5 is Slither-only: Mythril/SmartCheck through SmartBugs were removed on 2026-11-12.)
"""

from __future__ import annotations

import platform
import tempfile
from pathlib import Path
from typing import Any

from E2E.common.security_tools import (
    _classify_category,
    _copy_source_tree,
    _run,
    _run_native_slither,
)
from E2E.common.regex_patterns import RE_LOCATION_IN_MESSAGE, RE_WHITESPACE
from E2E.common.solc import compiler_error_signatures, select_best_solc
from E2E.config import RISK_ORDER
from E2E.phase0_parsing.code_units import parse_code_units


# Đổi mức độ của tool (High/Medium/Low/Informational...) thành số để so sánh (thang RISK_ORDER của config).
def _severity_rank(value: Any) -> int:
    return RISK_ORDER.get(str(value or "").strip().capitalize(), 0)


# Nhóm lỗ hổng (theo _classify_category của common/security_tools.py) ứng với từng SWC mục tiêu.
SWC_CATEGORY = {
    "SWC-100": "access_control",
    "SWC-101": "arithmetic",
    "SWC-104": "unchecked_low_level_calls",
    "SWC-105": "prodigal",
    "SWC-106": "selfdestruct",
    "SWC-107": "reentrancy",
    "SWC-113": "denial_of_service",
    "SWC-115": "tx_origin",
    "SWC-116": "timestamp",
    "SWC-128": "denial_of_service",
}


# Nhóm lỗ hổng Slither phát hiện được. Lỗi mục tiêu ngoài nhóm này: Slither chạy sạch cũng không chứng minh đã sửa.
SLITHER_COVERAGE = {
    "access_control",
    "arithmetic",
    "denial_of_service",
    "reentrancy",
    "selfdestruct",
    "timestamp",
    "tx_origin",
    "unchecked_low_level_calls",
}


# Biên dịch mã (kèm file phụ thuộc) bằng solc trong thư mục tạm; trả trạng thái compile (theo mã thoát của solc).
# Không yêu cầu output nào (như vòng sửa): solc < 0.4.12 không có --ast-compact-json, nếu yêu cầu thì mọi contract
# 0.4.11 bị coi là "không biên dịch được" và Slither không bao giờ chạy ở Phase 5.
def _compile_source(
    solc: Path,
    code: str,
    file_name: str,
    source_root: str | Path | None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="e2e-compile-") as temp_dir:
        source_path, dependency_count = _copy_source_tree(Path(temp_dir), code, file_name, source_root)
        result = _run([str(solc), str(source_path)])
    result.pop("stdout", None)
    result["compiler"] = str(solc)
    result["host_architecture"] = platform.machine()
    result["copied_solidity_dependencies"] = dependency_count
    return result


# Từ các finding mục tiêu -> tập nhóm lỗ hổng (reentrancy, arithmetic...) qua SWC_CATEGORY + từ khoá.
# Mục đích: biết finding nào của tool là "cùng loại lỗi đã sửa".
def _target_categories(target_findings: list[dict[str, Any]] | None) -> set[str]:
    categories: set[str] = set()
    for finding in target_findings or []:
        swc = str(finding.get("swc") or finding.get("vuln_id") or "").upper()
        if swc in SWC_CATEGORY:
            categories.add(SWC_CATEGORY[swc])
        categories.add(_classify_category(finding.get("title"), finding.get("description"), swc))
    categories.discard("other")
    return categories


# Lấy tập số dòng từ một finding của tool (line có thể là int, list hoặc chuỗi).
def _finding_lines(finding: dict[str, Any]) -> set[int]:
    value = finding.get("line") if finding.get("line") is not None else finding.get("lines")
    if isinstance(value, int):
        return {value}
    if isinstance(value, (list, tuple, set)):
        return {item for item in value if isinstance(item, int)}
    if isinstance(value, str) and value.strip().isdigit():
        return {int(value.strip())}
    return set()


# Tập các dòng của bản vá thuộc những hàm đã được nhắm sửa.
def _target_line_ranges(fixed_code: str, target_findings: list[dict[str, Any]] | None) -> set[int]:
    """Lines of the patched source that belong to the units the findings targeted."""
    unit_ids = {str(item.get("unit_id")) for item in target_findings or [] if item.get("unit_id")}
    lines: set[int] = set()
    for unit in parse_code_units(fixed_code):
        if unit.unit_id in unit_ids:
            lines.update(range(unit.start_line, unit.end_line + 1))
    return lines


# Finding của tool là "còn sót" (residual) khi cùng nhóm lỗi mục tiêu VÀ (nếu có số dòng) nằm trong hàm đã vá.
def _finding_matches_target(
    finding: dict[str, Any],
    targets: set[str],
    target_lines: set[int] | None = None,
) -> bool:
    """A tool finding is residual when its category is a target category and, if the
    finding carries line information, it lies inside a targeted (patched) unit."""
    if not targets:
        return False
    category = str(finding.get("category") or "").lower()
    if category == "other" or not category:
        category = _classify_category(finding.get("name"), finding.get("check"), finding.get("message"))
    if category not in targets:
        return False
    lines = _finding_lines(finding)
    if target_lines and lines:
        return bool(lines & target_lines)
    return True


# Định danh finding không phụ thuộc vị trí (nhóm, tên check, thông điệp đã bỏ số dòng/đường dẫn).
# Mục đích: bản vá làm lệch số dòng thì finding cũ không bị coi là "mới phát sinh".
def _finding_signature(finding: dict[str, Any]) -> tuple[str, str, str]:
    """Location-independent identity of a tool finding: a patch that shifts line numbers
    must not make every pre-existing finding look newly introduced."""
    message = str(finding.get("message") or finding.get("description") or "")
    message = RE_LOCATION_IN_MESSAGE.sub("", message)
    return (
        str(finding.get("category") or "other").lower(),
        str(finding.get("name") or finding.get("check") or "").strip().lower(),
        RE_WHITESPACE.sub(" ", message).strip().lower()[:160],
    )


# Chấm kết quả Slither: residual (lỗi mục tiêu còn sót) hoặc introduced (lỗi High+ mới) -> FAIL; còn finding khác
# -> WARN; sạch -> PASS; tool lỗi -> ERROR; không chạy -> SKIPPED. Ghi thêm Slither có "phủ" loại lỗi mục tiêu
# không (applicable).
def _evaluate_tool_result(
    tool: str,
    run_status: str,
    fixed_findings: list[dict[str, Any]],
    original_findings: list[dict[str, Any]],
    target_categories: set[str],
    error: str | None = None,
    target_lines: set[int] | None = None,
) -> dict[str, Any]:
    original_signatures = {_finding_signature(item) for item in original_findings}
    residual = [
        item for item in fixed_findings
        if _finding_matches_target(item, target_categories, target_lines)
    ]
    introduced = [
        item
        for item in fixed_findings
        if _finding_signature(item) not in original_signatures
        and _severity_rank(item.get("severity")) >= RISK_ORDER["High"]
    ]
    applicable = not target_categories or bool(SLITHER_COVERAGE & target_categories)
    has_fatal_error = bool(error and any(err in error.lower() for err in ("compilation failed", "exception in constructor", "invalid solc compilation")))
    if run_status in {"unavailable", "skipped"}:
        verdict = "SKIPPED"
    elif run_status != "completed" or has_fatal_error:
        verdict = "ERROR"
    elif residual or introduced:
        verdict = "FAIL"
    elif fixed_findings:
        verdict = "WARN"
    else:
        verdict = "PASS"
    detail = "No findings after fix."
    if run_status in {"unavailable", "skipped"}:
        detail = error or f"Tool {run_status}."
    elif verdict == "ERROR":
        detail = error or "Tool execution or compilation error."
    elif residual:
        detail = f"{len(residual)} residual target finding(s)."
    elif introduced:
        detail = f"{len(introduced)} introduced high-impact finding(s)."
    elif fixed_findings:
        detail = f"{len(fixed_findings)} non-target finding(s) remain."
    elif not applicable and verdict == "PASS":
        detail = "No finding; target category is outside this tool's main coverage."
    return {
        "tool": tool,
        "run_status": run_status,
        "verdict": verdict,
        "applicable": applicable,
        "findings": fixed_findings,
        "original_findings": original_findings,
        "residual_target_findings": residual,
        "introduced_findings": introduced,
        "detail": detail,
        "error": error,
    }


# Kết luận của tool: có FAIL -> failed; tool lỗi hoặc không chạy xong -> inconclusive; tool chạy sạch nhưng
# không phủ loại lỗi mục tiêu -> inconclusive (không phải bằng chứng đã sửa); còn lại -> passed.
def _policy_verdict(
    tool_results: dict[str, dict[str, Any]],
    require_applicable: bool = False,
) -> dict[str, Any]:
    failed = [name for name, result in tool_results.items() if result.get("verdict") == "FAIL"]
    errors = [name for name, result in tool_results.items() if result.get("verdict") == "ERROR"]
    completed = [
        name
        for name, result in tool_results.items()
        if result.get("run_status") == "completed" and result.get("verdict") not in {"ERROR", "SKIPPED"}
    ]
    if failed:
        return {
            "verdict": "failed",
            "reason": f"Security tools still report residual or introduced findings: {', '.join(failed)}",
        }
    if errors:
        return {
            "verdict": "inconclusive",
            "reason": f"Tool(s) did not complete: {', '.join(errors)}",
        }
    if not completed:
        return {
            "verdict": "inconclusive",
            "reason": "No security tools completed analysis.",
        }
    if require_applicable and not any(tool_results[name].get("applicable") for name in completed):
        return {
            "verdict": "inconclusive",
            "reason": "No completed tool covers the targeted vulnerability category; a clean run is not evidence of a fix.",
        }
    return {
        "verdict": "passed",
        "reason": f"{len(completed)}/{len(tool_results)} tools completed without residual target findings.",
    }


# Trạng thái ngắn của Slither cho báo cáo (passed / failed / warn / error / skipped / not run).
def _slither_status(result: dict[str, Any]) -> str:
    verdict = str(result.get("verdict") or "").upper()
    status = {"PASS": "passed", "FAIL": "failed", "WARN": "warn", "ERROR": "error", "SKIPPED": "skipped"}
    return status.get(verdict) or result.get("run_status", "not run")


# [HÀM CHÍNH] Compile bản vá (so baseline với bản gốc) -> chạy Slither native trên bản gốc và bản vá
# -> chấm (_evaluate_tool_result) -> _policy_verdict -> policy_verdict + tóm tắt riêng cho Slither.
# Gọi bởi: final_verification.verify_final_patch.
def verify_security_tool_matrix(
    original_code: str,
    fixed_code: str,
    *,
    file_name: str = "Contract.sol",
    source_root: str | Path | None = None,
    target_findings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    # Same compiler choice as the repair loop and the AST check (common/solc.py): the lowest
    # installed solc that satisfies the ORIGINAL pragma, so original and patch are compared fairly.
    solc, selection_reason, _ = select_best_solc(original_code)
    if solc is None:
        compile_result = {"status": "unavailable", "reason": selection_reason}
    else:
        compile_result = _compile_source(solc, fixed_code, file_name, source_root)
        compile_result["selection_reason"] = selection_reason
        if compile_result["status"] != "passed":
            baseline = _compile_source(solc, original_code, file_name, source_root)
            compile_result["baseline_status"] = baseline["status"]
            if baseline["status"] != "passed":
                new_errors = compiler_error_signatures(compile_result.get("stderr", "")) - compiler_error_signatures(
                    baseline.get("stderr", "")
                )
                if not new_errors:
                    # The original itself does not compile here (e.g. unresolved imports):
                    # the patch cannot be blamed, but nothing can be verified either.
                    compile_result["status"] = "baseline_failed"

    target_categories = _target_categories(target_findings)
    target_lines = _target_line_ranges(fixed_code, target_findings)
    compile_status = compile_result.get("status")

    # Slither is only run on a patch that compiles.
    if compile_status != "passed":
        original_run = fixed_run = {
            "run_status": "skipped",
            "findings": [],
            "error": "fixed source did not compile",
            "elapsed_ms": 0.0,
        }
    else:
        original_run = _run_native_slither(original_code, file_name, source_root, solc, selection_reason)
        fixed_run = _run_native_slither(fixed_code, file_name, source_root, solc, selection_reason)
    slither = _evaluate_tool_result(
        "slither",
        str(fixed_run.get("run_status") or "error"),
        list(fixed_run.get("findings", [])),
        list(original_run.get("findings", [])),
        target_categories,
        str(fixed_run.get("error") or original_run.get("error") or "") or None,
        target_lines,
    )
    slither["elapsed_ms"] = round(
        float(original_run.get("elapsed_ms") or 0.0) + float(fixed_run.get("elapsed_ms") or 0.0),
        3,
    )
    slither["command"] = fixed_run.get("command")
    tool_results = {"slither": slither}

    if compile_status == "passed":
        policy_result = _policy_verdict(tool_results, require_applicable=bool(target_categories))
    elif compile_status == "failed":
        policy_result = {"verdict": "failed", "reason": "Fixed source does not compile."}
    elif compile_status == "baseline_failed":
        policy_result = {
            "verdict": "inconclusive",
            "reason": "The original source does not compile standalone either; security tools could not run.",
        }
    else:
        policy_result = {
            "verdict": "inconclusive",
            "reason": f"Compiler unavailable ({compile_result.get('reason') or compile_status}).",
        }

    return {
        "component": "tool_verification",
        "status": "ok" if policy_result["verdict"] == "passed" else "partial",
        "engine": "slither",
        "policy_verdict": policy_result["verdict"],
        "policy_reason": policy_result["reason"],
        "compile": compile_result,
        "tools": tool_results,
        "slither": {
            "status": _slither_status(slither),
            "findings": slither.get("findings", []),
            "verdict": slither.get("verdict", "Not run"),
        },
    }
