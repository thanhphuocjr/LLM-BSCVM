"""Phase 5 - compile bản vá (so với bản gốc), Slither trên bản gốc và bản vá -> passed / failed / inconclusive."""

import platform
import tempfile
from functools import lru_cache
from pathlib import Path

from E2E.common.regex_patterns import RE_LOCATION_IN_MESSAGE, RE_WHITESPACE
from E2E.common.security_tools import _classify_category, _copy_source_tree, _run, _run_native_slither
from E2E.common.solc import compiler_error_signatures, select_best_solc
from E2E.config import RISK_ORDER
from E2E.phase0_parsing.code_units import parse_code_units

# Nhóm lỗ hổng (theo _classify_category) ứng với từng SWC mục tiêu.
SWC_CATEGORY = {"SWC-100": "access_control", "SWC-101": "arithmetic", "SWC-104": "unchecked_low_level_calls",
                "SWC-105": "prodigal", "SWC-106": "selfdestruct", "SWC-107": "reentrancy",
                "SWC-113": "denial_of_service", "SWC-115": "tx_origin", "SWC-116": "timestamp",
                "SWC-128": "denial_of_service"}
# Nhóm Slither phát hiện được; lỗi mục tiêu ngoài nhóm này thì Slither chạy sạch cũng không chứng minh đã sửa.
SLITHER_COVERAGE = {"access_control", "arithmetic", "denial_of_service", "reentrancy", "selfdestruct", "timestamp",
                    "tx_origin", "unchecked_low_level_calls"}


# Slither trên bản GỐC (có nhớ): vòng sửa kiểm chứng tối đa 2 bản vá của cùng một contract.
_original_slither = lru_cache(maxsize=8)(_run_native_slither)


# Đổi mức độ của tool thành số để so sánh (thang RISK_ORDER).
def _severity_rank(value) -> int:
    return RISK_ORDER.get(str(value or "").strip().capitalize(), 0)


# Biên dịch mã (kèm file phụ thuộc) bằng solc trong thư mục tạm; trạng thái theo mã thoát (không đòi AST).
def _compile_source(solc: Path, code: str, file_name: str, source_root: str | Path | None) -> dict:
    with tempfile.TemporaryDirectory(prefix="e2e-compile-") as temp_dir:
        source_path, dependency_count = _copy_source_tree(Path(temp_dir), code, file_name, source_root)
        result = _run([str(solc), str(source_path)])
    result.pop("stdout", None)
    return {**result, "compiler": str(solc), "host_architecture": platform.machine(),
            "copied_solidity_dependencies": dependency_count}


# Finding mục tiêu -> tập nhóm lỗ hổng (reentrancy, arithmetic...) qua SWC_CATEGORY + từ khoá.
def _target_categories(target_findings: list[dict] | None) -> set[str]:
    categories = set()
    for finding in target_findings or []:
        swc = str(finding.get("swc") or finding.get("vuln_id") or "").upper()
        if swc in SWC_CATEGORY:
            categories.add(SWC_CATEGORY[swc])
        categories.add(_classify_category(finding.get("title"), finding.get("description"), swc))
    return categories - {"other"}


# Tập số dòng của một finding của tool (line có thể là int, list hoặc chuỗi số).
def _finding_lines(finding: dict) -> set[int]:
    value = finding.get("line") if finding.get("line") is not None else finding.get("lines")
    if isinstance(value, int):
        return {value}
    if isinstance(value, (list, tuple, set)):
        return {item for item in value if isinstance(item, int)}
    return {int(value.strip())} if isinstance(value, str) and value.strip().isdigit() else set()


# Các dòng của bản vá thuộc những hàm đã được nhắm sửa.
def _target_line_ranges(fixed_code: str, target_findings: list[dict] | None) -> set[int]:
    unit_ids = {str(item.get("unit_id")) for item in target_findings or [] if item.get("unit_id")}
    return {line for unit in parse_code_units(fixed_code) if unit.unit_id in unit_ids
            for line in range(unit.start_line, unit.end_line + 1)}


# Finding của tool là "còn sót" khi cùng nhóm lỗi mục tiêu VÀ (nếu có số dòng) nằm trong hàm đã vá.
def _finding_matches_target(finding: dict, targets: set[str], target_lines: set[int] | None = None) -> bool:
    category = str(finding.get("category") or "").lower()
    if category in {"other", ""}:
        category = _classify_category(finding.get("name"), finding.get("check"), finding.get("message"))
    lines = _finding_lines(finding)
    return bool(targets) and category in targets and (not (target_lines and lines) or bool(lines & target_lines))


# Định danh finding không phụ thuộc vị trí (nhóm, tên check, thông điệp đã bỏ số dòng/đường dẫn).
def _finding_signature(finding: dict) -> tuple[str, str, str]:
    message = RE_LOCATION_IN_MESSAGE.sub("", str(finding.get("message") or finding.get("description") or ""))
    return (str(finding.get("category") or "other").lower(),
            str(finding.get("name") or finding.get("check") or "").strip().lower(),
            RE_WHITESPACE.sub(" ", message).strip().lower()[:160])


# Chấm kết quả Slither: lỗi mục tiêu còn / lỗi High+ mới -> FAIL; finding khác -> WARN; sạch -> PASS; lỗi -> ERROR.
def _evaluate_tool_result(tool: str, run_status: str, fixed_findings: list[dict], original_findings: list[dict],
                          target_categories: set[str], error: str | None = None,
                          target_lines: set[int] | None = None) -> dict:
    original_signatures = {_finding_signature(item) for item in original_findings}
    residual = [item for item in fixed_findings if _finding_matches_target(item, target_categories, target_lines)]
    introduced = [item for item in fixed_findings if _finding_signature(item) not in original_signatures
                  and _severity_rank(item.get("severity")) >= RISK_ORDER["High"]]
    applicable = not target_categories or bool(SLITHER_COVERAGE & target_categories)
    fatal = bool(error and any(text in error.lower() for text in
                               ("compilation failed", "exception in constructor", "invalid solc compilation")))
    if run_status in {"unavailable", "skipped"}:
        verdict, detail = "SKIPPED", error or f"Tool {run_status}."
    elif run_status != "completed" or fatal:
        verdict, detail = "ERROR", error or "Tool execution or compilation error."
    elif residual:
        verdict, detail = "FAIL", f"{len(residual)} residual target finding(s)."
    elif introduced:
        verdict, detail = "FAIL", f"{len(introduced)} introduced high-impact finding(s)."
    elif fixed_findings:
        verdict, detail = "WARN", f"{len(fixed_findings)} non-target finding(s) remain."
    else:
        verdict = "PASS"
        detail = "No findings after fix." if applicable \
            else "No finding; target category is outside this tool's main coverage."
    return {"tool": tool, "run_status": run_status, "verdict": verdict, "applicable": applicable,
            "findings": fixed_findings, "original_findings": original_findings, "residual_target_findings": residual,
            "introduced_findings": introduced, "detail": detail, "error": error}


# Kết luận các tool: FAIL -> failed; lỗi / chưa chạy xong / không phủ loại lỗi mục tiêu -> inconclusive; còn lại passed.
def _policy_verdict(tool_results: dict[str, dict], require_applicable: bool = False) -> dict:
    failed = [name for name, result in tool_results.items() if result.get("verdict") == "FAIL"]
    errors = [name for name, result in tool_results.items() if result.get("verdict") == "ERROR"]
    completed = [name for name, result in tool_results.items()
                 if result.get("run_status") == "completed" and result.get("verdict") not in {"ERROR", "SKIPPED"}]
    if failed:
        return {"verdict": "failed",
                "reason": f"Security tools still report residual or introduced findings: {', '.join(failed)}"}
    if errors:
        return {"verdict": "inconclusive", "reason": f"Tool(s) did not complete: {', '.join(errors)}"}
    if not completed:
        return {"verdict": "inconclusive", "reason": "No security tools completed analysis."}
    if require_applicable and not any(tool_results[name].get("applicable") for name in completed):
        return {"verdict": "inconclusive",
                "reason": "No completed tool covers the targeted vulnerability category; "
                          "a clean run is not evidence of a fix."}
    return {"verdict": "passed",
            "reason": f"{len(completed)}/{len(tool_results)} tools completed without residual target findings."}


# Compile bản vá; nếu lỗi thì compile bản gốc: bản gốc cũng lỗi và bản vá không thêm lỗi mới -> baseline_failed.
def _compile_with_baseline(solc: Path | None, reason: str, original_code: str, fixed_code: str, file_name: str,
                           source_root: str | Path | None) -> dict:
    if solc is None:
        return {"status": "unavailable", "reason": reason}
    result = {**_compile_source(solc, fixed_code, file_name, source_root), "selection_reason": reason}
    if result["status"] != "passed":
        baseline = _compile_source(solc, original_code, file_name, source_root)
        result["baseline_status"] = baseline["status"]
        new_errors = compiler_error_signatures(result.get("stderr", "")) \
            - compiler_error_signatures(baseline.get("stderr", ""))
        if baseline["status"] != "passed" and not new_errors:
            result["status"] = "baseline_failed"
    return result


# [HÀM CHÍNH] Compile (so với bản gốc) -> Slither trên bản gốc + bản vá (khi compile được) -> chấm -> policy_verdict.
def verify_security_tool_matrix(original_code: str, fixed_code: str, *, file_name: str = "Contract.sol",
                                source_root: str | Path | None = None,
                                target_findings: list[dict] | None = None) -> dict:
    solc, reason, _ = select_best_solc(original_code)
    compile_result = _compile_with_baseline(solc, reason, original_code, fixed_code, file_name, source_root)
    compile_status = compile_result.get("status")
    target_categories = _target_categories(target_findings)
    if compile_status == "passed":
        original_run = _original_slither(original_code, file_name, source_root, solc, reason)
        fixed_run = _run_native_slither(fixed_code, file_name, source_root, solc, reason)
    else:
        original_run = fixed_run = {"run_status": "skipped", "findings": [], "error": "fixed source did not compile",
                                    "elapsed_ms": 0.0}
    slither = _evaluate_tool_result("slither", str(fixed_run.get("run_status") or "error"),
                                    list(fixed_run.get("findings", [])), list(original_run.get("findings", [])),
                                    target_categories, str(fixed_run.get("error") or original_run.get("error") or "") or None,
                                    _target_line_ranges(fixed_code, target_findings))
    slither["elapsed_ms"] = round(float(original_run.get("elapsed_ms") or 0.0) + float(fixed_run.get("elapsed_ms") or 0.0),
                                  3)
    slither["command"] = fixed_run.get("command")
    if compile_status == "passed":
        policy = _policy_verdict({"slither": slither}, require_applicable=bool(target_categories))
    elif compile_status == "failed":
        policy = {"verdict": "failed", "reason": "Fixed source does not compile."}
    elif compile_status == "baseline_failed":
        policy = {"verdict": "inconclusive",
                  "reason": "The original source does not compile standalone either; security tools could not run."}
    else:
        policy = {"verdict": "inconclusive",
                  "reason": f"Compiler unavailable ({compile_result.get('reason') or compile_status})."}
    short_status = {"PASS": "passed", "FAIL": "failed", "WARN": "warn", "ERROR": "error", "SKIPPED": "skipped"}
    return {
        "component": "tool_verification", "status": "ok" if policy["verdict"] == "passed" else "partial",
        "engine": "slither",
        "policy_verdict": policy["verdict"], "policy_reason": policy["reason"], "compile": compile_result,
        "tools": {"slither": slither},
        "slither": {"status": short_status.get(str(slither.get("verdict") or "").upper())
                    or slither.get("run_status", "not run"),
                    "findings": slither.get("findings", []), "verdict": slither.get("verdict", "Not run")},
    }
