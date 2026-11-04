"""
Phase 5 - security tool matrix on the original and the patched contract.

Slither (+ Mythril, SmartCheck through SmartBugs/Docker when enabled) runs on both
versions. Per tool: a finding of the targeted vulnerability class still present on the
patched lines -> FAIL; a new finding the original did not have -> FAIL; a clean run of a
tool that does not cover the class -> inconclusive, not PASS. `_policy_verdict` combines
the tools into passed / failed / inconclusive. The patch must also compile (baseline
compared with the original, so a contract that never compiled does not blame the patch).
"""

from __future__ import annotations

import concurrent.futures
import os
import platform
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from E2E.common.security_tools import (
    _classify_category,
    _copy_source_tree,
    _run,
    _run_native_slither,
    _run_smartbugs_tool,
    _smartbugs_executable,
)


SOLC_SELECT_ARTIFACTS = Path.home() / ".solc-select" / "artifacts"


SECURITY_TOOL_NAMES = [
    "slither",
    "mythril",
    "smartcheck",
]


CRITICAL_SECURITY_TOOLS = {"slither", "mythril"}


DEFAULT_TOOL_TIMEOUTS = {
    "honeybadger": 180,
    "maian": 420,
    "manticore": 900,
    "mythril": 240,
    "osiris": 180,
    "oyente": 180,
    "securify": 600,
    "slither": 180,
    "smartcheck": 180,
}


SEVERITY_ORDER = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
    "informational": 0,
    "info": 0,
    "optimization": 0,
    "unknown": 0,
}


def _severity_rank(value: Any) -> int:
    return SEVERITY_ORDER.get(str(value or "unknown").strip().lower(), 0)


SWC_CATEGORY_ALIASES = {
    "SWC-100": {"visibility", "access_control"},
    "SWC-101": {"arithmetic", "overflow", "underflow"},
    "SWC-104": {"unchecked_low_level_calls", "unchecked_call", "low_level_call"},
    "SWC-105": {"prodigal", "unprotected_withdrawal", "ether_withdrawal"},
    "SWC-106": {"suicidal", "selfdestruct"},
    "SWC-107": {"reentrancy"},
    "SWC-113": {"denial_of_service", "dos"},
    "SWC-115": {"tx_origin"},
    "SWC-116": {"timestamp", "time_manipulation"},
    "SWC-128": {"denial_of_service", "dos"},
}


TOOL_COVERAGE = {
    "honeybadger": {"honeypot", "prodigal", "reentrancy"},
    "maian": {"prodigal", "suicidal", "greedy", "selfdestruct", "ether_withdrawal"},
    "manticore": {
        "access_control",
        "arithmetic",
        "overflow",
        "reentrancy",
        "timestamp",
        "unchecked_low_level_calls",
    },
    "mythril": {
        "access_control",
        "arithmetic",
        "overflow",
        "underflow",
        "reentrancy",
        "selfdestruct",
        "unchecked_low_level_calls",
        "tx_origin",
    },
    "osiris": {"arithmetic", "overflow", "underflow"},
    "oyente": {"arithmetic", "overflow", "underflow", "reentrancy", "timestamp"},
    "securify": {
        "access_control",
        "front_running",
        "reentrancy",
        "unchecked_low_level_calls",
    },
    "slither": {
        "access_control",
        "arithmetic",
        "denial_of_service",
        "reentrancy",
        "selfdestruct",
        "timestamp",
        "tx_origin",
        "unchecked_low_level_calls",
    },
    "smartcheck": {
        "access_control",
        "arithmetic",
        "denial_of_service",
        "reentrancy",
        "timestamp",
        "unchecked_low_level_calls",
    },
}


def _pragma_version(code: str) -> tuple[int, int, int] | None:
    match = re.search(r"pragma\s+solidity\s+([^;]+);", code)
    if not match:
        return None
    version = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", match.group(1))
    if not version:
        return None
    return int(version.group(1)), int(version.group(2)), int(version.group(3) or 0)


def _installed_solc() -> list[tuple[tuple[int, int, int], Path]]:
    versions = []
    for path in SOLC_SELECT_ARTIFACTS.glob("solc-*/solc-*"):
        match = re.search(r"solc-(\d+)\.(\d+)\.(\d+)$", path.name)
        if match and os.access(path, os.X_OK):
            versions.append((tuple(map(int, match.groups())), path))
    return sorted(versions)


def choose_solc(code: str) -> tuple[Path | None, str]:
    installed = _installed_solc()
    if not installed:
        executable = shutil.which("solc")
        return (Path(executable), "system solc") if executable else (None, "solc not installed")
    requested = _pragma_version(code)
    if requested:
        same_minor = [item for item in installed if item[0][:2] == requested[:2]]
        if same_minor:
            return same_minor[-1][1], f"matched pragma {requested[0]}.{requested[1]}"
        compatible = [item for item in installed if item[0][0] == requested[0] and item[0] >= requested]
        if compatible:
            return compatible[0][1], "nearest installed compiler"
    return installed[-1][1], "latest installed compiler"


_ERROR_LOCATION_RE = re.compile(r"(-->\s*\S+:\d+:\d+:?|\S+\.sol:\d+:\d+:?)")


def _error_lines(text: str) -> set[str]:
    return {
        _ERROR_LOCATION_RE.sub("", line).strip()
        for line in (text or "").splitlines()
        if "Error" in line
    }


def _compile_source(
    solc: Path,
    code: str,
    file_name: str,
    source_root: str | Path | None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="e2e-compile-") as temp_dir:
        source_path, dependency_count = _copy_source_tree(Path(temp_dir), code, file_name, source_root)
        result = _run([str(solc), "--ast-compact-json", str(source_path)])
    result.pop("stdout", None)  # the AST is large and not needed here
    result["compiler"] = str(solc)
    result["host_architecture"] = platform.machine()
    result["copied_solidity_dependencies"] = dependency_count
    return result


def _target_categories(target_findings: list[dict[str, Any]] | None) -> set[str]:
    categories: set[str] = set()
    for finding in target_findings or []:
        swc = str(finding.get("swc") or finding.get("vuln_id") or "").upper()
        categories.update(SWC_CATEGORY_ALIASES.get(swc, set()))
        categories.add(_classify_category(finding.get("title"), finding.get("description"), swc))
    categories.discard("other")
    return categories


def _finding_lines(finding: dict[str, Any]) -> set[int]:
    value = finding.get("line") if finding.get("line") is not None else finding.get("lines")
    if isinstance(value, int):
        return {value}
    if isinstance(value, (list, tuple, set)):
        return {item for item in value if isinstance(item, int)}
    if isinstance(value, str) and value.strip().isdigit():
        return {int(value.strip())}
    return set()


def _target_line_ranges(fixed_code: str, target_findings: list[dict[str, Any]] | None) -> set[int]:
    """Lines of the patched source that belong to the units the findings targeted."""
    from E2E.phase0_parsing.code_units import parse_code_units

    unit_ids = {str(item.get("unit_id")) for item in target_findings or [] if item.get("unit_id")}
    lines: set[int] = set()
    for unit in parse_code_units(fixed_code):
        if unit.unit_id in unit_ids:
            lines.update(range(unit.start_line, unit.end_line + 1))
    return lines


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


# Source paths (temporary directories differ per run) and line references.
_LOCATION_IN_MESSAGE_RE = re.compile(r"(\S*\.sol(?:#\d+(?:-\d+)?|:\d+(?::\d+)?)?|#\d+(?:-\d+)?|\blines?\s+\d+(?:-\d+)?)")


def _finding_signature(finding: dict[str, Any]) -> tuple[str, str, str]:
    """Location-independent identity of a tool finding: a patch that shifts line numbers
    must not make every pre-existing finding look newly introduced."""
    message = str(finding.get("message") or finding.get("description") or "")
    message = _LOCATION_IN_MESSAGE_RE.sub("", message)
    return (
        str(finding.get("category") or "other").lower(),
        str(finding.get("name") or finding.get("check") or "").strip().lower(),
        re.sub(r"\s+", " ", message).strip().lower()[:160],
    )


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
        and _severity_rank(item.get("severity")) >= SEVERITY_ORDER["high"]
    ]
    applicable = not target_categories or bool(TOOL_COVERAGE.get(tool, set()) & target_categories)
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


def _policy_verdict(
    tool_results: dict[str, dict[str, Any]],
    policy: str = "balanced",
    require_applicable: bool = False,
) -> dict[str, Any]:
    failed = [name for name, result in tool_results.items() if result.get("verdict") == "FAIL"]
    errors = [name for name, result in tool_results.items() if result.get("verdict") == "ERROR"]
    completed = [
        name
        for name, result in tool_results.items()
        if result.get("run_status") == "completed" and result.get("verdict") not in {"ERROR", "SKIPPED"}
    ]
    critical_errors = sorted(CRITICAL_SECURITY_TOOLS & set(errors))
    if failed:
        return {
            "verdict": "failed",
            "reason": f"Security tools still report residual or introduced findings: {', '.join(failed)}",
        }
    if policy == "strict" and errors:
        return {
            "verdict": "inconclusive",
            "reason": f"Strict policy requires every tool to complete; errors: {', '.join(errors)}",
        }
    if critical_errors:
        return {
            "verdict": "inconclusive",
            "reason": f"Critical tool(s) did not complete: {', '.join(critical_errors)}",
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


def _run_security_tool_pair(
    tool: str,
    original_code: str,
    fixed_code: str,
    file_name: str,
    source_root: str | Path | None,
    smartbugs_executable: Path | None,
    results_root: Path,
    solc: Path | None,
    solc_reason: str,
    timeout: int,
    prefer_docker_slither: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    # When Docker / SmartBugs is preferred and available, run Slither in Docker
    use_docker_slither = prefer_docker_slither and smartbugs_executable is not None
    if tool == "slither" and not use_docker_slither and shutil.which("slither") and solc is not None:
        original = _run_native_slither(
            original_code,
            file_name,
            source_root,
            solc,
            solc_reason,
        )
        fixed = _run_native_slither(
            fixed_code,
            file_name,
            source_root,
            solc,
            solc_reason,
        )
        return original, fixed
    if smartbugs_executable is not None:
        original = _run_smartbugs_tool(
            tool,
            original_code,
            file_name,
            source_root,
            smartbugs_executable,
            results_root,
            "original",
            timeout,
        )
        fixed = _run_smartbugs_tool(
            tool,
            fixed_code,
            file_name,
            source_root,
            smartbugs_executable,
            results_root,
            "fixed",
            timeout,
        )
        return original, fixed
    if tool == "slither" and shutil.which("slither") and solc is not None:
        original = _run_native_slither(
            original_code,
            file_name,
            source_root,
            solc,
            solc_reason,
        )
        fixed = _run_native_slither(
            fixed_code,
            file_name,
            source_root,
            solc,
            solc_reason,
        )
        return original, fixed
    unavailable = {
        "run_status": "unavailable",
        "findings": [],
        "error": "SmartBugs (Docker) is unavailable and no native executable exists for this tool",
        "elapsed_ms": 0.0,
    }
    return unavailable, unavailable


def verify_security_tool_matrix(
    original_code: str,
    fixed_code: str,
    *,
    file_name: str = "Contract.sol",
    source_root: str | Path | None = None,
    target_findings: list[dict[str, Any]] | None = None,
    smartbugs_home: str | Path | None = None,
    tools: list[str] | None = None,
    policy: str = "balanced",
    results_dir: str | Path | None = None,
    use_smartbugs: bool = True,
    prefer_docker_slither: bool = True,
) -> dict[str, Any]:
    selected_tools = tools or list(SECURITY_TOOL_NAMES)
    solc, selection_reason = choose_solc(fixed_code)
    if solc is None:
        compile_result = {"status": "unavailable", "reason": selection_reason}
    else:
        compile_result = _compile_source(solc, fixed_code, file_name, source_root)
        compile_result["selection_reason"] = selection_reason
        if compile_result["status"] != "passed":
            baseline = _compile_source(solc, original_code, file_name, source_root)
            compile_result["baseline_status"] = baseline["status"]
            if baseline["status"] != "passed":
                new_errors = _error_lines(compile_result.get("stderr", "")) - _error_lines(baseline.get("stderr", ""))
                if not new_errors:
                    # The original itself does not compile here (e.g. unresolved imports):
                    # the patch cannot be blamed, but nothing can be verified either.
                    compile_result["status"] = "baseline_failed"

    target_categories = _target_categories(target_findings)
    target_lines = _target_line_ranges(fixed_code, target_findings)
    if use_smartbugs:
        smartbugs_executable, smartbugs_reason = _smartbugs_executable(smartbugs_home)
    else:
        smartbugs_executable, smartbugs_reason = None, "SmartBugs disabled for this run"
    result_root = Path(results_dir) if results_dir else Path(tempfile.mkdtemp(prefix="e2e-smartbugs-results-"))
    tool_results: dict[str, dict[str, Any]] = {}

    def _execute_single_tool(tool: str) -> tuple[str, dict[str, Any]]:
        timeout = DEFAULT_TOOL_TIMEOUTS.get(tool, 240)
        if compile_result.get("status") != "passed":
            original_run = fixed_run = {
                "run_status": "skipped",
                "findings": [],
                "error": "fixed source did not compile",
                "elapsed_ms": 0.0,
            }
        else:
            original_run, fixed_run = _run_security_tool_pair(
                tool,
                original_code,
                fixed_code,
                file_name,
                source_root,
                smartbugs_executable,
                result_root,
                solc,
                selection_reason,
                timeout,
                prefer_docker_slither=prefer_docker_slither,
            )
        evaluated = _evaluate_tool_result(
            tool,
            str(fixed_run.get("run_status") or "error"),
            list(fixed_run.get("findings", [])),
            list(original_run.get("findings", [])),
            target_categories,
            str(fixed_run.get("error") or original_run.get("error") or "") or None,
            target_lines,
        )
        evaluated["elapsed_ms"] = round(
            float(original_run.get("elapsed_ms") or 0.0)
            + float(fixed_run.get("elapsed_ms") or 0.0),
            3,
        )
        evaluated["original_run_status"] = original_run.get("run_status")
        evaluated["fixed_run_status"] = fixed_run.get("run_status")
        evaluated["command"] = fixed_run.get("command")
        return tool, evaluated

    max_workers = min(4, len(selected_tools)) if len(selected_tools) > 1 else 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_execute_single_tool, tool) for tool in selected_tools]
        for future in concurrent.futures.as_completed(futures):
            tool_name, evaluated = future.result()
            tool_results[tool_name] = evaluated

    compile_status = compile_result.get("status")
    if compile_status == "passed":
        policy_result = _policy_verdict(tool_results, policy, require_applicable=bool(target_categories))
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
    summary = {
        "pass": sum(1 for item in tool_results.values() if item.get("verdict") == "PASS"),
        "fail": sum(1 for item in tool_results.values() if item.get("verdict") == "FAIL"),
        "warn": sum(1 for item in tool_results.values() if item.get("verdict") == "WARN"),
        "error": sum(1 for item in tool_results.values() if item.get("verdict") == "ERROR"),
    }
    slither_result = tool_results.get("slither", {})
    slither_verdict = str(slither_result.get("verdict") or "").upper()
    if slither_verdict == "PASS":
        slither_status = "passed"
    elif slither_verdict == "FAIL":
        slither_status = "failed"
    elif slither_verdict == "WARN":
        slither_status = "warn"
    elif slither_verdict == "ERROR":
        slither_status = "error"
    elif slither_verdict == "SKIPPED":
        slither_status = "skipped"
    elif slither_result.get("run_status") == "completed":
        slither_status = (
            "failed"
            if slither_result.get("residual_target_findings")
            or slither_result.get("introduced_findings")
            else "passed"
        )
    else:
        slither_status = slither_result.get("run_status", "not run")

    return {
        "component": "tool_verification",
        "status": "ok" if policy_result["verdict"] == "passed" else "partial",
        "engine": "security_3_tool_matrix",
        "policy": policy,
        "policy_verdict": policy_result["verdict"],
        "policy_reason": policy_result["reason"],
        "compile": compile_result,
        "smartbugs": {
            "status": "available" if smartbugs_executable else "unavailable",
            "reason": smartbugs_reason,
            "results_dir": str(result_root),
        },
        "tools": tool_results,
        "summary": summary,
        "slither": {
            "status": slither_status,
            "findings": slither_result.get("findings", []),
            "verdict": slither_result.get("verdict", "Not run"),
        },
    }
