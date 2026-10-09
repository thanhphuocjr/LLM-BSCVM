from __future__ import annotations

import concurrent.futures
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


SOLC_SELECT_ARTIFACTS = Path.home() / ".solc-select" / "artifacts"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SMARTBUGS_HOME = PROJECT_ROOT / ".tools" / "smartbugs"

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

SWC_CATEGORY_ALIASES = {
    "SWC-100": {"visibility", "access_control"},
    "SWC-101": {"arithmetic", "overflow", "underflow", "access_control"},
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


def _run(command: list[str], timeout: int = 120, cwd: Path | None = None) -> dict[str, Any]:
    try:
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
            cwd=str(cwd) if cwd else None,
        )
        return {
            "status": "passed" if completed.returncode == 0 else "failed",
            "return_code": completed.returncode,
            "stdout": completed.stdout[-12000:],
            "stderr": completed.stderr[-12000:],
            "command": command,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
    except subprocess.TimeoutExpired as error:
        return {
            "status": "error",
            "error": f"timeout after {timeout}s",
            "stdout": str(error.stdout or "")[-4000:],
            "stderr": str(error.stderr or "")[-4000:],
            "command": command,
            "elapsed_ms": round(timeout * 1000, 3),
        }
    except OSError as error:
        return {"status": "error", "error": str(error), "command": command, "elapsed_ms": 0.0}


def _slither_findings(payload: dict[str, Any]) -> list[dict[str, Any]]:
    detectors = payload.get("results", {}).get("detectors", [])
    findings = []
    for detector in detectors:
        lines = []
        for element in detector.get("elements", []):
            lines.extend(element.get("source_mapping", {}).get("lines") or [])
        findings.append(
            {
                "check": detector.get("check"),
                "impact": detector.get("impact"),
                "confidence": detector.get("confidence"),
                "description": str(detector.get("description") or "").strip(),
                "lines": sorted(set(lines)),
            }
        )
    return findings


def _copy_source_tree(
    temp_root: Path,
    code: str,
    file_name: str,
    source_root: str | Path | None,
) -> tuple[Path, int]:
    dependency_count = 0
    if source_root:
        root = Path(source_root).resolve()
        if root.is_dir():
            for dependency in root.rglob("*.sol"):
                relative = dependency.relative_to(root)
                destination = temp_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dependency, destination)
                dependency_count += 1
    source_path = temp_root / Path(file_name).name
    source_path.write_text(code, encoding="utf-8")
    return source_path, dependency_count


def _normalise_severity(value: Any) -> str:
    text = str(value or "Unknown").strip()
    if not text:
        return "Unknown"
    lowered = text.lower()
    if lowered in {"optimization", "informational", "info"}:
        return "Informational"
    return lowered.capitalize()


def _severity_rank(value: Any) -> int:
    return SEVERITY_ORDER.get(str(value or "unknown").strip().lower(), 0)


def _tokenise(value: Any) -> set[str]:
    text = str(value or "").lower()
    return {
        token
        for token in re.split(r"[^a-z0-9_]+", text)
        if token and len(token) > 2
    }


def _classify_category(*parts: Any) -> str:
    text = " ".join(str(part or "") for part in parts).lower()
    checks = (
        ("reentrancy", ("reentr",)),
        ("unchecked_low_level_calls", ("unchecked", "low level", "low_level", ".call")),
        ("access_control", ("access control", "authorization", "permission", "owner")),
        ("selfdestruct", ("selfdestruct", "suicid")),
        ("prodigal", ("prodigal", "leak ether", "ether leak", "unprotected withdrawal")),
        ("greedy", ("greedy", "locked ether", "lock ether")),
        ("arithmetic", ("overflow", "underflow", "arithmetic", "integer")),
        ("timestamp", ("timestamp", "time manipulation", "block.timestamp", "now")),
        ("tx_origin", ("tx.origin", "tx origin")),
        ("front_running", ("front running", "transaction order", "tod")),
        ("denial_of_service", ("denial of service", "dos", "gas limit")),
        ("honeypot", ("honeypot",)),
    )
    for category, needles in checks:
        if any(needle in text for needle in needles):
            return category
    return "other"


def _target_categories(target_findings: list[dict[str, Any]] | None) -> set[str]:
    categories: set[str] = set()
    for finding in target_findings or []:
        swc = str(finding.get("swc") or finding.get("vuln_id") or "").upper()
        categories.update(SWC_CATEGORY_ALIASES.get(swc, set()))
        categories.add(_classify_category(finding.get("title"), finding.get("description"), swc))
        categories.update(_tokenise(finding.get("title")))
    categories.discard("other")
    return categories


def _finding_matches_target(finding: dict[str, Any], targets: set[str]) -> bool:
    if not targets:
        return False
    category = str(finding.get("category") or "").lower()
    if category in targets:
        return True
    text_tokens = set()
    for field in ("name", "check", "message", "description", "swc"):
        text_tokens.update(_tokenise(finding.get(field)))
    return bool(text_tokens & targets)


def _finding_signature(finding: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(finding.get("category") or "other").lower(),
        str(finding.get("name") or finding.get("check") or "").strip().lower(),
        str(finding.get("message") or finding.get("description") or "").strip().lower()[:160],
    )


def _is_docker_available() -> bool:
    docker_bin = shutil.which("docker")
    if not docker_bin:
        return False
    try:
        completed = subprocess.run(
            [docker_bin, "info"],
            capture_output=True,
            timeout=2,
            check=False,
        )
        return completed.returncode == 0
    except Exception:
        return False


def _smartbugs_executable(smartbugs_home: str | Path | None = None) -> tuple[Path | None, str]:
    if not _is_docker_available():
        return None, "SmartBugs unavailable: Docker daemon is not running"
    candidates: list[Path] = []
    if smartbugs_home:
        candidates.append(Path(smartbugs_home) / "smartbugs")
    if os.environ.get("SMARTBUGS_HOME"):
        candidates.append(Path(os.environ["SMARTBUGS_HOME"]) / "smartbugs")
    candidates.append(DEFAULT_SMARTBUGS_HOME / "smartbugs")
    discovered = shutil.which("smartbugs")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            venv_activate = candidate.parent / ".venv" / "bin" / "activate"
            if candidate.parent == DEFAULT_SMARTBUGS_HOME and not venv_activate.exists():
                return None, f"SmartBugs found at {candidate.parent}, but its virtualenv is not installed"
            return candidate.resolve(), f"SmartBugs executable: {candidate.resolve()}"
    return None, "SmartBugs executable not found"


def _normalise_smartbugs_finding(tool: str, item: dict[str, Any]) -> dict[str, Any]:
    name = item.get("name") or item.get("check") or item.get("title") or "unknown"
    message = item.get("message") or item.get("description") or item.get("extra") or ""
    severity = _normalise_severity(item.get("severity") or item.get("impact"))
    return {
        "tool": tool,
        "name": str(name),
        "severity": severity,
        "line": item.get("line") or item.get("lines"),
        "message": str(message),
        "category": _classify_category(name, message),
        "raw": item,
    }


def _read_smartbugs_results(tool: str, result_dir: Path) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    findings: list[dict[str, Any]] = []
    errors: list[str] = []
    fails: list[str] = []
    for result_path in result_dir.rglob("result.json"):
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{result_path}: {error}")
            continue
        for item in payload.get("findings", []) or []:
            if isinstance(item, dict):
                findings.append(_normalise_smartbugs_finding(tool, item))
        errors.extend(str(item) for item in payload.get("errors", []) or [])
        fails.extend(str(item) for item in payload.get("fails", []) or [])
    return findings, errors, fails


def _normalise_slither_finding(item: dict[str, Any]) -> dict[str, Any]:
    check = item.get("check") or "slither"
    description = item.get("description") or ""
    severity = _normalise_severity(item.get("impact"))
    return {
        "tool": "slither",
        "name": str(check),
        "check": str(check),
        "severity": severity,
        "line": item.get("lines"),
        "message": str(description),
        "category": _classify_category(check, description),
        "raw": item,
    }


def _evaluate_tool_result(
    tool: str,
    run_status: str,
    fixed_findings: list[dict[str, Any]],
    original_findings: list[dict[str, Any]],
    target_categories: set[str],
    error: str | None = None,
) -> dict[str, Any]:
    original_signatures = {_finding_signature(item) for item in original_findings}
    residual = [
        item for item in fixed_findings if _finding_matches_target(item, target_categories)
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


def _policy_verdict(tool_results: dict[str, dict[str, Any]], policy: str = "balanced") -> dict[str, Any]:
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
    return {
        "verdict": "passed",
        "reason": f"{len(completed)}/{len(tool_results)} tools completed without residual target findings.",
    }


def verify_tools(
    code: str,
    file_name: str = "Contract.sol",
    source_root: str | Path | None = None,
) -> dict[str, Any]:
    solc, selection_reason = choose_solc(code)
    if solc is None:
        return {
            "component": "tool_verification",
            "status": "partial",
            "compile": {"status": "unavailable", "reason": selection_reason},
            "slither": {"status": "skipped", "reason": "compiler unavailable"},
        }

    with tempfile.TemporaryDirectory(prefix="e2e-solidity-") as temp_dir:
        temp_root = Path(temp_dir)
        dependency_count = 0
        if source_root:
            root = Path(source_root).resolve()
            if root.is_dir():
                for dependency in root.rglob("*.sol"):
                    relative = dependency.relative_to(root)
                    destination = temp_root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(dependency, destination)
                    dependency_count += 1
        source_path = temp_root / Path(file_name).name
        source_path.write_text(code, encoding="utf-8")
        compile_result = _run([str(solc), "--ast-compact-json", str(source_path)])
        compile_result["compiler"] = str(solc)
        compile_result["selection_reason"] = selection_reason
        compile_result["host_architecture"] = platform.machine()
        compile_result["copied_solidity_dependencies"] = dependency_count

        if compile_result["status"] != "passed":
            return {
                "component": "tool_verification",
                "status": "partial",
                "compile": compile_result,
                "slither": {"status": "skipped", "reason": "source did not compile"},
            }

        slither_executable = shutil.which("slither")
        if not slither_executable:
            return {
                "component": "tool_verification",
                "status": "partial",
                "compile": compile_result,
                "slither": {"status": "unavailable", "reason": "slither not installed"},
            }
        json_path = Path(temp_dir) / "slither.json"
        slither_result = _run(
            [
                slither_executable,
                str(source_path),
                "--solc",
                str(solc),
                "--json",
                str(json_path),
            ],
            timeout=180,
        )
        if json_path.exists():
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                findings = _slither_findings(payload)
                slither_result["findings"] = findings
                slither_result["status"] = "failed" if findings else "passed"
                slither_result["verdict"] = "FAIL" if findings else "PASS"
            except (json.JSONDecodeError, OSError) as error:
                slither_result["parse_error"] = str(error)
                slither_result["status"] = "error"
                slither_result["verdict"] = "ERROR"
        status = "ok" if slither_result.get("status") == "passed" else "partial"
        return {
            "component": "tool_verification",
            "status": status,
            "compile": compile_result,
            "slither": slither_result,
        }


def _run_native_slither(
    code: str,
    file_name: str,
    source_root: str | Path | None,
    solc: Path,
    selection_reason: str,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="e2e-slither-") as temp_dir:
        temp_root = Path(temp_dir)
        source_path, dependency_count = _copy_source_tree(
            temp_root,
            code,
            file_name,
            source_root,
        )
        slither_executable = shutil.which("slither")
        if not slither_executable:
            return {
                "run_status": "unavailable",
                "findings": [],
                "error": "slither not installed",
                "elapsed_ms": 0.0,
            }
        json_path = temp_root / "slither.json"
        result = _run(
            [
                slither_executable,
                str(source_path),
                "--solc",
                str(solc),
                "--json",
                str(json_path),
            ],
            timeout=180,
        )
        findings: list[dict[str, Any]] = []
        parse_error = None
        if json_path.exists():
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                findings = [_normalise_slither_finding(item) for item in _slither_findings(payload)]
            except (json.JSONDecodeError, OSError) as error:
                parse_error = str(error)
        return {
            "run_status": "completed" if result["status"] in {"passed", "failed"} else "error",
            "findings": findings,
            "error": parse_error or result.get("error") or (result.get("stderr") if result["status"] == "error" else None),
            "elapsed_ms": result.get("elapsed_ms", 0.0),
            "command": result.get("command"),
            "compiler": str(solc),
            "selection_reason": selection_reason,
            "copied_solidity_dependencies": dependency_count,
        }


def _run_smartbugs_tool(
    tool: str,
    code: str,
    file_name: str,
    source_root: str | Path | None,
    smartbugs_executable: Path,
    results_root: Path,
    label: str,
    timeout: int,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"e2e-{tool}-{label}-") as temp_dir:
        temp_root = Path(temp_dir)
        source_path, dependency_count = _copy_source_tree(
            temp_root,
            code,
            file_name,
            source_root,
        )
        result_dir = results_root / label / tool
        result_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(smartbugs_executable),
            "-t",
            tool,
            "-f",
            str(source_path),
            "--processes",
            "1",
            "--timeout",
            str(timeout),
            "--mem-limit",
            "4g",
            "--json",
            "--results",
            str(result_dir),
            "--overwrite",
            "--quiet",
        ]
        run_result = _run(command, timeout=timeout + 120, cwd=smartbugs_executable.parent)
        has_result = any(result_dir.rglob("result.json"))
        findings, errors, fails = _read_smartbugs_results(tool, result_dir)
        run_status = "completed" if has_result or run_result["status"] == "passed" else "error"
        error = None
        if errors or fails:
            error = "; ".join([*errors, *fails])[:4000]
        elif run_result["status"] == "error":
            error = run_result.get("error") or run_result.get("stderr")
        elif run_status == "error":
            error = run_result.get("stderr") or "SmartBugs did not produce a parsed result"
        return {
            "run_status": run_status,
            "findings": findings,
            "error": error,
            "elapsed_ms": run_result.get("elapsed_ms", 0.0),
            "command": command,
            "copied_solidity_dependencies": dependency_count,
            "stdout": run_result.get("stdout", ""),
            "stderr": run_result.get("stderr", ""),
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
        with tempfile.TemporaryDirectory(prefix="e2e-compile-fixed-") as temp_dir:
            source_path, dependency_count = _copy_source_tree(
                Path(temp_dir),
                fixed_code,
                file_name,
                source_root,
            )
            compile_result = _run([str(solc), "--ast-compact-json", str(source_path)])
            compile_result["compiler"] = str(solc)
            compile_result["selection_reason"] = selection_reason
            compile_result["host_architecture"] = platform.machine()
            compile_result["copied_solidity_dependencies"] = dependency_count

    target_categories = _target_categories(target_findings)
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

    policy_result = (
        {"verdict": "failed", "reason": "Fixed source does not compile."}
        if compile_result.get("status") != "passed"
        else _policy_verdict(tool_results, policy)
    )
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
