"""
Run external security tools on one Solidity source and normalise their findings.

* Slither, natively (`slither --solc <solc picked by pragma>`): used by Phase 1A
  (static evidence) and by the Phase 5 tool matrix.
* SmartBugs (Docker) for Mythril, SmartCheck and the other wrapped tools: used by the
  Phase 5 tool matrix and by evaluation/benchmark_solodit_9tools.py.
Findings are normalised to {tool, name, check, severity, line, message, category}.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_SMARTBUGS_HOME = PROJECT_ROOT / ".tools" / "smartbugs"


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
