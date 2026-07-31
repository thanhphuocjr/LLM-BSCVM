from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


SOLC_SELECT_ARTIFACTS = Path.home() / ".solc-select" / "artifacts"


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


def _run(command: list[str], timeout: int = 120) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "status": "passed" if completed.returncode == 0 else "failed",
            "return_code": completed.returncode,
            "stdout": completed.stdout[-12000:],
            "stderr": completed.stderr[-12000:],
            "command": command,
        }
    except subprocess.TimeoutExpired as error:
        return {
            "status": "error",
            "error": f"timeout after {timeout}s",
            "stdout": str(error.stdout or "")[-4000:],
            "stderr": str(error.stderr or "")[-4000:],
            "command": command,
        }
    except OSError as error:
        return {"status": "error", "error": str(error), "command": command}


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
                slither_result["findings"] = _slither_findings(payload)
                # Slither may return non-zero when detectors find issues.
                slither_result["status"] = "passed"
            except (json.JSONDecodeError, OSError) as error:
                slither_result["parse_error"] = str(error)
        status = "ok" if slither_result["status"] == "passed" else "partial"
        return {
            "component": "tool_verification",
            "status": status,
            "compile": compile_result,
            "slither": slither_result,
        }
