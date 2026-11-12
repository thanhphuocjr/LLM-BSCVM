"""
Run Slither natively on one Solidity source and normalise its findings.

`slither <file> --solc <solc picked by pragma> --json` is used by Phase 1A (static
evidence) and by the Phase 5 tool matrix. Findings are normalised to
{tool, name, check, severity, line, message, category}.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


# Chạy một lệnh ngoài (subprocess) có timeout, trả dict status/return_code/stdout/stderr/thời gian.
# Không bao giờ ném lỗi (timeout, thiếu lệnh -> status="error").
def _run(command: list[str], timeout: int = 120) -> dict[str, Any]:
    try:
        started = time.perf_counter()
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


# Rút danh sách detector từ JSON của Slither: check, impact, confidence, mô tả, các dòng code liên quan.
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


# Chép mã nguồn (và các file .sol phụ thuộc trong source_root, nếu có) vào thư mục tạm.
# Mục đích: Slither cần file thật trên đĩa và cần thấy các file được import.
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


# Chuẩn hoá mức độ của tool về High/Medium/Low/Informational/Unknown.
def _normalise_severity(value: Any) -> str:
    text = str(value or "Unknown").strip()
    if not text:
        return "Unknown"
    lowered = text.lower()
    if lowered in {"optimization", "informational", "info"}:
        return "Informational"
    return lowered.capitalize()


# Gom finding của mọi tool vào nhóm lỗ hổng chung (reentrancy, arithmetic, access_control, tx_origin...)
# bằng từ khoá trong tên/mô tả. Mục đích: so sánh được kết quả giữa các tool khác nhau.
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


# Chuẩn hoá một finding Slither về dạng chung {tool, name, check, severity, line, message, category}.
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


# Nhiệm vụ: chạy Slither trực tiếp trên máy (slither <file> --solc <solc chọn theo pragma> --json).
# Trả run_status (completed/unavailable/error) + danh sách finding đã chuẩn hoá.
# Mục đích: dùng ở Phase 1A (bằng chứng static) và trong ma trận tool Phase 5.
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
