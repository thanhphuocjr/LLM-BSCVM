"""Chạy Slither native trên một mã nguồn Solidity (Phase 1A, Phase 5) và chuẩn hoá finding."""

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

# Nhóm lỗ hổng chung -> từ khoá nhận diện trong tên/mô tả finding của tool (thứ tự = độ ưu tiên).
_CATEGORY_KEYWORDS = (
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


# Chạy lệnh ngoài có timeout, không bao giờ ném lỗi: trả status passed/failed/error, return_code, stdout, stderr.
def _run(command: list[str], timeout: int = 120) -> dict:
    started = time.perf_counter()
    try:
        done = subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as error:
        return {"status": "error", "error": f"timeout after {timeout}s", "stdout": str(error.stdout or "")[-4000:],
                "stderr": str(error.stderr or "")[-4000:], "command": command, "elapsed_ms": round(timeout * 1000, 3)}
    except OSError as error:
        return {"status": "error", "error": str(error), "command": command, "elapsed_ms": 0.0}
    return {"status": "passed" if done.returncode == 0 else "failed", "return_code": done.returncode,
            "stdout": done.stdout[-12000:], "stderr": done.stderr[-12000:], "command": command,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3)}


# Chép mã nguồn (và các file .sol trong source_root, nếu có) vào thư mục tạm cho compiler/Slither.
def _copy_source_tree(temp_root: Path, code: str, file_name: str, source_root: str | Path | None) -> tuple[Path, int]:
    root = Path(source_root).resolve() if source_root else None
    dependencies = list(root.rglob("*.sol")) if root and root.is_dir() else []
    for dependency in dependencies:
        destination = temp_root / dependency.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dependency, destination)
    source_path = temp_root / Path(file_name).name
    source_path.write_text(code, encoding="utf-8")
    return source_path, len(dependencies)


# Chuẩn hoá mức độ của tool về High/Medium/Low/Informational/Unknown.
def _normalise_severity(value: Any) -> str:
    text = str(value or "Unknown").strip()
    if text.lower() in {"optimization", "informational", "info"}:
        return "Informational"
    return text.capitalize() if text else "Unknown"


# Gom finding của tool vào nhóm lỗ hổng chung (reentrancy, arithmetic, tx_origin...) theo từ khoá.
def _classify_category(*parts: Any) -> str:
    text = " ".join(str(part or "") for part in parts).lower()
    return next((category for category, needles in _CATEGORY_KEYWORDS if any(n in text for n in needles)), "other")


# Đổi JSON của Slither thành danh sách finding chuẩn {tool, name, check, severity, line, message, category, raw}.
def _slither_findings(payload: dict) -> list[dict]:
    findings = []
    for detector in payload.get("results", {}).get("detectors", []):
        lines = [line for element in detector.get("elements", [])
                 for line in element.get("source_mapping", {}).get("lines") or []]
        raw = {"check": detector.get("check"), "impact": detector.get("impact"),
               "confidence": detector.get("confidence"), "description": str(detector.get("description") or "").strip(),
               "lines": sorted(set(lines))}
        check = str(raw["check"] or "slither")
        findings.append({"tool": "slither", "name": check, "check": check,
                         "severity": _normalise_severity(raw["impact"]), "line": raw["lines"],
                         "message": raw["description"], "category": _classify_category(check, raw["description"]),
                         "raw": raw})
    return findings


# Chạy `slither <file> --solc <solc> --json` trong thư mục tạm -> run_status (completed/unavailable/error) + finding.
def _run_native_slither(code: str, file_name: str, source_root: str | Path | None, solc: Path,
                        selection_reason: str) -> dict:
    slither = shutil.which("slither")
    if not slither:
        return {"run_status": "unavailable", "findings": [], "error": "slither not installed", "elapsed_ms": 0.0}
    with tempfile.TemporaryDirectory(prefix="e2e-slither-") as temp_dir:
        source_path, dependency_count = _copy_source_tree(Path(temp_dir), code, file_name, source_root)
        json_path = Path(temp_dir) / "slither.json"
        result = _run([slither, str(source_path), "--solc", str(solc), "--json", str(json_path)], timeout=180)
        findings, parse_error = [], None
        if json_path.exists():
            try:
                findings = _slither_findings(json.loads(json_path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError) as error:
                parse_error = str(error)
    return {
        "run_status": "completed" if result["status"] in {"passed", "failed"} else "error", "findings": findings,
        "error": parse_error or result.get("error") or (result.get("stderr") if result["status"] == "error" else None),
        "elapsed_ms": result.get("elapsed_ms", 0.0), "command": result.get("command"), "compiler": str(solc),
        "selection_reason": selection_reason, "copied_solidity_dependencies": dependency_count,
    }
