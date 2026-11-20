"""Chọn solc theo pragma (bản thấp nhất thoả pragma), biên dịch, so lỗi của bản vá với bản gốc."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import semantic_version

from E2E.common.regex_patterns import (
    RE_MINOR_VERSION, RE_PRAGMA_SPEC, RE_SOLC_BINARY_VERSION, RE_SOLC_ERROR_LOCATION, RE_WHITESPACE,
)

SOLC_SELECT_ARTIFACTS = Path.home() / ".solc-select" / "artifacts"


# Các bản solc đã cài qua solc-select (~/.solc-select/artifacts), sắp xếp tăng dần theo phiên bản.
def get_installed_solc_versions() -> list[tuple[semantic_version.Version, Path]]:
    installed = []
    for path in SOLC_SELECT_ARTIFACTS.glob("solc-*/solc-*") if SOLC_SELECT_ARTIFACTS.exists() else []:
        match = RE_SOLC_BINARY_VERSION.search(path.name)
        if match and path.is_file() and os.access(path, os.X_OK):
            try:
                installed.append((semantic_version.Version(match.group(1)), path))
            except ValueError:
                pass
    return sorted(installed, key=lambda item: item[0])


# Đọc "pragma solidity ...;" -> (chuỗi pragma gốc, NpmSpec / SimpleSpec để so khớp phiên bản, hoặc None).
def parse_pragma_spec(code: str) -> tuple[str | None, object]:
    match = RE_PRAGMA_SPEC.search(code)
    if not match:
        return None, None
    raw_spec = match.group(1).strip()
    for spec_class in (semantic_version.NpmSpec, semantic_version.SimpleSpec):
        try:
            return raw_spec, spec_class(RE_WHITESPACE.sub(" ", raw_spec))
        except Exception:
            pass
    return raw_spec, None


# Chọn solc: bản THẤP NHẤT thoả pragma -> bản thấp nhất cùng minor -> bản mới nhất. Trả (đường dẫn, lý do, pragma).
def select_best_solc(code: str) -> tuple[Path | None, str, str | None]:
    installed = get_installed_solc_versions()
    raw_pragma, spec = parse_pragma_spec(code)
    if not installed:
        system_solc = shutil.which("solc")
        if system_solc:
            return Path(system_solc), "system solc (solc-select empty)", raw_pragma
        return None, "solc not installed", raw_pragma
    matching = [item for item in installed if spec and spec.match(item[0])]
    if matching:
        return matching[0][1], f"lowest satisfying version {matching[0][0]} for pragma '{raw_pragma}'", raw_pragma
    minor = RE_MINOR_VERSION.search(raw_pragma) if raw_pragma else None
    same_minor = [item for item in installed if minor and item[0].major == 0 and item[0].minor == int(minor.group(1))]
    if same_minor:
        return same_minor[0][1], f"fallback same minor version {same_minor[0][0]} for pragma '{raw_pragma}'", raw_pragma
    return installed[-1][1], f"default latest available {installed[-1][0]}", raw_pragma


# Lọc output solc: chỉ giữ các khối "Error:" (bỏ Warning), tối đa 5 khối; không có thì 1500 ký tự đầu.
def extract_solc_errors(output: str) -> str:
    if not output:
        return ""
    blocks, current = [], None
    for line in output.strip().splitlines():
        if "Error:" in line or "Fatal error:" in line:
            if current:
                blocks.append("\n".join(current))
            current = [line]
        elif current is not None:
            if line.startswith("Warning:") or (line.startswith("-->") and "Warning" in line):
                blocks.append("\n".join(current))
                current = None
            else:
                current.append(line)
    if current:
        blocks.append("\n".join(current))
    return "\n\n".join(blocks[:5]) if blocks else output.strip()[:1500]


# "Chữ ký" lỗi compile: mỗi dòng có "Error", bỏ vị trí file:dòng:cột (để so lỗi bản vá với lỗi sẵn có của bản gốc).
def compiler_error_signatures(text: str) -> set[str]:
    return {RE_SOLC_ERROR_LOCATION.sub("", line).strip() for line in (text or "").splitlines() if "Error" in line}


# Biên dịch một chuỗi mã bằng solc (tuỳ chọn xuất AST) -> {passed (None = không có solc), status, compiler, error, ast}.
def compile_solidity_code(code: str, solc_path: Path | None = None, with_ast: bool = False, timeout: int = 15) -> dict:
    if not code or not code.strip():
        return {"passed": False, "status": "failed", "compiler": "none", "error": "Empty source code", "ast": None}
    solc_path = solc_path or select_best_solc(code)[0]
    if solc_path is None or not solc_path.exists():
        return {"passed": None, "status": "skipped", "compiler": "none",
                "error": "solc compiler not found or not installed", "ast": None}
    with tempfile.NamedTemporaryFile("w", suffix=".sol", delete=False) as source:
        source.write(code)
    try:
        proc = subprocess.run([str(solc_path), *(["--ast-compact-json"] if with_ast else []), source.name],
                              capture_output=True, text=True, timeout=timeout)
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        failed = proc.returncode != 0 or "Error:" in output or "Fatal error:" in output
        return {"passed": not failed, "status": "failed" if failed else "passed", "compiler": solc_path.name,
                "error": extract_solc_errors(proc.stderr or proc.stdout) if failed else "",
                "ast": proc.stdout if with_ast and not failed and proc.stdout else None}
    except subprocess.TimeoutExpired:
        return {"passed": False, "status": "timeout", "compiler": solc_path.name,
                "error": f"Compilation timed out after {timeout} seconds", "ast": None}
    except Exception as exc:
        return {"passed": None, "status": "skipped", "compiler": str(solc_path), "error": f"Execution error: {exc}",
                "ast": None}
    finally:
        Path(source.name).unlink(missing_ok=True)


# Biên dịch CẢ bản gốc lẫn bản vá bằng cùng một solc; bản gốc không compile được -> baseline_failed.
def check_compilation_with_baseline(original_code: str, fixed_code: str, timeout: int = 15) -> dict:
    solc_path, reason, _ = select_best_solc(original_code)
    baseline = compile_solidity_code(original_code, solc_path=solc_path, timeout=timeout)
    fixed = compile_solidity_code(fixed_code, solc_path=solc_path, timeout=timeout)
    return {"baseline": baseline, "fixed": fixed, "compiler": fixed.get("compiler"), "selection_reason": reason,
            "baseline_failed": baseline.get("passed") is False}
