"""
E2E/shared/solc.py
──────────────────────────────────────────────────────────────────────────────
Trình quản lý và biên dịch Solidity Compiler (solc) chuẩn hóa cho pipeline LLM-BSCVM.
Tuân thủ các yêu cầu trong kế hoạch thẩm định (B6, R-07):
  - Phân tích ràng buộc pragma bằng semantic_version.
  - Chọn phiên bản THẤP NHẤT thỏa mãn trong số các bản đã cài đặt trong solc-select.
  - Compile kiểm tra cả contract gốc (baseline): nếu gốc lỗi thì đánh dấu baseline_failed.
  - Trích xuất thông báo lỗi chính xác (các dòng 'Error:'), loại bỏ cảnh báo (warning) gây nhiễu.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Tuple

import semantic_version


SOLC_SELECT_ARTIFACTS = Path.home() / ".solc-select" / "artifacts"


def get_installed_solc_versions() -> List[Tuple[semantic_version.Version, Path]]:
    """Liệt kê tất cả các phiên bản solc thực thi được trong ~/.solc-select/artifacts."""
    installed: List[Tuple[semantic_version.Version, Path]] = []
    if not SOLC_SELECT_ARTIFACTS.exists():
        return installed

    for path in SOLC_SELECT_ARTIFACTS.glob("solc-*/solc-*"):
        if not (path.is_file() and os.access(path, os.X_OK)):
            continue
        m = re.search(r"solc-(\d+\.\d+\.\d+)$", path.name)
        if m:
            try:
                ver = semantic_version.Version(m.group(1))
                installed.append((ver, path))
            except ValueError:
                continue

    # Sắp xếp theo thứ tự phiên bản tăng dần
    installed.sort(key=lambda item: item[0])
    return installed


def parse_pragma_spec(code: str) -> Tuple[Optional[str], Optional[semantic_version.NpmSpec]]:
    """Trích xuất chuỗi pragma và tạo NpmSpec từ code Solidity."""
    match = re.search(r"pragma\s+solidity\s+([^;]+);", code)
    if not match:
        return None, None
    raw_spec = match.group(1).strip()
    # Chuẩn hóa khoảng trắng
    clean_spec = re.sub(r"\s+", " ", raw_spec)
    try:
        npm_spec = semantic_version.NpmSpec(clean_spec)
        return raw_spec, npm_spec
    except Exception:
        # Fallback thử SimpleSpec
        try:
            simple_spec = semantic_version.SimpleSpec(clean_spec)
            # Chuyển sang NpmSpec hoặc tương đương
            return raw_spec, simple_spec
        except Exception:
            return raw_spec, None


def select_best_solc(code: str) -> Tuple[Optional[Path], str, Optional[str]]:
    """
    Chọn trình biên dịch solc thích hợp nhất cho code.
    Quy tắc:
      1. Nếu tìm thấy pragma và có spec hợp lệ: chọn phiên bản THẤP NHẤT thỏa mãn spec.
      2. Nếu không có phiên bản nào thỏa mãn hoàn toàn spec, chọn phiên bản cùng minor (ví dụ 0.4.x).
      3. Fallback: dùng system solc hoặc phiên bản cao nhất có sẵn.
    Trả về: (solc_path, reason, raw_pragma)
    """
    installed = get_installed_solc_versions()
    raw_pragma, spec = parse_pragma_spec(code)

    if not installed:
        system_solc = shutil.which("solc")
        if system_solc:
            return Path(system_solc), "system solc (solc-select empty)", raw_pragma
        return None, "solc not installed", raw_pragma

    if spec:
        matching = [item for item in installed if spec.match(item[0])]
        if matching:
            # Chọn bản THẤP NHẤT thỏa mãn spec
            chosen = matching[0]
            return chosen[1], f"lowest satisfying version {chosen[0]} for pragma '{raw_pragma}'", raw_pragma

    # Nếu không match chính xác, thử match theo minor version (0.4, 0.5, 0.6, 0.7, 0.8)
    if raw_pragma:
        minor_match = re.search(r"0\.(\d+)", raw_pragma)
        if minor_match:
            target_minor = int(minor_match.group(1))
            same_minor = [item for item in installed if item[0].major == 0 and item[0].minor == target_minor]
            if same_minor:
                # Chọn bản thấp nhất trong cùng minor
                chosen = same_minor[0]
                return chosen[1], f"fallback same minor version {chosen[0]} for pragma '{raw_pragma}'", raw_pragma

    # Fallback bản mặc định có sẵn (hoặc bản đầu tiên trong danh sách)
    fallback = installed[-1]
    return fallback[1], f"default latest available {fallback[0]}", raw_pragma


def extract_solc_errors(stderr_or_output: str) -> str:
    """
    Trích xuất các dòng lỗi thực sự ('Error:') và ngữ cảnh kèm theo từ stderr của solc.
    Bỏ qua các cảnh báo (Warning) không làm fail compilation.
    """
    if not stderr_or_output:
        return ""

    lines = stderr_or_output.strip().splitlines()
    error_blocks: List[str] = []
    current_block: List[str] = []
    capturing_error = False

    for line in lines:
        if "Error:" in line or "Fatal error:" in line:
            if current_block:
                error_blocks.append("\n".join(current_block))
                current_block = []
            capturing_error = True
            current_block.append(line)
        elif capturing_error:
            if line.startswith("Warning:") or line.startswith("-->") and "Warning" in line:
                capturing_error = False
                if current_block:
                    error_blocks.append("\n".join(current_block))
                    current_block = []
            else:
                current_block.append(line)

    if current_block:
        error_blocks.append("\n".join(current_block))

    if error_blocks:
        return "\n\n".join(error_blocks[:5])  # Giới hạn 5 lỗi chính

    # Nếu không tìm thấy chữ Error cụ thể nhưng compiler fail, trả về 1000 ký tự đầu của stderr
    return stderr_or_output.strip()[:1500]


def compile_solidity_code(
    code: str,
    solc_path: Optional[Path] = None,
    with_ast: bool = False,
    timeout: int = 15,
) -> dict[str, Any]:
    """
    Biên dịch chuỗi mã nguồn Solidity bằng solc.
    Trả về dict chuẩn: passed, status, compiler, error, ast (nếu có).
    """
    if not code or not code.strip():
        return {
            "passed": False,
            "status": "failed",
            "compiler": "none",
            "error": "Empty source code",
            "ast": None,
        }

    if solc_path is None:
        solc_path, reason, _ = select_best_solc(code)

    if solc_path is None or not solc_path.exists():
        return {
            "passed": None,
            "status": "skipped",
            "compiler": "none",
            "error": "solc compiler not found or not installed",
            "ast": None,
        }

    cmd = [str(solc_path)]
    if with_ast:
        cmd.append("--ast-compact-json")

    with tempfile.NamedTemporaryFile("w", suffix=".sol", delete=False) as tf:
        tf.write(code)
        tf_path = tf.name

    cmd.append(tf_path)

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        combined_output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        has_error = proc.returncode != 0 or "Error:" in combined_output or "Fatal error:" in combined_output
        err_msg = extract_solc_errors(proc.stderr or proc.stdout) if has_error else ""

        ast_data = None
        if with_ast and not has_error and proc.stdout:
            ast_data = proc.stdout

        return {
            "passed": not has_error,
            "status": "passed" if not has_error else "failed",
            "compiler": solc_path.name,
            "compiler_path": str(solc_path),
            "error": err_msg,
            "raw_output": combined_output[:2000],
            "ast": ast_data,
        }
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "status": "timeout",
            "compiler": solc_path.name,
            "error": f"Compilation timed out after {timeout} seconds",
            "ast": None,
        }
    except Exception as exc:
        return {
            "passed": None,
            "status": "skipped",
            "compiler": str(solc_path),
            "error": f"Execution error: {exc}",
            "ast": None,
        }
    finally:
        try:
            Path(tf_path).unlink(missing_ok=True)
        except Exception:
            pass


def check_compilation_with_baseline(
    original_code: str,
    fixed_code: str,
    timeout: int = 15,
) -> dict[str, Any]:
    """
    Biên dịch kiểm tra cả contract gốc và contract đã vá.
    Nếu contract gốc đã không compile được, đánh dấu baseline_failed.
    """
    solc_path, reason, raw_pragma = select_best_solc(original_code)

    base_res = compile_solidity_code(original_code, solc_path=solc_path, timeout=timeout)
    fixed_res = compile_solidity_code(fixed_code, solc_path=solc_path, timeout=timeout)

    baseline_failed = base_res.get("passed") is False
    return {
        "baseline": base_res,
        "fixed": fixed_res,
        "compiler": fixed_res.get("compiler"),
        "selection_reason": reason,
        "raw_pragma": raw_pragma,
        "baseline_failed": baseline_failed,
        # Nếu baseline_failed mà fixed_code cũng fail do cùng nguyên nhân, ta không quy lỗi hoàn toàn cho fixer
        "passed": fixed_res.get("passed"),
        "status": "baseline_failed" if baseline_failed and not fixed_res.get("passed") else fixed_res.get("status"),
        "error": fixed_res.get("error"),
    }
