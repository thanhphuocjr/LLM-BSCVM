"""Orchestrator - chạy một contract qua Phase 0 -> 6, ghi một artifact cho mỗi phase.

    python -m E2E.run demo_contracts/VulnerableBank.sol
"""

import argparse
import concurrent.futures
import json
import time
from pathlib import Path

from E2E.common.io_utils import log, read_json, sha256_text, utc_now, write_json, write_text
from E2E.common.regex_patterns import RE_RUN_ID, RE_UNSAFE_NAME_CHARS
from E2E.common.safe_run import safe_run
from E2E.config import (
    DEFAULT_DETECTOR_THRESHOLD, DETECTOR_ADAPTER_DIR, MAX_FIX_ATTEMPTS, PIPELINE_VERSION, RUNS_DIR, SCHEMA_VERSION,
)
from E2E.phase0_parsing.code_units import parse_code_units
from E2E.phase1_fusion.fusion import fuse_detection
from E2E.phase1a_static.static_runner import run_static
from E2E.phase1b_rag.rag_runner import run_rag
from E2E.phase5_verifier.verifier import build_verification_channels, decide_overall_verdict, skipped_verification
from E2E.phase6_reporter.report import apply_verification_to_report, build_report_data, render_markdown, write_reports


# Ngưỡng P(vulnerable) của detector: tham số > best_threshold trong threshold_config.json của model > mặc định 0.70.
def resolve_detector_threshold(configured: float | None = None) -> float:
    if configured is not None:
        return float(configured)
    try:
        return float(read_json(DETECTOR_ADAPTER_DIR / "threshold_config.json")["best_threshold"])
    except Exception:
        return DEFAULT_DETECTOR_THRESHOLD


# run_id "<YYYYmmdd-HHMMSS>-<tên file>-<8 ký tự SHA-256 của mã>" (mỗi lần audit một thư mục riêng).
def make_run_id(file_name: str, code: str) -> str:
    stem = RE_UNSAFE_NAME_CHARS.sub("-", Path(file_name).stem).strip("-") or "contract"
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{stem}-{sha256_text(code)[:8]}"


# run_id chỉ gồm chữ/số/._- và <= 128 ký tự (dùng làm tên thư mục -> chặn path traversal).
def validated_run_id(value: str) -> str:
    if not RE_RUN_ID.fullmatch(value):
        raise ValueError("run_id must contain only letters, digits, dot, underscore, or hyphen "
                         "and must not exceed 128 characters")
    return value


# Tóm tắt detector của bản "original" / "fixed" trong server_result (chưa có kết quả GPU -> None).
def _remote_detector(remote: dict, target: str) -> dict | None:
    summary = remote.get("phases", {}).get("detector", {}).get(target)
    return dict(summary) if isinstance(summary, dict) else None


# Kết quả GPU phải khớp request: schema_version, có "phases", status hợp lệ, cùng run_id và cùng SHA-256 mã nguồn.
def validate_remote_result(request: dict, remote: dict) -> None:
    if remote.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Remote schema mismatch: expected {SCHEMA_VERSION}, got {remote.get('schema_version')}")
    if not isinstance(remote.get("phases"), dict):
        raise ValueError("Remote result is missing the phases object")
    if remote.get("status") not in {"ok", "partial", "error"}:
        raise ValueError(f"Invalid remote status: {remote.get('status')}")
    if remote.get("run_id") != request["run_id"]:
        raise ValueError(f"Remote run_id mismatch: expected {request['run_id']}, got {remote.get('run_id')}")
    if remote.get("source_sha256") != request["source"]["sha256"]:
        raise ValueError("Remote source SHA-256 does not match the submitted source")


# Phase 1A (static) và 1B (RAG) chạy song song; lượt 2 dùng lại kết quả lượt 1 (cùng mã nguồn).
def _local_phase1(code: str, units: list, local_phase1: dict | None) -> tuple[dict, dict]:
    if local_phase1 and local_phase1.get("static") and local_phase1.get("rag"):
        return local_phase1["static"], local_phase1["rag"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        static = executor.submit(safe_run, "static", lambda: run_static(code, units))
        rag = executor.submit(safe_run, "rag", lambda: run_rag(code, units))
        return static.result(), rag.result()


# Phase 5 cuối: lấy kết quả kiểm chứng của bản vá tốt nhất (đã chạy trong vòng sửa) -> overall_verdict.
def _final_verification(code: str, remote: dict, fused: dict, run_dir: Path) -> tuple[dict, dict]:
    fixer = remote.get("phases", {}).get("fixer", {})
    verifier = remote.get("phases", {}).get("verifier", {}) if remote else {}
    fixed_code = fixer.get("fixed_code") or code
    patch_outcomes = fixer.get("patch_outcomes", []) if remote else []
    patch_applied = any(item.get("status") == "applied" for item in patch_outcomes)
    final = verifier.get("final") if fixed_code != code else None
    if final:
        tool_result, behavioral, fixed_static = (dict(final["tool_verification"]), final["behavioral_verification"],
                                                 final["fixed_static"])
        write_json(run_dir / "phase5_fixed_static.json", fixed_static)
    else:
        (tool_result, behavioral), fixed_static = skipped_verification("No verified patch."), None
    tool_result["behavioral"] = behavioral
    verdict = decide_overall_verdict(remote=remote, fused=fused, code=code, fixed_code=fixed_code,
                                     patch_applied=patch_applied, tool_result=tool_result, behavioral_result=behavioral,
                                     fixed_static=fixed_static)
    remote_fixed = _remote_detector(remote, "fixed")
    complete = tool_result.get("policy_verdict", "inconclusive") in {"passed", "failed"} \
        and tool_result.get("status") in {"ok", "partial"}
    verification = {
        "status": "complete" if complete else "inconclusive" if not remote or remote.get("status") != "ok" else "partial",
        "original_detection": fused, "fixed_detection": remote_fixed or {}, "tool_verification": tool_result,
        "behavioral_verification": behavioral,
        "verification_channels": build_verification_channels(tool_result, static_result=fixed_static,
                                                             detector_fixed=remote_fixed),
        "attempts_count": verifier.get("attempts_count", 1) if remote else 0,
        "max_attempts": verifier.get("max_attempts", MAX_FIX_ATTEMPTS),
        "fixed_unit_results": verifier.get("fixed_unit_results", []),
        "detector_disagreement": bool(remote_fixed) and remote_fixed.get("verdict") == "Vulnerable"
        and verdict.startswith("Patch accepted"),
        "overall_verdict": verdict,
    }
    write_json(run_dir / "phase5_verification.json", verification)
    return tool_result, verification


# [HÀM TRUNG TÂM] Lượt 1 (chưa có remote_result): Phase 0, 1A, 1B -> request.json; lượt 2: fusion, Phase 5, Phase 6.
def run_pipeline(code: str, *, file_name: str = "Contract.sol", run_id: str | None = None,
                 runs_dir: str | Path = RUNS_DIR, remote_result: dict | None = None, detector_threshold: float | None = None,
                 source_root: str | Path | None = None, max_fix_attempts: int = MAX_FIX_ATTEMPTS,
                 local_phase1: dict | None = None) -> dict:
    remote = remote_result or {}
    progress = (lambda message: None) if remote else log
    run_id = validated_run_id(run_id or make_run_id(file_name, code))
    run_dir = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    progress("[PHASE 0] Normalizing source code and extracting AST code units...")
    units = parse_code_units(code)
    source = {"file_name": file_name, "sha256": sha256_text(code), "code": code}
    header = {"schema_version": SCHEMA_VERSION, "pipeline_version": PIPELINE_VERSION, "run_id": run_id}
    write_json(run_dir / "phase0_input.json", {**header, "created_at": utc_now(), "source": source,
                                               "code_units": [unit.to_dict() for unit in units]})
    progress(f"[PHASE 0] Completed ({len(units)} code unit(s) extracted).")
    progress("[PHASE 1A] Running static vulnerability rules (Slither / Regex AST)...")
    progress("[PHASE 1B] Running RAG empirical vulnerability retrieval...")
    static_result, rag_result = _local_phase1(code, units, local_phase1)
    write_json(run_dir / "phase1a_static.json", static_result)
    progress(f"[PHASE 1A] Completed: Static analysis finished ({len(static_result.get('findings', []))} finding(s)).")
    write_json(run_dir / "phase1b_rag.json", rag_result)
    progress(f"[PHASE 1B] Completed: RAG retrieval finished ({len(rag_result.get('findings', []))} finding(s)).")
    provisional = fuse_detection(static_result, rag_result, None)
    write_json(run_dir / "phase1_local_provisional.json", provisional)
    request = {
        **header, "created_at": utc_now(), "source": source, "code_units": [unit.to_dict() for unit in units],
        "local_phase1": {"static": static_result, "rag": rag_result, "provisional_fusion": provisional},
        "execution": {"detector_threshold": resolve_detector_threshold(detector_threshold),
                      "source_root": str(source_root) if source_root else None,
                      "repair": {"attempt": 1, "max_attempts": min(max(1, max_fix_attempts), MAX_FIX_ATTEMPTS)}},
    }
    request_path = write_json(run_dir / "request.json", request)
    if remote:
        validate_remote_result(request, remote)
        write_json(run_dir / "server_result.json", remote)
    fused = fuse_detection(static_result, rag_result, _remote_detector(remote, "original"))
    write_json(run_dir / "phase1_fused.json", fused)
    if remote:
        log(f"[PHASE 1 FUSED] Completed: Multi-modal fusion decision is {fused.get('verdict')} "
            f"(score: {fused.get('score')}).")
    for number, name in ((2, "advisor"), (3, "assessor"), (4, "fixer")):
        write_json(run_dir / f"phase{number}_{name}.json", remote.get("phases", {}).get(
            name, {"status": "pending", "reason": "GPU server result has not been supplied"}))
    tool_result, verification = _final_verification(code, remote, fused, run_dir)
    if remote:
        log(f"[PHASE 5] Completed: Verification verdict: {verification['overall_verdict']}.")
        log("[PHASE 6] Generating multi-format audit reports (HTML, PDF, Markdown)...")
        report_data = build_report_data(request, fused, remote, tool_result)
        apply_verification_to_report(report_data, verification)
        write_json(run_dir / "phase6_report_data.json", report_data)
        report_paths = write_reports(report_data, run_dir)
    else:
        report_data = build_report_data(request, fused, None, tool_result)
        preview = write_text(run_dir / "phase1_preview.md", render_markdown(report_data))
        report_paths = {"markdown_preview": str(preview)}
    complete = remote.get("status") == "ok" and fused["status"] == "complete" and verification["status"] == "complete"
    manifest = {
        **header, "status": "complete" if complete else "pending_remote" if not remote else "partial",
        "request_path": str(request_path), "run_dir": str(run_dir),
        "detection": {"verdict": fused["verdict"], "status": fused["status"], "score": fused["score"],
                      "risk_level": report_data["risk_level"], "finding_count": len(report_data["findings"]),
                      "dismissed_count": len(report_data.get("dismissed_findings", []))},
        "verification": {"status": verification["status"], "overall_verdict": verification["overall_verdict"]},
        "reports": report_paths,
        "next_action": "Waiting for the GPU phases (pipeline/gpu_pipeline.py, Phase 1C-5) on request.json." if not remote
        else "Review findings, generated patch, compiler output, and behavioral verification checks.",
        "completed_at": utc_now(),
    }
    write_json(run_dir / "manifest.json", manifest)
    if remote:
        log("[PHASE 6] Completed: Audit reports compiled.")
        log("Run completed successfully.")
    return manifest


# Điểm vào CLI (web app gọi lại đúng lệnh này): lượt 1 local -> Phase 1C-5 (gpu_pipeline, cùng tiến trình) -> lượt 2.
def main() -> None:
    parser = argparse.ArgumentParser(description="Audit one Solidity contract (Phase 0 -> 6) through the GPU server.")
    parser.add_argument("code_file", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--detector-threshold", type=float, default=None,
                        help="Vulnerability probability threshold. Default is auto from threshold_config.json "
                             f"({DEFAULT_DETECTOR_THRESHOLD}).")
    parser.add_argument("--max-fix-attempts", type=int, default=MAX_FIX_ATTEMPTS,
                        help="Maximum repair attempts (1-2) of the Fixer -> Verifier loop.")
    parser.add_argument("--solidity-root", type=Path,
                        help="Root containing imported .sol files; defaults to the input file directory.")
    args = parser.parse_args()
    code = args.code_file.read_text(encoding="utf-8")
    common = dict(file_name=args.code_file.name, runs_dir=args.runs_dir, detector_threshold=args.detector_threshold,
                  source_root=args.solidity_root or args.code_file.parent, max_fix_attempts=args.max_fix_attempts)
    manifest = run_pipeline(code, run_id=args.run_id, **common)
    from E2E.pipeline.gpu_pipeline import run_server_gpu_pipeline
    request = read_json(manifest["request_path"])
    manifest = run_pipeline(code, run_id=manifest["run_id"], remote_result=run_server_gpu_pipeline(request),
                            local_phase1=request.get("local_phase1"), **common)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
