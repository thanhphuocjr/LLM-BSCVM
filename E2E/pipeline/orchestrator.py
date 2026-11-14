"""
Orchestrator - runs one contract through Phase 0 -> 6 and writes one artifact per phase.

    python -m E2E.run demo_contracts/VulnerableBank.sol

  Phase 0   parse the source into code units             phase0_parsing/code_units.py
  Phase 1A  static analysis + Slither                    phase1a_static/static_runner.py
  Phase 1B  RAG over the vulnerability knowledge base    phase1b_rag/rag_runner.py
            -> request.json (input of the GPU phases)
  Phase 1C-5 GPU phases: detector, per-finding fusion, Advisor, Assessor, Fixer + repair loop
            (self-check)                                  pipeline/gpu_pipeline.py
            The loop itself runs on THIS machine, in the same process as main(); only the
            model calls go to the GPU server over HTTP (/api/v1/detect, /api/v1/generate).
  Phase 1F  contract-level fusion of static + RAG + detector  phase1_fusion/fusion.py
  Phase 5   final verification of the patch             phase5_verifier/final_verification.py
  Phase 6   audit report (Markdown / HTML / PDF)         phase6_reporter/report.py

Artifacts in data/results/audit_runs/<run_id>/: phase0_input.json, phase1a_static.json,
phase1b_rag.json, request.json, server_result.json, phase1_fused.json, phase2_advisor.json,
phase3_assessor.json, phase4_fixer.json, phase5_verification.json,
phase6_audit_report.{md,html,pdf}, manifest.json.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from pathlib import Path
from typing import Any

from E2E.common.io_utils import read_json, sha256_text, utc_now, write_json, write_text
from E2E.common.regex_patterns import RE_RUN_ID, RE_UNSAFE_NAME_CHARS
from E2E.common.safe_run import safe_run
from E2E.config import (
    DEFAULT_DETECTOR_THRESHOLD,
    DETECTOR_ADAPTER_DIR,
    MAX_FIX_ATTEMPTS,
    PIPELINE_VERSION,
    RUNS_DIR,
    SCHEMA_VERSION,
)
from E2E.phase0_parsing.code_units import parse_code_units
from E2E.phase1_fusion.fusion import fuse_detection
from E2E.phase1a_static.static_runner import run_static
from E2E.phase1b_rag.rag_runner import run_rag
from E2E.phase5_verifier.final_verification import (
    build_verification_channels,
    decide_overall_verdict,
    skipped_verification,
    verify_final_patch,
)
from E2E.phase6_reporter.report import apply_verification_to_report, build_report_data, render_markdown, write_reports


# Nhiệm vụ: chọn ngưỡng P(vulnerable) cho detector Phase 1C.
# Thứ tự ưu tiên: tham số truyền vào (--detector-threshold) > "best_threshold" trong
# models/qwen25-v13-14b-detector-final/threshold_config.json > DEFAULT_DETECTOR_THRESHOLD (0.70).
# Mục đích: ngưỡng được ghi vào request.json để GPU server dùng khi phân loại từng hàm.
def resolve_detector_threshold(configured: float | None = None) -> float:
    if configured is not None:
        return float(configured)
    config_path = DETECTOR_ADAPTER_DIR / "threshold_config.json"
    if config_path.exists():
        try:
            data = read_json(config_path)
            if "best_threshold" in data:
                return float(data["best_threshold"])
        except Exception:
            pass
    return DEFAULT_DETECTOR_THRESHOLD


# Nhiệm vụ: sinh run_id dạng "<YYYYmmdd-HHMMSS>-<tên file>-<8 ký tự SHA-256 của code>".
# Mục đích: mỗi lần audit có một thư mục riêng data/results/audit_runs/<run_id>/, nhìn tên là biết
# chạy lúc nào, file nào, nội dung nào.
def make_run_id(file_name: str, code: str) -> str:
    stem = RE_UNSAFE_NAME_CHARS.sub("-", Path(file_name).stem).strip("-") or "contract"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{stem}-{sha256_text(code)[:8]}"


# Nhiệm vụ: kiểm tra run_id chỉ gồm chữ/số/._- và tối đa 128 ký tự, sai thì raise ValueError.
# Mục đích: run_id được dùng làm tên thư mục -> chặn path traversal (vd "../../etc") từ CLI/web app.
def validated_run_id(value: str) -> str:
    if not RE_RUN_ID.fullmatch(value):
        raise ValueError(
            "run_id must contain only letters, digits, dot, underscore, or hyphen "
            "and must not exceed 128 characters"
        )
    return value


# Nhiệm vụ: lấy tóm tắt detector (Phase 1C) của bản "original" hoặc "fixed" từ server_result
# (phases.detector.original / .fixed, do summarize_detector tạo); chưa có kết quả GPU -> None.
# Mục đích: đưa điểm detector của bản gốc vào fuse_detection để ra kết luận cuối của contract.
def _remote_detector(remote: dict[str, Any], target: str) -> dict[str, Any] | None:
    summary = remote.get("phases", {}).get("detector", {}).get(target)
    return dict(summary) if isinstance(summary, dict) else None


# Nhiệm vụ: kiểm tra kết quả GPU trả về khớp với request: đúng schema_version, có "phases",
# status hợp lệ, cùng run_id và cùng SHA-256 mã nguồn.
# Mục đích: không bao giờ ghép nhầm kết quả của contract/lần chạy khác vào báo cáo.
def validate_remote_result(request: dict[str, Any], remote: dict[str, Any]) -> None:
    if remote.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Remote schema mismatch: expected {SCHEMA_VERSION}, got {remote.get('schema_version')}"
        )
    if not isinstance(remote.get("phases"), dict):
        raise ValueError("Remote result is missing the phases object")
    if remote.get("status") not in {"ok", "partial", "error"}:
        raise ValueError(f"Invalid remote status: {remote.get('status')}")
    if remote.get("run_id") != request["run_id"]:
        raise ValueError(
            f"Remote run_id mismatch: expected {request['run_id']}, got {remote.get('run_id')}"
        )
    if remote.get("source_sha256") != request["source"]["sha256"]:
        raise ValueError("Remote source SHA-256 does not match the submitted source")


# [HÀM TRUNG TÂM] Chạy một contract qua Phase 0 -> 6 và ghi một file JSON cho mỗi phase.
# Được gọi 2 lần trong một lần audit (xem main):
#   - Lượt 1 (remote_result=None): Phase 0 tách hàm, Phase 1A static + Phase 1B RAG chạy song song,
#     fusion tạm (chưa có detector), ghi request.json + bản xem trước phase1_preview.md.
#   - Lượt 2 (có remote_result từ gpu_pipeline): kiểm tra kết quả GPU, fusion đầy đủ (static + RAG +
#     detector), ghi phase2/3/4, chạy kiểm chứng cuối Phase 5 (verify_final_patch), quyết định
#     overall_verdict, rồi xuất báo cáo Phase 6 (MD/HTML/PDF) và manifest.json.
# Mục đích: là "xương sống" của hệ thống; CLI và web app (gọi lại CLI) đều đi qua hàm này.
def run_pipeline(
    code: str,
    *,
    file_name: str = "Contract.sol",
    run_id: str | None = None,
    runs_dir: str | Path = RUNS_DIR,
    remote_result: dict[str, Any] | None = None,
    detector_threshold: float | None = None,
    source_root: str | Path | None = None,
    max_fix_attempts: int = MAX_FIX_ATTEMPTS,
    local_phase1: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write every phase artifact of one contract to runs_dir/<run_id>/.

    Called twice per audit (see main): without `remote_result` it runs Phase 0, 1A, 1B and
    writes request.json; with the GPU-phase result it merges Phases 1C-4 and runs Phase 5 and 6.
    """
    actual_run_id = validated_run_id(run_id or make_run_id(file_name, code))
    run_dir = Path(runs_dir) / actual_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if not remote_result:
        print(f"[{time.strftime('%H:%M:%S')}] [PHASE 0] Normalizing source code and extracting AST code units...", flush=True)
    units = parse_code_units(code)
    source = {
        "file_name": file_name,
        "sha256": sha256_text(code),
        "code": code,
    }
    write_json(
        run_dir / "phase0_input.json",
        {
            "schema_version": SCHEMA_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            "run_id": actual_run_id,
            "created_at": utc_now(),
            "source": source,
            "code_units": [unit.to_dict() for unit in units],
        },
    )
    if not remote_result:
        print(f"[{time.strftime('%H:%M:%S')}] [PHASE 0] Completed ({len(units)} code unit(s) extracted).", flush=True)

    if not remote_result:
        print(f"[{time.strftime('%H:%M:%S')}] [PHASE 1A] Running static vulnerability rules (Slither / Regex AST)...", flush=True)
        print(f"[{time.strftime('%H:%M:%S')}] [PHASE 1B] Running RAG empirical vulnerability retrieval...", flush=True)
    if local_phase1 and local_phase1.get("static") and local_phase1.get("rag"):
        # Re-entry after the remote/GPU phases: Phase 1A/1B already ran on this exact source.
        static_result = local_phase1["static"]
        rag_result = local_phase1["rag"]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_static = executor.submit(safe_run, "static", lambda: run_static(code, units))
            future_rag = executor.submit(safe_run, "rag", lambda: run_rag(code, units))
            static_result = future_static.result()
            rag_result = future_rag.result()

    write_json(run_dir / "phase1a_static.json", static_result)
    if not remote_result:
        print(f"[{time.strftime('%H:%M:%S')}] [PHASE 1A] Completed: Static analysis finished ({len(static_result.get('findings', []))} finding(s)).", flush=True)
    write_json(run_dir / "phase1b_rag.json", rag_result)
    if not remote_result:
        print(f"[{time.strftime('%H:%M:%S')}] [PHASE 1B] Completed: RAG retrieval finished ({len(rag_result.get('findings', []))} finding(s)).", flush=True)
    provisional = fuse_detection(static_result, rag_result, None)
    write_json(run_dir / "phase1_local_provisional.json", provisional)

    request = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "run_id": actual_run_id,
        "created_at": utc_now(),
        "source": source,
        "code_units": [unit.to_dict() for unit in units],
        "local_phase1": {
            "static": static_result,
            "rag": rag_result,
            "provisional_fusion": provisional,
        },
        "execution": {
            "detector_threshold": resolve_detector_threshold(detector_threshold),
            "repair": {
                "attempt": 1,
                "max_attempts": min(max(1, max_fix_attempts), MAX_FIX_ATTEMPTS),
            },
        },
    }
    request_path = write_json(run_dir / "request.json", request)

    remote = remote_result or {}
    if remote:
        validate_remote_result(request, remote)
        write_json(run_dir / "server_result.json", remote)
    detector_original = _remote_detector(remote, "original")
    fused = fuse_detection(static_result, rag_result, detector_original)
    write_json(run_dir / "phase1_fused.json", fused)
    if remote:
        print(f"[{time.strftime('%H:%M:%S')}] [PHASE 1 FUSED] Completed: Multi-modal fusion decision is {fused.get('verdict')} (score: {fused.get('score')}).", flush=True)

    for number, phase_name in (
        (2, "advisor"),
        (3, "assessor"),
        (4, "fixer"),
    ):
        payload = remote.get("phases", {}).get(phase_name, {
            "status": "pending",
            "reason": "GPU server result has not been supplied",
        })
        write_json(run_dir / f"phase{number}_{phase_name}.json", payload)

    fixed_code = remote.get("phases", {}).get("fixer", {}).get("fixed_code") or code
    patch_outcomes = remote.get("phases", {}).get("fixer", {}).get("patch_outcomes", []) if remote else []
    patch_applied = any(item.get("status") == "applied" for item in patch_outcomes)

    if remote and remote.get("status") in {"ok", "partial"} and fixed_code and fixed_code != code:
        print(f"[{time.strftime('%H:%M:%S')}] [PHASE 5] Verifying patch: compile + Slither, AST behavior check, static/RAG re-run...", flush=True)
        tool_result, behavioral_result, fixed_static, fixed_rag = verify_final_patch(
            code,
            fixed_code,
            patch_outcomes=patch_outcomes,
            fused=fused,
            file_name=file_name,
            source_root=source_root,
        )
        write_json(run_dir / "phase5_fixed_static.json", fixed_static)
        write_json(run_dir / "phase5_fixed_rag.json", fixed_rag)
    else:
        fixed_static = fixed_rag = None
        tool_result, behavioral_result = skipped_verification()

    tool_result["behavioral"] = behavioral_result
    slither_verdict = tool_result.get("policy_verdict", "inconclusive")
    remote_fixed = _remote_detector(remote, "fixed")
    detector_still_flags = bool(remote_fixed) and remote_fixed.get("verdict") == "Vulnerable"
    overall_verdict = decide_overall_verdict(
        remote=remote,
        fused=fused,
        code=code,
        fixed_code=fixed_code,
        patch_applied=patch_applied,
        tool_result=tool_result,
        behavioral_result=behavioral_result,
        fixed_static=fixed_static,
        fixed_rag=fixed_rag,
    )

    v_info = remote.get("phases", {}).get("verifier", {}) if remote else {}
    verification = {
        "status": (
            "complete"
            if slither_verdict in {"passed", "failed"} and tool_result.get("status") in {"ok", "partial"}
            else "inconclusive" if not remote or remote.get("status") != "ok" else "partial"
        ),
        "original_detection": fused,
        # Detector re-score of the patched units (GPU repair loop); the report reads its threshold.
        "fixed_detection": remote_fixed or {},
        "tool_verification": tool_result,
        "behavioral_verification": behavioral_result,
        "verification_channels": build_verification_channels(
            tool_result,
            static_result=fixed_static,
            rag_result=fixed_rag,
            detector_fixed=remote_fixed,
        ),
        # Per-attempt details of the repair loop stay in server_result.json (phases.verifier.attempts).
        "attempts_count": v_info.get("attempts_count", 1) if remote else 0,
        "max_attempts": v_info.get("max_attempts", MAX_FIX_ATTEMPTS),
        "fixed_unit_results": v_info.get("fixed_unit_results", []),
        "detector_disagreement": detector_still_flags and overall_verdict.startswith("Patch accepted"),
        "overall_verdict": overall_verdict,
    }
    write_json(run_dir / "phase5_verification.json", verification)
    if remote:
        print(f"[{time.strftime('%H:%M:%S')}] [PHASE 5] Completed: Verification verdict: {overall_verdict}.", flush=True)
        print(f"[{time.strftime('%H:%M:%S')}] [PHASE 6] Generating multi-format audit reports (HTML, PDF, Markdown)...", flush=True)
        report_data = build_report_data(request, fused, remote, tool_result)
        apply_verification_to_report(report_data, verification)
        write_json(run_dir / "phase6_report_data.json", report_data)
        report_paths = write_reports(report_data, run_dir)
    else:
        # Local Phase 1 preview only - do not emit phase6 full reports before phase 2-5 run
        report_data = build_report_data(request, fused, None, tool_result)
        preview_md = render_markdown(report_data)
        preview_path = run_dir / "phase1_preview.md"
        write_text(preview_path, preview_md)
        report_paths = {"markdown_preview": str(preview_path)}

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "run_id": actual_run_id,
        "status": (
            "complete"
            if (
                remote.get("status") == "ok"
                and fused["status"] == "complete"
                and verification["status"] == "complete"
            )
            else "pending_remote" if not remote else "partial"
        ),
        "request_path": str(request_path),
        "run_dir": str(run_dir),
        "detection": {
            "verdict": fused["verdict"],
            "status": fused["status"],
            "score": fused["score"],
            "risk_level": report_data["risk_level"],  # highest severity among the reported findings
            # Same list as the report: after the GPU phases, findings the Advisor dismissed are excluded.
            "finding_count": len(report_data["findings"]),
            "dismissed_count": len(report_data.get("dismissed_findings", [])),
        },
        "verification": {
            "status": verification["status"],
            "overall_verdict": verification["overall_verdict"],
        },
        "reports": report_paths,
        "next_action": (
            "Waiting for the GPU phases (pipeline/gpu_pipeline.py, Phase 1C-5) on request.json."
            if not remote
            else "Review findings, generated patch, compiler output, and behavioral verification checks."
        ),
        "completed_at": utc_now(),
    }
    write_json(run_dir / "manifest.json", manifest)
    if remote:
        print(f"[{time.strftime('%H:%M:%S')}] [PHASE 6] Completed: Audit reports compiled.", flush=True)
        print(f"[{time.strftime('%H:%M:%S')}] Run completed successfully.", flush=True)
    return manifest


# Nhiệm vụ: khai báo tham số CLI cho `python -m E2E.run <file.sol>` (--run-id, --detector-threshold,
# --max-fix-attempts, --solidity-root).
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit one Solidity contract (Phase 0 -> 6) through the GPU server.")
    parser.add_argument("code_file", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument(
        "--detector-threshold",
        type=float,
        default=None,
        help=f"Vulnerability probability threshold. Default is auto from threshold_config.json ({DEFAULT_DETECTOR_THRESHOLD}).",
    )
    parser.add_argument(
        "--max-fix-attempts",
        type=int,
        default=MAX_FIX_ATTEMPTS,
        help="Maximum repair attempts (1-3) of the Fixer -> Verifier loop.",
    )
    parser.add_argument(
        "--solidity-root",
        type=Path,
        help="Root containing imported .sol files; defaults to the input file directory.",
    )
    return parser.parse_args()


# Nhiệm vụ: điểm vào CLI. Đọc file .sol, rồi:
#   1) run_pipeline lượt 1 (local: Phase 0, 1A, 1B) -> request.json
#   2) run_server_gpu_pipeline(request): Phase 1C -> 5, vòng sửa <= 3 lần. Hàm này chạy NGAY TRÊN MÁY NÀY
#      (cùng tiến trình); chỉ các lời gọi model đi qua HTTP tới GPU server (/api/v1/detect, /api/v1/generate).
#   3) run_pipeline lượt 2 với kết quả GPU -> fusion, kiểm chứng cuối, báo cáo.
# Mục đích: web app cũng gọi lại đúng lệnh CLI này trong một subprocess.
def main() -> None:
    args = parse_args()
    code = args.code_file.read_text(encoding="utf-8")

    common = dict(
        file_name=args.code_file.name,
        runs_dir=args.runs_dir,
        detector_threshold=args.detector_threshold,
        source_root=args.solidity_root or args.code_file.parent,
        max_fix_attempts=args.max_fix_attempts,
    )
    # Pass 1: Phase 0, 1A, 1B on this machine -> request.json.
    manifest = run_pipeline(code, run_id=args.run_id, **common)

    # Phases 1C-5: runs in this process; only the model calls go to the GPU server over HTTP.
    # The GPU pipeline owns the single repair loop
    # (compiler + detector + static/RAG feedback, at most MAX_FIX_ATTEMPTS attempts).
    from E2E.pipeline.gpu_pipeline import run_server_gpu_pipeline

    request_obj = read_json(manifest["request_path"])
    server_result = run_server_gpu_pipeline(request_obj)

    # Pass 2: fusion with the detector, Phase 5 verification, Phase 6 report.
    manifest = run_pipeline(
        code,
        run_id=manifest["run_id"],
        remote_result=server_result,
        local_phase1=request_obj.get("local_phase1"),
        **common,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
