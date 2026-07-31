from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from E2E.config import PIPELINE_VERSION, RUNS_DIR, SCHEMA_VERSION
from E2E.local.behavior_verifier import verify_patch_behavior
from E2E.local.rag_runner import run_rag
from E2E.local.report import apply_verification_to_report, build_report_data, write_reports
from E2E.local.static_runner import run_static
from E2E.local.tool_verifier import verify_tools
from E2E.shared.function_parser import parse_code_units
from E2E.shared.fusion import fuse_detection
from E2E.shared.io_utils import read_json, sha256_text, utc_now, write_json


def safe_run(component: str, action: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = action()
        result.setdefault("component", component)
        result.setdefault("status", "ok")
    except Exception as error:
        result = {
            "component": component,
            "status": "error",
            "score": 0.0,
            "verdict": "Error",
            "error": f"{type(error).__name__}: {error}",
        }
    result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return result


def make_run_id(file_name: str, code: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "-", Path(file_name).stem).strip("-") or "contract"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{stem}-{sha256_text(code)[:8]}"


def validated_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError(
            "run_id must contain only letters, digits, dot, underscore, or hyphen "
            "and must not exceed 128 characters"
        )
    return value


def _remote_detector(remote: dict[str, Any], target: str) -> dict[str, Any] | None:
    detector = remote.get("phases", {}).get("detector", {})
    if target in detector and isinstance(detector[target], dict):
        return detector[target]
    return detector if target == "original" and detector else None


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


def run_pipeline(
    code: str,
    *,
    file_name: str = "Contract.sol",
    run_id: str | None = None,
    runs_dir: str | Path = RUNS_DIR,
    remote_result: dict[str, Any] | None = None,
    execution_mode: str = "full",
    detector_threshold: float = 0.25,
    source_root: str | Path | None = None,
) -> dict[str, Any]:
    actual_run_id = validated_run_id(run_id or make_run_id(file_name, code))
    run_dir = Path(runs_dir) / actual_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
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

    static_result = safe_run("static", lambda: run_static(code, units))
    write_json(run_dir / "phase1a_static.json", static_result)
    rag_result = safe_run("rag", lambda: run_rag(code, units))
    write_json(run_dir / "phase1b_rag.json", rag_result)
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
            "mode": execution_mode,
            "detector_threshold": detector_threshold,
            "generation": {
                "temperature": 0.0,
                "max_new_tokens": 1200,
                "max_retries": 2,
            },
        },
    }
    request_path = write_json(run_dir / "request.json", request)

    remote = remote_result or {}
    if remote:
        validate_remote_result(request, remote)
        write_json(run_dir / "kaggle_result.json", remote)
    detector_original = _remote_detector(remote, "original")
    fused = fuse_detection(static_result, rag_result, detector_original)
    write_json(run_dir / "phase1_fused.json", fused)

    for number, phase_name in (
        (2, "advisor"),
        (3, "assessor"),
        (4, "fixer"),
    ):
        payload = remote.get("phases", {}).get(phase_name, {
            "status": "pending",
            "reason": "Kaggle result has not been supplied",
        })
        write_json(run_dir / f"phase{number}_{phase_name}.json", payload)

    fixed_code = remote.get("phases", {}).get("fixer", {}).get("fixed_code") or code
    fixed_units = parse_code_units(fixed_code)
    fixed_static = safe_run("static_fixed", lambda: run_static(fixed_code, fixed_units))
    fixed_rag = safe_run("rag_fixed", lambda: run_rag(fixed_code, fixed_units))
    detector_fixed = _remote_detector(remote, "fixed")
    fixed_fusion = fuse_detection(fixed_static, fixed_rag, detector_fixed)
    tool_result = safe_run(
        "tool_verification",
        lambda: verify_tools(
            fixed_code,
            file_name=file_name,
            source_root=source_root,
        ),
    )
    remote_verifier = remote.get("phases", {}).get("verifier", {})
    patch_outcomes = remote.get("phases", {}).get("fixer", {}).get("patch_outcomes", [])
    patch_applied = any(item.get("status") == "applied" for item in patch_outcomes)
    behavioral_result = verify_patch_behavior(
        code,
        fixed_code,
        patch_outcomes,
        tool_result,
    )
    tool_result["behavioral"] = behavioral_result
    compile_status = tool_result.get("compile", {}).get("status")
    local_remaining = bool(
        fixed_static.get("findings")
        or fixed_rag.get("findings")
    )
    llm_remaining = bool(
        detector_fixed
        and detector_fixed.get("status") == "ok"
        and detector_fixed.get("verdict") == "Vulnerable"
    )
    if not remote:
        overall_verdict = "Patch verification pending Kaggle result"
    elif (
        fused["verdict"] == "Safe"
        and fixed_fusion["verdict"] == "Safe"
        and not patch_applied
    ):
        overall_verdict = "No patch required: all detectors report Safe"
    elif behavioral_result["status"] == "failed":
        overall_verdict = "Patch rejected: behavioral regression detected"
    elif compile_status == "failed":
        overall_verdict = "Patch rejected: fixed code does not compile"
    elif local_remaining:
        overall_verdict = "Patch rejected: local static/RAG findings remain after redetection"
    elif not patch_applied or fixed_code == code:
        overall_verdict = "Patch rejected: no effective patch was applied"
    elif llm_remaining:
        overall_verdict = (
            "Patch inconclusive: CodeLlama detector remains positive while deterministic checks pass"
        )
    elif fixed_fusion["verdict"] == "Safe" and compile_status == "passed":
        overall_verdict = (
            remote_verifier.get("overall_verdict")
            or "Patch accepted by automated checks"
        )
    else:
        overall_verdict = "Patch inconclusive: manual review required"
    verification = {
        "status": (
            "complete"
            if fixed_fusion["status"] == "complete" and tool_result.get("status") == "ok"
            else "partial"
        ),
        "original_detection": fused,
        "fixed_detection": fixed_fusion,
        "tool_verification": tool_result,
        "behavioral_verification": behavioral_result,
        "llm_verifier": remote_verifier,
        "risk_delta": round(float(fused["score"]) - float(fixed_fusion["score"]), 6),
        "overall_verdict": overall_verdict,
    }
    write_json(run_dir / "phase5_verification.json", verification)

    report_data = build_report_data(request, fused, remote, tool_result)
    apply_verification_to_report(report_data, verification)
    write_json(run_dir / "phase6_report_data.json", report_data)
    report_paths = write_reports(report_data, run_dir)
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
            else "awaiting_kaggle" if not remote else "partial"
        ),
        "request_path": str(request_path),
        "run_dir": str(run_dir),
        "detection": {
            "verdict": fused["verdict"],
            "status": fused["status"],
            "score": fused["score"],
            "risk_level": fused["risk_level"],
            "finding_count": len(fused["findings"]),
        },
        "verification": {
            "status": verification["status"],
            "overall_verdict": verification["overall_verdict"],
            "risk_delta": verification["risk_delta"],
        },
        "reports": report_paths,
        "next_action": (
            "Submit request.json to the private Kaggle CodeLlama notebook and rerun with --remote-result."
            if not remote
            else "Review findings, generated patch, compiler output, and Slither evidence."
        ),
        "completed_at": utc_now(),
    }
    write_json(run_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the self-contained smart-contract E2E audit.")
    parser.add_argument("code_file", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--remote-result", type=Path)
    parser.add_argument(
        "--submit-kaggle",
        action="store_true",
        help="Push a private Kaggle job, wait, download result.json, and finish the report.",
    )
    parser.add_argument(
        "--kernel-id",
        default="thanhphuocjr/codellama-e2e-smart-contract-audit",
    )
    parser.add_argument(
        "--mode",
        choices=("detector_only", "agents_only", "full"),
        default="full",
    )
    parser.add_argument("--detector-threshold", type=float, default=0.25)
    parser.add_argument(
        "--solidity-root",
        type=Path,
        help="Root containing imported .sol files; defaults to the input file directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    code = args.code_file.read_text(encoding="utf-8")
    remote = read_json(args.remote_result) if args.remote_result else None
    if args.submit_kaggle and args.remote_result:
        raise SystemExit("--submit-kaggle and --remote-result are mutually exclusive")
    manifest = run_pipeline(
        code,
        file_name=args.code_file.name,
        run_id=args.run_id,
        runs_dir=args.runs_dir,
        remote_result=remote,
        execution_mode=args.mode,
        detector_threshold=args.detector_threshold,
        source_root=args.solidity_root or args.code_file.parent,
    )
    if args.submit_kaggle:
        from E2E.local.kaggle_job_client import KaggleJobClient

        client = KaggleJobClient(kernel_id=args.kernel_id)
        remote_job = client.submit_and_wait(
            manifest["request_path"],
            manifest["run_dir"],
        )
        manifest = run_pipeline(
            code,
            file_name=args.code_file.name,
            run_id=manifest["run_id"],
            runs_dir=args.runs_dir,
            remote_result=remote_job["result"],
            execution_mode=args.mode,
            detector_threshold=args.detector_threshold,
            source_root=args.solidity_root or args.code_file.parent,
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
