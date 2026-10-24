from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from E2E.config import (
    DEFAULT_DETECTOR_THRESHOLD,
    DETECTOR_ADAPTER_DIR,
    PIPELINE_VERSION,
    RUNS_DIR,
    SCHEMA_VERSION,
)
from E2E.local.behavior_verifier import verify_patch_behavior
from E2E.local.rag_runner import run_rag
from E2E.local.report import apply_verification_to_report, build_report_data, render_markdown, write_reports
from E2E.local.static_runner import run_static
from E2E.local.tool_verifier import verify_security_tool_matrix
from E2E.shared.function_parser import parse_code_units
from E2E.shared.fusion import fuse_detection
from E2E.shared.io_utils import read_json, sha256_text, utc_now, write_json, write_text


MAX_FIX_ATTEMPTS = 3


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
        res = dict(detector[target])
        if "status" not in res:
            res["status"] = detector.get("status", "ok")
        return res
    if target == "original" and detector and isinstance(detector, dict):
        res = dict(detector)
        if "status" not in res:
            res["status"] = "ok"
        return res
    return None


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


def _verification_passed(result: dict[str, Any] | None) -> bool:
    if not result or result.get("status") != "ok":
        return False
    verdict = result.get("verdict") or result.get("final_verdict")
    return verdict == "Safe"


def _compact_feedback_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "finding_id",
        "swc",
        "unit_id",
        "title",
        "severity",
        "confidence",
        "line",
        "start_line",
        "end_line",
        "matched_text",
        "description",
        "remediation",
        "evidence",
        "notes",
    )
    return [
        {key: finding[key] for key in fields if key in finding}
        for finding in findings
    ]


def _llm_vulnerability_reason(
    detector: dict[str, Any] | None,
    verifier: dict[str, Any] | None,
) -> str:
    detector = detector or {}
    verifier = verifier or {}
    for value in (
        detector.get("vulnerability_reason"),
        verifier.get("vulnerability_reason"),
        verifier.get("reason"),
    ):
        if str(value or "").strip():
            return str(value).strip()

    evidence = [
        str(item.get("evidence") or "").strip()
        for item in verifier.get("verifications", [])
        if str(item.get("status") or "").lower() not in {"fixed", "safe"}
        and str(item.get("evidence") or "").strip()
    ]
    if evidence:
        return " ".join(evidence)

    vulnerable_units = [
        (
            str(item.get("unit_id") or "fixed source"),
            float(item.get("vulnerability_probability") or 0.0),
        )
        for item in detector.get("unit_results", [])
        if item.get("label") == "Vulnerable"
    ]
    if vulnerable_units:
        details = ", ".join(f"{unit_id} ({probability:.3f})" for unit_id, probability in vulnerable_units)
        return f"The neural detector still flags: {details}."
    return str(detector.get("error") or "The neural detector verification did not return a Safe verdict.")


def build_repair_feedback(
    fixed_static: dict[str, Any],
    fixed_rag: dict[str, Any],
    detector_fixed: dict[str, Any] | None,
    verifier: dict[str, Any] | None,
    *,
    attempt: int,
    max_attempts: int = MAX_FIX_ATTEMPTS,
) -> dict[str, Any]:
    static_passed = _verification_passed(fixed_static)
    rag_passed = _verification_passed(fixed_rag)
    llm_passed = _verification_passed(detector_fixed)
    return {
        "attempt": attempt,
        "max_attempts": min(max(1, max_attempts), MAX_FIX_ATTEMPTS),
        "passed": static_passed and rag_passed and llm_passed,
        "components": {
            "static": {
                "passed": static_passed,
                "status": fixed_static.get("status", "unknown"),
                "verdict": fixed_static.get("verdict", "Unknown"),
                "findings": _compact_feedback_findings(fixed_static.get("findings", [])),
                "error": fixed_static.get("error"),
            },
            "rag": {
                "passed": rag_passed,
                "status": fixed_rag.get("status", "unknown"),
                "verdict": fixed_rag.get("verdict")
                or fixed_rag.get("final_verdict")
                or "Unknown",
                "findings": _compact_feedback_findings(fixed_rag.get("findings", [])),
                "error": fixed_rag.get("error"),
            },
            "llm": {
                "passed": llm_passed,
                "status": (detector_fixed or {}).get("status", "missing"),
                "verdict": (detector_fixed or {}).get("verdict", "Unknown"),
                "reason": (
                    ""
                    if llm_passed
                    else _llm_vulnerability_reason(detector_fixed, verifier)
                ),
                "unit_results": (detector_fixed or {}).get("unit_results", []),
                "error": (detector_fixed or {}).get("error"),
            },
        },
    }


def _compact_tool_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "tool",
        "name",
        "check",
        "category",
        "severity",
        "line",
        "message",
        "swc",
    )
    return [
        {key: finding[key] for key in fields if key in finding}
        for finding in findings
    ]


def build_tool_repair_feedback(
    tool_result: dict[str, Any],
    *,
    attempt: int,
    max_attempts: int = MAX_FIX_ATTEMPTS,
    static_result: dict[str, Any] | None = None,
    rag_result: dict[str, Any] | None = None,
    detector_fixed: dict[str, Any] | None = None,
    verifier: dict[str, Any] | None = None,
) -> dict[str, Any]:
    components: dict[str, Any] = {}
    for tool_name, result in tool_result.get("tools", {}).items():
        residual = result.get("residual_target_findings", [])
        introduced = result.get("introduced_findings", [])
        findings = _compact_tool_findings([*residual, *introduced])
        verdict = str(result.get("verdict") or "ERROR")
        components[tool_name] = {
            "passed": verdict in {"PASS", "WARN"},
            "status": result.get("run_status", "unknown"),
            "verdict": verdict,
            "applicable": bool(result.get("applicable", True)),
            "findings": findings,
            "residual_target_findings": _compact_tool_findings(residual),
            "introduced_findings": _compact_tool_findings(introduced),
            "reason": result.get("detail"),
            "error": result.get("error"),
        }
    channel_passed = True
    if any([static_result, rag_result, detector_fixed, verifier]):
        channel_feedback = build_repair_feedback(
            static_result or {},
            rag_result or {},
            detector_fixed,
            verifier,
            attempt=attempt,
            max_attempts=max_attempts,
        )
        for ch_name, ch_data in channel_feedback.get("components", {}).items():
            components[ch_name] = ch_data
        channel_passed = channel_feedback.get("passed", True)

    compile_info = tool_result.get("compile", {})
    compile_status = compile_info.get("status", "unknown")
    compile_passed = compile_status == "passed"
    compile_error = (
        compile_info.get("stderr")
        or compile_info.get("error")
        or compile_info.get("reason")
        or ""
    )
    components["compile"] = {
        "passed": compile_passed,
        "status": compile_status,
        "verdict": "PASS" if compile_passed else "FAIL",
        "applicable": True,
        "findings": [],
        "residual_target_findings": [],
        "introduced_findings": [],
        "reason": "Compilation succeeded" if compile_passed else "Compilation failed",
        "error": compile_error,
    }
    failed_tools = [
        name
        for name, component in components.items()
        if component.get("verdict") == "FAIL"
    ]
    errored_tools = [
        name
        for name, component in components.items()
        if component.get("verdict") == "ERROR"
    ]
    return {
        "attempt": attempt,
        "max_attempts": min(max(1, max_attempts), MAX_FIX_ATTEMPTS),
        "passed": tool_result.get("policy_verdict") == "passed" and compile_passed,
        "channel_passed": channel_passed,
        "policy_verdict": tool_result.get("policy_verdict", "inconclusive"),
        "policy_reason": tool_result.get("policy_reason", ""),
        "compile_passed": compile_passed,
        "compile_status": compile_status,
        "compiler_error": compile_error,
        "compile": compile_info,
        "failed_tools": failed_tools,
        "errored_tools": errored_tools,
        "summary": tool_result.get("summary", {}),
        "components": components,
    }


def _aggregate_remote_attempts(
    original_request: dict[str, Any],
    remotes: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    if not remotes:
        return {}
    first = remotes[0]
    final = copy.deepcopy(remotes[-1])
    first_phases = first.get("phases", {})
    final_phases = final.setdefault("phases", {})
    first_detector = first_phases.get("detector", {})
    final_detector = final_phases.get("detector", {})
    final_phases["detector"] = {
        "status": final_detector.get("status", first_detector.get("status", "pending")),
        "original": first_detector.get("original"),
        "fixed": final_detector.get("fixed"),
    }
    for phase_name in ("advisor", "assessor"):
        if phase_name in first_phases:
            final_phases[phase_name] = first_phases[phase_name]

    cumulative_outcomes = []
    cumulative_patches = []
    for attempt_number, remote in enumerate(remotes, 1):
        fixer = remote.get("phases", {}).get("fixer", {})
        cumulative_outcomes.extend(
            {**item, "attempt": attempt_number}
            for item in fixer.get("patch_outcomes", [])
        )
        cumulative_patches.extend(
            {**item, "attempt": attempt_number}
            for item in fixer.get("patches", [])
        )
    final_fixer = final_phases.setdefault("fixer", {})
    final_fixer["patch_outcomes"] = cumulative_outcomes
    final_fixer["patches"] = cumulative_patches
    final_fixer["repair_attempt_count"] = len(attempts)
    final["run_id"] = original_request["run_id"]
    final["source_sha256"] = original_request["source"]["sha256"]
    final["repair_attempts"] = attempts
    final["errors"] = [
        str(error)
        for remote in remotes
        for error in remote.get("errors", [])
    ]
    final.setdefault("timings", {})["repair_attempts"] = len(attempts)
    final["timings"]["total_ms"] = sum(
        int(remote.get("timings", {}).get("total_ms") or 0)
        for remote in remotes
    )
    return final


def run_kaggle_repair_loop(
    request_path: str | Path,
    run_dir: str | Path,
    client: Any,
    *,
    max_attempts: int = MAX_FIX_ATTEMPTS,
    source_root: str | Path | None = None,
    tools: list[str] | None = None,
    use_smartbugs: bool = True,
) -> dict[str, Any]:
    limit = min(max(1, max_attempts), MAX_FIX_ATTEMPTS)
    original_request = read_json(request_path)
    current_request = copy.deepcopy(original_request)
    current_request_path = Path(request_path)
    attempt_root = Path(run_dir) / "repair_attempts"
    remotes: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    for attempt_number in range(1, limit + 1):
        attempt_dir = attempt_root / f"attempt-{attempt_number}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        if attempt_number > 1:
            current_request_path = write_json(attempt_dir / "request.json", current_request)
        existing_result = attempt_dir / "kaggle_result.json"
        if existing_result.exists():
            print(f"[Orchestrator] Found existing Kaggle result at {existing_result}. Loading without re-pushing...")
            remote = read_json(existing_result)
            remote_job = {
                "result": remote,
                "push_output": "(resumed/cached)",
                "status_output": "cached",
                "result_path": str(existing_result),
            }
        elif (attempt_dir / "kaggle-kernel-build").exists():
            print(f"[Orchestrator] Found existing kernel build in {attempt_dir}. Resuming wait for Kaggle run...")
            remote_job = client.wait_and_download(current_request_path, attempt_dir)
            remote = remote_job["result"]
        else:
            remote_job = client.submit_and_wait(current_request_path, attempt_dir)
            remote = remote_job["result"]
        validate_remote_result(current_request, remote)
        remotes.append(remote)

        fixed_code = (
            remote.get("phases", {}).get("fixer", {}).get("fixed_code")
            or current_request["source"]["code"]
        )
        target_findings = (
            current_request.get("local_phase1", {})
            .get("provisional_fusion", {})
            .get("findings", [])
        )
        tool_result = safe_run(
            "tool_verification",
            lambda: verify_security_tool_matrix(
                current_request["source"]["code"],
                fixed_code,
                file_name=current_request["source"]["file_name"],
                source_root=source_root,
                target_findings=target_findings,
                tools=tools,
                use_smartbugs=use_smartbugs,
            ),
        )
        feedback = build_tool_repair_feedback(
            tool_result,
            attempt=attempt_number,
            max_attempts=limit,
            static_result=current_request.get("local_phase1", {}).get("static"),
            rag_result=current_request.get("local_phase1", {}).get("rag"),
            detector_fixed=remote.get("phases", {}).get("detector", {}).get("fixed"),
            verifier=remote.get("phases", {}).get("verifier"),
        )
        attempt_result = {
            "attempt": attempt_number,
            "input_sha256": current_request["source"]["sha256"],
            "output_sha256": sha256_text(fixed_code),
            **feedback,
        }
        attempts.append(attempt_result)
        write_json(attempt_dir / "verification_feedback.json", attempt_result)

        # Retry only if the candidate fixed code failed compilation
        compile_status = tool_result.get("compile", {}).get("status")
        if compile_status != "passed" and attempt_number < limit:
            current_request = copy.deepcopy(original_request)
            current_request["repair_feedback"] = feedback
            current_request["repair"] = {
                "attempt": attempt_number + 1,
                "max_attempts": limit,
            }
            continue

        break

    return {
        "result": _aggregate_remote_attempts(original_request, remotes, attempts),
        "attempts": attempts,
    }


def run_pipeline(
    code: str,
    *,
    file_name: str = "Contract.sol",
    run_id: str | None = None,
    runs_dir: str | Path = RUNS_DIR,
    remote_result: dict[str, Any] | None = None,
    execution_mode: str = "full",
    detector_threshold: float | None = None,
    source_root: str | Path | None = None,
    tools: list[str] | None = None,
    use_smartbugs: bool = True,
    max_new_tokens: int = 900,
    max_llm_retries: int = 1,
    skip_reporter_llm: bool = True,
    skip_detector_explanation: bool = True,
    deterministic_advisor_assessor: bool = False,
) -> dict[str, Any]:
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
            "mode": execution_mode,
            "detector_threshold": resolve_detector_threshold(detector_threshold),
            "repair": {
                "attempt": 1,
                "max_attempts": MAX_FIX_ATTEMPTS,
            },
            "generation": {
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
                "max_retries": max_llm_retries,
                "skip_reporter_llm": skip_reporter_llm,
                "skip_detector_explanation": skip_detector_explanation,
                "deterministic_advisor_assessor": deterministic_advisor_assessor,
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
    if remote:
        print(f"[{time.strftime('%H:%M:%S')}] [PHASE 1 FUSED] Completed: Multi-modal fusion decision is {fused.get('verdict')} (score: {fused.get('score')}).", flush=True)

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
    patch_outcomes = remote.get("phases", {}).get("fixer", {}).get("patch_outcomes", []) if remote else []
    patch_applied = any(item.get("status") == "applied" for item in patch_outcomes)

    if remote and remote.get("status") in {"ok", "partial"} and fixed_code and fixed_code != code:
        print(f"[{time.strftime('%H:%M:%S')}] [PHASE 5] Verifying patch integrity & compiler behavioral matrix...", flush=True)
        tool_result = safe_run(
            "tool_verification",
            lambda: verify_security_tool_matrix(
                code,
                fixed_code,
                file_name=file_name,
                source_root=source_root,
                target_findings=fused.get("findings", []),
                results_dir=run_dir / "security_tool_matrix",
                tools=tools,
                use_smartbugs=use_smartbugs,
            ),
        )
        behavioral_result = safe_run(
            "behavioral_verification",
            lambda: verify_patch_behavior(
                code,
                fixed_code,
                patch_outcomes,
                tool_result,
            ),
        )
    else:
        tool_result = {
            "component": "tool_verification",
            "status": "inconclusive",
            "policy_verdict": "inconclusive",
            "policy_reason": "Patch execution incomplete or remote runner did not finish.",
            "compile": {"status": "skipped", "reason": "remote runner incomplete"},
            "smartbugs": {"status": "skipped", "reason": "remote runner incomplete"},
            "tools": {},
            "summary": {"pass": 0, "fail": 0, "warn": 0, "error": 0},
            "slither": {"status": "skipped", "findings": [], "verdict": "Not run"},
        }
        behavioral_result = {
            "status": "inconclusive",
            "findings": [],
            "checks": ["Patch execution incomplete: behavioral verification skipped."],
        }

    tool_result["behavioral"] = behavioral_result
    compile_status = tool_result.get("compile", {}).get("status")
    matrix_verdict = tool_result.get("policy_verdict", "inconclusive")

    if not remote or remote.get("status") in {"pending", "crashed", "awaiting_kaggle"}:
        overall_verdict = "Inconclusive: patch execution incomplete"
    elif fused.get("verdict") == "Safe" and not patch_applied and fixed_code == code:
        overall_verdict = "No patch required: original detection reported Safe"
    elif behavioral_result.get("status") == "failed":
        overall_verdict = "Patch rejected: behavioral regression detected"
    elif compile_status in {"failed", "unavailable"}:
        overall_verdict = "Patch rejected: fixed code does not compile"
    elif not patch_applied or fixed_code == code:
        overall_verdict = "Patch rejected: no effective patch was applied"
    elif matrix_verdict == "failed":
        overall_verdict = "Patch rejected: security tool matrix found residual or introduced vulnerabilities"
    elif matrix_verdict == "passed":
        overall_verdict = "Patch accepted by security tool matrix"
    else:
        overall_verdict = "Inconclusive: security tool matrix did not complete enough checks"

    # Fixed detection & risk delta: KHÔNG bịa điểm số 0.0/0.8 (Task A1)
    remote_fixed = remote.get("phases", {}).get("detector", {}).get("fixed") if remote else None
    if remote_fixed and remote_fixed.get("score") is not None:
        fixed_detection = remote_fixed
        fixed_score = float(remote_fixed["score"])
        risk_delta = round(float(fused.get("score") or 0.0) - fixed_score, 6)
    elif matrix_verdict == "passed" and behavioral_result.get("status") == "passed":
        fixed_detection = {
            "component": "security_tool_matrix",
            "status": "ok",
            "verdict": "Safe",
            "score": 0.0,
            "risk_level": "None",
            "policy_reason": tool_result.get("policy_reason", ""),
            "summary": tool_result.get("summary", {}),
        }
        risk_delta = round(float(fused.get("score") or 0.0), 6)
    else:
        fixed_detection = {
            "component": "verification",
            "status": "inconclusive",
            "verdict": overall_verdict,
            "score": None,
            "risk_level": "Unknown",
            "policy_reason": "Verification inconclusive or incomplete.",
            "summary": {},
        }
        risk_delta = None

    v_info = remote.get("phases", {}).get("verifier", {}) if remote else {}
    final_feedback = build_tool_repair_feedback(
        tool_result,
        attempt=len(remote.get("repair_attempts", [])) if remote else 1,
        max_attempts=MAX_FIX_ATTEMPTS,
        static_result=request.get("local_phase1", {}).get("static"),
        rag_result=request.get("local_phase1", {}).get("rag"),
        detector_fixed=remote_fixed,
        verifier=v_info,
    )
    verification = {
        "status": (
            "complete"
            if matrix_verdict in {"passed", "failed"} and tool_result.get("status") in {"ok", "partial"}
            else "inconclusive" if not remote or remote.get("status") != "ok" else "partial"
        ),
        "verification_model": v_info.get("verification_model", "Multi-Tier AST & Security Tool Matrix"),
        "original_detection": fused,
        "fixed_detection": fixed_detection,
        "tool_verification": tool_result,
        "behavioral_verification": behavioral_result,
        "verifications": v_info.get("verifications", []),
        "verification_channels": final_feedback.get("components", {}),
        "repair_attempts": remote.get("repair_attempts", []) or v_info.get("attempts", []),
        "attempts_count": v_info.get("attempts_count", len(remote.get("repair_attempts", [])) or 1) if remote else 0,
        "max_attempts": v_info.get("max_attempts", 3),
        "attempts": v_info.get("attempts", []),
        "fixed_unit_results": v_info.get("fixed_unit_results", []),
        "vulnerability_probability": v_info.get("vulnerability_probability"),
        "risk_delta": risk_delta,
        "overall_verdict": overall_verdict,
        "notes": f"Phase 5 verification finished with verdict: {overall_verdict}.",
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
                and verification["status"] in {"complete", "ok"}
            )
            else "pending_remote" if not remote else "partial"
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
            "Run server_gpu_runner or local_llm_runner with request.json to complete Phase 2-5."
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the self-contained smart-contract E2E audit.")
    parser.add_argument("code_file", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--remote-result", type=Path)
    parser.add_argument(
        "--backend",
        choices=("server_gpu", "local", "ollama", "local_fast", "gemini", "kaggle", "auto"),
        default="server_gpu",
        help="Execution backend for LLM phases. Default is server_gpu (Dedicated RTX 4070 Ti SUPER v13); kaggle, local, ollama, gemini.",
    )
    parser.add_argument(
        "--submit-kaggle",
        action="store_true",
        help="Push a private Kaggle job, wait, download result.json, and finish the report. This is the default unless another backend is selected.",
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
    parser.add_argument(
        "--detector-threshold",
        type=float,
        default=None,
        help=f"Vulnerability probability threshold. Default is auto from threshold_config.json ({DEFAULT_DETECTOR_THRESHOLD}).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=900,
        help="Maximum new tokens for each Kaggle CodeLlama generation call. Lower is faster.",
    )
    parser.add_argument(
        "--max-llm-retries",
        type=int,
        default=1,
        help="Maximum correction retries for Kaggle CodeLlama JSON/text generation.",
    )
    parser.add_argument(
        "--llm-reporter",
        action="store_true",
        help="Use CodeLlama for the reporter narrative. Default skips this slow non-critical generation.",
    )
    parser.add_argument(
        "--llm-detector-explanation",
        action="store_true",
        help="Use CodeLlama to explain a vulnerable fixed-code detector result. Default uses the detector signal directly.",
    )
    parser.add_argument(
        "--fast-agents",
        action="store_true",
        help="Use deterministic advisor/assessor fallbacks and reserve CodeLlama generation for fixer/verifier.",
    )
    parser.add_argument("--poll-seconds", type=int, default=8)
    parser.add_argument(
        "--max-fix-attempts",
        type=int,
        default=MAX_FIX_ATTEMPTS,
        help="Deprecated compatibility option; the pipeline now runs exactly one fix attempt.",
    )
    parser.add_argument(
        "--tools",
        nargs="*",
        default=None,
        help="List of security tools to run in Phase 5 (e.g. slither mythril smartcheck). Default: slither, mythril, smartcheck for full runs, native slither for local debug.",
    )
    parser.add_argument(
        "--full-matrix",
        action="store_true",
        help="Run all 3 verified security tools (slither, mythril, smartcheck) in Phase 5 verification.",
    )
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

    local_debug_backend = args.backend in {"server_gpu", "local", "ollama", "gemini", "local_fast", "auto"} and not args.submit_kaggle
    selected_tools = args.tools
    if selected_tools is None and local_debug_backend and not args.full_matrix:
        selected_tools = ["slither"]
    use_smartbugs = (
        args.full_matrix
        or selected_tools is None
        or bool(args.tools and any(t != "slither" for t in args.tools))
    )

    manifest = run_pipeline(
        code,
        file_name=args.code_file.name,
        run_id=args.run_id,
        runs_dir=args.runs_dir,
        remote_result=remote,
        execution_mode=args.mode,
        detector_threshold=args.detector_threshold,
        source_root=args.solidity_root or args.code_file.parent,
        tools=selected_tools,
        use_smartbugs=use_smartbugs,
        max_new_tokens=args.max_new_tokens,
        max_llm_retries=args.max_llm_retries,
        skip_reporter_llm=not args.llm_reporter,
        skip_detector_explanation=not args.llm_detector_explanation,
        deterministic_advisor_assessor=args.fast_agents,
    )

    if args.backend == "server_gpu" and not remote and not args.submit_kaggle:
        from E2E.local.server_gpu_runner import run_server_gpu_pipeline

        limit = min(max(1, args.max_fix_attempts), MAX_FIX_ATTEMPTS)
        request_obj = read_json(manifest["request_path"])
        for attempt in range(1, limit + 1):
            if attempt > 1:
                request_obj["repair"] = {"attempt": attempt, "max_attempts": limit}
            server_result = run_server_gpu_pipeline(request_obj)
            manifest = run_pipeline(
                code,
                file_name=args.code_file.name,
                run_id=manifest["run_id"],
                runs_dir=args.runs_dir,
                remote_result=server_result,
                execution_mode=args.mode,
                detector_threshold=args.detector_threshold,
                source_root=args.solidity_root or args.code_file.parent,
                tools=selected_tools,
                use_smartbugs=use_smartbugs,
            )
            v_verdict = manifest.get("verification", {}).get("overall_verdict", "")
            if not v_verdict.startswith("Patch rejected") or attempt >= limit:
                break
            run_dir = Path(manifest["run_dir"])
            v_path = run_dir / "phase5_verification.json"
            v_data = read_json(v_path) if v_path.exists() else {}
            tool_data = v_data.get("tool_verification", {})
            feedback = build_tool_repair_feedback(
                tool_data,
                attempt=attempt,
                max_attempts=limit,
                static_result=request_obj.get("local_phase1", {}).get("static"),
                rag_result=request_obj.get("local_phase1", {}).get("rag"),
                detector_fixed=manifest.get("verification", {}).get("fixed_detection"),
                verifier=v_data.get("verifier"),
            )
            request_obj["repair_feedback"] = feedback
    elif args.backend in {"local", "ollama", "gemini", "local_fast", "auto"} and not remote and not args.submit_kaggle:
        from E2E.local.fast_agent_runner import run_fast_local_pipeline

        request_obj = read_json(manifest["request_path"])
        fast_result = run_fast_local_pipeline(request_obj, backend=args.backend)
        manifest = run_pipeline(
            code,
            file_name=args.code_file.name,
            run_id=manifest["run_id"],
            runs_dir=args.runs_dir,
            remote_result=fast_result,
            execution_mode=args.mode,
            detector_threshold=args.detector_threshold,
            source_root=args.solidity_root or args.code_file.parent,
            tools=selected_tools,
            use_smartbugs=use_smartbugs,
        )
    elif (args.submit_kaggle or args.backend == "kaggle") and not remote:
        from E2E.local.kaggle_job_client import KaggleJobClient

        client = KaggleJobClient(kernel_id=args.kernel_id, poll_seconds=args.poll_seconds)
        repair_result = run_kaggle_repair_loop(
            manifest["request_path"],
            manifest["run_dir"],
            client,
            max_attempts=args.max_fix_attempts,
            source_root=args.solidity_root or args.code_file.parent,
            tools=selected_tools,
            use_smartbugs=use_smartbugs,
        )
        manifest = run_pipeline(
            code,
            file_name=args.code_file.name,
            run_id=manifest["run_id"],
            runs_dir=args.runs_dir,
            remote_result=repair_result["result"],
            execution_mode=args.mode,
            detector_threshold=args.detector_threshold,
            source_root=args.solidity_root or args.code_file.parent,
            tools=selected_tools,
            use_smartbugs=use_smartbugs,
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
