"""
Phases 1C -> 5 of one contract, driven through the GPU server (Qwen2.5-Coder-14B v13).

Input: request.json written by the orchestrator (source, code units, Phase 1A/1B results).

  1C  detector scores every unit                      -> phase1c_detector/detector.py
  1F  per-finding fusion of static + RAG + detector   -> phase1_fusion/fusion.py
  2   Advisor: triage (drops false positives) + fix strategy -> phase2_advisor/
  3   Assessor: CVSS v3.1 + repair priority            -> phase3_assessor/
  4   Fixer: patch each target unit                     -> phase4_fixer/fixer.py
  5   check the candidate (compiler, detector, static/RAG); on failure feed the
      problems back to the Fixer, at most MAX_VERIFICATION_ATTEMPTS attempts
                                                        -> phase5_verifier/candidate_check.py

single_pass (batch runs, execution.repair.single_pass): one Fixer call per target unit,
no Phase 3 and no Phase 5; the model's patch is the result (scored by an external judge).
"""

from __future__ import annotations

import time
from typing import Any

from E2E.common.findings import finding_key
from E2E.common.gpu_client import check_server_health, get_server_url
from E2E.config import DEFAULT_DETECTOR_THRESHOLD
from E2E.phase0_parsing.code_units import as_code_unit
from E2E.phase1_fusion.fusion import fuse_findings
from E2E.phase1c_detector.detector import score_units_via_server, summarize_detector
from E2E.phase2_advisor.advisor import RepairKnowledgeBase
from E2E.phase2_advisor.advisor_phase import _generate_advisor_section
from E2E.phase3_assessor.assessor_phase import _generate_assessor_section
from E2E.phase4_fixer.fixer import _build_rag_knowledge_section, _feedback_from_request, _splice_unit, patch_unit
from E2E.phase4_fixer.heuristic_fallback import _apply_heuristic_security_patch
from E2E.phase5_verifier.candidate_check import _finding_pairs, _verification_feedback, verify_candidate


MAX_VERIFICATION_ATTEMPTS = 3


# In log có kèm giờ (web app đọc dòng "[PHASE x]" này để cập nhật tiến độ).
def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# [HÀM TRUNG TÂM PHẦN GPU] Chạy Phase 1C -> 5 cho một contract từ request.json (do orchestrator ghi).
#   1C: kiểm tra server sống (health) -> detector chấm từng unit (score_units_via_server + summarize_detector).
#   1F: fuse_findings -> finding confirmed / candidate; RepairKnowledgeBase gắn tri thức sửa lỗi theo SWC;
#       (tuỳ chọn) hoãn finding chỉ-detector có xác suất thấp (detector_only_min_prob).
#   2 : Advisor -> loại false positive khỏi tập cần sửa, sinh hướng sửa (suggestions).
#   3 : Assessor -> CVSS v3.1 (bỏ qua ở chế độ single_pass của batch).
#   4-5: VÒNG SỬA (tối đa 3 lần): Fixer vá từng hàm mục tiêu (patch_unit); nếu model không vá được gì thì
#       dùng vá heuristic; rồi verify_candidate kiểm tra compile + detector + static/RAG chạy lại.
#       Chưa đạt -> tạo feedback; lần sau giữ lại các bản vá đã đạt, chỉ vá lại các hàm còn lỗi TỪ MÃ GỐC.
#       Chọn attempt tốt nhất theo (compile được, ít hàm lỗi, nhiều hàm vá, ít thay đổi độ dài).
# Trả về server_result: phases {detector, advisor, assessor, fixer, verifier, reporter}, fixed_code,
# patch_outcomes, verdict PASS/WARN/UNVERIFIED/NOT_APPLICABLE, thời gian từng phần.
# single_pass (dùng cho batch vòng 1): mỗi hàm chỉ gọi Fixer 1 lần, không Phase 3 và không Phase 5.
# Gọi bởi: orchestrator.main, batch_audit, repair_round2.
def run_server_gpu_pipeline(
    request: dict[str, Any],
    *,
    server_url: str | None = None,
) -> dict[str, Any]:
    """Thực thi pipeline phase 1C-5 thông qua dịch vụ AI trên GPU Server.

    Vòng sửa lỗi duy nhất của backend server_gpu (tối đa 3 attempt). Mỗi attempt được
    kiểm tra bởi ba kênh: compiler (so với baseline), neural detector trên unit đã vá,
    và static + RAG chạy lại trên code đã vá.
    """
    started_at = time.perf_counter()
    server_url = get_server_url(server_url)
    run_id = request.get("run_id", "server-gpu-run")
    source_code = request.get("source", {}).get("code", "")
    unit_objs = [as_code_unit(unit) for unit in request.get("code_units", [])]
    units_by_id = {unit.unit_id: unit for unit in unit_objs}
    execution = request.get("execution", {})
    detector_threshold = float(execution.get("detector_threshold") or DEFAULT_DETECTOR_THRESHOLD)
    # single_pass: one fixer call per target unit, no Phase 5; the model's patch is the result.
    single_pass = bool((execution.get("repair") or {}).get("single_pass"))
    max_attempts = 1 if single_pass else min(
        max(1, int((execution.get("repair") or {}).get("max_attempts") or MAX_VERIFICATION_ATTEMPTS)),
        MAX_VERIFICATION_ATTEMPTS,
    )
    local_phase1 = request.get("local_phase1", {})
    static_p1 = local_phase1.get("static", {}) or request.get("static", {})
    rag_p1 = local_phase1.get("rag", {}) or request.get("rag", {})
    rag_context = rag_p1.get("retrieved_context", [])

    health = check_server_health(server_url)
    if health.get("status") != "ready":
        raise ConnectionError(
            f"GPU Server tại {server_url} chưa sẵn sàng (trạng thái: {health.get('status')}). "
            "Vui lòng đảm bảo bạn đã khởi chạy `bash run_api_server.sh` trên server."
        )

    # Phase 1C: neural detector on every unit with a body.
    _log(f"[PHASE 1C] Running Neural Detector v13 on GPU server ({len(unit_objs)} code unit(s))...")
    detector_started = time.perf_counter()
    unit_results = score_units_via_server(unit_objs, detector_threshold, server_url)
    detector_original = summarize_detector(unit_results, detector_threshold)
    detector_elapsed_ms = round((time.perf_counter() - detector_started) * 1000, 2)
    _log(
        f"[PHASE 1C] Completed: {len(unit_results)} unit(s) evaluated, status={detector_original['status']}, "
        f"max_prob={detector_original['vulnerability_probability']}."
    )

    # Phase 1 fused: per-finding fusion decides what reaches the agents.
    _log("[PHASE 1 FUSED] Synchronizing multi-modal detection signals...")
    findings, candidate_findings = fuse_findings(unit_results, static_p1, rag_p1, tau_agent=0.45)
    try:
        findings = RepairKnowledgeBase().enrich_findings(
            findings, code=source_code, code_units=request.get("code_units", [])
        )
    except Exception:
        pass
    # Units flagged only by the neural detector (no SWC from static/RAG) are optional repair
    # targets: `detector_only_min_prob` (execution.repair) raises the bar for them, and a value
    # above 1 keeps them out of repair entirely. They are still reported.
    detector_only_min = (execution.get("repair") or {}).get("detector_only_min_prob")
    detector_only_deferred: list[dict[str, Any]] = []
    if detector_only_min is not None:
        detector_only_deferred = [
            item for item in findings
            if item.get("swc") == "UNMAPPED" and float(item.get("p_detector") or 0.0) < float(detector_only_min)
        ]
        deferred_ids = {id(item) for item in detector_only_deferred}
        findings = [item for item in findings if id(item) not in deferred_ids]
    _log(
        f"[PHASE 1 FUSED] Completed: {len(findings)} confirmed finding(s), {len(candidate_findings)} candidate(s)"
        + (f", {len(detector_only_deferred)} detector-only finding(s) not repaired." if detector_only_deferred else ".")
    )

    # Phase 2: Advisor (triage + remediation). False positives are removed from the repair set.
    _log("[PHASE 2] Running Security Advisor agent (root cause & remediation analysis)...")
    advisor = _generate_advisor_section(findings, source_code, use_llm=True)
    dismissed_ids = set(advisor.get("false_positives", []))
    dismissed = [item for item in findings if finding_key(item) in dismissed_ids]
    findings = [item for item in findings if finding_key(item) not in dismissed_ids]
    _log(
        f"[PHASE 2] Completed: {len(advisor.get('suggestions', []))} suggestion(s), "
        f"{len(dismissed)} finding(s) dismissed as false positive."
    )

    # Phase 3: Assessor (CVSS only feeds the report, so single-pass batch runs skip it).
    if single_pass:
        _log("[PHASE 3] Skipped (single-pass mode).")
        assessor = {"status": "skipped", "assessments": []}
    else:
        _log("[PHASE 3] Running Risk Assessor agent (CVSS v3.1 scoring & priority ranking)...")
        assessor = _generate_assessor_section(
            findings,
            source_code,
            use_llm=True,
            advisor_suggestions=advisor.get("suggestions", []),
        )
        _log(f"[PHASE 3] Completed: {len(assessor.get('assessments', []))} item(s) assessed.")

    # Phase 4-5: repair loop.
    finding_units = {item.get("unit_id") for item in findings if item.get("unit_id")}
    target_units = [
        unit for unit in unit_objs
        if unit.unit_id in finding_units and unit.has_body and not unit.is_view_or_pure
    ]
    # Optional budget for very large contracts: keep the units with the strongest fused
    # evidence; the rest are reported as `skipped_targets`, never silently dropped.
    max_target_units = (execution.get("repair") or {}).get("max_target_units")
    skipped_targets: list[str] = []
    if max_target_units and len(target_units) > int(max_target_units):
        strength = {
            unit.unit_id: max(
                float(item.get("p_final") or 0.0) for item in findings if item.get("unit_id") == unit.unit_id
            )
            for unit in target_units
        }
        ranked = sorted(target_units, key=lambda unit: strength[unit.unit_id], reverse=True)
        target_units = ranked[: int(max_target_units)]
        skipped_targets = [unit.unit_id for unit in ranked[int(max_target_units):]]
        _log(f"[PHASE 4] Target budget: repairing {len(target_units)} unit(s), skipping {len(skipped_targets)}.")
    target_unit_ids = {unit.unit_id for unit in target_units}
    target_pairs = {pair for pair in _finding_pairs(findings) if pair[1] in target_unit_ids}
    known_pairs = _finding_pairs(
        [
            *static_p1.get("findings", []),
            *static_p1.get("candidates", []),
            *rag_p1.get("findings", []),
        ]
    )
    suggestions = advisor.get("suggestions", [])
    patch_candidates = list(findings) + suggestions
    first_swc = next((item.get("swc") for item in findings if item.get("swc") not in {None, "UNMAPPED"}), None)
    rag_section = _build_rag_knowledge_section(rag_context, target_swc=first_swc)
    feedback_note = _feedback_from_request(request.get("repair_feedback"))

    fixer_started = time.perf_counter()
    attempts: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    base_code = source_code
    carried_outcomes: list[dict[str, Any]] = []
    active_targets = list(target_units)

    for attempt in range(1, max_attempts + 1 if target_units else 1):
        label = f"Attempt {attempt}/{max_attempts}"
        _log(f"[PHASE 4] Running Fixer agent via Qwen2.5-Coder-14B ({label}, {len(active_targets)} unit(s))...")
        current_code = base_code
        outcomes = list(carried_outcomes)
        for unit in active_targets:
            current_code, outcome = patch_unit(
                current_code,
                unit,
                unit_findings=[item for item in findings if item.get("unit_id") == unit.unit_id],
                unit_suggestions=[item for item in suggestions if item.get("unit_id") == unit.unit_id],
                source_code=source_code,
                feedback_note=feedback_note,
                rag_section=rag_section,
                attempt=attempt,
                server_url=server_url,
                single_pass=single_pass,
            )
            outcomes.append(outcome)

        if (
            not single_pass
            and not any(item.get("status") == "applied" for item in outcomes)
            and base_code == source_code
        ):
            heuristic_code, heuristic_outcomes = _apply_heuristic_security_patch(
                source_code, patch_candidates, code_units=unit_objs
            )
            if heuristic_outcomes:
                current_code = heuristic_code
                outcomes.extend(heuristic_outcomes)

        patched_ids = {
            str(item.get("unit_id")) for item in outcomes
            if item.get("status") == "applied" and item.get("unit_id")
        }
        if single_pass:
            _log("[PHASE 5] Skipped (single-pass mode): the fixer output is taken as the final patch.")
            verification = {
                "passed": False,
                "compilation": {"state": "skipped", "failed": False},
                "unit_results": [],
                "detector_failing_units": [],
                "unverified_units": [],
                "residual_findings": [],
                "introduced_findings": [],
                "channel_errors": {},
                "failing_units": [],
                "compile_culprit_units": [],
                "max_vulnerability_probability": None,
            }
        else:
            _log(f"[PHASE 5] Verifying candidate ({label}): compiler + neural detector + static/RAG re-run...")
            verification = verify_candidate(
                source_code,
                current_code,
                patched_unit_ids=patched_ids,
                target_unit_ids=target_unit_ids,
                original_unit_results=unit_results,
                target_pairs=target_pairs,
                known_pairs=known_pairs,
                threshold=detector_threshold,
                server_url=server_url,
            )
        record = {
            "attempt": attempt,
            "max_attempts": max_attempts,
            "threshold": detector_threshold,
            "verdict": "Unverified" if single_pass else ("Safe" if verification["passed"] else "Vulnerable"),
            "code": current_code,
            "patch_outcomes": outcomes,
            **verification,
        }
        attempts.append(record)
        score = (
            0 if verification["compilation"]["failed"] else 1,
            -len(verification["failing_units"]),
            len(patched_ids),
            -abs(len(current_code) - len(source_code)),
        )
        if best is None or score > best["_score"]:
            best = {**record, "_score": score}
        _log(
            f"[PHASE 5] {label}: {'PASSED' if verification['passed'] else 'FAILED'} "
            f"(compile={verification['compilation']['state']}, failing={verification['failing_units']})."
        )
        if verification["passed"]:
            break
        if attempt == max_attempts:
            break

        record["feedback"] = _verification_feedback(attempt, max_attempts, verification)
        feedback_note = record["feedback"]
        failing = set(verification["failing_units"])
        failing.update(
            unit.unit_id for unit in active_targets
            if unit.unit_id not in patched_ids
        )
        compile_localized = not verification["compilation"]["failed"] or bool(
            set(verification["compile_culprit_units"]) & patched_ids
        )
        base_code, carried_outcomes, active_targets = source_code, [], list(target_units)
        if compile_localized and failing:
            # Next attempt = original source + the patches that already verified; the failing
            # units are re-patched from their original code (never on top of a failed patch).
            keep = [
                item for item in outcomes
                if item.get("status") == "applied" and item.get("unit_id") not in failing
            ]
            rebuilt = source_code
            try:
                for item in keep:
                    unit = units_by_id[item["unit_id"]]
                    rebuilt = _splice_unit(rebuilt, unit, item["patched_code"], str(item.get("reason") or ""))
            except (KeyError, ValueError):
                pass  # cannot rebuild (e.g. rule-based patch without unit code): restart fully
            else:
                base_code, carried_outcomes = rebuilt, keep
                active_targets = [unit for unit in target_units if unit.unit_id in failing]

    fixer_elapsed_ms = round((time.perf_counter() - fixer_started) * 1000, 2)

    if best is None:
        fixed_code, patch_outcomes, final_verification = source_code, [], None
        overall_verdict = "NOT_APPLICABLE"
    else:
        fixed_code = best["code"]
        patch_outcomes = best["patch_outcomes"]
        final_verification = best
        overall_verdict = "UNVERIFIED" if single_pass else ("PASS" if best["passed"] else "WARN")
    _log(f"[PHASE 5] Completed: verdict {overall_verdict} after {len(attempts)} attempt(s).")

    if final_verification is not None and not single_pass:
        fixed_detector = summarize_detector(
            [item for item in final_verification["unit_results"] if item["unit_id"] in target_unit_ids],
            detector_threshold,
        )
    else:
        fixed_detector = {
            "status": "not_run",
            "verdict": "Not run",
            "score": None,
            "vulnerability_probability": None,
            "threshold": detector_threshold,
            "unit_results": [],
            "reason": "Verification skipped (single-pass mode)." if single_pass and final_verification
            else "No function-level repair target.",
        }

    patched_ids = {
        str(item.get("unit_id")) for item in patch_outcomes
        if item.get("status") == "applied" and item.get("unit_id")
    }
    failing_final = set(final_verification["failing_units"]) if final_verification else set()
    verifier_items = []
    for item in findings:
        unit_id = item.get("unit_id")
        if unit_id in failing_final:
            status = "FLAGGED"
        elif unit_id in patched_ids:
            status = "SKIPPED" if single_pass else "PASS"
        else:
            status = "NOT_PATCHED"
        verifier_items.append({
            "finding_id": finding_key(item),
            "unit_id": unit_id,
            "status": status,
            "residual_risk": "None" if status == "PASS" else "Unverified or residual risk remains",
        })

    attempts_public = [{k: v for k, v in record.items() if k != "code"} for record in attempts]
    compilation = final_verification["compilation"] if final_verification else {}
    applied_count = sum(1 for item in patch_outcomes if item.get("status") == "applied")
    phases = {
        "detector": {
            "status": detector_original["status"],
            "original": detector_original,
            "fixed": fixed_detector,
            "candidate_findings": candidate_findings,
            "confirmed_findings": findings,
            "dismissed_findings": dismissed,
            "skipped_targets": skipped_targets,
            "detector_only_deferred": detector_only_deferred,
        },
        "advisor": advisor,
        "assessor": assessor,
        "fixer": {
            "status": "ok",
            "fixed_code": fixed_code,
            "patch_outcomes": patch_outcomes,
            "patches": [
                {"finding_id": item.get("finding_id"), "description": item.get("reason")}
                for item in patch_outcomes
            ],
        },
        "verifier": {
            "status": "ok",
            "verification_model": "solc baseline + Qwen2.5-Coder-14B v13 detector + static/RAG re-run",
            "verdict": overall_verdict,
            "overall_verdict": overall_verdict,
            "verifications": verifier_items,
            "attempts_count": len(attempts),
            "max_attempts": max_attempts,
            "attempts": attempts_public,
            "compilation": compilation,
            "fixed_unit_results": fixed_detector.get("unit_results", []),
            "vulnerability_probability": fixed_detector.get("vulnerability_probability"),
            "threshold": detector_threshold,
            "notes": f"Self-verification over {len(attempts)} attempt(s). Final verdict: {overall_verdict}.",
        },
        "reporter": {
            "status": "ok",
            "executive_summary": (
                f"Audit completed using the GPU server (Qwen2.5-Coder-14B v13) for run {run_id}. "
                f"{len(findings)} confirmed finding(s), {len(dismissed)} dismissed by the Advisor, "
                f"{applied_count} patch(es) applied."
            ),
            "recommendations": advisor["recommendations"],
        },
    }
    return {
        "schema_version": "1.0.0",
        "pipeline_version": "1.0.0",
        "status": "ok" if detector_original["status"] == "ok" else "partial",
        "run_id": run_id,
        "backend": "server_gpu_rtx4070ti",
        "source_sha256": request.get("source", {}).get("sha256", ""),
        "fixed_code": fixed_code,
        "patch_outcomes": patch_outcomes,
        "verification_verdict": overall_verdict,
        "phases": phases,
        "timings": {
            "total_ms": round((time.perf_counter() - started_at) * 1000, 2),
            "detector_ms": detector_elapsed_ms,
            "fixer_ms": fixer_elapsed_ms,
        },
    }
