"""
Phases 1C -> 5 of one contract, driven through the GPU server (Qwen2.5-Coder-14B v13).

Where it runs: this module runs on the LOCAL machine, in the same process as
orchestrator.main(). Only the model calls are remote: gpu_client sends HTTP requests to
the GPU server's /api/v1/detect (Phase 1C, detector re-scoring in Phase 5) and
/api/v1/generate (Advisor, Assessor, Fixer). Compiler and static/RAG re-runs are local.

Input: request.json written by the orchestrator (source, code units, Phase 1A/1B results).

  1C  detector scores every unit                      -> phase1c_detector/detector.py
  1F  per-finding fusion of static + RAG + detector   -> phase1_fusion/fusion.py
  2   Advisor: triage (drops false positives) + fix strategy -> phase2_advisor/
  3   Assessor: CVSS v3.1 + repair priority            -> phase3_assessor/
  4   Fixer: patch each target unit, in repair-priority order -> phase4_fixer/fixer.py
  5   check the candidate (compiler, detector, static/RAG); on failure feed the
      problems back to the Fixer, at most MAX_FIX_ATTEMPTS (config.py) attempts
                                                        -> phase5_verifier/candidate_check.py
The final verification (Slither, AST) and the report (Phase 6) run afterwards in the orchestrator.
"""

from __future__ import annotations

import time
from typing import Any

from E2E.common.findings import finding_key
from E2E.common.gpu_client import check_server_health, get_server_url
from E2E.config import DEFAULT_DETECTOR_THRESHOLD, MAX_FIX_ATTEMPTS, PIPELINE_VERSION, SCHEMA_VERSION
from E2E.phase0_parsing.code_units import as_code_unit
from E2E.phase1_fusion.fusion import fuse_findings
from E2E.phase1c_detector.detector import score_units_via_server, summarize_detector
from E2E.phase2_advisor.advisor import RepairKnowledgeBase
from E2E.phase2_advisor.advisor_phase import _generate_advisor_section
from E2E.phase3_assessor.assessor_phase import _generate_assessor_section
from E2E.phase4_fixer.fixer import _build_rag_knowledge_section, _splice_unit, patch_unit
from E2E.phase5_verifier.candidate_check import _finding_pairs, _verification_feedback, verify_candidate


# In log có kèm giờ (web app đọc dòng "[PHASE x]" này để cập nhật tiến độ).
def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# [HÀM TRUNG TÂM PHẦN GPU] Chạy Phase 1C -> 5 cho một contract từ request.json (do orchestrator ghi).
# LƯU Ý: hàm này chạy ở MÁY LOCAL, cùng tiến trình với orchestrator.main; chỉ lời gọi model đi qua HTTP tới
# GPU server (/api/v1/detect cho detector, /api/v1/generate cho Advisor/Assessor/Fixer). solc, static, RAG
# trong vòng tự kiểm đều chạy local.
#   1C: kiểm tra server sống (health) -> detector chấm từng unit (score_units_via_server + summarize_detector).
#   1F: fuse_findings -> finding confirmed / candidate; RepairKnowledgeBase gắn tri thức sửa lỗi theo SWC
#       (một KB dùng chung cho Phase 1F, Advisor và Assessor).
#   2 : Advisor -> loại false positive khỏi tập cần sửa, sinh hướng sửa (suggestions).
#   3 : Assessor -> CVSS v3.1 + repair_priority (điểm CVSS cao sửa trước).
#   4-5: VÒNG SỬA (tối đa 3 lần): Fixer vá từng hàm mục tiêu theo repair_priority (patch_unit);
#       rồi verify_candidate kiểm tra compile + detector + static/RAG chạy lại.
#       Chưa đạt -> tạo feedback; lần sau giữ lại các bản vá đã đạt, chỉ vá lại các hàm còn lỗi TỪ MÃ GỐC.
#       Chọn attempt tốt nhất theo (compile được, ít hàm lỗi, nhiều hàm vá, ít thay đổi độ dài).
# Trả về server_result: phases {detector, advisor, assessor, fixer, verifier} (bản vá ở phases.fixer, kết luận vòng sửa
# PASS/WARN/NOT_APPLICABLE ở phases.verifier.overall_verdict), thời gian từng phần.
# Gọi bởi: orchestrator.main.
def run_server_gpu_pipeline(
    request: dict[str, Any],
    *,
    server_url: str | None = None,
) -> dict[str, Any]:
    """Thực thi pipeline phase 1C-5 (chạy local) bằng cách gọi dịch vụ AI trên GPU Server.

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
    max_attempts = min(
        max(1, int((execution.get("repair") or {}).get("max_attempts") or MAX_FIX_ATTEMPTS)),
        MAX_FIX_ATTEMPTS,
    )
    local_phase1 = request.get("local_phase1", {})
    static_p1 = local_phase1.get("static", {})
    rag_p1 = local_phase1.get("rag", {})
    rag_context = rag_p1.get("retrieved_context", [])

    health = check_server_health(server_url)
    if health.get("status") != "ready":
        raise ConnectionError(
            f"GPU Server tại {server_url} chưa sẵn sàng (trạng thái: {health.get('status')}). "
            "Hãy khởi động server: ./scripts/gpu_server.sh start"
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
    findings, candidate_findings = fuse_findings(unit_results, static_p1, rag_p1)  # tau_agent from fusion_calibration.json
    knowledge_base = RepairKnowledgeBase()
    findings = knowledge_base.enrich_findings(findings, code=source_code, code_units=request.get("code_units", []))
    _log(f"[PHASE 1 FUSED] Completed: {len(findings)} confirmed finding(s), {len(candidate_findings)} candidate(s).")

    # Phase 2: Advisor (triage + remediation). False positives are removed from the repair set.
    _log("[PHASE 2] Running Security Advisor agent (root cause & remediation analysis)...")
    advisor = _generate_advisor_section(findings, source_code, use_llm=True, knowledge_base=knowledge_base)
    dismissed_ids = set(advisor.get("false_positives", []))
    reasons = advisor.get("false_positive_reasons", {})
    dismissed = [
        {**item, "dismissal_reason": reasons.get(finding_key(item), "")}
        for item in findings if finding_key(item) in dismissed_ids
    ]
    findings = [item for item in findings if finding_key(item) not in dismissed_ids]
    _log(
        f"[PHASE 2] Completed: {len(advisor.get('suggestions', []))} suggestion(s), "
        f"{len(dismissed)} finding(s) dismissed as false positive."
    )

    # Phase 3: Assessor (CVSS v3.1 + repair priority, feeds the report).
    _log("[PHASE 3] Running Risk Assessor agent (CVSS v3.1 scoring & priority ranking)...")
    assessor = _generate_assessor_section(
        findings,
        source_code,
        use_llm=True,
        advisor_suggestions=advisor.get("suggestions", []),
        knowledge_base=knowledge_base,
    )
    _log(f"[PHASE 3] Completed: {len(assessor.get('assessments', []))} item(s) assessed.")

    # Phase 4-5: repair loop.
    finding_units = {item.get("unit_id") for item in findings if item.get("unit_id")}
    target_units = [
        unit for unit in unit_objs
        if unit.unit_id in finding_units and unit.has_body and not unit.is_view_or_pure
    ]
    # Phase 3 -> Phase 4: repair the units in the Assessor's repair-priority order (highest CVSS first).
    unit_priority: dict[str, int] = {}
    for item in assessor.get("assessments", []):
        unit_id = item.get("unit_id")
        unit_priority[unit_id] = min(unit_priority.get(unit_id, 10**6), int(item.get("repair_priority") or 10**6))
    target_units.sort(key=lambda unit: unit_priority.get(unit.unit_id, 10**6))
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
    first_swc = next((item.get("swc") for item in findings if item.get("swc") not in {None, "UNMAPPED"}), None)
    rag_section = _build_rag_knowledge_section(rag_context, target_swc=first_swc)
    feedback_note = ""

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
            )
            outcomes.append(outcome)

        patched_ids = {
            str(item.get("unit_id")) for item in outcomes
            if item.get("status") == "applied" and item.get("unit_id")
        }
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
            "verdict": "Safe" if verification["passed"] else "Vulnerable",
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
                pass  # a verified patch no longer splices into the original: restart from the original source
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
        overall_verdict = "PASS" if best["passed"] else "WARN"
    _log(f"[PHASE 5] Completed: verdict {overall_verdict} after {len(attempts)} attempt(s).")

    if final_verification is not None:
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
            "reason": "No function-level repair target.",
        }

    patched_ids = {
        str(item.get("unit_id")) for item in patch_outcomes
        if item.get("status") == "applied" and item.get("unit_id")
    }
    failing_final = set(final_verification["failing_units"]) if final_verification else set()
    verifier_items = []
    for item in findings:
        unit_id = item.get("unit_id")
        # An unpatched unit keeps its original (Vulnerable) detector score, so it is also in
        # failing_final; it must be reported as NOT_PATCHED, not as a patch that failed.
        if unit_id not in patched_ids:
            status = "NOT_PATCHED"
        elif unit_id in failing_final:
            status = "FLAGGED"
        else:
            status = "PASS"
        verifier_items.append({
            "finding_id": finding_key(item),
            "unit_id": unit_id,
            "status": status,
            "residual_risk": "None" if status == "PASS" else "Unverified or residual risk remains",
        })

    attempts_public = [{k: v for k, v in record.items() if k != "code"} for record in attempts]
    compilation = final_verification["compilation"] if final_verification else {}
    phases = {
        "detector": {
            "status": detector_original["status"],
            "original": detector_original,
            "fixed": fixed_detector,
            "candidate_findings": candidate_findings,
            "confirmed_findings": findings,
            "dismissed_findings": dismissed,
        },
        "advisor": advisor,
        "assessor": assessor,
        "fixer": {
            "status": "ok",
            "fixed_code": fixed_code,
            "patch_outcomes": patch_outcomes,
        },
        "verifier": {
            "status": "ok",
            "verification_model": "solc baseline + Qwen2.5-Coder-14B v13 detector + static/RAG re-run",
            "overall_verdict": overall_verdict,
            "verifications": verifier_items,
            "attempts_count": len(attempts),
            "max_attempts": max_attempts,
            "attempts": attempts_public,
            "compilation": compilation,
            "fixed_unit_results": fixed_detector.get("unit_results", []),
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "status": "ok" if detector_original["status"] == "ok" else "partial",
        "run_id": run_id,
        "backend": "server_gpu_rtx4070ti",
        "source_sha256": request.get("source", {}).get("sha256", ""),
        "phases": phases,
        "timings": {
            "total_ms": round((time.perf_counter() - started_at) * 1000, 2),
            "detector_ms": detector_elapsed_ms,
            "fixer_ms": fixer_elapsed_ms,
        },
    }
