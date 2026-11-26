"""Phase 1C -> 5 của một contract: chạy ở máy local, chỉ lời gọi model đi qua HTTP tới GPU server."""

import time
from collections import Counter

from E2E.common.findings import finding_key
from E2E.common.gpu_client import check_server_health, get_server_url
from E2E.common.io_utils import log
from E2E.common.knowledge_base import default_knowledge_base
from E2E.config import DEFAULT_DETECTOR_THRESHOLD, MAX_FIX_ATTEMPTS, PIPELINE_VERSION, SCHEMA_VERSION
from E2E.phase0_parsing.code_units import CodeUnit, as_code_unit
from E2E.phase1_fusion.fusion import fuse_findings
from E2E.phase1c_detector.detector import score_units_via_server, summarize_detector
from E2E.phase2_advisor.advisor_phase import _generate_advisor_section
from E2E.phase3_assessor.assessor_phase import _generate_assessor_section
from E2E.phase4_fixer.fixer import _splice_unit, patch_unit
from E2E.phase5_verifier.verifier import finding_pairs, verification_feedback, verify_patch


# Gắn mã của chính hàm (unit_code) vào mỗi finding để Advisor xét đúng hàm đó.
def _with_unit_code(findings: list[dict], units: list[CodeUnit]) -> list[dict]:
    code = {unit.unit_id: unit.code for unit in units if unit.code}
    return [{**item, "unit_code": code[item["unit_id"]]} if item.get("unit_id") in code and not item.get("unit_code")
            else dict(item) for item in findings]


# Tách finding theo kết luận Advisor: bị loại (kèm lý do Advisor đưa ra) / giữ lại.
def _apply_advisor_triage(findings: list[dict], advisor: dict) -> tuple[list[dict], list[dict]]:
    dismissed_ids = set(advisor["false_positives"])
    kept, dismissed = [], []
    for item in findings:
        key = finding_key(item)
        if key in dismissed_ids:
            dismissed.append({**item, "dismissal_reason": advisor["false_positive_reasons"].get(key, "")})
        else:
            kept.append(item)
    return kept, dismissed


# Hàm cần vá: có finding, có thân, không view/pure (trừ khi detector gắn Vulnerable); xếp theo repair_priority.
def _repair_targets(units: list[CodeUnit], findings: list[dict], unit_results: list[dict],
                    assessor: dict) -> list[CodeUnit]:
    finding_units = {item.get("unit_id") for item in findings if item.get("unit_id")}
    flagged = {item.get("unit_id") for item in unit_results if item.get("label") == "Vulnerable"}
    priority = {}
    for item in assessor.get("assessments", []):
        unit_id = item.get("unit_id")
        priority[unit_id] = min(priority.get(unit_id, 10**6), int(item.get("repair_priority") or 10**6))
    targets = [unit for unit in units if unit.unit_id in finding_units and unit.has_body
               and (not unit.is_view_or_pure or unit.unit_id in flagged)]
    return sorted(targets, key=lambda unit: priority.get(unit.unit_id, 10**6))


# Lần sửa sau: mã gốc + các bản vá đã đạt; chỉ vá lại các hàm còn lỗi TỪ MÃ GỐC -> (base_code, outcomes giữ, targets).
def _next_attempt_plan(source_code: str, targets: list[CodeUnit], active: list[CodeUnit], outcomes: list[dict],
                       verification: dict, patched_ids: set[str],
                       units_by_id: dict) -> tuple[str, list[dict], list[CodeUnit]]:
    failing = set(verification["failing_units"]) | {unit.unit_id for unit in active if unit.unit_id not in patched_ids}
    compile_localized = not verification["compilation"]["failed"] \
        or bool(set(verification["compile_culprit_units"]) & patched_ids)
    if not (compile_localized and failing):
        return source_code, [], list(targets)
    keep = [item for item in outcomes if item.get("status") == "applied" and item.get("unit_id") not in failing]
    rebuilt = source_code
    try:
        for item in keep:
            unit = units_by_id[item["unit_id"]]
            rebuilt = _splice_unit(rebuilt, unit, item["patched_code"], str(item.get("reason") or ""))
    except (KeyError, ValueError):
        return source_code, [], list(targets)
    return rebuilt, keep, [unit for unit in targets if unit.unit_id in failing]


# [VÒNG SỬA Phase 4-5, tối đa max_attempts] Fixer vá từng hàm -> verify_patch (đủ mọi kiểm tra) -> feedback; chọn tốt nhất.
def _repair_loop(source_code: str, targets: list[CodeUnit], units_by_id: dict, findings: list[dict],
                 suggestions: list[dict], unit_results: list[dict], known_pairs: set, threshold: float,
                 max_attempts: int, server_url: str, file_name: str,
                 source_root: str | None) -> tuple[list[dict], dict | None]:
    target_ids = {unit.unit_id for unit in targets}
    kb = default_knowledge_base()
    attempts, best, feedback = [], None, ""
    base_code, carried, active = source_code, [], list(targets)
    for attempt in range(1, max_attempts + 1 if targets else 1):
        label = f"Attempt {attempt}/{max_attempts}"
        log(f"[PHASE 4] Running Fixer agent via Qwen2.5-Coder-14B ({label}, {len(active)} unit(s))...")
        code, outcomes = base_code, list(carried)
        for unit in active:
            code, outcome = patch_unit(
                code, unit, unit_findings=[item for item in findings if item.get("unit_id") == unit.unit_id],
                unit_suggestions=[item for item in suggestions if item.get("unit_id") == unit.unit_id],
                source_code=source_code, feedback_note=feedback, attempt=attempt, server_url=server_url,
                knowledge_base=kb)
            outcomes.append(outcome)
        patched_ids = {str(item.get("unit_id")) for item in outcomes
                       if item.get("status") == "applied" and item.get("unit_id")}
        log(f"[PHASE 5] Verifying candidate ({label}): compile + Slither, AST behavior, static re-run, detector...")
        verification = verify_patch(source_code, code, patch_outcomes=outcomes, target_findings=findings,
                                    original_unit_results=unit_results, target_unit_ids=target_ids,
                                    known_pairs=known_pairs, threshold=threshold, server_url=server_url,
                                    file_name=file_name, source_root=source_root)
        record = {"attempt": attempt, "max_attempts": max_attempts, "threshold": threshold,
                  "verdict": "Safe" if verification["passed"] else "Vulnerable", "code": code,
                  "patch_outcomes": outcomes, **verification}
        attempts.append(record)
        score = (0 if verification["compilation"]["failed"] else 1, -len(verification["failing_units"]),
                 len(patched_ids), -abs(len(code) - len(source_code)))
        if best is None or score > best["_score"]:
            best = {**record, "_score": score}
        log(f"[PHASE 5] {label}: {'PASSED' if verification['passed'] else 'FAILED'} "
            f"(compile={verification['compilation']['state']}, failing={verification['failing_units']}).")
        if verification["passed"] or attempt == max_attempts:
            break
        record["feedback"] = feedback = verification_feedback(attempt, max_attempts, verification)
        base_code, carried, active = _next_attempt_plan(source_code, targets, active, outcomes, verification,
                                                        patched_ids, units_by_id)
    return attempts, best


# Trạng thái cuối của từng finding: hàm không được vá -> NOT_PATCHED; vá nhưng vẫn lỗi -> FLAGGED; còn lại PASS.
def _verifier_items(findings: list[dict], patch_outcomes: list[dict], best: dict | None) -> list[dict]:
    patched = {str(item.get("unit_id")) for item in patch_outcomes
               if item.get("status") == "applied" and item.get("unit_id")}
    failing = set(best["failing_units"]) if best else set()
    items = []
    for item in findings:
        unit_id = item.get("unit_id")
        status = "NOT_PATCHED" if unit_id not in patched else "FLAGGED" if unit_id in failing else "PASS"
        items.append({"finding_id": finding_key(item), "unit_id": item.get("unit_id"), "status": status,
                      "residual_risk": "None" if status == "PASS" else "Unverified or residual risk remains"})
    return items


# [HÀM TRUNG TÂM PHẦN GPU] request.json -> 1C detector -> fusion -> 2 Advisor -> 3 Assessor -> 4-5 vòng sửa + kiểm chứng.
def run_server_gpu_pipeline(request: dict, *, server_url: str | None = None) -> dict:
    started_at = time.perf_counter()
    server_url = get_server_url(server_url)
    source_code = request.get("source", {}).get("code", "")
    units = [as_code_unit(unit) for unit in request.get("code_units", [])]
    execution = request.get("execution", {})
    threshold = float(execution.get("detector_threshold") or DEFAULT_DETECTOR_THRESHOLD)
    requested = int((execution.get("repair") or {}).get("max_attempts") or MAX_FIX_ATTEMPTS)
    max_attempts = min(max(1, requested), MAX_FIX_ATTEMPTS)
    static_p1 = request.get("local_phase1", {}).get("static", {})
    rag_p1 = request.get("local_phase1", {}).get("rag", {})
    health = check_server_health(server_url)
    if health.get("status") != "ready":
        raise ConnectionError(f"GPU Server tại {server_url} chưa sẵn sàng (trạng thái: {health.get('status')}). "
                              "Hãy khởi động server: ./scripts/gpu_server.sh start (server 176) hoặc "
                              ".venv/bin/python scripts/kaggle_server.py start (Kaggle)")

    log(f"[PHASE 1C] Running Neural Detector v13 on GPU server ({len(units)} code unit(s))...")
    detector_started = time.perf_counter()
    unit_results = score_units_via_server(units, threshold, server_url)
    detector_original = summarize_detector(unit_results, threshold)
    detector_ms = round((time.perf_counter() - detector_started) * 1000, 2)
    log(f"[PHASE 1C] Completed: {len(unit_results)} unit(s) evaluated, status={detector_original['status']}, "
        f"max_prob={detector_original['vulnerability_probability']}.")

    log("[PHASE 1 FUSED] Synchronizing multi-modal detection signals...")
    findings, candidates = fuse_findings(unit_results, static_p1, rag_p1)
    findings = _with_unit_code(findings, units)
    classified = sum(1 for item in findings if item.get("classification"))
    log(f"[PHASE 1 FUSED] Completed: {len(findings)} confirmed finding(s) ({classified} classified from the "
        f"nearest known case), {len(candidates)} candidate(s).")

    log("[PHASE 2] Running Security Advisor agent (triage, root cause & remediation analysis)...")
    knowledge_base = default_knowledge_base()
    advisor = _generate_advisor_section(findings, source_code, use_llm=True, knowledge_base=knowledge_base)
    findings, dismissed = _apply_advisor_triage(findings, advisor)
    verdicts = Counter(item.get("advisor_verdict", "no LLM verdict") for item in advisor["suggestions"])
    log(f"[PHASE 2] Completed: {len(advisor['suggestions'])} suggestion(s) {dict(verdicts)}, "
        f"{len(dismissed)} finding(s) dismissed as false positive.")

    log("[PHASE 3] Running Risk Assessor agent (CVSS v3.1 scoring & priority ranking)...")
    assessor = _generate_assessor_section(findings, source_code, use_llm=True,
                                          advisor_suggestions=advisor.get("suggestions", []), knowledge_base=knowledge_base)
    log(f"[PHASE 3] Completed: {len(assessor.get('assessments', []))} item(s) assessed.")

    targets = _repair_targets(units, findings, unit_results, assessor)
    known_pairs = finding_pairs([*static_p1.get("findings", []), *static_p1.get("candidates", []),
                                 *rag_p1.get("findings", [])])
    fixer_started = time.perf_counter()
    attempts, best = _repair_loop(source_code, targets, {unit.unit_id: unit for unit in units}, findings,
                                  advisor.get("suggestions", []), unit_results, known_pairs, threshold, max_attempts,
                                  server_url, request.get("source", {}).get("file_name") or "Contract.sol",
                                  execution.get("source_root"))
    fixer_ms = round((time.perf_counter() - fixer_started) * 1000, 2)
    fixed_code, patch_outcomes = (best["code"], best["patch_outcomes"]) if best else (source_code, [])
    overall_verdict = "NOT_APPLICABLE" if best is None else "PASS" if best["passed"] else "WARN"
    log(f"[PHASE 5] Completed: verdict {overall_verdict} after {len(attempts)} attempt(s).")
    target_ids = {unit.unit_id for unit in targets}
    if best:
        fixed_detector = summarize_detector([item for item in best["unit_results"] if item["unit_id"] in target_ids],
                                            threshold)
    else:
        fixed_detector = {"status": "not_run", "verdict": "Not run", "score": None, "vulnerability_probability": None,
                          "threshold": threshold, "unit_results": [], "reason": "No function-level repair target."}
    return {
        "schema_version": SCHEMA_VERSION, "pipeline_version": PIPELINE_VERSION,
        "status": "ok" if detector_original["status"] == "ok" else "partial",
        "run_id": request.get("run_id", "server-gpu-run"), "backend": "server_gpu_rtx4070ti",
        "source_sha256": request.get("source", {}).get("sha256", ""),
        "phases": {
            "detector": {"status": detector_original["status"], "original": detector_original, "fixed": fixed_detector,
                         "candidate_findings": candidates, "confirmed_findings": findings,
                         "dismissed_findings": dismissed},
            "advisor": advisor,
            "assessor": assessor,
            "fixer": {"status": "ok", "fixed_code": fixed_code, "patch_outcomes": patch_outcomes},
            "verifier": {
                "status": "ok",
                "verification_model": "solc + Slither + AST behavior + static re-run + Qwen2.5-Coder-14B v13 detector",
                "overall_verdict": overall_verdict, "verifications": _verifier_items(findings, patch_outcomes, best),
                "attempts_count": len(attempts), "max_attempts": max_attempts,
                "attempts": [{key: value for key, value in record.items() if key != "code"} for record in attempts],
                "compilation": best["compilation"] if best else {},
                "fixed_unit_results": fixed_detector.get("unit_results", []),
                "final": {key: best[key] for key in ("tool_verification", "behavioral_verification", "fixed_static")}
                if best else None,
            },
        },
        "timings": {"total_ms": round((time.perf_counter() - started_at) * 1000, 2), "detector_ms": detector_ms,
                    "fixer_ms": fixer_ms},
    }
