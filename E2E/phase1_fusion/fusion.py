"""
Phase 1 fusion - combine static (1A), RAG (1B) and the neural detector (1C).

* fuse_findings (used by the GPU pipeline): one score per (SWC, unit) finding,
  p_final = logistic(intercept + w_det*p_detector + w_static*conf + w_rag*conf
  + corroboration + severity) with the weights of fusion_calibration.json (set by hand,
  not fitted). Findings with p_final >= tau_agent (0.45) go to the agents (confirmed);
  the rest stay as candidates. A unit flagged only by the detector becomes an UNMAPPED
  finding so it can still be repaired.
* fuse_detection (contract verdict for the report): weighted sum of the three channels
  (DEFAULT_COMPONENT_WEIGHTS, re-balanced towards static/RAG when the detector is offline
  or below its threshold while they found something) plus evidence floors (corroborated or
  High/Critical local findings). Fail-closed: Safe only when every required channel ran
  and the detector scored every unit; otherwise Inconclusive.
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
from typing import Any

from E2E.config import (
    DEFAULT_COMPONENT_WEIGHTS,
    DEFAULT_DETECTOR_THRESHOLD,
    DEFAULT_FINAL_THRESHOLD,
    FUSION_CALIBRATION_PATH,
    RISK_ORDER,
)


# Lấy điểm của một kênh (static/RAG/detector) từ score / vulnerability_probability.
def _score(component: dict[str, Any]) -> float:
    return float(component.get("score") or component.get("vulnerability_probability") or 0.0)


# Kênh có "hoạt động" không: status="ok" hoặc đã có verdict kèm điểm. Kênh lỗi/pending không được tính trọng số.
def _active(component: dict[str, Any]) -> bool:
    if not isinstance(component, dict):
        return False
    if component.get("status") == "ok":
        return True
    if component.get("verdict") in {"Vulnerable", "Safe", "Inconclusive"} and (
        component.get("vulnerability_probability") is not None
        or component.get("score") is not None
    ):
        return True
    return False


# Detector chỉ được coi là đầy đủ khi MỌI unit đều có xác suất (không unit nào error/not_run).
# Mục đích: chặn kết luận "Safe" khi detector chấm thiếu hàm (fail-closed).
def _detector_complete(component: dict[str, Any]) -> bool:
    """A detector only counts as complete when every scored unit returned a probability.

    A run where some (or all) units errored must never be allowed to produce `Safe`.
    """
    if not _active(component):
        return False
    return all(unit.get("vulnerability_probability") is not None for unit in component.get("unit_results") or [])


# Slither is a static tool: its findings belong to the same evidence channel as the
# regex rules, so static + slither agreement is not cross-channel corroboration.
_CHANNEL_BY_SOURCE = {"static": "static", "slither": "static", "rag": "rag"}


# Đổi danh sách nguồn thành kênh bằng chứng: "slither" và "static" là CÙNG kênh static, "rag" là kênh riêng.
# Mục đích: static + Slither cùng báo không được tính là "hai kênh xác nhận nhau" (corroborated).
def _channels(sources: list[str] | set[str]) -> set[str]:
    return {_CHANNEL_BY_SOURCE.get(source, source) for source in sources}


# Nhiệm vụ: gộp finding của static và RAG theo khoá (SWC, unit_id): lấy mức nghiêm trọng cao nhất, confidence
# cao nhất, danh sách nguồn, corroborated (>= 2 kênh khác nhau), dòng, mô tả, cách sửa, cặp mã lỗi/mã sửa từ RAG.
# Mục đích: mỗi lỗ hổng (một SWC tại một hàm) chỉ còn MỘT finding, kèm toàn bộ bằng chứng (evidence).
def merge_findings(static_result: dict[str, Any], rag_result: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for component in (static_result, rag_result):
        for finding in component.get("findings", []):
            grouped[
                (
                    str(finding.get("swc") or "UNMAPPED"),
                    str(finding.get("unit_id") or "contract"),
                )
            ].append(finding)

    merged = []
    for (swc, unit_id), evidence in grouped.items():
        severity = max(
            (item.get("severity", "Unknown") for item in evidence),
            key=lambda level: RISK_ORDER.get(level, 0),
        )
        sources = sorted({str(item.get("source") or "unknown") for item in evidence})
        best = max(evidence, key=lambda item: float(item.get("confidence") or 0.0))
        merged.append(
            {
                "finding_id": f"{swc}:{unit_id}",
                "swc": swc,
                "unit_id": None if unit_id == "contract" else unit_id,
                "title": best.get("title") or swc,
                "severity": severity,
                "confidence": round(max(float(item.get("confidence") or 0.0) for item in evidence), 6),
                "sources": sources,
                "corroborated": len(_channels(sources)) > 1,
                "line": next((item.get("line") for item in evidence if item.get("line")), None),
                "description": next(
                    (item.get("description") for item in evidence if item.get("description")),
                    "",
                ),
                "remediation": next(
                    (item.get("remediation") for item in evidence if item.get("remediation")),
                    "",
                ),
                "fixed_code": next(
                    (item.get("fixed_code") for item in evidence if item.get("fixed_code")),
                    "",
                ),
                "vulnerable_code": next(
                    (item.get("vulnerable_code") for item in evidence if item.get("vulnerable_code")),
                    "",
                ),
                "evidence": evidence,
            }
        )
    return sorted(
        merged,
        key=lambda item: (RISK_ORDER.get(item["severity"], 0), item["confidence"]),
        reverse=True,
    )


# Gắn vị trí từ kết quả detector của unit vào một finding: start_line / end_line của unit, line_scores
# (điểm từng dòng của line head) và dòng nghi ngờ nhất = dòng có line_score cao nhất (không vượt end_line).
#   overwrite=True  (fuse_findings): giá trị của detector thắng; không có line_scores thì dòng mặc định là start_line.
#   overwrite=False (fuse_detection): chỉ điền những trường finding còn thiếu.
def _apply_detector_lines(finding: dict[str, Any], unit_result: dict[str, Any], *, overwrite: bool) -> None:
    start_line = unit_result.get("start_line")
    end_line = unit_result.get("end_line")
    line_scores = unit_result.get("line_scores")
    for key, value in (("start_line", start_line), ("end_line", end_line)):
        if value and (overwrite or not finding.get(key)):
            finding[key] = value
    if isinstance(line_scores, list) and line_scores:
        finding["line_scores"] = line_scores
        if start_line and (overwrite or not finding.get("line")):
            line = start_line + line_scores.index(max(line_scores))
            finding["line"] = min(line, end_line) if end_line else line
    elif overwrite and start_line and not finding.get("line"):
        finding["line"] = start_line


# Trọng số logistic + ngưỡng tau_agent, đặt tay trong fusion_calibration.json (chưa fit trên tập hiệu chuẩn).
def _load_calibration() -> dict[str, Any]:
    return json.loads(FUSION_CALIBRATION_PATH.read_text(encoding="utf-8"))


# [FUSION THEO TỪNG FINDING - dùng trong gpu_pipeline trước Phase 2]
# Với mỗi finding (SWC, unit) đã gộp: lấy p_detector của chính unit đó (lỗi -> 0.5, không chạy -> 0.05), rồi
#   p_final = logistic(intercept + w_det*p_detector + w_static*static_conf + w_rag*rag_conf + corroborated + severity)
# với trọng số trong fusion_calibration.json (đặt tay). p_final >= tau_agent (0.45) -> confirmed (đi tiếp tới
# Advisor/Assessor/Fixer); còn lại -> candidates. Dùng line_scores của detector để chỉ ra dòng nghi ngờ nhất.
# Unit chỉ detector báo lỗi (static/RAG không thấy) vẫn thành finding "UNMAPPED" để có thể được sửa.
# Trả về (confirmed, candidates).
def fuse_findings(
    unit_results: list[dict[str, Any]],
    static_result: dict[str, Any],
    rag_result: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Hợp nhất đa phương thức theo từng finding riêng biệt trước khi chuyển sang Phase 2.

    - Gộp static + rag và dedupe duy nhất theo (swc, unit_id), không tính line vào key.
    - Tính điểm p_final cho từng (swc, unit) dựa trên điểm của chính unit đó từ Neural Detector.
    - Chỉ các finding có p_final >= tau_agent mới được chuyển sang Phase 2 (confirmed_findings).
    - Các finding còn lại được đưa vào candidate_findings (Candidates).
    - Đảm bảo số dòng (line) không vượt quá end_line của unit.
    """
    calibration = _load_calibration()
    intercept = float(calibration["intercept"])
    weights = calibration["weights"]
    tau_agent = float(calibration["tau_agent"])

    local_merged = merge_findings(static_result, rag_result)

    unit_map = {str(item.get("unit_id")): item for item in unit_results if item.get("unit_id")}
    valid_probs = [
        float(item["vulnerability_probability"])
        for item in unit_results
        if item.get("vulnerability_probability") is not None
    ]
    errored_units = any(
        item.get("vulnerability_probability") is None and item.get("status") == "error"
        for item in unit_results
    )
    # A unit whose detector call failed gets a neutral 0.5 (unknown), never an implicit
    # "safe" score. Without any detector run, only strong local evidence passes (0.05 prior).
    contract_max_prob = max(valid_probs) if valid_probs else (0.5 if errored_units else 0.0)

    confirmed: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for finding in local_merged:
        u_id = finding.get("unit_id")
        u_res = unit_map.get(str(u_id)) if u_id else None

        if u_res and u_res.get("vulnerability_probability") is not None:
            p_detector = float(u_res["vulnerability_probability"])
        elif not u_id:
            p_detector = contract_max_prob
        elif u_res is not None:
            # Detector errored on this unit.
            p_detector = 0.5
            finding["p_detector_missing"] = True
        else:
            # Detector did not run, or the unit was not scored (declaration without body).
            p_detector = 0.05

        evidence = finding.get("evidence", [])
        static_conf = max(
            [
                float(item.get("confidence") or 0.0)
                for item in evidence
                if _CHANNEL_BY_SOURCE.get(str(item.get("source"))) == "static"
            ]
            or [0.0]
        )
        rag_conf = max(
            [float(item.get("confidence") or 0.0) for item in evidence if item.get("source") == "rag"] or [0.0]
        )
        corroborated = bool(static_conf > 0.0 and rag_conf > 0.0)
        severity = finding.get("severity", "Medium")

        z = (
            intercept
            + float(weights["p_detector"]) * p_detector
            + float(weights["static_conf"]) * static_conf
            + float(weights["rag_conf"]) * rag_conf
            + (float(weights["corroborated"]) if corroborated else 0.0)
            + float(weights.get(f"severity_{severity.lower()}", 0.0))
        )
        p_final = 1.0 / (1.0 + math.exp(-z))  # z is bounded (inputs in [0, 1]): exp cannot overflow

        p_final = max(0.0, min(1.0, round(p_final, 6)))
        finding["confidence"] = p_final
        finding["p_final"] = p_final
        finding["p_detector"] = round(p_detector, 6)
        finding["corroborated"] = corroborated

        if u_res:
            _apply_detector_lines(finding, u_res, overwrite=True)

        if p_final >= tau_agent:
            confirmed.append(finding)
        else:
            candidates.append(finding)

    # Units flagged only by the neural detector must still reach the repair agents;
    # otherwise a detector-driven Vulnerable verdict can never be repaired.
    local_units = {item.get("unit_id") for item in local_merged}
    for finding in detector_findings(unit_results, local_units):
        p_detector = float(finding.get("confidence") or 0.0)
        finding["p_detector"] = round(p_detector, 6)
        finding["p_final"] = round(p_detector, 6)
        confirmed.append(finding)

    return confirmed, candidates


# Tạo finding "UNMAPPED:<unit>" cho các unit detector gắn nhãn Vulnerable mà static/RAG không có finding nào
# (local_units = tập unit_id đã có finding static/RAG).
# Mục đích: detector là phân loại nhị phân (không biết SWC) nhưng tín hiệu của nó vẫn không bị bỏ qua.
def detector_findings(
    unit_results: list[dict[str, Any]],
    local_units: set[Any],
) -> list[dict[str, Any]]:
    findings = []
    for item in unit_results:
        if item.get("label") == "Vulnerable" and item.get("unit_id") not in local_units:
            start_line = item.get("start_line")
            det_title = "Qwen2.5-14B neural vulnerability signal"
            findings.append({
                "finding_id": f"UNMAPPED:{item.get('unit_id') or 'source'}",
                "swc": "UNMAPPED",
                "unit_id": item.get("unit_id"),
                "title": det_title,
                "severity": "Unknown",
                "confidence": round(float(item.get("vulnerability_probability") or 0.0), 6),
                "sources": ["llm_detector"],
                "corroborated": False,
                "line": start_line,
                "start_line": start_line,
                "end_line": item.get("end_line"),
                "line_scores": item.get("line_scores"),
                "description": (
                    "The fine-tuned binary classifier crossed its vulnerability threshold. "
                    "Static/RAG did not assign a supported SWC category to this unit."
                ),
                "remediation": "Perform targeted manual review and use the Advisor phase for root-cause analysis.",
                "evidence": [item],
            })
    return findings


# [KẾT LUẬN CẤP CONTRACT - dùng cho báo cáo]
# Điểm = tổng có trọng số của 3 kênh (detector 0.65, static 0.20, RAG 0.15), trọng số tự cân lại khi detector
# offline (0/0.5/0.5) hoặc dưới ngưỡng mà static/RAG có phát hiện (0.2/0.4/0.4).
# Sau đó áp "sàn bằng chứng" (decision_rules): static+RAG cùng xác nhận >= 0.72, lỗi High/Critical mạnh >= 0.65,
# detector vượt ngưỡng >= 0.58 (có thêm finding local >= 0.62)...
# Verdict: điểm >= 0.50 và có bằng chứng -> Vulnerable; mọi kênh bắt buộc chạy đủ -> Safe; còn lại -> Inconclusive.
# Gọi bởi: orchestrator (lượt 1 = tạm thời không detector; lượt 2 = có detector) và final_verification.
def fuse_detection(
    static_result: dict[str, Any],
    rag_result: dict[str, Any],
    llm_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    threshold = DEFAULT_FINAL_THRESHOLD
    detector = llm_result or {
        "component": "llm_detector",
        "status": "pending",
        "score": 0.0,
        "verdict": "Pending",
    }
    components = {
        "llm_detector": detector,
        "rag": rag_result,
        "static": static_result,
    }
    component_scores = {name: _score(result) for name, result in components.items()}
    local_findings = merge_findings(static_result, rag_result)
    detector_threshold = float(detector.get("threshold") or DEFAULT_DETECTOR_THRESHOLD)

    # Adaptive weighting: khi Neural Detector offline hoặc có confidence thấp (< threshold)
    # trong khi RAG/Static phát hiện bằng chứng, tự động tái phân bổ trọng số cho RAG và Static
    # nhằm tránh việc trọng số 65% của detector dập tắt các phát hiện bảo mật đã được xác thực.
    if not _active(detector):
        # Detector offline/pending -> chia đều 50/50 cho RAG và Static
        weights = {"llm_detector": 0.0, "rag": 0.50, "static": 0.50}
    elif component_scores["llm_detector"] < detector_threshold and local_findings:
        # Detector không kích hoạt nhưng RAG/Static có phát hiện -> tăng trọng số RAG & Static
        weights = {"llm_detector": 0.20, "rag": 0.40, "static": 0.40}
    else:
        weights = DEFAULT_COMPONENT_WEIGHTS

    active_names = [name for name, result in components.items() if _active(result)]
    weight_total = sum(weights.get(name, 0.0) for name in active_names)
    effective_weights = {
        name: (weights.get(name, 0.0) / weight_total if name in active_names and weight_total else 0.0)
        for name in components
    }
    weighted_score = sum(effective_weights[name] * component_scores[name] for name in components)

    score = weighted_score
    decision_rules = ["active_weighted_fusion"] if active_names else ["no_active_component"]
    if not _active(detector) and any(
        finding["severity"] in {"Critical", "High"} for finding in local_findings
    ):
        is_crit = any(finding["severity"] == "Critical" for finding in local_findings)
        score = max(score, 0.78 if is_crit else 0.68)
        decision_rules.append("provisional_critical_local_evidence_floor" if is_crit else "provisional_high_local_evidence_floor")
    if any(finding["corroborated"] for finding in local_findings):
        score = max(score, 0.72)
        decision_rules.append("static_rag_corroboration_floor")
    if any(
        finding.get("severity") in {"Critical", "High"} and float(finding.get("confidence") or 0.0) >= 0.65
        for finding in local_findings
    ):
        score = max(score, 0.65)
        decision_rules.append("strong_local_evidence_floor")
    if _active(detector) and component_scores["llm_detector"] >= detector_threshold:
        score = max(score, 0.58)
        decision_rules.append("llm_detector_threshold_floor")
        if local_findings:
            score = max(score, 0.62)
            decision_rules.append("llm_local_corroboration_floor")
    score = max(0.0, min(1.0, score))

    all_required_active = (
        _active(static_result)
        and _active(rag_result)
        and _detector_complete(detector)
    )
    if _active(detector) and not _detector_complete(detector):
        decision_rules.append("detector_incomplete_no_safe_verdict")
    positive_evidence = bool(local_findings) or (
        _active(detector) and component_scores["llm_detector"] >= detector_threshold
    )
    if score >= threshold and positive_evidence:
        verdict = "Vulnerable"
    elif all_required_active:
        verdict = "Safe"
    else:
        verdict = "Inconclusive"

    if all_required_active:
        analysis_status = "complete"
    elif active_names:
        analysis_status = "partial"
    else:
        analysis_status = "inconclusive"
    unit_results = detector.get("unit_results", [])
    binary_findings = detector_findings(unit_results, {item.get("unit_id") for item in local_findings})
    evidence_findings = [*local_findings, *binary_findings]
    unit_map = {item.get("unit_id"): item for item in unit_results if item.get("unit_id")}
    for finding in evidence_findings:
        uid = finding.get("unit_id")
        if uid and uid in unit_map:
            _apply_detector_lines(finding, unit_map[uid], overwrite=False)
    findings = evidence_findings if verdict == "Vulnerable" else []
    candidate_findings = evidence_findings if verdict != "Vulnerable" else []
    risk_level = findings[0]["severity"] if findings else "None"
    if verdict == "Vulnerable" and not local_findings:
        risk_level = "Unknown"

    return {
        "component": "fusion",
        "status": analysis_status,
        "verdict": verdict,
        "is_vulnerable": verdict == "Vulnerable",
        "score": round(score, 6),
        "threshold": threshold,
        "risk_level": risk_level,
        "findings": findings,
        "candidate_findings": candidate_findings,
        "decision_rules": decision_rules,
        "required_components": ["static", "rag", "llm_detector"],
        "active_components": active_names,
        "configured_weights": weights,
        "effective_weights": {
            name: round(value, 6) for name, value in effective_weights.items()
        },
        "score_breakdown": {
            name: {
                "status": components[name].get("status", "unknown"),
                "raw_score": round(component_scores[name], 6),
                "weight": round(effective_weights[name], 6),
                "weighted_score": round(component_scores[name] * effective_weights[name], 6),
                "verdict": components[name].get("verdict") or "Unknown",
            }
            for name in components
        },
    }
