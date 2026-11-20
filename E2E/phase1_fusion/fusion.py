"""Phase 1 (hợp nhất) - kết hợp static (1A) + RAG (1B) + detector (1C): theo từng finding và kết luận cả contract."""

import json
import math
from collections import defaultdict

from E2E.config import (
    DEFAULT_COMPONENT_WEIGHTS, DEFAULT_DETECTOR_THRESHOLD, DEFAULT_FINAL_THRESHOLD, FUSION_CALIBRATION_PATH, RISK_ORDER,
)

# Slither cùng kênh bằng chứng với luật static: static + Slither cùng báo KHÔNG tính là hai kênh xác nhận nhau.
_CHANNEL_BY_SOURCE = {"static": "static", "slither": "static", "rag": "rag"}


# Điểm của một kênh (score, không có thì vulnerability_probability).
def _score(component: dict) -> float:
    return float(component.get("score") or component.get("vulnerability_probability") or 0.0)


# Kênh có hoạt động không: status ok, hoặc đã có verdict kèm điểm (kênh lỗi / pending không được tính trọng số).
def _active(component: dict) -> bool:
    return isinstance(component, dict) and (component.get("status") == "ok" or (
        component.get("verdict") in {"Vulnerable", "Safe", "Inconclusive"}
        and (component.get("vulnerability_probability") is not None or component.get("score") is not None)))


# Detector chỉ "đầy đủ" khi MỌI unit đều có xác suất (chặn kết luận Safe khi chấm thiếu hàm).
def _detector_complete(component: dict) -> bool:
    return _active(component) and all(unit.get("vulnerability_probability") is not None
                                      for unit in component.get("unit_results") or [])


# Giá trị khác rỗng đầu tiên của `key` trong danh sách bằng chứng.
def _first(evidence: list[dict], key: str, default=""):
    return next((item.get(key) for item in evidence if item.get(key)), default)


# Gộp finding static + RAG theo (SWC, unit): mức độ / confidence cao nhất, nguồn, corroborated (>= 2 kênh).
def merge_findings(static_result: dict, rag_result: dict) -> list[dict]:
    grouped = defaultdict(list)
    for finding in [*static_result.get("findings", []), *rag_result.get("findings", [])]:
        grouped[(str(finding.get("swc") or "UNMAPPED"), str(finding.get("unit_id") or "contract"))].append(finding)
    merged = []
    for (swc, unit_id), evidence in grouped.items():
        sources = sorted({str(item.get("source") or "unknown") for item in evidence})
        merged.append({
            "finding_id": f"{swc}:{unit_id}", "swc": swc, "unit_id": None if unit_id == "contract" else unit_id,
            "title": max(evidence, key=lambda item: float(item.get("confidence") or 0.0)).get("title") or swc,
            "severity": max((item.get("severity", "Unknown") for item in evidence),
                            key=lambda level: RISK_ORDER.get(level, 0)),
            "confidence": round(max(float(item.get("confidence") or 0.0) for item in evidence), 6),
            "sources": sources, "corroborated": len({_CHANNEL_BY_SOURCE.get(source, source) for source in sources}) > 1,
            "line": _first(evidence, "line", None), "description": _first(evidence, "description"),
            "remediation": _first(evidence, "remediation"), "fixed_code": _first(evidence, "fixed_code"),
            "vulnerable_code": _first(evidence, "vulnerable_code"), "evidence": evidence,
        })
    return sorted(merged, key=lambda item: (RISK_ORDER.get(item["severity"], 0), item["confidence"]), reverse=True)


# Gắn vị trí từ detector vào finding (start/end_line, line_scores, dòng nghi nhất); overwrite=False chỉ điền chỗ trống.
def _apply_detector_lines(finding: dict, unit_result: dict, *, overwrite: bool) -> None:
    start_line, end_line, line_scores = (unit_result.get(key) for key in ("start_line", "end_line", "line_scores"))
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


# Finding "UNMAPPED:<unit>" cho unit detector gắn Vulnerable mà static/RAG không có finding nào (để vẫn được sửa).
def detector_findings(unit_results: list[dict], local_units: set) -> list[dict]:
    return [{
        "finding_id": f"UNMAPPED:{item.get('unit_id') or 'source'}", "swc": "UNMAPPED", "unit_id": item.get("unit_id"),
        "title": "Qwen2.5-14B neural vulnerability signal", "severity": "Unknown",
        "confidence": round(float(item.get("vulnerability_probability") or 0.0), 6), "sources": ["llm_detector"],
        "corroborated": False, "line": item.get("start_line"), "start_line": item.get("start_line"),
        "end_line": item.get("end_line"), "line_scores": item.get("line_scores"),
        "description": "The fine-tuned binary classifier crossed its vulnerability threshold. "
                       "Static/RAG did not assign a supported SWC category to this unit.",
        "remediation": "Perform targeted manual review and use the Advisor phase for root-cause analysis.",
        "evidence": [item],
    } for item in unit_results if item.get("label") == "Vulnerable" and item.get("unit_id") not in local_units]


# [FUSION THEO TỪNG FINDING] p_final = logistic(intercept + w·x) (fusion_calibration.json); >= tau_agent -> confirmed.
def fuse_findings(unit_results: list[dict], static_result: dict, rag_result: dict) -> tuple[list[dict], list[dict]]:
    calibration = json.loads(FUSION_CALIBRATION_PATH.read_text(encoding="utf-8"))
    weights = calibration["weights"]
    local = merge_findings(static_result, rag_result)
    unit_map = {str(item.get("unit_id")): item for item in unit_results if item.get("unit_id")}
    probabilities = [float(item["vulnerability_probability"]) for item in unit_results
                     if item.get("vulnerability_probability") is not None]
    errored = any(item.get("vulnerability_probability") is None and item.get("status") == "error"
                  for item in unit_results)
    contract_p = max(probabilities) if probabilities else (0.5 if errored else 0.0)
    confirmed, candidates = [], []
    for finding in local:
        unit_id = finding.get("unit_id")
        unit = unit_map.get(str(unit_id)) if unit_id else None
        if unit and unit.get("vulnerability_probability") is not None:
            p_detector = float(unit["vulnerability_probability"])
        elif not unit_id:
            p_detector = contract_p
        elif unit is not None:
            p_detector, finding["p_detector_missing"] = 0.5, True
        else:
            p_detector = 0.05
        evidence = finding.get("evidence", [])
        static_conf = max([float(item.get("confidence") or 0.0) for item in evidence
                           if _CHANNEL_BY_SOURCE.get(str(item.get("source"))) == "static"] or [0.0])
        rag_conf = max([float(item.get("confidence") or 0.0) for item in evidence
                        if item.get("source") == "rag"] or [0.0])
        corroborated = bool(static_conf > 0.0 and rag_conf > 0.0)
        z = (float(calibration["intercept"]) + float(weights["p_detector"]) * p_detector
             + float(weights["static_conf"]) * static_conf + float(weights["rag_conf"]) * rag_conf
             + (float(weights["corroborated"]) if corroborated else 0.0)
             + float(weights.get(f"severity_{finding.get('severity', 'Medium').lower()}", 0.0)))
        p_final = max(0.0, min(1.0, round(1.0 / (1.0 + math.exp(-z)), 6)))
        finding.update(confidence=p_final, p_final=p_final, p_detector=round(p_detector, 6), corroborated=corroborated)
        if unit:
            _apply_detector_lines(finding, unit, overwrite=True)
        (confirmed if p_final >= float(calibration["tau_agent"]) else candidates).append(finding)
    for finding in detector_findings(unit_results, {item.get("unit_id") for item in local}):
        finding["p_detector"] = finding["p_final"] = round(float(finding.get("confidence") or 0.0), 6)
        confirmed.append(finding)
    return confirmed, candidates


# [KẾT LUẬN CẤP CONTRACT] Tổng có trọng số 3 kênh (tự cân lại khi detector offline / dưới ngưỡng) + sàn bằng chứng.
def fuse_detection(static_result: dict, rag_result: dict, llm_result: dict | None = None) -> dict:
    detector = llm_result or {"component": "llm_detector", "status": "pending", "score": 0.0, "verdict": "Pending"}
    components = {"llm_detector": detector, "rag": rag_result, "static": static_result}
    scores = {name: _score(result) for name, result in components.items()}
    local = merge_findings(static_result, rag_result)
    threshold = float(detector.get("threshold") or DEFAULT_DETECTOR_THRESHOLD)
    detector_active = _active(detector)
    detector_fired = detector_active and scores["llm_detector"] >= threshold
    if not detector_active:
        weights = {"llm_detector": 0.0, "rag": 0.50, "static": 0.50}
    elif scores["llm_detector"] < threshold and local:
        weights = {"llm_detector": 0.20, "rag": 0.40, "static": 0.40}
    else:
        weights = DEFAULT_COMPONENT_WEIGHTS
    active = [name for name, result in components.items() if _active(result)]
    total = sum(weights.get(name, 0.0) for name in active)
    effective = {name: weights.get(name, 0.0) / total if name in active and total else 0.0 for name in components}
    score = sum(effective[name] * scores[name] for name in components)
    rules = ["active_weighted_fusion"] if active else ["no_active_component"]
    critical = any(finding["severity"] == "Critical" for finding in local)
    floors = (
        (not detector_active and any(finding["severity"] in {"Critical", "High"} for finding in local),
         0.78 if critical else 0.68,
         "provisional_critical_local_evidence_floor" if critical else "provisional_high_local_evidence_floor"),
        (any(finding["corroborated"] for finding in local), 0.72, "static_rag_corroboration_floor"),
        (any(finding.get("severity") in {"Critical", "High"} and float(finding.get("confidence") or 0.0) >= 0.65
             for finding in local), 0.65, "strong_local_evidence_floor"),
        (detector_fired, 0.58, "llm_detector_threshold_floor"),
        (detector_fired and bool(local), 0.62, "llm_local_corroboration_floor"),
    )
    for condition, floor, rule in floors:
        if condition:
            score = max(score, floor)
            rules.append(rule)
    score = max(0.0, min(1.0, score))
    complete = _active(static_result) and _active(rag_result) and _detector_complete(detector)
    if detector_active and not _detector_complete(detector):
        rules.append("detector_incomplete_no_safe_verdict")
    if score >= DEFAULT_FINAL_THRESHOLD and (local or detector_fired):
        verdict = "Vulnerable"
    else:
        verdict = "Safe" if complete else "Inconclusive"
    unit_results = detector.get("unit_results", [])
    evidence = [*local, *detector_findings(unit_results, {finding.get("unit_id") for finding in local})]
    unit_map = {item.get("unit_id"): item for item in unit_results if item.get("unit_id")}
    for finding in evidence:
        if finding.get("unit_id") in unit_map:
            _apply_detector_lines(finding, unit_map[finding["unit_id"]], overwrite=False)
    vulnerable = verdict == "Vulnerable"
    return {
        "component": "fusion", "status": "complete" if complete else "partial" if active else "inconclusive",
        "verdict": verdict, "is_vulnerable": vulnerable, "score": round(score, 6), "threshold": DEFAULT_FINAL_THRESHOLD,
        "risk_level": "None" if not vulnerable else evidence[0]["severity"] if local else "Unknown",
        "findings": evidence if vulnerable else [], "candidate_findings": [] if vulnerable else evidence,
        "decision_rules": rules, "required_components": ["static", "rag", "llm_detector"], "active_components": active,
        "configured_weights": weights,
        "effective_weights": {name: round(value, 6) for name, value in effective.items()},
        "score_breakdown": {name: {
            "status": components[name].get("status", "unknown"), "raw_score": round(scores[name], 6),
            "weight": round(effective[name], 6), "weighted_score": round(scores[name] * effective[name], 6),
            "verdict": components[name].get("verdict") or "Unknown",
        } for name in components},
    }
