from __future__ import annotations

from collections import defaultdict
from typing import Any

from E2E.config import (
    DEFAULT_COMPONENT_WEIGHTS,
    DEFAULT_DETECTOR_THRESHOLD,
    DEFAULT_FINAL_THRESHOLD,
    RISK_ORDER,
)


def _score(component: dict[str, Any]) -> float:
    return float(
        component.get("score")
        or component.get("final_score")
        or component.get("vulnerability_probability")
        or 0.0
    )


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
                "corroborated": len(sources) > 1,
                "line": next((item.get("line") for item in evidence if item.get("line")), None),
                "description": next(
                    (item.get("description") for item in evidence if item.get("description")),
                    "",
                ),
                "remediation": next(
                    (item.get("remediation") for item in evidence if item.get("remediation")),
                    "",
                ),
                "checklist": next(
                    (item.get("checklist") for item in evidence if item.get("checklist")),
                    [],
                ),
                "fixed_code": next(
                    (item.get("fixed_code") for item in evidence if item.get("fixed_code")),
                    "",
                ),
                "vulnerable_code": next(
                    (item.get("vulnerable_code") for item in evidence if item.get("vulnerable_code")),
                    "",
                ),
                "best_match_id": next(
                    (item.get("best_match_id") for item in evidence if item.get("best_match_id")),
                    None,
                ),
                "similarity_score": next(
                    (item.get("similarity_score") for item in evidence if item.get("similarity_score") is not None),
                    None,
                ),
                "has_code_patch": any(bool(item.get("has_code_patch", bool(item.get("fixed_code")))) for item in evidence),
                "evidence": evidence,
            }
        )
    return sorted(
        merged,
        key=lambda item: (RISK_ORDER.get(item["severity"], 0), item["confidence"]),
        reverse=True,
    )


def detector_findings(
    detector: dict[str, Any],
    local_findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    local_units = {item.get("unit_id") for item in local_findings}
    results = detector.get("unit_results", [])
    findings = []
    for item in results:
        if item.get("label") == "Vulnerable" and item.get("unit_id") not in local_units:
            start_line = item.get("start_line")
            line_scores = item.get("line_scores")
            line = item.get("line") or start_line
            if line_scores and isinstance(line_scores, list) and line_scores and start_line and not line:
                max_idx = line_scores.index(max(line_scores))
                line = start_line + max_idx
            findings.append({
                "finding_id": f"UNMAPPED:{item.get('unit_id') or 'source'}",
                "swc": "UNMAPPED",
                "unit_id": item.get("unit_id"),
                "title": "CodeLlama binary vulnerability signal",
                "severity": "Unknown",
                "confidence": round(float(item.get("vulnerability_probability") or 0.0), 6),
                "sources": ["llm_detector"],
                "corroborated": False,
                "line": line,
                "start_line": start_line,
                "end_line": item.get("end_line"),
                "line_scores": line_scores,
                "description": (
                    "The fine-tuned binary classifier crossed its vulnerability threshold. "
                    "Static/RAG did not assign a supported SWC category to this unit."
                ),
                "remediation": "Perform targeted manual review and use the Advisor phase for root-cause analysis.",
                "checklist": [],
                "evidence": [item],
            })
    return findings


def fuse_detection(
    static_result: dict[str, Any],
    rag_result: dict[str, Any],
    llm_result: dict[str, Any] | None = None,
    *,
    threshold: float = DEFAULT_FINAL_THRESHOLD,
    configured_weights: dict[str, float] | None = None,
    require_llm_for_safe: bool = True,
) -> dict[str, Any]:
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
    if configured_weights is not None:
        weights = dict(configured_weights)
    elif not _active(detector):
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

    all_required_active = all(_active(components[name]) for name in ("static", "rag", "llm_detector"))
    positive_evidence = bool(local_findings) or (
        _active(detector) and component_scores["llm_detector"] >= detector_threshold
    )
    if score >= threshold and positive_evidence:
        verdict = "Vulnerable"
    elif all_required_active or not require_llm_for_safe:
        verdict = "Safe"
    else:
        verdict = "Inconclusive"

    if all_required_active:
        analysis_status = "complete"
    elif active_names:
        analysis_status = "partial"
    else:
        analysis_status = "inconclusive"
    binary_findings = detector_findings(detector, local_findings)
    evidence_findings = [*local_findings, *binary_findings]
    unit_map = {item.get("unit_id"): item for item in detector.get("unit_results", []) if item.get("unit_id")}
    for finding in evidence_findings:
        uid = finding.get("unit_id")
        if uid and uid in unit_map:
            u_res = unit_map[uid]
            st_line = u_res.get("start_line")
            end_line = u_res.get("end_line")
            l_scores = u_res.get("line_scores")
            if st_line and not finding.get("start_line"):
                finding["start_line"] = st_line
            if end_line and not finding.get("end_line"):
                finding["end_line"] = end_line
            if l_scores and isinstance(l_scores, list) and l_scores:
                finding["line_scores"] = l_scores
                if not finding.get("line") and st_line:
                    max_idx = l_scores.index(max(l_scores))
                    finding["line"] = st_line + max_idx
    findings = evidence_findings if verdict == "Vulnerable" else []
    candidate_findings = evidence_findings if verdict != "Vulnerable" else []
    risk_level = findings[0]["severity"] if findings and verdict == "Vulnerable" else "None"
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
                "verdict": components[name].get("verdict")
                or components[name].get("final_verdict")
                or "Unknown",
            }
            for name in components
        },
    }
