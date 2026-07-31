from __future__ import annotations

from collections import defaultdict
from typing import Any

from E2E.config import DEFAULT_COMPONENT_WEIGHTS, DEFAULT_FINAL_THRESHOLD, RISK_ORDER


def _score(component: dict[str, Any]) -> float:
    return float(
        component.get("score")
        or component.get("final_score")
        or component.get("vulnerability_probability")
        or 0.0
    )


def _active(component: dict[str, Any]) -> bool:
    return component.get("status") == "ok"


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
    return [
        {
            "finding_id": f"UNMAPPED:{item.get('unit_id') or 'source'}",
            "swc": "UNMAPPED",
            "unit_id": item.get("unit_id"),
            "title": "CodeLlama binary vulnerability signal",
            "severity": "Unknown",
            "confidence": round(float(item.get("vulnerability_probability") or 0.0), 6),
            "sources": ["llm_detector"],
            "corroborated": False,
            "line": item.get("start_line"),
            "description": (
                "The fine-tuned binary classifier crossed its vulnerability threshold. "
                "Static/RAG did not assign a supported SWC category to this unit."
            ),
            "remediation": "Perform targeted manual review and use the Advisor phase for root-cause analysis.",
            "checklist": [],
            "evidence": [item],
        }
        for item in results
        if item.get("label") == "Vulnerable" and item.get("unit_id") not in local_units
    ]


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
    weights = configured_weights or DEFAULT_COMPONENT_WEIGHTS
    active_names = [name for name, result in components.items() if _active(result)]
    weight_total = sum(weights.get(name, 0.0) for name in active_names)
    effective_weights = {
        name: (weights.get(name, 0.0) / weight_total if name in active_names and weight_total else 0.0)
        for name in components
    }
    component_scores = {name: _score(result) for name, result in components.items()}
    weighted_score = sum(effective_weights[name] * component_scores[name] for name in components)
    local_findings = merge_findings(static_result, rag_result)

    score = weighted_score
    decision_rules = ["active_weighted_fusion"] if active_names else ["no_active_component"]
    if not _active(detector) and any(
        finding["severity"] == "Critical" for finding in local_findings
    ):
        score = max(score, 0.78)
        decision_rules.append("provisional_critical_local_evidence_floor")
    if any(finding["corroborated"] for finding in local_findings):
        score = max(score, 0.72)
        decision_rules.append("static_rag_corroboration_floor")
    detector_threshold = float(detector.get("threshold") or 0.25)
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
