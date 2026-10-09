from __future__ import annotations

import json
from typing import Any

from E2E.local.local_llm_runner import run_local_llm_pipeline


def _derive_findings(request: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    local_p1 = request.get("local_phase1", {})
    for src in ("static", "rag", "provisional_fusion"):
        for f in local_p1.get(src, {}).get("findings", []):
            findings.append(dict(f))
    return findings


def run_fast_local_pipeline(
    request: dict[str, Any],
    backend: str = "local_fast",
) -> dict[str, Any]:
    """Runs local inference pipeline without Kaggle dependency.
    
    Supports:
    - 'local' / 'auto': Apple Silicon M2 native fast-path with heuristic and AST security patching.
    - 'ollama': Local Ollama service (codellama:7b).
    - 'local_fast': Ultra-fast grounded analysis.
    """
    if backend in {"local", "ollama", "local_mps", "auto"}:
        return run_local_llm_pipeline(request, backend=backend)

    run_id = request.get("run_id", "local-run")
    source_code = request.get("source", {}).get("code", "")
    findings = _derive_findings(request)

    advisor_suggestions = []
    assessor_items = []
    verifier_items = []
    recommendations = ["Run the complete project test suite.", "Review patch with security tooling."]

    for finding in findings:
        swc = finding.get("swc", "SWC-100")
        unit_id = finding.get("unit_id") or "contract"
        title = finding.get("title", "Potential Vulnerability")
        remediation = finding.get("remediation") or f"Apply secure pattern for {title}."

        advisor_suggestions.append({
            "finding_id": finding.get("finding_id") or f"{swc}:{unit_id}",
            "swc": swc,
            "unit_id": unit_id,
            "title": title,
            "root_cause": finding.get("description") or f"Detected unsafe usage matching {title}.",
            "repair_steps": [remediation],
            "recommendations": [remediation],
        })

        assessor_items.append({
            "finding_id": finding.get("finding_id") or f"{swc}:{unit_id}",
            "swc": swc,
            "unit_id": unit_id,
            "title": title,
            "severity": finding.get("severity", "Medium"),
            "impact": f"Risk of {title} exploit.",
            "cvss_score": 7.5 if finding.get("severity") in {"High", "Critical"} else 5.0,
            "repair_priority": 1 if finding.get("severity") == "Critical" else 2,
        })

        verifier_items.append({
            "finding_id": finding.get("finding_id") or f"{swc}:{unit_id}",
            "status": "Verified",
            "residual_risk": "",
        })

    phases = {
        "advisor": {
            "status": "ok",
            "suggestions": advisor_suggestions,
            "recommendations": recommendations,
        },
        "assessor": {
            "status": "ok",
            "assessments": assessor_items,
        },
        "fixer": {
            "status": "ok",
            "fixed_code": source_code,
            "patch_outcomes": [],
            "patches": [],
        },
        "verifier": {
            "status": "ok",
            "verifications": verifier_items,
            "verdict": "Safe" if not findings else "Vulnerable",
            "overall_verdict": "Local fast pass completed.",
        },
        "reporter": {
            "status": "ok",
            "executive_summary": f"Audit completed for run {run_id}. Identified {len(findings)} findings.",
            "recommendations": recommendations,
        },
    }

    return {
        "schema_version": "1.0.0",
        "pipeline_version": "1.0.0",
        "status": "ok",
        "run_id": run_id,
        "backend": backend,
        "source_sha256": request.get("source", {}).get("sha256", ""),
        "phases": phases,
    }
