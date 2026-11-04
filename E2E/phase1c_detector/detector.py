"""
Phase 1C - neural detector: Qwen2.5-Coder-14B + LoRA v13 (sequence classification).

Each code unit with a body is sent to the GPU server (/api/v1/detect) and gets
P(vulnerable); label = Vulnerable when P >= threshold (0.70 from threshold_config.json).
The detector is binary: it does not name an SWC (SWC labels come from Phase 1A/1B).

Fail-closed: a unit the server could not score is recorded as an error with no
probability, and a contract with unscored units is never summarised as Safe.

Model training and serving code: gpu_server/train_qwen25_v13.py, gpu_server/serve_v13_api.py.
"""

from __future__ import annotations

from typing import Any

from E2E.common.gpu_client import detect_via_server
from E2E.phase0_parsing.code_units import CodeUnit


def score_units_via_server(
    units: list[CodeUnit],
    threshold: float,
    server_url: str | None = None,
) -> list[dict[str, Any]]:
    """Score every unit with a body. A failed request is recorded as an error, never as a probability."""
    results: list[dict[str, Any]] = []
    for unit in units:
        if not unit.has_body:
            continue
        base = {
            "unit_id": unit.unit_id,
            "threshold": threshold,
            "start_line": unit.start_line,
            "end_line": unit.end_line,
        }
        try:
            response = detect_via_server(unit.code, threshold=threshold, server_url=server_url)
            probability = float(response["vulnerable_probability"])
        except Exception as error:
            results.append({
                **base,
                "status": "error",
                "label": "Inconclusive",
                "vulnerability_probability": None,
                "error": f"{type(error).__name__}: {error}",
            })
            continue
        results.append({
            **base,
            "status": "ok",
            "label": "Vulnerable" if probability >= threshold else "Safe",
            "vulnerability_probability": round(probability, 4),
            "line_scores": response.get("line_scores"),
        })
    return results


def summarize_detector(unit_results: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """Contract-level detector summary that stays honest about failed units."""
    scored = [item for item in unit_results if item.get("vulnerability_probability") is not None]
    errored = [item for item in unit_results if item.get("status") == "error"]
    if errored and not scored:
        status = "error"
    elif errored:
        status = "partial"
    else:
        status = "ok"
    max_probability = max((float(item["vulnerability_probability"]) for item in scored), default=None)
    if any(item.get("label") == "Vulnerable" for item in scored):
        verdict = "Vulnerable"
    elif status == "ok":
        verdict = "Safe"
    else:
        verdict = "Inconclusive"
    summary = {
        "status": status,
        "verdict": verdict,
        "vulnerability_probability": round(max_probability, 4) if max_probability is not None else None,
        "score": round(max_probability, 4) if max_probability is not None else None,
        "threshold": threshold,
        "unit_results": unit_results,
    }
    if errored:
        summary["error"] = f"{len(errored)}/{len(unit_results)} unit(s) could not be scored by the detector"
    return summary
