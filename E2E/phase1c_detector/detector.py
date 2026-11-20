"""Phase 1C - detector Qwen2.5-Coder-14B + LoRA v13: P(vulnerable) cho từng hàm qua GPU server (fail-closed)."""

from E2E.common.gpu_client import detect_via_server
from E2E.phase0_parsing.code_units import CodeUnit


# Tóm tắt detector (fail-closed): có hàm Vulnerable -> Vulnerable; chấm đủ và đều Safe -> Safe; còn lại Inconclusive.
def summarize_detector(unit_results: list[dict], threshold: float) -> dict:
    scored = [item for item in unit_results if item.get("vulnerability_probability") is not None]
    errored = [item for item in unit_results if item.get("status") == "error"]
    status = "ok" if not errored else "partial" if scored else "error"
    best = max((float(item["vulnerability_probability"]) for item in scored), default=None)
    if any(item.get("label") == "Vulnerable" for item in scored):
        verdict = "Vulnerable"
    else:
        verdict = "Safe" if status == "ok" else "Inconclusive"
    best = None if best is None else round(best, 4)
    summary = {"status": status, "verdict": verdict, "vulnerability_probability": best, "score": best,
               "threshold": threshold, "unit_results": unit_results}
    if errored:
        summary["error"] = f"{len(errored)}/{len(unit_results)} unit(s) could not be scored by the detector"
    return summary


# [HÀM CHÍNH PHASE 1C] Gửi từng unit có thân lên /api/v1/detect; Vulnerable nếu P >= threshold; lỗi -> P=None.
def score_units_via_server(units: list[CodeUnit], threshold: float, server_url: str | None = None) -> list[dict]:
    results = []
    for unit in (unit for unit in units if unit.has_body):
        base = {"unit_id": unit.unit_id, "threshold": threshold, "start_line": unit.start_line,
                "end_line": unit.end_line}
        try:
            response = detect_via_server(unit.code, threshold=threshold, server_url=server_url)
            probability = float(response["vulnerable_probability"])
        except Exception as error:
            results.append({**base, "status": "error", "label": "Inconclusive", "vulnerability_probability": None,
                            "error": f"{type(error).__name__}: {error}"})
            continue
        results.append({**base, "status": "ok", "label": "Vulnerable" if probability >= threshold else "Safe",
                        "vulnerability_probability": round(probability, 4), "line_scores": response.get("line_scores")})
    return results
