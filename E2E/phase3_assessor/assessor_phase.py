"""
Phase 3 - risk assessment of every confirmed finding (CVSS v3.1 -> Critical/High/Medium/Low).

The LLM Assessor (assessor.py) picks the CVSS metrics; the base score is computed from
the vector, never taken from the model. Findings it could not assess get the default
vector of their SWC (SWC_CVSS_MAP), scored with the same formula. The result is sorted
into a repair priority (1 = repair first).
"""

from __future__ import annotations

import concurrent.futures
import os
from typing import Any

from E2E.common.findings import finding_key
from E2E.phase3_assessor.assessor import AssessorAgent, compute_cvss_31_base_score, cvss_to_level


MAX_LLM_ASSESSOR_CALLS = int(os.getenv("MAX_LLM_ASSESSOR_CALLS", "5"))


SWC_CVSS_MAP: dict[str, dict[str, Any]] = {
    "SWC-107": {
        "severity": "Critical",
        "cvss_score": 8.6,
        "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:H/A:H",
        "likelihood": "High",
        "impact_rating": "Critical",
        "funds_at_risk": True,
        "justification": "Reentrant call before state deduction allows malicious contracts to drain funds recursively.",
        "repair_priority": 1,
    },
    "SWC-105": {
        "severity": "Critical",
        "cvss_score": 8.6,
        "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:H/A:H",
        "likelihood": "High",
        "impact_rating": "Critical",
        "funds_at_risk": True,
        "justification": "Unprotected state-changing function allows any external account to drain contract funds or change owner.",
        "repair_priority": 1,
    },
    "SWC-115": {
        "severity": "High",
        "cvss_score": 7.5,
        "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:U/C:N/I:H/A:H",
        "likelihood": "Medium",
        "impact_rating": "High",
        "funds_at_risk": True,
        "justification": "Authorization check using tx.origin can be bypassed via phishing attacks with intermediate contracts.",
        "repair_priority": 1,
    },
    "SWC-104": {
        "severity": "High",
        "cvss_score": 7.5,
        "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:H/A:N",
        "likelihood": "Medium",
        "impact_rating": "High",
        "funds_at_risk": True,
        "justification": "Failure to check return values of low-level calls allows execution to proceed after failed transactions.",
        "repair_priority": 2,
    },
    "SWC-101": {
        "severity": "High",
        "cvss_score": 7.5,
        "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:H/A:N",
        "likelihood": "Medium",
        "impact_rating": "High",
        "funds_at_risk": True,
        "justification": "Integer overflow/underflow in arithmetic calculations causes incorrect token or balance accounting.",
        "repair_priority": 2,
    },
    "SWC-114": {
        "severity": "High",
        "cvss_score": 7.0,
        "cvss_vector": "CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:U/C:N/I:H/A:N",
        "likelihood": "Medium",
        "impact_rating": "High",
        "funds_at_risk": True,
        "justification": "Transaction ordering dependence allows miners or front-running bots to extract value.",
        "repair_priority": 2,
    },
    "SWC-116": {
        "severity": "Medium",
        "cvss_score": 5.3,
        "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:L/A:N",
        "likelihood": "Medium",
        "impact_rating": "Low",
        "funds_at_risk": False,
        "justification": "Relying on block.timestamp allows miners to manipulate timestamps within small drift windows.",
        "repair_priority": 3,
    },
    "SWC-120": {
        "severity": "Medium",
        "cvss_score": 5.3,
        "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:L/A:N",
        "likelihood": "Medium",
        "impact_rating": "Low",
        "funds_at_risk": False,
        "justification": "Blockhash dependence for randomness can be predicted or biased by miners.",
        "repair_priority": 3,
    },
    "SWC-102": {
        "severity": "Low",
        "cvss_score": 3.1,
        "cvss_vector": "CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:U/C:N/I:L/A:N",
        "likelihood": "Low",
        "impact_rating": "Low",
        "funds_at_risk": False,
        "justification": "Outdated compiler pragma version may contain unpatched compiler bugs.",
        "repair_priority": 4,
    },
}


# Từ vector CVSS mặc định, tính lại điểm bằng công thức (không tin con số viết tay trong bảng) và mức rủi ro tương ứng.
def _with_computed_score(meta: dict[str, Any]) -> dict[str, Any]:
    """Derive the CVSS v3.1 base score and its qualitative rating from the vector itself."""
    metrics = dict(part.split(":", 1) for part in meta["cvss_vector"].split("/")[1:])
    score, vector = compute_cvss_31_base_score(**{key.lower(): value for key, value in metrics.items()})
    return {**meta, "cvss_score": score, "cvss_vector": vector, "severity": cvss_to_level(score)}


# CVSS mặc định cho một SWC (bảng SWC_CVSS_MAP, hoặc mẫu theo mức độ) với điểm đã tính lại bằng công thức.
def get_cvss_defaults(swc: str, severity: str = "Medium") -> dict[str, Any]:
    return _with_computed_score(_cvss_template(swc, severity))


# Chọn mẫu CVSS: theo SWC nếu có trong SWC_CVSS_MAP (107, 105, 115, 104, 101, 114, 116, 120, 102),
# không thì theo mức độ Critical/High/Medium/Low.
def _cvss_template(swc: str, severity: str = "Medium") -> dict[str, Any]:
    if swc in SWC_CVSS_MAP:
        return SWC_CVSS_MAP[swc]
    sev = str(severity or "Medium").capitalize()
    if sev in {"Critical"}:
        return {
            "severity": "Critical",
            "cvss_score": 8.6,
            "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:H/A:H",
            "likelihood": "High",
            "impact_rating": "Critical",
            "funds_at_risk": True,
            "justification": "Critical vulnerability posing direct threat to contract integrity or funds.",
            "repair_priority": 1,
        }
    if sev in {"High"}:
        return {
            "severity": "High",
            "cvss_score": 7.5,
            "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:H/A:N",
            "likelihood": "Medium",
            "impact_rating": "High",
            "funds_at_risk": True,
            "justification": "High severity flaw that could lead to significant unintended behavior or loss.",
            "repair_priority": 2,
        }
    if sev in {"Low"}:
        return {
            "severity": "Low",
            "cvss_score": 3.1,
            "cvss_vector": "CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:U/C:N/I:L/A:N",
            "likelihood": "Low",
            "impact_rating": "Low",
            "funds_at_risk": False,
            "justification": "Low severity code quality or best-practice violation.",
            "repair_priority": 4,
        }
    return {
        "severity": "Medium",
        "cvss_score": 5.3,
        "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:L/A:N",
        "likelihood": "Medium",
        "impact_rating": "Medium",
        "funds_at_risk": False,
        "justification": "Medium severity vulnerability requiring code remediation.",
        "repair_priority": 3,
    }


# [HÀM CHÍNH PHASE 3] Đánh giá rủi ro mọi finding:
#   1) gọi LLM song song (4 luồng) cho tối đa MAX_LLM_ASSESSOR_CALLS (5) finding đầu, có kèm root cause/impact từ Advisor;
#   2) finding còn lại hoặc LLM lỗi -> dùng CVSS mặc định theo SWC (get_cvss_defaults);
#   3) sắp xếp theo repair_priority rồi đánh số lại 1, 2, 3... (1 = sửa trước).
# Gọi bởi: gpu_pipeline.run_server_gpu_pipeline (bỏ qua ở chế độ single_pass).
def _generate_assessor_section(
    findings: list[dict[str, Any]],
    code: str = "",
    *,
    use_llm: bool = False,
    advisor_suggestions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    assessments = []
    agent = None
    if use_llm:
        try:
            agent = AssessorAgent()
        except Exception:
            pass

    # Map suggestions by finding_id and SWC to carry forward Phase 2 insights
    suggestions_by_id = {s.get("finding_id"): s for s in (advisor_suggestions or []) if s.get("finding_id")}
    suggestions_by_swc = {s.get("swc"): s for s in (advisor_suggestions or []) if s.get("swc")}

    # 1. Parallelize LLM risk assessments for top findings
    llm_assessor_results = {}
    if agent is not None and findings:
        findings_to_query = findings[:MAX_LLM_ASSESSOR_CALLS]
        # Gọi assess_one cho một finding (đã bổ sung root_cause/impact từ suggestion của Advisor).
        def _call_assessor(finding_dict):
            f_key = finding_key(finding_dict)
            f_enr = dict(finding_dict)
            s = suggestions_by_id.get(finding_dict.get("finding_id")) or suggestions_by_swc.get(finding_dict.get("swc")) or {}
            if s:
                if s.get("root_cause") and not f_enr.get("root_cause"):
                    f_enr["root_cause"] = s["root_cause"]
                if s.get("impact") and not f_enr.get("impact"):
                    f_enr["impact"] = s["impact"]
            try:
                res = agent.assess_one(code, f_enr)
                return f_key, res
            except Exception:
                return f_key, None

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(findings_to_query), 4)) as executor:
            for f_key, res in executor.map(_call_assessor, findings_to_query):
                if res and getattr(res, "cvss_score", 0) > 0 and not getattr(res, "error", None):
                    llm_assessor_results[f_key] = res

    for f in findings:
        swc = f.get("swc", "SWC-100")
        unit_id = f.get("unit_id") or "contract"
        title = f.get("title") or f"Vulnerability {swc}"
        severity = f.get("severity") or ("High" if swc in {"SWC-107", "SWC-105", "SWC-115"} else "Medium")
        fid = finding_key(f)

        # 1. Check if parallel LLM risk assessment succeeded
        res = llm_assessor_results.get(fid)
        if res:
            assessments.append({
                "finding_id": fid,
                "swc": swc,
                "unit_id": unit_id,
                "title": title,
                "severity": res.risk_level or severity,
                "risk_level": res.risk_level or severity,
                "impact": res.justification or f.get("impact") or f.get("description") or f"Risk evaluated with CVSS score {res.cvss_score} ({res.risk_level}).",
                "cvss_score": res.cvss_score,
                "cvss_vector": res.cvss_vector,
                "likelihood": res.likelihood,
                "impact_rating": res.impact_rating,
                "funds_at_risk": res.funds_at_risk,
                "justification": res.justification,
                "repair_priority": res.repair_priority or 1,
            })
            continue

        # 2. Grounded CVSS calculation based on calibrated SWC standard
        cvss_meta = get_cvss_defaults(swc, severity)
        assessments.append({
            "finding_id": fid,
            "swc": swc,
            "unit_id": unit_id,
            "title": title,
            "severity": cvss_meta["severity"],
            "risk_level": cvss_meta["severity"],
            "impact": f"High risk of unauthorized state modification or asset drain via {title}.",
            "cvss_score": cvss_meta["cvss_score"],
            "cvss_vector": cvss_meta["cvss_vector"],
            "likelihood": cvss_meta["likelihood"],
            "impact_rating": cvss_meta["impact_rating"],
            "funds_at_risk": cvss_meta["funds_at_risk"],
            "justification": cvss_meta["justification"],
            "repair_priority": cvss_meta["repair_priority"],
        })

    assessments.sort(key=lambda item: item.get("repair_priority", 100))
    for idx, item in enumerate(assessments, 1):
        item["repair_priority"] = idx

    return {
        "status": "ok",
        "assessments": assessments,
    }
