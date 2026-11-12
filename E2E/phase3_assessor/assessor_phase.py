"""
Phase 3 - risk assessment of every confirmed finding (CVSS v3.1 -> Critical/High/Medium/Low).

The LLM Assessor (assessor.py) picks the CVSS metrics; the base score is always computed
from the vector, never taken from the model. Findings it could not assess (LLM error, or
beyond MAX_LLM_ASSESSOR_CALLS in config.py) get the default vector of their SWC (SWC_CVSS_MAP),
scored with the same formula; `cvss_source` records which path was used.

Output order = repair priority (1 = repair first, highest CVSS score first). The Fixer
(Phase 4) repairs the functions in this order.
"""

from __future__ import annotations

import concurrent.futures
from typing import Any

from E2E.common.findings import finding_key
from E2E.config import MAX_LLM_ASSESSOR_CALLS
from E2E.phase2_advisor.advisor import RepairKnowledgeBase
from E2E.phase3_assessor.assessor import AssessorAgent, compute_cvss_31_base_score, cvss_to_level


# Vector CVSS mặc định theo SWC (điểm và mức rủi ro luôn được TÍNH từ vector, không gõ tay).
SWC_CVSS_MAP: dict[str, dict[str, str]] = {
    "SWC-107": {
        "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:H/A:H",
        "justification": "Reentrant call before state deduction allows malicious contracts to drain funds recursively.",
    },
    "SWC-105": {
        "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:H/A:H",
        "justification": "Unprotected state-changing function allows any external account to drain contract funds or change owner.",
    },
    "SWC-115": {
        "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:U/C:N/I:H/A:H",
        "justification": "Authorization check using tx.origin can be bypassed via phishing attacks with intermediate contracts.",
    },
    "SWC-104": {
        "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:H/A:N",
        "justification": "Failure to check return values of low-level calls allows execution to proceed after failed transactions.",
    },
    "SWC-101": {
        "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:H/A:N",
        "justification": "Integer overflow/underflow in arithmetic calculations causes incorrect token or balance accounting.",
    },
    "SWC-114": {
        "cvss_vector": "CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:U/C:N/I:H/A:N",
        "justification": "Transaction ordering dependence allows miners or front-running bots to extract value.",
    },
    "SWC-116": {
        "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:L/A:N",
        "justification": "Relying on block.timestamp allows miners to manipulate timestamps within small drift windows.",
    },
    "SWC-120": {
        "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:L/A:N",
        "justification": "Blockhash dependence for randomness can be predicted or biased by miners.",
    },
    "SWC-102": {
        "cvss_vector": "CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:U/C:N/I:L/A:N",
        "justification": "Outdated compiler pragma version may contain unpatched compiler bugs.",
    },
}


# SWC ngoài bảng: vector mặc định theo mức độ của finding.
SEVERITY_CVSS_MAP: dict[str, dict[str, str]] = {
    "Critical": {
        "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:H/A:H",
        "justification": "Critical vulnerability posing direct threat to contract integrity or funds.",
    },
    "High": {
        "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:H/A:N",
        "justification": "High severity flaw that could lead to significant unintended behavior or loss.",
    },
    "Medium": {
        "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:L/A:N",
        "justification": "Medium severity vulnerability requiring code remediation.",
    },
    "Low": {
        "cvss_vector": "CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:U/C:N/I:L/A:N",
        "justification": "Low severity code quality or best-practice violation.",
    },
}


# CVSS mặc định cho một SWC: chọn vector (theo SWC, không có thì theo mức độ), rồi TÍNH điểm + mức rủi ro.
def get_cvss_defaults(swc: str, severity: str = "Medium") -> dict[str, Any]:
    template = SWC_CVSS_MAP.get(swc) or SEVERITY_CVSS_MAP.get(
        str(severity or "Medium").capitalize(), SEVERITY_CVSS_MAP["Medium"]
    )
    metrics = dict(part.split(":", 1) for part in template["cvss_vector"].split("/")[1:])
    score, vector = compute_cvss_31_base_score(**{key.lower(): value for key, value in metrics.items()})
    return {
        "cvss_score": score,
        "cvss_vector": vector,
        "severity": cvss_to_level(score),
        "justification": template["justification"],
    }


# [HÀM CHÍNH PHASE 3] Đánh giá rủi ro mọi finding:
#   1) gọi LLM song song (4 luồng) cho tối đa MAX_LLM_ASSESSOR_CALLS (5, config.py) finding đầu, có kèm root cause/impact
#      từ suggestion của Advisor (cùng finding_id);
#   2) finding còn lại hoặc LLM lỗi -> CVSS mặc định theo SWC (get_cvss_defaults); cvss_source ghi rõ nguồn;
#   3) sắp xếp theo điểm CVSS giảm dần -> repair_priority 1, 2, 3... (1 = sửa trước; Fixer vá theo thứ tự này).
# knowledge_base: KB dùng chung của gpu_pipeline (không truyền thì tự nạp).
# Gọi bởi: gpu_pipeline.run_server_gpu_pipeline (use_llm=True); test dùng use_llm=False (không gọi LLM).
def _generate_assessor_section(
    findings: list[dict[str, Any]],
    code: str = "",
    *,
    use_llm: bool = False,
    advisor_suggestions: list[dict[str, Any]] | None = None,
    knowledge_base: RepairKnowledgeBase | None = None,
) -> dict[str, Any]:
    agent = AssessorAgent(knowledge_base=knowledge_base) if use_llm else None

    # Carry the Advisor's root cause / impact (Phase 2) into the Assessor prompt.
    suggestions_by_id = {s.get("finding_id"): s for s in (advisor_suggestions or []) if s.get("finding_id")}

    llm_results = {}
    if agent is not None and findings:
        findings_to_query = findings[:MAX_LLM_ASSESSOR_CALLS]

        # Gọi assess_one cho một finding (đã bổ sung root_cause/impact từ suggestion của Advisor).
        def _call_assessor(finding: dict[str, Any]):
            enriched = dict(finding)
            suggestion = suggestions_by_id.get(finding_key(finding), {})
            for field in ("root_cause", "impact"):
                if suggestion.get(field) and not enriched.get(field):
                    enriched[field] = suggestion[field]
            try:
                return finding_key(finding), agent.assess_one(code, enriched)
            except Exception:
                return finding_key(finding), None

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(findings_to_query), 4)) as executor:
            for key, result in executor.map(_call_assessor, findings_to_query):
                if result is not None and result.cvss_score > 0:
                    llm_results[key] = result

    assessments = []
    for finding in findings:
        swc = finding.get("swc", "SWC-100")
        key = finding_key(finding)
        result = llm_results.get(key)
        if result is not None:
            cvss = {
                "cvss_score": result.cvss_score,
                "cvss_vector": result.cvss_vector,
                "severity": result.risk_level,
                "justification": result.justification,
                "cvss_source": "llm",
            }
        else:
            default_severity = finding.get("severity") or ("High" if swc in {"SWC-107", "SWC-105", "SWC-115"} else "Medium")
            cvss = {**get_cvss_defaults(swc, default_severity), "cvss_source": "swc_default"}
        assessments.append({
            "finding_id": key,
            "swc": swc,
            "unit_id": finding.get("unit_id") or "contract",
            "title": finding.get("title") or f"Vulnerability {swc}",
            **cvss,
        })

    # Repair priority: highest CVSS score first (stable sort keeps the detection order on ties).
    assessments.sort(key=lambda item: -float(item["cvss_score"]))
    for index, item in enumerate(assessments, 1):
        item["repair_priority"] = index

    return {
        "status": "ok",
        "assessments": assessments,
    }
