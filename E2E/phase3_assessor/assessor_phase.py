"""Phase 3 - đánh giá rủi ro mọi finding (CVSS v3.1) và thứ tự ưu tiên sửa (repair_priority) cho Fixer."""

import concurrent.futures

from E2E.common.findings import finding_key
from E2E.common.knowledge_base import KnowledgeBase
from E2E.config import MAX_LLM_ASSESSOR_CALLS
from E2E.phase3_assessor.assessor import AssessorAgent, compute_cvss_31_base_score, cvss_to_level

# Vector CVSS mặc định theo SWC (điểm luôn TÍNH từ vector) + lý giải.
SWC_CVSS_MAP: dict[str, tuple[str, str]] = {
    "SWC-107": ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:H/A:H",
                "Reentrant call before state deduction allows malicious contracts to drain funds recursively."),
    "SWC-105": ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:H/A:H",
                "Unprotected state-changing function allows any external account to drain contract funds "
                "or change owner."),
    "SWC-115": ("CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:U/C:N/I:H/A:H",
                "Authorization check using tx.origin can be bypassed via phishing attacks "
                "with intermediate contracts."),
    "SWC-104": ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:H/A:N",
                "Failure to check return values of low-level calls allows execution to proceed "
                "after failed transactions."),
    "SWC-101": ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:H/A:N",
                "Integer overflow/underflow in arithmetic calculations causes incorrect token or balance accounting."),
    "SWC-114": ("CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:U/C:N/I:H/A:N",
                "Transaction ordering dependence allows miners or front-running bots to extract value."),
    "SWC-116": ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:L/A:N",
                "Relying on block.timestamp allows miners to manipulate timestamps within small drift windows."),
    "SWC-120": ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:L/A:N",
                "Blockhash dependence for randomness can be predicted or biased by miners."),
    "SWC-102": ("CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:U/C:N/I:L/A:N",
                "Outdated compiler pragma version may contain unpatched compiler bugs."),
}
# SWC ngoài bảng: vector mặc định theo mức độ của finding.
SEVERITY_CVSS_MAP: dict[str, tuple[str, str]] = {
    "Critical": ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:H/A:H",
                 "Critical vulnerability posing direct threat to contract integrity or funds."),
    "High": ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:H/A:N",
             "High severity flaw that could lead to significant unintended behavior or loss."),
    "Medium": ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:L/A:N",
               "Medium severity vulnerability requiring code remediation."),
    "Low": ("CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:U/C:N/I:L/A:N", "Low severity code quality or best-practice violation."),
}


# CVSS mặc định: vector theo SWC (không có thì theo mức độ) -> TÍNH điểm + mức rủi ro.
def get_cvss_defaults(swc: str, severity: str = "Medium") -> dict:
    vector, justification = SWC_CVSS_MAP.get(swc) or SEVERITY_CVSS_MAP.get(str(severity or "Medium").capitalize(),
                                                                            SEVERITY_CVSS_MAP["Medium"])
    metrics = dict(part.split(":", 1) for part in vector.split("/")[1:])
    score, vector = compute_cvss_31_base_score(**{key.lower(): value for key, value in metrics.items()})
    return {"cvss_score": score, "cvss_vector": vector, "severity": cvss_to_level(score),
            "justification": justification}


# [HÀM CHÍNH PHASE 3] LLM (4 luồng) cho <= MAX_LLM_ASSESSOR_CALLS finding, còn lại CVSS mặc định; CVSS giảm dần = ưu tiên.
def _generate_assessor_section(findings: list[dict], code: str = "", *, use_llm: bool = False,
                               advisor_suggestions: list[dict] | None = None,
                               knowledge_base: KnowledgeBase | None = None) -> dict:
    suggestions = {item.get("finding_id"): item for item in advisor_suggestions or [] if item.get("finding_id")}
    agent = AssessorAgent(knowledge_base=knowledge_base) if use_llm else None

    # Gọi assess_one cho một finding (bổ sung root_cause / impact của Advisor nếu finding chưa có).
    def assess(finding: dict):
        enriched, suggestion = dict(finding), suggestions.get(finding_key(finding), {})
        for field in ("root_cause", "impact"):
            if suggestion.get(field) and not enriched.get(field):
                enriched[field] = suggestion[field]
        try:
            return finding_key(finding), agent.assess_one(code, enriched)
        except Exception:
            return finding_key(finding), None

    llm_results = {}
    if agent is not None and findings:
        queried = findings[:MAX_LLM_ASSESSOR_CALLS]
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(queried), 4)) as executor:
            llm_results = {key: result for key, result in executor.map(assess, queried)
                           if result is not None and result.cvss_score > 0}
    assessments = []
    for finding in findings:
        swc, key = finding.get("swc", "SWC-100"), finding_key(finding)
        if result := llm_results.get(key):
            cvss = {"cvss_score": result.cvss_score, "cvss_vector": result.cvss_vector, "severity": result.risk_level,
                    "justification": result.justification, "cvss_source": "llm"}
        else:
            severity = finding.get("severity") or ("High" if swc in {"SWC-107", "SWC-105", "SWC-115"} else "Medium")
            cvss = {**get_cvss_defaults(swc, severity), "cvss_source": "swc_default"}
        assessments.append({"finding_id": key, "swc": swc, "unit_id": finding.get("unit_id") or "contract",
                            "title": finding.get("title") or f"Vulnerability {swc}", **cvss})
    assessments.sort(key=lambda item: -float(item["cvss_score"]))
    for index, item in enumerate(assessments, 1):
        item["repair_priority"] = index
    return {"status": "ok", "assessments": assessments}
