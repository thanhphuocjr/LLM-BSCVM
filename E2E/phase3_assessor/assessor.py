"""Phase 3 - Assessor (agent #3 của LLM-BSCVM): LLM chọn 8 metric CVSS v3.1, điểm luôn TÍNH bằng công thức FIRST."""

import math
from dataclasses import dataclass

from E2E.common.gpu_client import ServerGPUClient
from E2E.common.knowledge_base import KnowledgeBase, default_knowledge_base

ASSESSOR_SYSTEM_INSTRUCTION = (
    "You are a smart-contract security risk analyst. For a single confirmed "
    "vulnerability, select each CVSS v3.1 metric based on the specific smart-contract context: "
    "AV (Attack Vector), AC (Attack Complexity), PR (Privileges Required), UI (User Interaction), "
    "S (Scope), C (Confidentiality), I (Integrity), A (Availability). "
    "The final CVSS score will be computed algorithmically from your metric selections. "
    "Return ONLY valid JSON matching the requested schema."
)
# Schema JSON yêu cầu Assessor trả về, kèm giải thích từng metric theo ngữ cảnh blockchain.
_SCHEMA = (
    "{\n"
    '  "attack_vector": "N|A|L|P",      // N: callable via any blockchain tx, A: adjacent, L: local, P: physical\n'
    '  "attack_complexity": "L|H",      // L: standard tx exploit, H: requires front-running/race condition/miner\n'
    '  "privileges_required": "N|L|H",  // N: public permissionless, L: regular caller, H: contract owner/admin\n'
    '  "user_interaction": "N|R",       // N: direct exploit, R: requires victim transaction / approve\n'
    '  "scope": "U|C",                  // U: this contract only, C: impacts external tokens / protocols\n'
    '  "confidentiality": "N|L|H",      // information disclosure\n'
    '  "integrity": "N|L|H",            // unauthorized token transfer, state corruption\n'
    '  "availability": "N|L|H",         // frozen ether, contract bricked, locked balances\n'
    '  "funds_at_risk": boolean,        // true if assets or token balances can be drained or frozen\n'
    '  "justification": string          // technical rationale for metric choices\n'
    "}"
)
# Trọng số chuẩn CVSS v3.1 của từng metric (PR khác nhau khi scope Unchanged / Changed).
_WEIGHTS = {"AV": {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.20}, "AC": {"L": 0.77, "H": 0.44},
            "PR_U": {"N": 0.85, "L": 0.62, "H": 0.27}, "PR_C": {"N": 0.85, "L": 0.68, "H": 0.50},
            "UI": {"N": 0.85, "R": 0.62}, "CIA": {"N": 0.0, "L": 0.22, "H": 0.56}}


# Đổi điểm CVSS thành mức rủi ro: >= 9 Critical, >= 7 High, >= 4 Medium, còn lại Low.
def cvss_to_level(score: float) -> str:
    return "Critical" if score >= 9.0 else "High" if score >= 7.0 else "Medium" if score >= 4.0 else "Low"


# Chuẩn hoá một metric về một chữ cái hợp lệ ("Network" -> "N"); không hợp lệ -> mặc định.
def _metric_letter(value: str, allowed: str, default: str) -> str:
    letter = str(value or "").strip()[:1].upper()
    return letter if letter and letter in allowed else default


# Điểm CVSS v3.1 base score theo công thức FIRST từ 8 metric (AV, AC, PR, UI, S, C, I, A) -> (điểm, vector).
def compute_cvss_31_base_score(av: str = "N", ac: str = "L", pr: str = "N", ui: str = "N", s: str = "U",
                               c: str = "N", i: str = "H", a: str = "H") -> tuple[float, str]:
    av, ac, pr, ui, s = (_metric_letter(value, allowed, default) for value, allowed, default in
                         ((av, "NALP", "N"), (ac, "LH", "L"), (pr, "NLH", "N"), (ui, "NR", "N"), (s, "UC", "U")))
    c, i, a = (_metric_letter(value, "NLH", default) for value, default in ((c, "N"), (i, "H"), (a, "H")))
    changed = s == "C"
    iss = 1.0 - ((1.0 - _WEIGHTS["CIA"][c]) * (1.0 - _WEIGHTS["CIA"][i]) * (1.0 - _WEIGHTS["CIA"][a]))
    impact = 7.52 * (iss - 0.029) - 3.25 * ((iss - 0.02) ** 15) if changed else 6.42 * iss
    pr_weight = _WEIGHTS["PR_C" if changed else "PR_U"][pr]
    exploitability = 8.22 * _WEIGHTS["AV"][av] * _WEIGHTS["AC"][ac] * pr_weight * _WEIGHTS["UI"][ui]
    if impact <= 0:
        base_score = 0.0
    else:
        base_score = math.ceil(min((1.08 if changed else 1.0) * (impact + exploitability), 10.0) * 10.0) / 10.0
    return round(base_score, 1), f"CVSS:3.1/AV:{av}/AC:{ac}/PR:{pr}/UI:{ui}/S:{s}/C:{c}/I:{i}/A:{a}"


# Đánh giá rủi ro của MỘT lỗ hổng: mức rủi ro (từ điểm CVSS), điểm + vector CVSS, lý giải của model.
@dataclass
class RiskAssessment:
    swc: str
    risk_level: str
    cvss_score: float
    cvss_vector: str
    justification: str


# Agent Assessor: LLM chọn các metric CVSS v3.1 theo ngữ cảnh smart contract.
class AssessorAgent:
    # Gắn client LLM và KB dùng chung (lấy risk_guidance theo SWC).
    def __init__(self, llm: ServerGPUClient | None = None, knowledge_base: KnowledgeBase | None = None) -> None:
        self.llm = llm or ServerGPUClient()
        self.knowledge_base = knowledge_base or default_knowledge_base()

    # Prompt: lỗ hổng (+ root cause / impact của Advisor), hướng dẫn KB, mã contract (<= 10k ký tự), schema JSON.
    @staticmethod
    def _build_prompt(code: str, item: dict, risk_guidance: dict | None = None) -> str:
        root_cause = str(item.get("root_cause") or item.get("description") or "").strip()
        impact = str(item.get("impact") or "").strip()
        guidance = []
        if risk_guidance:
            cwe_ids = risk_guidance.get("cwe_ids") or []
            cwe = f" (CWE: {', '.join(cwe_ids)})" if cwe_ids else ""
            guidance = ["<VulnerabilityKnowledgeGuidance>",
                        f"SWC Benchmark Severity: {risk_guidance.get('benchmark_severity') or 'Unknown'}{cwe}"]
            if remediation := risk_guidance.get("remediation_summary") or "":
                guidance.append(f"Remediation Standard: {remediation[:300]}")
            if checks := risk_guidance.get("checklist") or []:
                guidance += ["Risk Evaluation Checklist:", *(f"  - {check}" for check in checks[:4])]
            guidance += ["</VulnerabilityKnowledgeGuidance>", ""]
        return "\n".join([
            "Assess the risk of the following smart-contract vulnerability.", "",
            "<Vulnerability>", f"SWC: {item.get('swc') or 'Unknown'}",
            f"Name: {item.get('vulnerability_name') or item.get('title') or 'Unknown vulnerability'}",
            f"Detector-assigned severity: {item.get('severity') or 'Unknown'}",
            f"Root cause: {root_cause[:1200] or '(not provided)'}",
            f"Impact (from repair analysis): {impact[:1200] or '(not provided)'}", "</Vulnerability>", "",
            *guidance,
            "<Contract>", "```solidity", code.strip()[:10000], "```", "</Contract>", "",
            "Return ONLY a JSON object with exactly this schema:", _SCHEMA,
            "Carefully evaluate smart-contract exploitability (front-running, owner privileges) and impact. "
            "Do not wrap the JSON in markdown fences.",
        ])

    # [HÀM CHÍNH PHASE 3 - một lỗ hổng] LLM chọn 8 metric -> compute_cvss_31_base_score -> mức rủi ro; LLM lỗi -> None.
    def assess_one(self, code: str, item: dict) -> RiskAssessment | None:
        swc = item.get("swc") or "Unknown"
        guidance = self.knowledge_base.risk_guidance(swc)
        try:
            data = self.llm.generate_json(self._build_prompt(code, item, guidance), ASSESSOR_SYSTEM_INSTRUCTION)
        except Exception:
            return None
        metric = lambda key, default: str(data.get(key) or default)
        score, vector = compute_cvss_31_base_score(
            av=metric("attack_vector", "N"), ac=metric("attack_complexity", "L"), pr=metric("privileges_required", "N"),
            ui=metric("user_interaction", "N"), s=metric("scope", "U"), c=metric("confidentiality", "N"),
            i=metric("integrity", "H"), a=metric("availability", "H"))
        return RiskAssessment(swc=swc, risk_level=cvss_to_level(score), cvss_score=score, cvss_vector=vector,
                              justification=str(data.get("justification") or "").strip())
