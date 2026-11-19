"""
advisor.py
──────────────────────────────────────────────────────────────────────────────
Repair Suggestion Agent (Advisor) — agent #2 of the LLM-BSCVM framework.

For ONE fused Phase 1 finding the Advisor:
    Retrieve  -> remediation knowledge for the finding's SWC from the knowledge base
                 (RepairKnowledgeBase: 37-SWC registry + 11,869 cases; offline, no model reload)
    Prompt    -> finding + Phase 1 evidence + triage rules + target function + contract context + knowledge
    Generate  -> JSON from the instruct LLM on the GPU server:
                 verdict (confirmed / false_positive / uncertain), guard_quote, root cause,
                 attack scenario, fix strategy; an UNMAPPED finding also gets a vulnerability class.

Triage is recall-first: missing a real vulnerability costs more than a false positive, so the
prompt asks for proof before a dismissal and advisor_phase.py checks that proof.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List

from E2E.common.gpu_client import ServerGPUClient
from E2E.common.regex_patterns import RE_PRAGMA_SPEC, RE_SWC_ID
from E2E.config import KNOWLEDGE_BASE_DIR
from E2E.phase0_parsing.contract_context import build_contract_context
from E2E.phase1b_rag.swc_agents import normalize_swc_id


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
# Neutral framing: the Advisor is an independent reviewer, not a false-positive filter. An earlier wording
# ("Candidate findings may be false positives") made the model dismiss whatever it could not fully verify.
ADVISOR_SYSTEM_INSTRUCTION = (
    "You are a senior smart-contract security auditor specialising in Solidity. "
    "You review ONE finding reported by automated detectors and decide whether it is a real, exploitable "
    "vulnerability in the supplied code. Base every claim on code you can see and never assume a protection "
    "that is not shown. Return ONLY valid JSON matching the requested schema."
)

# The three verdicts. Only a false_positive that passes advisor_phase's check removes a finding.
VERDICT_CONFIRMED = "confirmed"
VERDICT_FALSE_POSITIVE = "false_positive"
VERDICT_UNCERTAIN = "uncertain"
VERDICTS = {VERDICT_CONFIRMED, VERDICT_FALSE_POSITIVE, VERDICT_UNCERTAIN}

# Classes the Advisor may assign to an UNMAPPED (detector-only) finding; titles follow swc_registry.json.
UNMAPPED_CLASSES = (
    ("SWC-101", "Arithmetic Overflow/Underflow"),
    ("SWC-104", "Unchecked Return Value"),
    ("SWC-105", "Access Control"),
    ("SWC-107", "Reentrancy"),
    ("SWC-112", "Delegatecall Flaw"),
    ("SWC-113", "DoS with Failed Call"),
    ("SWC-114", "Front-Running / Slippage"),
    ("SWC-115", "Oracle Manipulation"),
    ("SWC-116", "Timestamp Dependence"),
    ("SWC-120", "Weak Sources of Randomness"),
    ("SWC-121", "Signature Replay"),
    ("SWC-128", "DoS With Block Gas Limit"),
    ("SWC-135", "Business Logic Flaw"),
)
UNMAPPED_CLASS_TITLES = dict(UNMAPPED_CLASSES)

# Verdict definitions + triage rules of the prompt. Rule 3 exists because the model used to dismiss
# reentrancy / logic findings with "protected by onlyOwner / notPaused".
TRIAGE_RULES = "\n".join([
    "Verdict:",
    '- "confirmed": the code shows the flaw and no visible code prevents it.',
    '- "false_positive": (a) visible code makes this flaw impossible - copy that exact code into `guard_quote`; '
    'or (b) the reported flaw is simply not present in <TargetFunction> - explain why in `root_cause` and leave `guard_quote` empty.',
    '- "uncertain": the answer depends on code that is not shown (an undefined modifier, library, external contract or caller).',
    "",
    "Rules:",
    "1. Cite only lines that exist in <TargetFunction>.",
    "2. Missing code is not a guard: never assume what an unseen modifier, library or caller does.",
    "3. Access restrictions (onlyOwner, role checks, `require(msg.sender == ...)`, internal/private visibility) "
    "only answer ACCESS-CONTROL findings (an unauthorised caller). They do NOT make reentrancy, arithmetic, "
    "accounting/business-logic, oracle/price, front-running, DoS or signature findings false positives: "
    "those still harm users when an allowed caller, or a public function that calls this one, runs the code.",
    "4. A modifier prevents only what it implements: nonReentrant/lock stop reentrancy; pause (whenNotPaused, "
    "notFrozen), initializer and state-machine modifiers stop nothing else.",
    "5. A check on `tx.origin` is NOT a valid guard: for tx.origin authentication that check is the "
    "vulnerability itself.",
])

_SOURCE_NAMES = {
    "static": "static rule",
    "slither": "Slither",
    "rag": "RAG similar case",
    "llm_detector": "fine-tuned detector",
}


# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────
# Kết quả Advisor cho MỘT finding: verdict, guard_quote (đoạn mã guard LLM trích khi kết luận false positive -
# advisor_phase kiểm tra nó có thật trong mã), nguyên nhân gốc, tác động (= kịch bản tấn công), chiến lược sửa cho
# Fixer, vulnerability_class (chỉ finding UNMAPPED: mã SWC LLM gán, "" = không gán); error khi gọi LLM thất bại.
@dataclass
class RepairSuggestion:
    verdict: str = VERDICT_CONFIRMED
    guard_quote: str = ""
    root_cause: str = ""
    impact: str = ""
    fix_strategy: str = ""
    vulnerability_class: str = ""
    error: str | None = None


# ──────────────────────────────────────────────────────────────────────────────
# Repair knowledge base (RAG retrieval by SWC id — offline, deterministic)
# ──────────────────────────────────────────────────────────────────────────────
# Điểm giống nhau giữa mã đang xét và mã lỗi của một ca trong KB: trùng hẳn = 10, chứa nhau = 5 + tỉ lệ độ dài,
# còn lại SequenceMatcher trên 500 ký tự đầu. Dùng để xếp hạng ca thực tế trong RepairKnowledgeBase._retrieve.
def _code_match_score(query_code: str, item: Dict[str, Any]) -> float:
    v_code = str(item.get("vulnerable_code") or "").strip()
    if not v_code:
        return 0.0
    if v_code == query_code:
        return 10.0
    if v_code in query_code or query_code in v_code:
        return 5.0 + min(len(v_code), len(query_code)) / max(len(v_code), len(query_code))
    return SequenceMatcher(None, query_code[:500], v_code[:500]).ratio()


# Thêm một chuỗi (đã strip) vào danh sách nếu khác rỗng và chưa có (giữ thứ tự, không trùng).
def _add_unique(bucket: List[str], value: Any) -> None:
    text = str(value or "").strip()
    if text and text not in bucket:
        bucket.append(text)


# "Bước RAG của Advisor": tra tri thức sửa lỗi THEO MÃ SWC (không chạy lại TF-IDF/embedding) từ
# swc_registry.json (37 SWC) + knowledge_items.json (11,869 ca). Dùng ở Phase 1F (enrich), Phase 2, Phase 3.
class RepairKnowledgeBase:
    """Indexes the vulnerability knowledge store by SWC id for repair retrieval."""

    # Nạp registry và các ca lỗi, lập chỉ mục theo SWC (thiếu file / file hỏng thì bỏ qua, không lỗi).
    def __init__(self, knowledge_dir: str | Path = KNOWLEDGE_BASE_DIR) -> None:
        self._registry_by_swc: Dict[str, Dict[str, Any]] = {}
        self._items_by_swc: Dict[str, List[Dict[str, Any]]] = {}
        self._retrieved: Dict[tuple, Dict[str, Any]] = {}
        for record in self._load(Path(knowledge_dir) / "swc_registry.json"):
            swc = normalize_swc_id(record.get("id"))
            if swc:
                self._registry_by_swc[swc] = record
        for record in self._load(Path(knowledge_dir) / "knowledge_items.json"):
            swc = normalize_swc_id(record.get("swc") or record.get("id"))
            if swc:
                self._items_by_swc.setdefault(swc, []).append(record)

    # Đọc một file JSON danh sách; thiếu file hoặc không đọc được -> [].
    @staticmethod
    def _load(path: Path) -> List[Dict[str, Any]]:
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:  # noqa: BLE001 - a missing / broken KB file leaves that part of the index empty
            return []

    # Nhiệm vụ: gom tri thức cho một SWC: mô tả, cách sửa, checklist, ví dụ mã đã sửa từ registry; cộng thêm `limit`
    # ca thực tế cùng SWC giống mã đang xét nhất (_code_match_score). Trả best_pair (cặp mã lỗi/mã sửa gần nhất) và
    # risk_guidance (cho Assessor).
    # Dùng bởi: enrich_finding (Phase 1F), AdvisorAgent / advisor_phase (Phase 2), Assessor (risk_guidance).
    # Kết quả được nhớ theo (SWC, limit, code): Phase 1F và Phase 2 tra cùng một finding với cùng tham số, và
    # việc xếp hạng bằng SequenceMatcher trên ~2,000 ca mất khoảng 1 giây. Người gọi chỉ đọc kết quả.
    def retrieve(self, swc: str, limit: int = 3, code: str | None = None) -> Dict[str, Any]:
        key = (normalize_swc_id(swc), limit, code)
        if key not in self._retrieved:
            self._retrieved[key] = self._retrieve(*key)
        return self._retrieved[key]

    # Phần tính toán của retrieve (không nhớ kết quả).
    def _retrieve(self, norm_swc: str, limit: int, code: str | None) -> Dict[str, Any]:
        titles: List[str] = []
        descriptions: List[str] = []
        remediations: List[str] = []
        checklist: List[str] = []
        registry_fixes: List[str] = []
        empirical_fixes: List[str] = []
        best_pair: Dict[str, Any] = {}

        # 1. Canonical knowledge from swc_registry.json
        reg = self._registry_by_swc.get(norm_swc)
        if reg:
            _add_unique(titles, reg.get("title") or norm_swc)
            _add_unique(descriptions, reg.get("description") or reg.get("root_cause"))
            _add_unique(remediations, reg.get("remediation"))
            for text in reg.get("checklist", []) or []:
                _add_unique(checklist, text)
            for sample in reg.get("samples", []) or []:
                if isinstance(sample, dict):
                    _add_unique(registry_fixes, sample.get("fixed"))

        # 2. The `limit` empirical cases closest to the code under review
        items = list(self._items_by_swc.get(norm_swc, []))
        if code and items:
            items.sort(key=lambda item: _code_match_score(code.strip(), item), reverse=True)
        for item in items[:limit]:
            _add_unique(titles, item.get("title"))
            _add_unique(descriptions, item.get("description") or item.get("root_cause"))
            remediation = str(item.get("remediation") or "").strip()
            _add_unique(remediations, remediation)
            for text in item.get("checklist", []) or []:
                _add_unique(checklist, text)
            fixed = str(item.get("fixed_code") or "").strip()
            vulnerable = str(item.get("vulnerable_code") or "").strip()
            _add_unique(empirical_fixes, fixed)
            if (fixed or vulnerable) and not best_pair:
                best_pair = {
                    "vulnerable_code": vulnerable,
                    "fixed_code": fixed,
                    "root_cause": str(item.get("root_cause") or "").strip(),
                    "remediation": remediation,
                    "severity": str(item.get("severity") or (reg.get("severity") if reg else "Medium")),
                    "cwe_ids": item.get("cwe_ids") or [],
                }

        # No empirical pair: fall back to the first registry sample
        if not best_pair and reg:
            sample = next((s for s in reg.get("samples", []) or [] if isinstance(s, dict)), None)
            if sample:
                best_pair = {
                    "vulnerable_code": str(sample.get("vulnerable") or "").strip(),
                    "fixed_code": str(sample.get("fixed") or "").strip(),
                    "root_cause": str(sample.get("root_cause") or "").strip(),
                    "remediation": str(reg.get("remediation") or "").strip(),
                    "severity": str(reg.get("severity") or "Medium"),
                    "cwe_ids": [],
                }

        # Closest empirical fixes first, then the registry samples
        code_examples = empirical_fixes + [x for x in registry_fixes if x not in empirical_fixes]
        risk_guidance = {
            "swc": norm_swc,
            "title": titles[0] if titles else norm_swc,
            "benchmark_severity": reg.get("severity") if reg else (best_pair.get("severity") if best_pair else "Medium"),
            "cwe_ids": best_pair.get("cwe_ids", []) if best_pair else [],
            "remediation_summary": remediations[0] if remediations else "",
            "root_cause_summary": descriptions[0] if descriptions else "",
            "checklist": checklist[:6],
        }
        return {
            "descriptions": descriptions,
            "remediations": remediations,
            "checklist": checklist,
            "code_examples": code_examples[:limit],
            "best_pair": best_pair,
            "risk_guidance": risk_guidance,
        }

    # Bổ sung cho một finding: mã lỗi/mã sửa mẫu, cách sửa, root cause (chỉ điền trường finding chưa có), risk_guidance.
    def enrich_finding(self, finding: Dict[str, Any], code: str = "") -> Dict[str, Any]:
        target_code = (
            finding.get("unit_code")
            or finding.get("vulnerable_code")
            or finding.get("matched_text")
            or code
        )
        retrieved = self.retrieve(normalize_swc_id(finding.get("swc")), limit=3, code=target_code)
        enriched = dict(finding)
        best_pair = retrieved.get("best_pair", {})
        for field in ("vulnerable_code", "fixed_code", "remediation", "root_cause"):
            if not enriched.get(field) and best_pair.get(field):
                enriched[field] = best_pair[field]
        if not enriched.get("remediation") and retrieved.get("remediations"):
            enriched["remediation"] = retrieved["remediations"][0]
        if not enriched.get("root_cause") and retrieved.get("descriptions"):
            enriched["root_cause"] = retrieved["descriptions"][0]
        enriched["risk_guidance"] = retrieved.get("risk_guidance", {})
        return enriched

    # Làm giàu hàng loạt finding; gắn thêm unit_code (mã của chính hàm) để tra ca giống nhất.
    # Gọi bởi: gpu_pipeline ngay sau fuse_findings.
    def enrich_findings(
        self,
        findings: List[Dict[str, Any]],
        code: str = "",
        code_units: List[Dict[str, Any]] | None = None,
    ) -> List[Dict[str, Any]]:
        unit_code_map = {cu["unit_id"]: cu["code"] for cu in code_units or [] if cu.get("unit_id") and cu.get("code")}
        enriched_list = []
        for f in findings:
            f_copy = dict(f)
            u_id = f_copy.get("unit_id")
            if u_id and u_id in unit_code_map and not f_copy.get("unit_code"):
                f_copy["unit_code"] = unit_code_map[u_id]
            enriched_list.append(self.enrich_finding(f_copy, code=code))
        return enriched_list


# ──────────────────────────────────────────────────────────────────────────────
# Prompt blocks
# ──────────────────────────────────────────────────────────────────────────────
# Nhiệm vụ: dựng khối <Phase1Evidence>: kênh nào báo, static và RAG có cùng báo không, dòng mã mà luật static /
# Slither khớp (kèm ghi chú), mô tả ca tương tự của RAG (cắt 400 ký tự); finding UNMAPPED -> dòng detector nghi nhất.
# KHÔNG đưa xác suất của detector: Advisor là người kiểm tra độc lập, không được neo theo điểm detector.
def _phase1_evidence(vulnerability: Dict[str, Any], code: str) -> str:
    source_lines = code.splitlines()

    # Dòng mã nguồn thứ `line` (đếm từ 1, cắt 160 ký tự); số dòng không hợp lệ -> "".
    def code_at(line: Any) -> str:
        if isinstance(line, int) and 1 <= line <= len(source_lines):
            return source_lines[line - 1].strip()[:160]
        return ""

    lines = []
    sources = [str(item) for item in vulnerability.get("sources") or []]
    if sources:
        lines.append("Reported by: " + ", ".join(_SOURCE_NAMES.get(item, item) for item in sources))
    if vulnerability.get("corroborated"):
        lines.append("Static rules and RAG both reported this finding independently.")

    shown_lines: set = set()
    rag_shown = False
    for item in vulnerability.get("evidence") or []:
        source = str(item.get("source") or "")
        line = item.get("line")
        if source in {"static", "slither"} and line not in shown_lines:
            shown_lines.add(line)
            text = f"{_SOURCE_NAMES[source]} at line {line}" if line else _SOURCE_NAMES[source]
            if code_at(line):
                text += f": `{code_at(line)}`"
            matched = str(item.get("matched_text") or item.get("check") or "").strip()[:80]
            if matched:
                text += f" (matched `{matched}`)"
            notes = [str(note) for note in item.get("notes") or [] if note]
            if notes:
                text += f" - {notes[0][:120]}"
            lines.append(text)
        elif source == "rag" and not rag_shown:
            description = " ".join(str(item.get("description") or item.get("root_cause") or "").split())
            if description:
                rag_shown = True
                lines.append(
                    "A similar known vulnerable case from the knowledge base (a different contract) was "
                    f"described as: {description[:400]}"
                )

    if vulnerability.get("swc") == "UNMAPPED" and code_at(vulnerability.get("line")):
        lines.append(
            f"Most suspicious line according to the detector: line {vulnerability['line']}: "
            f"`{code_at(vulnerability['line'])}`"
        )
    return "\n".join(lines) if lines else "No additional evidence"


# Nhiệm vụ: với finding UNMAPPED, liệt kê tối đa 2 ca tương tự (theo độ giống mã, từ Phase 1B retrieved_context)
# làm gợi ý phân loại, ghi rõ có thể không liên quan. Không có ngữ cảnh RAG -> "".
def _rag_hints(rag_context: List[Dict[str, Any]] | None) -> str:
    hints = []
    for case in (rag_context or [])[:2]:
        swc = normalize_swc_id(case.get("swc"))
        title = str(case.get("title") or swc).strip()
        description = " ".join(str(case.get("description") or case.get("root_cause") or "").split())[:200]
        try:
            similarity = f"{float(case.get('similarity')):.2f}"
        except (TypeError, ValueError):
            similarity = "?"
        hints.append(f"- {swc} {title} (similarity {similarity}): {description}")
    if not hints:
        return ""
    return "Most similar known cases by code similarity (they may be unrelated):\n" + "\n".join(hints)


# Dựng khối <RetrievedSecurityKnowledge>: tối đa 3 cách sửa, 8 mục checklist, và cặp "mã lỗi -> mã đã sửa" gần nhất
# (từ finding đã enrich hoặc best_pair); không có cặp thì tối đa 2 ví dụ mã đã sửa.
def _knowledge_block(vulnerability: Dict[str, Any], retrieved: Dict[str, Any]) -> str:
    best_pair = retrieved.get("best_pair", {})
    rag_fixed = str(vulnerability.get("fixed_code") or best_pair.get("fixed_code") or "").strip()
    rag_vuln = str(vulnerability.get("vulnerable_code") or best_pair.get("vulnerable_code") or "").strip()
    remediations = [item[:800] for item in retrieved.get("remediations", [])[:3]]
    checklist = [str(item)[:200] for item in retrieved.get("checklist", [])[:8]]
    code_examples = [item[:800] for item in retrieved.get("code_examples", [])[:3]]
    if rag_fixed and rag_fixed not in code_examples:
        code_examples.insert(0, rag_fixed[:800])

    lines = ["<RetrievedSecurityKnowledge>"]
    lines.extend(f"Remediation {idx}: {item}" for idx, item in enumerate(remediations, 1))
    if checklist:
        lines.append("Checklist:")
        lines.extend(f"  - {item}" for item in checklist)
    if rag_vuln and rag_fixed:
        lines.append(
            "Most Relevant Empirical Code Repair Pair (from 11,869-case knowledge base):\n"
            f"  Vulnerable pattern:\n```solidity\n{rag_vuln[:600]}\n```\n"
            f"  Validated fixed pattern:\n```solidity\n{rag_fixed[:600]}\n```"
        )
    else:
        lines.extend(
            f"Reference fix example {idx} (from vulnerability knowledge base):\n```solidity\n{item}\n```"
            for idx, item in enumerate(code_examples[:2], 1)
        )
    lines.append("</RetrievedSecurityKnowledge>")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Advisor agent
# ──────────────────────────────────────────────────────────────────────────────
# Agent Advisor (Phase 2): dùng LLM trên GPU server để đánh giá một finding (confirmed / false_positive / uncertain),
# giải thích nguyên nhân và đề xuất chiến lược sửa, dựa trên tri thức RAG + bằng chứng Phase 1 + ngữ cảnh contract.
class AdvisorAgent:
    # Gắn client LLM (ServerGPUClient) và kho tri thức sửa lỗi (RepairKnowledgeBase).
    def __init__(
        self,
        llm: ServerGPUClient | None = None,
        knowledge_base: RepairKnowledgeBase | None = None,
    ) -> None:
        self.llm = llm or ServerGPUClient()
        self.knowledge_base = knowledge_base or RepairKnowledgeBase()

    # Dựng prompt Advisor theo thứ tự: đầu bài (finding thường: tên lỗi + SWC + mức độ; finding UNMAPPED: yêu cầu tự
    # xác định lớp lỗi trong UNMAPPED_CLASSES), phiên bản Solidity, <Phase1Evidence> (+ gợi ý RAG cho UNMAPPED),
    # schema JSON, TRIAGE_RULES, <TargetFunction> (mã của hàm; finding cấp contract -> 2,000 ký tự đầu của contract),
    # <Context> (build_contract_context) và <Knowledge>.
    def _build_prompt(
        self,
        code: str,
        vulnerability: Dict[str, Any],
        knowledge_block: str,
        rag_context: List[Dict[str, Any]] | None = None,
    ) -> str:
        unmapped = vulnerability.get("swc") == "UNMAPPED"
        unit_code = str(vulnerability.get("unit_code") or "")
        if unit_code:
            target_code = unit_code
            start_line = vulnerability.get("start_line") or vulnerability.get("line") or 1
            end_line = vulnerability.get("end_line") or (start_line + target_code.count("\n"))
        else:
            target_code = code[:2000]
            start_line, end_line = 1, 1 + target_code.count("\n")

        sol_match = RE_PRAGMA_SPEC.search(code)
        compiler_note = sol_match.group(1).strip() if sol_match else "0.4.x"

        # unit_id = "<Contract>.<function>[#n]": the context comes from the enclosing contract. A contract-level
        # finding (no unit) matches no contract name, so build_contract_context falls back to the whole source.
        unit_id = str(vulnerability.get("unit_id") or "")
        contract_name = unit_id.split(".", 1)[0] if "." in unit_id else "Contract"
        try:
            context_str = build_contract_context(
                {"contract_name": contract_name, "code": target_code}, code, budget_tokens=1500
            )
        except Exception:  # noqa: BLE001 - the prompt still works without context
            context_str = ""

        if unmapped:
            header = [
                "Review ONE finding: the fine-tuned detector flagged <TargetFunction> as vulnerable but did not "
                "name the vulnerability.",
                "First find the most likely vulnerability in <TargetFunction> and set `vulnerability_class` to one of: "
                + ", ".join(f"{swc_id} {name}" for swc_id, name in UNMAPPED_CLASSES)
                + '. If you find no vulnerability, set it to "none" and the verdict to "false_positive".',
                "Then judge that vulnerability with the rules below.",
            ]
            evidence = [_phase1_evidence(vulnerability, code), _rag_hints(rag_context)]
        else:
            header = [
                "Review ONE finding reported by the Phase 1 detectors.",
                f"Finding: {vulnerability.get('title') or 'Unknown vulnerability'} "
                f"({vulnerability.get('swc') or 'Unknown'}) - Severity: {vulnerability.get('severity') or 'Unknown'}",
            ]
            evidence = [_phase1_evidence(vulnerability, code)]

        schema = (
            "{\n"
            + ('  "vulnerability_class": string,    // one SWC id from the class list below, or "none"\n' if unmapped else "")
            + '  "verdict": "confirmed" | "false_positive" | "uncertain",\n'
            '  "guard_quote": string,            // false_positive caused by a guard: the exact guard code copied from <TargetFunction> or <Context>; otherwise ""\n'
            '  "evidence_lines": [int],          // line numbers within <TargetFunction> exhibiting the flaw\n'
            '  "root_cause": string,             // technical explanation of the flaw, or why it is not exploitable\n'
            '  "attack_scenario": string,        // concrete step-by-step exploit scenario (or "None" if false_positive)\n'
            '  "preconditions": [string],        // prerequisites for exploitation\n'
            '  "fix_strategy": string            // high-level conceptual remediation strategy for Fixer\n'
            "}"
        )
        return "\n".join([
            *header,
            f"Solidity version: {compiler_note}",
            "",
            "<Phase1Evidence>",
            *[block for block in evidence if block],
            "</Phase1Evidence>",
            "",
            "Return ONLY valid JSON matching this schema:",
            schema,
            "",
            TRIAGE_RULES,
            "",
            f'<TargetFunction lines="{start_line}-{end_line}">',
            target_code,
            "</TargetFunction>",
            "",
            "<Context>",
            context_str or "No additional context",
            "</Context>",
            "",
            "<Knowledge>",
            knowledge_block,
            "</Knowledge>",
        ])

    # [HÀM CHÍNH] Xử lý MỘT finding: tra tri thức theo SWC (mã của chính hàm, finding cấp contract -> cả contract; KHÔNG
    # dùng `vulnerable_code` vì đó là ca mẫu của contract khác) -> dựng prompt -> gọi LLM -> RepairSuggestion.
    # verdict ngoài 3 giá trị hợp lệ -> uncertain (không đọc được thì giữ finding). LLM lỗi / không trả object JSON ->
    # RepairSuggestion có error (advisor_phase dùng gợi ý từ KB, không làm hỏng cả batch).
    # rag_context: top ca tương tự của Phase 1B (gợi ý phân loại cho finding UNMAPPED).
    def advise_one(
        self,
        code: str,
        vulnerability: Dict[str, Any],
        rag_context: List[Dict[str, Any]] | None = None,
    ) -> RepairSuggestion:
        retrieved = self.knowledge_base.retrieve(
            vulnerability.get("swc") or "Unknown", code=str(vulnerability.get("unit_code") or "") or code
        )
        prompt = self._build_prompt(code, vulnerability, _knowledge_block(vulnerability, retrieved), rag_context)
        try:
            data = self.llm.generate_json(prompt, ADVISOR_SYSTEM_INSTRUCTION)
        except Exception as error:  # noqa: BLE001 - reported per finding, never aborts the batch
            return RepairSuggestion(error=f"{type(error).__name__}: {error}")
        if not isinstance(data, dict):
            return RepairSuggestion(error=f"Advisor returned {type(data).__name__}, expected a JSON object")

        verdict = str(data.get("verdict") or "").strip().lower()
        class_match = RE_SWC_ID.search(str(data.get("vulnerability_class") or "").upper())
        vulnerability_class = class_match.group(0) if class_match else ""
        return RepairSuggestion(
            verdict=verdict if verdict in VERDICTS else VERDICT_UNCERTAIN,
            guard_quote=str(data.get("guard_quote") or "").strip(),
            root_cause=str(data.get("root_cause") or "").strip(),
            impact=str(data.get("attack_scenario") or "").strip() or str(data.get("impact") or "").strip(),
            fix_strategy=str(data.get("fix_strategy") or "").strip(),
            vulnerability_class=(
                vulnerability_class
                if vulnerability.get("swc") == "UNMAPPED" and vulnerability_class in UNMAPPED_CLASS_TITLES else ""
            ),
        )
