"""
advisor.py
──────────────────────────────────────────────────────────────────────────────
Repair Suggestion Agent (Advisor) — agent #2 of the LLM-BSCVM framework.

Consumes the fused Phase 1 findings (pipeline/gpu_pipeline.py) and,
for each detected vulnerability, produces a targeted repair suggestion using
Retrieval-Augmented Generation:

    Decompose  -> one suggestion task per detected vulnerability
    Retrieve   -> remediation knowledge from the vulnerability knowledge base
                  (keyed by the detected SWC id; offline, no model reload)
    Generate   -> structured repair suggestion via the generative LLM on the GPU server

Following the paper (Fig. 6), each suggestion covers five aspects:
    1. Vulnerability name
    2. Root cause analysis
    3. Potential impact assessment
    4. Repair steps + fixed code example
    5. Preventive measures

Returns structured data so the next phases (Assessor, Fixer, report) can
consume it directly.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List

from E2E.common.gpu_client import ServerGPUClient
from E2E.config import KNOWLEDGE_BASE_DIR
from E2E.phase0_parsing.contract_context import build_contract_context
from E2E.phase1b_rag.swc_agents import normalize_swc_id

ADVISOR_SYSTEM_INSTRUCTION = (
    "You are a senior smart-contract security auditor specialising in Solidity. "
    "Your task is to audit candidate vulnerability findings. Candidate findings may be false positives. "
    "Carefully verify whether the finding is a genuine, exploitable vulnerability (true positive) "
    "or an invalid/guarded false positive. Ground your verdict and evidence strictly in the supplied "
    "function code and contract context. Return ONLY valid JSON matching the requested schema."
)


# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────
# Kết quả Advisor cho MỘT finding: đúng/sai (is_true_positive), nguyên nhân gốc, tác động (= kịch bản tấn công),
# chiến lược sửa (fix_strategy) cho Fixer; error khi gọi LLM thất bại.
@dataclass
class RepairSuggestion:
    is_true_positive: bool = True
    root_cause: str = ""
    impact: str = ""
    fix_strategy: str = ""
    error: str | None = None


# Mã mà Advisor xét cho một finding: mã của chính hàm (unit_code); finding cấp contract (không có hàm, vd
# pragma) xét chính contract đang audit - KHÔNG dùng `vulnerable_code` vì đó là ca mẫu của contract khác trong KB.
def _target_code(vulnerability: Dict[str, Any], code: str) -> str:
    return str(vulnerability.get("unit_code") or "") or code


# ──────────────────────────────────────────────────────────────────────────────
# Repair knowledge base (RAG retrieval by SWC id — offline, deterministic)
# ──────────────────────────────────────────────────────────────────────────────
# "Bước RAG của Advisor": tra tri thức sửa lỗi THEO MÃ SWC (không chạy lại TF-IDF/embedding) từ
# swc_registry.json (37 SWC) + knowledge_items.json (11,869 ca). Dùng ở Phase 1F (enrich) và Phase 2.
class RepairKnowledgeBase:
    """Indexes the vulnerability knowledge store by SWC id for repair retrieval.

    This is the Advisor's RAG step: rather than re-running the heavy embedding
    model, it looks remediation knowledge up directly from the 11,869-case
    knowledge-store and 37-SWC registry by the SWC id the detector already identified.
    """

    # Nạp registry và các ca lỗi, lập chỉ mục theo SWC (thiếu file thì bỏ qua, không lỗi).
    def __init__(self, knowledge_dir: str | Path = KNOWLEDGE_BASE_DIR) -> None:
        k_dir = Path(knowledge_dir)
        self._registry_by_swc: Dict[str, Dict[str, Any]] = {}
        self._items_by_swc: Dict[str, List[Dict[str, Any]]] = {}
        self._retrieved: Dict[tuple, Dict[str, Any]] = {}

        registry_path = k_dir / "swc_registry.json"
        if registry_path.exists():
            try:
                with registry_path.open("r", encoding="utf-8") as f:
                    for record in json.load(f):
                        swc = normalize_swc_id(record.get("id"))
                        if swc:
                            self._registry_by_swc[swc] = record
            except Exception:
                pass

        items_path = k_dir / "knowledge_items.json"
        if items_path.exists():
            try:
                with items_path.open("r", encoding="utf-8") as f:
                    for record in json.load(f):
                        swc = normalize_swc_id(record.get("swc") or record.get("id"))
                        if swc:
                            self._items_by_swc.setdefault(swc, []).append(record)
            except Exception:
                pass

    # Nhiệm vụ: gom tri thức cho một SWC: mô tả, cách sửa, checklist, ví dụ mã đã sửa từ registry;
    # cộng thêm các ca thực tế cùng SWC, xếp theo độ giống với mã đang xét (trùng hẳn = 10, chứa nhau = 5+,
    # còn lại SequenceMatcher). Trả best_pair (cặp mã lỗi/mã sửa gần nhất) và risk_guidance (cho Assessor).
    # Dùng bởi: enrich_finding (Phase 1F), AdvisorAgent / advisor_phase (Phase 2), Assessor (risk_guidance).
    # Kết quả được nhớ theo (SWC, limit, code): Phase 1F và Phase 2 tra cùng một finding với cùng tham số, và
    # việc xếp hạng bằng SequenceMatcher trên ~2,000 ca mất khoảng 1 giây. Người gọi chỉ đọc kết quả.
    def retrieve(self, swc: str, limit: int = 3, code: str | None = None) -> Dict[str, Any]:
        """Aggregate remediation knowledge for a given SWC id, prioritizing empirical cases closest to the input code."""
        key = (normalize_swc_id(swc), limit, code)
        if key not in self._retrieved:
            self._retrieved[key] = self._retrieve(*key)
        return self._retrieved[key]

    # Phần tính toán của retrieve (không nhớ kết quả).
    def _retrieve(self, norm_swc: str, limit: int, code: str | None) -> Dict[str, Any]:
        remediations: List[str] = []
        checklist: List[str] = []
        code_examples: List[str] = []
        descriptions: List[str] = []
        titles: List[str] = []

        # 1. Canonical knowledge from swc_registry.json
        reg = self._registry_by_swc.get(norm_swc)
        if reg:
            title = str(reg.get("title") or norm_swc).strip()
            if title and title not in titles:
                titles.append(title)
            desc = str(reg.get("description") or reg.get("root_cause") or "").strip()
            if desc and desc not in descriptions:
                descriptions.append(desc)
            remed = str(reg.get("remediation") or "").strip()
            if remed and remed not in remediations:
                remediations.append(remed)
            for item in reg.get("checklist", []) or []:
                text = str(item or "").strip()
                if text and text not in checklist:
                    checklist.append(text)
            for s in reg.get("samples", []) or []:
                if isinstance(s, dict):
                    fixed = str(s.get("fixed") or "").strip()
                    if fixed and fixed not in code_examples:
                        code_examples.append(fixed)

        # 2. Granular empirical cases from knowledge_items.json (11,869 cases)
        items = list(self._items_by_swc.get(norm_swc, []))
        if code and items:
            query_code = code.strip()

            # Điểm giống nhau giữa mã đang xét và mã lỗi của một ca trong KB (dùng để sắp xếp).
            def match_score(item: Dict[str, Any]) -> float:
                v_code = str(item.get("vulnerable_code") or "").strip()
                if not v_code:
                    return 0.0
                if v_code == query_code:
                    return 10.0
                if v_code in query_code or query_code in v_code:
                    return 5.0 + min(len(v_code), len(query_code)) / max(len(v_code), len(query_code))
                return SequenceMatcher(None, query_code[:500], v_code[:500]).ratio()

            items.sort(key=match_score, reverse=True)

        empirical_fixes: List[str] = []
        best_pair: Dict[str, Any] = {}
        for item in items[:limit]:
            title = str(item.get("title") or "").strip()
            if title and title not in titles:
                titles.append(title)
            desc = str(item.get("description") or item.get("root_cause") or "").strip()
            if desc and desc not in descriptions:
                descriptions.append(desc)
            remed = str(item.get("remediation") or "").strip()
            if remed and remed not in remediations:
                remediations.append(remed)
            for c_item in item.get("checklist", []) or []:
                text = str(c_item or "").strip()
                if text and text not in checklist:
                    checklist.append(text)
            fixed = str(item.get("fixed_code") or "").strip()
            v_code = str(item.get("vulnerable_code") or "").strip()
            if fixed and fixed not in empirical_fixes:
                empirical_fixes.append(fixed)
            if (fixed or v_code) and not best_pair:
                best_pair = {
                    "vulnerable_code": v_code,
                    "fixed_code": fixed,
                    "root_cause": str(item.get("root_cause") or "").strip(),
                    "remediation": remed,
                    "severity": str(item.get("severity") or (reg.get("severity") if reg else "Medium")),
                    "cwe_ids": item.get("cwe_ids") or [],
                }

        # Place the closest empirical fix at the top of code_examples
        code_examples = empirical_fixes + [x for x in code_examples if x not in empirical_fixes]

        # Fallback to the first registry sample if no empirical pair was found
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

    # Bổ sung cho một finding: mã lỗi/mã sửa mẫu, cách sửa, root cause (chỉ điền khi finding chưa có), risk_guidance.
    def enrich_finding(self, finding: Dict[str, Any], code: str = "") -> Dict[str, Any]:
        """Enrich a finding with the most relevant empirical code, fix patterns, and evaluation knowledge."""
        norm_swc = normalize_swc_id(finding.get("swc"))
        target_code = (
            finding.get("unit_code")
            or finding.get("vulnerable_code")
            or finding.get("matched_text")
            or code
        )
        retrieved = self.retrieve(norm_swc, limit=3, code=target_code)
        enriched = dict(finding)

        best_pair = retrieved.get("best_pair", {})
        if best_pair:
            if not enriched.get("vulnerable_code") and best_pair.get("vulnerable_code"):
                enriched["vulnerable_code"] = best_pair["vulnerable_code"]
            if not enriched.get("fixed_code") and best_pair.get("fixed_code"):
                enriched["fixed_code"] = best_pair["fixed_code"]
            if not enriched.get("remediation") and best_pair.get("remediation"):
                enriched["remediation"] = best_pair["remediation"]
            if not enriched.get("root_cause") and best_pair.get("root_cause"):
                enriched["root_cause"] = best_pair["root_cause"]

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
        """Batch-enrich all findings with empirical vulnerability knowledge from 11,869 cases."""
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
# Advisor agent
# ──────────────────────────────────────────────────────────────────────────────
# Agent Advisor (Phase 2): dùng LLM trên GPU server để kiểm tra một finding là thật hay báo nhầm,
# giải thích nguyên nhân và đề xuất chiến lược sửa, dựa trên tri thức RAG + ngữ cảnh contract.
class AdvisorAgent:
    # Gắn client LLM (ServerGPUClient) và kho tri thức sửa lỗi (RepairKnowledgeBase).
    def __init__(
        self,
        llm: ServerGPUClient | None = None,
        knowledge_base: RepairKnowledgeBase | None = None,
    ) -> None:
        self.llm = llm or ServerGPUClient()
        self.knowledge_base = knowledge_base or RepairKnowledgeBase()

    # ── knowledge formatting ────────────────────────────────────────────────
    # Cắt danh sách văn bản theo số mục và số ký tự tối đa (giữ prompt ngắn).
    @staticmethod
    def _truncate(items: List[str], max_items: int, max_chars: int) -> List[str]:
        return [item[:max_chars] for item in items[:max_items]]

    # Dựng khối <RetrievedSecurityKnowledge> cho prompt: best practice, cách sửa, checklist, mẫu mã lỗi,
    # và cặp "mã lỗi -> mã đã sửa" gần nhất từ KB 11,869 ca.
    def _build_knowledge_block(self, vulnerability: Dict[str, Any], retrieved: Dict[str, Any]) -> str:
        # Empirical RAG case match from Phase 1B or knowledge retrieval
        best_pair = retrieved.get("best_pair", {})
        rag_fixed = str(vulnerability.get("fixed_code") or best_pair.get("fixed_code") or "").strip()
        rag_vuln = str(vulnerability.get("vulnerable_code") or best_pair.get("vulnerable_code") or "").strip()

        # ... merged with a fresh lookup from the vulnerability knowledge base.
        remediations = self._truncate(retrieved.get("remediations", []), 3, 800)
        checklist = self._truncate([str(x) for x in retrieved.get("checklist", [])], 8, 200)
        code_examples = self._truncate(retrieved.get("code_examples", []), 3, 800)
        if rag_fixed and rag_fixed not in code_examples:
            code_examples.insert(0, rag_fixed[:800])

        lines = ["<RetrievedSecurityKnowledge>"]
        for idx, item in enumerate(remediations, 1):
            lines.append(f"Remediation {idx}: {item}")
        if checklist:
            lines.append("Checklist:")
            lines.extend(f"  - {item}" for item in checklist)
        if rag_vuln and rag_fixed:
            lines.append(
                "Most Relevant Empirical Code Repair Pair (from 11,869-case knowledge base):\n"
                f"  Vulnerable pattern:\n```solidity\n{rag_vuln[:600]}\n```\n"
                f"  Validated fixed pattern:\n```solidity\n{rag_fixed[:600]}\n```"
            )
        elif code_examples:
            for idx, item in enumerate(code_examples[:2], 1):
                lines.append(f"Reference fix example {idx} (from vulnerability knowledge base):\n```solidity\n{item}\n```")
        lines.append("</RetrievedSecurityKnowledge>")
        return "\n".join(lines)

    # Dựng prompt Advisor: thông tin finding (SWC, mức độ), phiên bản Solidity, schema JSON yêu cầu
    # (is_true_positive, evidence_lines, root_cause, attack_scenario, preconditions, fix_strategy),
    # luật triage (có onlyOwner/kiểm tra msg.sender thì lỗi access control là FALSE POSITIVE; kiểm tra tx.origin
    # KHÔNG phải guard - với SWC-115 nó chính là lỗ hổng),
    # <TargetFunction>, <Context> (build_contract_context) và <Knowledge>.
    def _build_prompt(self, code: str, vulnerability: Dict[str, Any], knowledge_block: str) -> str:
        swc = vulnerability.get("swc") or "Unknown"
        title = vulnerability.get("title") or "Unknown vulnerability"
        severity = vulnerability.get("severity") or "Unknown"

        unit_code = str(vulnerability.get("unit_code") or "")
        if unit_code:
            target_code = unit_code
            start_line = vulnerability.get("start_line") or vulnerability.get("line") or 1
            end_line = vulnerability.get("end_line") or (start_line + target_code.count("\n"))
        else:
            # Contract-level finding: the contract itself is the target (first 2,000 characters).
            target_code = code[:2000]
            start_line, end_line = 1, 1 + target_code.count("\n")

        sol_match = re.search(r"pragma\s+solidity\s+([^;]+);", code)
        compiler_note = sol_match.group(1).strip() if sol_match else "0.4.x"

        # unit_id = "<Contract>.<function>[#n]": the context comes from the enclosing contract. A contract-level
        # finding (no unit) matches no contract name, so build_contract_context falls back to the whole source.
        unit_id = str(vulnerability.get("unit_id") or "")
        contract_name = unit_id.split(".", 1)[0] if "." in unit_id else "Contract"
        try:
            context_str = build_contract_context(
                {"contract_name": contract_name, "code": target_code}, code, budget_tokens=1500
            )
        except Exception:
            context_str = ""

        schema_str = (
            "{\n"
            '  "is_true_positive": bool,         // true if genuinely exploitable, false if false positive or guarded\n'
            '  "evidence_lines": [int],          // line numbers within <TargetFunction> exhibiting the flaw\n'
            '  "root_cause": string,             // detailed technical explanation of the flaw or why it is false positive\n'
            '  "attack_scenario": string,        // concrete step-by-step exploit scenario (or "None" if false positive)\n'
            '  "preconditions": [string],        // prerequisites for exploitation\n'
            '  "fix_strategy": string            // high-level conceptual remediation strategy for Fixer\n'
            "}"
        )

        sections = [
            "You are auditing ONE candidate finding. It may be a false positive.",
            f"Candidate: {title} ({swc}) - Severity: {severity}",
            f"Solidity version: {compiler_note}",
            "",
            "Return ONLY valid JSON matching this schema:",
            schema_str,
            "",
            "Rules:",
            "1. Cite only lines that exist in <TargetFunction>.",
            "2. If the function is protected by an authorization modifier (e.g. onlyOwner, oO) or by `if (msg.sender != ...) revert/throw`, an access-control finding is a FALSE POSITIVE (`is_true_positive: false`). A check on `tx.origin` is NOT a valid guard: for SWC-115 (tx.origin authentication) that check is the vulnerability itself.",
            "3. If this candidate is a false positive, set `is_true_positive: false` and describe the protective guard in `root_cause`.",
            "",
            f'<TargetFunction lines="{start_line}-{end_line}">',
            target_code,
            "</TargetFunction>",
            "",
            "<Context>",
            context_str if context_str else "No additional context",
            "</Context>",
            "",
            "<Knowledge>",
            knowledge_block,
            "</Knowledge>",
        ]
        return "\n".join(sections)

    # ── per-vulnerability advice ────────────────────────────────────────────
    # [HÀM CHÍNH] Xử lý MỘT finding: tra tri thức -> dựng prompt -> gọi LLM (generate_json) -> parse thành RepairSuggestion.
    # LLM lỗi -> trả RepairSuggestion có error (không làm hỏng cả batch).
    def advise_one(self, code: str, vulnerability: Dict[str, Any]) -> RepairSuggestion:
        retrieved = self.knowledge_base.retrieve(
            vulnerability.get("swc") or "Unknown", code=_target_code(vulnerability, code)
        )
        knowledge_block = self._build_knowledge_block(vulnerability, retrieved)
        prompt = self._build_prompt(code, vulnerability, knowledge_block)

        try:
            data = self.llm.generate_json(prompt, ADVISOR_SYSTEM_INSTRUCTION)
        except Exception as error:  # noqa: BLE001 - reported per finding, never aborts the batch
            return RepairSuggestion(error=f"{type(error).__name__}: {error}")

        return RepairSuggestion(
            is_true_positive=bool(data.get("is_true_positive", True)),
            root_cause=str(data.get("root_cause") or "").strip(),
            impact=str(data.get("attack_scenario") or "").strip() or str(data.get("impact") or "").strip(),
            fix_strategy=str(data.get("fix_strategy") or "").strip(),
        )
