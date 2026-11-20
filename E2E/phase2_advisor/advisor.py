"""Phase 2 - Advisor (agent #2 của LLM-BSCVM): finding đúng hay sai, nguyên nhân gốc, kịch bản tấn công, hướng sửa."""

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from E2E.common.gpu_client import ServerGPUClient
from E2E.common.io_utils import read_json
from E2E.common.regex_patterns import RE_PRAGMA_SPEC, RE_SWC_ID
from E2E.config import KNOWLEDGE_BASE_DIR
from E2E.phase0_parsing.contract_context import build_contract_context
from E2E.phase1b_rag.swc_agents import normalize_swc_id

# Advisor là người kiểm tra độc lập (không phải bộ lọc false positive): chỉ dựa trên mã nhìn thấy được.
ADVISOR_SYSTEM_INSTRUCTION = (
    "You are a senior smart-contract security auditor specialising in Solidity. "
    "You review ONE finding reported by automated detectors and decide whether it is a real, exploitable "
    "vulnerability in the supplied code. Base every claim on code you can see and never assume a protection "
    "that is not shown. Return ONLY valid JSON matching the requested schema."
)
VERDICT_CONFIRMED, VERDICT_FALSE_POSITIVE, VERDICT_UNCERTAIN = "confirmed", "false_positive", "uncertain"
VERDICTS = {VERDICT_CONFIRMED, VERDICT_FALSE_POSITIVE, VERDICT_UNCERTAIN}
# Lớp lỗi Advisor được gán cho finding UNMAPPED (chỉ detector báo); tiêu đề theo swc_registry.json.
UNMAPPED_CLASSES = (
    ("SWC-101", "Arithmetic Overflow/Underflow"), ("SWC-104", "Unchecked Return Value"), ("SWC-105", "Access Control"),
    ("SWC-107", "Reentrancy"), ("SWC-112", "Delegatecall Flaw"), ("SWC-113", "DoS with Failed Call"),
    ("SWC-114", "Front-Running / Slippage"), ("SWC-115", "Oracle Manipulation"), ("SWC-116", "Timestamp Dependence"),
    ("SWC-120", "Weak Sources of Randomness"), ("SWC-121", "Signature Replay"), ("SWC-128", "DoS With Block Gas Limit"),
    ("SWC-135", "Business Logic Flaw"),
)
UNMAPPED_CLASS_TITLES = dict(UNMAPPED_CLASSES)
# Định nghĩa 3 verdict + luật triage trong prompt (luật 3: phân quyền không bác được reentrancy / logic...).
TRIAGE_RULES = "\n".join([
    "Verdict:",
    '- "confirmed": the code shows the flaw and no visible code prevents it.',
    '- "false_positive": (a) visible code makes this flaw impossible - copy that exact code into `guard_quote`; '
    'or (b) the reported flaw is simply not present in <TargetFunction> - explain why in `root_cause` '
    'and leave `guard_quote` empty.',
    '- "uncertain": the answer depends on code that is not shown '
    '(an undefined modifier, library, external contract or caller).',
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
_SOURCE_NAMES = {"static": "static rule", "slither": "Slither", "rag": "RAG similar case",
                 "llm_detector": "fine-tuned detector"}
# Schema JSON yêu cầu Advisor trả về (dòng vulnerability_class chỉ có ở finding UNMAPPED).
_SCHEMA_CLASS_LINE = '  "vulnerability_class": string,    // one SWC id from the class list below, or "none"\n'
_SCHEMA_BODY = (
    '  "verdict": "confirmed" | "false_positive" | "uncertain",\n'
    '  "guard_quote": string,            // false_positive caused by a guard: the exact guard code copied '
    'from <TargetFunction> or <Context>; otherwise ""\n'
    '  "evidence_lines": [int],          // line numbers within <TargetFunction> exhibiting the flaw\n'
    '  "root_cause": string,             // technical explanation of the flaw, or why it is not exploitable\n'
    '  "attack_scenario": string,        // concrete step-by-step exploit scenario (or "None" if false_positive)\n'
    '  "preconditions": [string],        // prerequisites for exploitation\n'
    '  "fix_strategy": string            // high-level conceptual remediation strategy for Fixer\n'
)


# Kết quả Advisor cho MỘT finding: verdict, guard_quote, nguyên nhân, tác động, hướng sửa, lớp lỗi, lỗi khi gọi LLM.
@dataclass
class RepairSuggestion:
    verdict: str = VERDICT_CONFIRMED
    guard_quote: str = ""
    root_cause: str = ""
    impact: str = ""
    fix_strategy: str = ""
    vulnerability_class: str = ""
    error: str | None = None


# Độ giống giữa mã đang xét và mã lỗi của ca KB: trùng = 10, chứa nhau = 5 + tỉ lệ độ dài, còn lại SequenceMatcher.
def _code_match_score(query_code: str, item: dict) -> float:
    known = str(item.get("vulnerable_code") or "").strip()
    if not known:
        return 0.0
    if known == query_code:
        return 10.0
    if known in query_code or query_code in known:
        return 5.0 + min(len(known), len(query_code)) / max(len(known), len(query_code))
    return SequenceMatcher(None, query_code[:500], known[:500]).ratio()


# Thêm chuỗi (đã strip) vào danh sách nếu khác rỗng và chưa có.
def _add_unique(bucket: list[str], value: Any) -> None:
    text = str(value or "").strip()
    if text and text not in bucket:
        bucket.append(text)


# Tri thức sửa lỗi tra THEO MÃ SWC từ swc_registry.json (37 SWC) + knowledge_items.json (11,869 ca); Phase 1F, 2, 3.
class RepairKnowledgeBase:
    # Nạp registry và các ca lỗi, lập chỉ mục theo SWC (thiếu / hỏng file thì phần đó rỗng).
    def __init__(self, knowledge_dir: str | Path = KNOWLEDGE_BASE_DIR) -> None:
        self._registry_by_swc, self._items_by_swc, self._retrieved = {}, {}, {}
        for record in self._load(Path(knowledge_dir) / "swc_registry.json"):
            if swc := normalize_swc_id(record.get("id")):
                self._registry_by_swc[swc] = record
        for record in self._load(Path(knowledge_dir) / "knowledge_items.json"):
            if swc := normalize_swc_id(record.get("swc") or record.get("id")):
                self._items_by_swc.setdefault(swc, []).append(record)

    # Đọc một file JSON danh sách; thiếu / không đọc được -> [].
    @staticmethod
    def _load(path: Path) -> list[dict]:
        try:
            return read_json(path)
        except Exception:
            return []

    # Tra tri thức cho (SWC, limit, code) - có nhớ kết quả vì Phase 1F và Phase 2 tra cùng một finding.
    def retrieve(self, swc: str, limit: int = 3, code: str | None = None) -> dict:
        key = (normalize_swc_id(swc), limit, code)
        if key not in self._retrieved:
            self._retrieved[key] = self._retrieve(*key)
        return self._retrieved[key]

    # Gom mô tả / cách sửa / checklist / mã sửa từ registry + `limit` ca giống mã nhất; kèm best_pair, risk_guidance.
    def _retrieve(self, swc: str, limit: int, code: str | None) -> dict:
        titles, descriptions, remediations, checklist, registry_fixes, empirical_fixes = [], [], [], [], [], []
        best_pair: dict = {}
        reg = self._registry_by_swc.get(swc)
        if reg:
            _add_unique(titles, reg.get("title") or swc)
            _add_unique(descriptions, reg.get("description") or reg.get("root_cause"))
            _add_unique(remediations, reg.get("remediation"))
            for text in reg.get("checklist", []) or []:
                _add_unique(checklist, text)
            for sample in reg.get("samples", []) or []:
                if isinstance(sample, dict):
                    _add_unique(registry_fixes, sample.get("fixed"))
        items = list(self._items_by_swc.get(swc, []))
        if code and items:
            items.sort(key=lambda item: _code_match_score(code.strip(), item), reverse=True)
        for item in items[:limit]:
            _add_unique(titles, item.get("title"))
            _add_unique(descriptions, item.get("description") or item.get("root_cause"))
            remediation = str(item.get("remediation") or "").strip()
            _add_unique(remediations, remediation)
            for text in item.get("checklist", []) or []:
                _add_unique(checklist, text)
            fixed, vulnerable = (str(item.get(key) or "").strip() for key in ("fixed_code", "vulnerable_code"))
            _add_unique(empirical_fixes, fixed)
            if (fixed or vulnerable) and not best_pair:
                best_pair = {"vulnerable_code": vulnerable, "fixed_code": fixed,
                             "root_cause": str(item.get("root_cause") or "").strip(), "remediation": remediation,
                             "severity": str(item.get("severity") or (reg.get("severity") if reg else "Medium")),
                             "cwe_ids": item.get("cwe_ids") or []}
        samples = (reg.get("samples", []) or []) if reg and not best_pair else []
        sample = next((item for item in samples if isinstance(item, dict)), None)
        if sample:
            best_pair = {"vulnerable_code": str(sample.get("vulnerable") or "").strip(),
                         "fixed_code": str(sample.get("fixed") or "").strip(),
                         "root_cause": str(sample.get("root_cause") or "").strip(),
                         "remediation": str(reg.get("remediation") or "").strip(),
                         "severity": str(reg.get("severity") or "Medium"), "cwe_ids": []}
        code_examples = empirical_fixes + [fix for fix in registry_fixes if fix not in empirical_fixes]
        return {
            "descriptions": descriptions, "remediations": remediations, "checklist": checklist,
            "code_examples": code_examples[:limit], "best_pair": best_pair,
            "risk_guidance": {
                "swc": swc, "title": titles[0] if titles else swc,
                "benchmark_severity": reg.get("severity") if reg
                else (best_pair.get("severity") if best_pair else "Medium"),
                "cwe_ids": best_pair.get("cwe_ids", []) if best_pair else [],
                "remediation_summary": remediations[0] if remediations else "",
                "root_cause_summary": descriptions[0] if descriptions else "", "checklist": checklist[:6],
            },
        }

    # Bổ sung cho một finding: mã lỗi/mã sửa mẫu, cách sửa, root cause (chỉ điền trường còn thiếu) + risk_guidance.
    def enrich_finding(self, finding: dict, code: str = "") -> dict:
        target = finding.get("unit_code") or finding.get("vulnerable_code") or finding.get("matched_text") or code
        retrieved = self.retrieve(normalize_swc_id(finding.get("swc")), limit=3, code=target)
        enriched, best_pair = dict(finding), retrieved.get("best_pair", {})
        for field in ("vulnerable_code", "fixed_code", "remediation", "root_cause"):
            if not enriched.get(field) and best_pair.get(field):
                enriched[field] = best_pair[field]
        if not enriched.get("remediation") and retrieved.get("remediations"):
            enriched["remediation"] = retrieved["remediations"][0]
        if not enriched.get("root_cause") and retrieved.get("descriptions"):
            enriched["root_cause"] = retrieved["descriptions"][0]
        enriched["risk_guidance"] = retrieved.get("risk_guidance", {})
        return enriched

    # Làm giàu hàng loạt finding (gpu_pipeline gọi ngay sau fuse_findings); gắn unit_code = mã của chính hàm.
    def enrich_findings(self, findings: list[dict], code: str = "", code_units: list[dict] | None = None) -> list[dict]:
        unit_code = {unit["unit_id"]: unit["code"] for unit in code_units or []
                     if unit.get("unit_id") and unit.get("code")}
        enriched = []
        for finding in map(dict, findings):
            if finding.get("unit_id") in unit_code and not finding.get("unit_code"):
                finding["unit_code"] = unit_code[finding["unit_id"]]
            enriched.append(self.enrich_finding(finding, code=code))
        return enriched


# Khối <Phase1Evidence>: kênh nào báo, dòng mã static / Slither khớp, ca tương tự của RAG (không đưa xác suất detector).
def _phase1_evidence(vulnerability: dict, code: str) -> str:
    source_lines = code.splitlines()
    valid = lambda line: isinstance(line, int) and 1 <= line <= len(source_lines)
    code_at = lambda line: source_lines[line - 1].strip()[:160] if valid(line) else ""
    lines = []
    if sources := [str(item) for item in vulnerability.get("sources") or []]:
        lines.append("Reported by: " + ", ".join(_SOURCE_NAMES.get(item, item) for item in sources))
    if vulnerability.get("corroborated"):
        lines.append("Static rules and RAG both reported this finding independently.")
    shown_lines, rag_shown = set(), False
    for item in vulnerability.get("evidence") or []:
        source, line = str(item.get("source") or ""), item.get("line")
        if source in {"static", "slither"} and line not in shown_lines:
            shown_lines.add(line)
            text = f"{_SOURCE_NAMES[source]} at line {line}" if line else _SOURCE_NAMES[source]
            text += f": `{code_at(line)}`" if code_at(line) else ""
            if matched := str(item.get("matched_text") or item.get("check") or "").strip()[:80]:
                text += f" (matched `{matched}`)"
            if notes := [str(note) for note in item.get("notes") or [] if note]:
                text += f" - {notes[0][:120]}"
            lines.append(text)
        elif source == "rag" and not rag_shown:
            if description := " ".join(str(item.get("description") or item.get("root_cause") or "").split()):
                rag_shown = True
                lines.append("A similar known vulnerable case from the knowledge base (a different contract) was "
                             f"described as: {description[:400]}")
    if vulnerability.get("swc") == "UNMAPPED" and code_at(vulnerability.get("line")):
        lines.append(f"Most suspicious line according to the detector: line {vulnerability['line']}: "
                     f"`{code_at(vulnerability['line'])}`")
    return "\n".join(lines) if lines else "No additional evidence"


# Gợi ý phân loại cho finding UNMAPPED: tối đa 2 ca tương tự theo độ giống mã (Phase 1B), ghi rõ có thể không liên quan.
def _rag_hints(rag_context: list[dict] | None) -> str:
    hints = []
    for case in (rag_context or [])[:2]:
        swc = normalize_swc_id(case.get("swc"))
        description = " ".join(str(case.get("description") or case.get("root_cause") or "").split())[:200]
        try:
            similarity = f"{float(case.get('similarity')):.2f}"
        except (TypeError, ValueError):
            similarity = "?"
        hints.append(f"- {swc} {str(case.get('title') or swc).strip()} (similarity {similarity}): {description}")
    return "Most similar known cases by code similarity (they may be unrelated):\n" + "\n".join(hints) if hints else ""


# Khối <RetrievedSecurityKnowledge>: <= 3 cách sửa, 8 mục checklist, cặp "mã lỗi -> mã đã sửa" gần nhất (hoặc 2 ví dụ).
def _knowledge_block(vulnerability: dict, retrieved: dict) -> str:
    best_pair = retrieved.get("best_pair", {})
    rag_fixed = str(vulnerability.get("fixed_code") or best_pair.get("fixed_code") or "").strip()
    rag_vuln = str(vulnerability.get("vulnerable_code") or best_pair.get("vulnerable_code") or "").strip()
    checklist = [str(item)[:200] for item in retrieved.get("checklist", [])[:8]]
    code_examples = [item[:800] for item in retrieved.get("code_examples", [])[:3]]
    if rag_fixed and rag_fixed not in code_examples:
        code_examples.insert(0, rag_fixed[:800])
    lines = ["<RetrievedSecurityKnowledge>"]
    lines += [f"Remediation {index}: {item[:800]}"
              for index, item in enumerate(retrieved.get("remediations", [])[:3], 1)]
    if checklist:
        lines += ["Checklist:", *(f"  - {item}" for item in checklist)]
    if rag_vuln and rag_fixed:
        lines.append("Most Relevant Empirical Code Repair Pair (from 11,869-case knowledge base):\n"
                     f"  Vulnerable pattern:\n```solidity\n{rag_vuln[:600]}\n```\n"
                     f"  Validated fixed pattern:\n```solidity\n{rag_fixed[:600]}\n```")
    else:
        lines += [f"Reference fix example {index} (from vulnerability knowledge base):\n```solidity\n{item}\n```"
                  for index, item in enumerate(code_examples[:2], 1)]
    return "\n".join(lines + ["</RetrievedSecurityKnowledge>"])


# Agent Advisor: LLM trên GPU server đánh giá một finding dựa trên bằng chứng Phase 1, ngữ cảnh contract và tri thức KB.
class AdvisorAgent:
    # Gắn client LLM và kho tri thức sửa lỗi (dùng chung KB của gpu_pipeline nếu được truyền vào).
    def __init__(self, llm: ServerGPUClient | None = None, knowledge_base: RepairKnowledgeBase | None = None) -> None:
        self.llm = llm or ServerGPUClient()
        self.knowledge_base = knowledge_base or RepairKnowledgeBase()

    # Prompt: đầu bài, phiên bản Solidity, <Phase1Evidence>, schema, TRIAGE_RULES, <TargetFunction>, <Context>, <Knowledge>.
    def _build_prompt(self, code: str, vulnerability: dict, knowledge_block: str,
                      rag_context: list[dict] | None = None) -> str:
        unmapped = vulnerability.get("swc") == "UNMAPPED"
        target_code = str(vulnerability.get("unit_code") or "")
        if target_code:
            start_line = vulnerability.get("start_line") or vulnerability.get("line") or 1
            end_line = vulnerability.get("end_line") or (start_line + target_code.count("\n"))
        else:
            target_code = code[:2000]
            start_line, end_line = 1, 1 + target_code.count("\n")
        pragma = RE_PRAGMA_SPEC.search(code)
        unit_id = str(vulnerability.get("unit_id") or "")
        contract_name = unit_id.split(".", 1)[0] if "." in unit_id else "Contract"
        try:
            context = build_contract_context({"contract_name": contract_name, "code": target_code}, code,
                                             budget_tokens=1500)
        except Exception:
            context = ""
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
        return "\n".join([
            *header, f"Solidity version: {pragma.group(1).strip() if pragma else '0.4.x'}", "",
            "<Phase1Evidence>", *[block for block in evidence if block], "</Phase1Evidence>", "",
            "Return ONLY valid JSON matching this schema:",
            "{\n" + (_SCHEMA_CLASS_LINE if unmapped else "") + _SCHEMA_BODY + "}", "",
            TRIAGE_RULES, "",
            f'<TargetFunction lines="{start_line}-{end_line}">', target_code, "</TargetFunction>", "",
            "<Context>", context or "No additional context", "</Context>", "",
            "<Knowledge>", knowledge_block, "</Knowledge>",
        ])

    # [HÀM CHÍNH PHASE 2 - một finding] Tra KB theo SWC -> prompt -> LLM -> RepairSuggestion (lỗi -> có error).
    def advise_one(self, code: str, vulnerability: dict, rag_context: list[dict] | None = None) -> RepairSuggestion:
        retrieved = self.knowledge_base.retrieve(vulnerability.get("swc") or "Unknown",
                                                 code=str(vulnerability.get("unit_code") or "") or code)
        prompt = self._build_prompt(code, vulnerability, _knowledge_block(vulnerability, retrieved), rag_context)
        try:
            data = self.llm.generate_json(prompt, ADVISOR_SYSTEM_INSTRUCTION)
        except Exception as error:
            return RepairSuggestion(error=f"{type(error).__name__}: {error}")
        if not isinstance(data, dict):
            return RepairSuggestion(error=f"Advisor returned {type(data).__name__}, expected a JSON object")
        verdict = str(data.get("verdict") or "").strip().lower()
        swc_class = RE_SWC_ID.search(str(data.get("vulnerability_class") or "").upper())
        swc_class = swc_class.group(0) if swc_class else ""
        unmapped = vulnerability.get("swc") == "UNMAPPED"
        text = lambda key: str(data.get(key) or "").strip()
        return RepairSuggestion(
            verdict=verdict if verdict in VERDICTS else VERDICT_UNCERTAIN, guard_quote=text("guard_quote"),
            root_cause=text("root_cause"), impact=text("attack_scenario") or text("impact"),
            fix_strategy=text("fix_strategy"),
            vulnerability_class=swc_class if unmapped and swc_class in UNMAPPED_CLASS_TITLES else "",
        )
