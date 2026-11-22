"""Phase 2 - Advisor (agent #2 của LLM-BSCVM): finding đúng hay sai, nguyên nhân gốc, kịch bản tấn công, hướng sửa."""

from dataclasses import dataclass

from E2E.common.gpu_client import ServerGPUClient
from E2E.common.knowledge_base import KnowledgeBase, default_knowledge_base
from E2E.common.regex_patterns import RE_PRAGMA_SPEC
from E2E.phase0_parsing.contract_context import build_contract_context

# Advisor là người kiểm tra độc lập (không phải bộ lọc false positive): chỉ dựa trên mã nhìn thấy được.
ADVISOR_SYSTEM_INSTRUCTION = (
    "You are a senior smart-contract security auditor specialising in Solidity. "
    "You review ONE finding reported by automated detectors and decide whether it is a real, exploitable "
    "vulnerability in the supplied code. Base every claim on code you can see and never assume a protection "
    "that is not shown. Return ONLY valid JSON matching the requested schema."
)
VERDICT_CONFIRMED, VERDICT_FALSE_POSITIVE, VERDICT_UNCERTAIN = "confirmed", "false_positive", "uncertain"
VERDICTS = {VERDICT_CONFIRMED, VERDICT_FALSE_POSITIVE, VERDICT_UNCERTAIN}
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
# Schema JSON yêu cầu Advisor trả về.
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


# Kết quả Advisor cho MỘT finding: verdict, guard_quote, nguyên nhân, tác động, hướng sửa, lỗi khi gọi LLM.
@dataclass
class RepairSuggestion:
    verdict: str = VERDICT_CONFIRMED
    guard_quote: str = ""
    root_cause: str = ""
    impact: str = ""
    fix_strategy: str = ""
    error: str | None = None


# Khối <Phase1Evidence>: kênh nào báo, dòng mã static / Slither khớp, ca tương tự của RAG (không đưa xác suất detector).
def _phase1_evidence(vulnerability: dict, code: str) -> str:
    source_lines = code.splitlines()
    valid = lambda line: isinstance(line, int) and 1 <= line <= len(source_lines)
    code_at = lambda line: source_lines[line - 1].strip()[:160] if valid(line) else ""
    lines = []
    if sources := [str(item) for item in vulnerability.get("sources") or []]:
        lines.append("Reported by: " + ", ".join(_SOURCE_NAMES.get(item, item) for item in sources))
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
    if classification := vulnerability.get("classification"):
        similarity = classification.get("similarity")
        lines.append(f"The class {vulnerability.get('swc')} was assigned in Phase 1 from the most similar known "
                     "vulnerable case in the knowledge base"
                     + (f" (code similarity {float(similarity):.2f})." if similarity is not None else "."))
        if code_at(vulnerability.get("line")):
            lines.append(f"Most suspicious line according to the detector: line {vulnerability['line']}: "
                         f"`{code_at(vulnerability['line'])}`")
    return "\n".join(lines) if lines else "No additional evidence"


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
    # Gắn client LLM và KB dùng chung (common/knowledge_base.py).
    def __init__(self, llm: ServerGPUClient | None = None, knowledge_base: KnowledgeBase | None = None) -> None:
        self.llm = llm or ServerGPUClient()
        self.knowledge_base = knowledge_base or default_knowledge_base()

    # Prompt: đầu bài, phiên bản Solidity, <Phase1Evidence>, schema, TRIAGE_RULES, <TargetFunction>, <Context>, <Knowledge>.
    def _build_prompt(self, code: str, vulnerability: dict, knowledge_block: str) -> str:
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
        return "\n".join([
            "Review ONE finding reported by the Phase 1 detectors.",
            f"Finding: {vulnerability.get('title') or 'Unknown vulnerability'} "
            f"({vulnerability.get('swc') or 'Unknown'}) - Severity: {vulnerability.get('severity') or 'Unknown'}",
            f"Solidity version: {pragma.group(1).strip() if pragma else '0.4.x'}", "",
            "<Phase1Evidence>", _phase1_evidence(vulnerability, code), "</Phase1Evidence>", "",
            "Return ONLY valid JSON matching this schema:", "{\n" + _SCHEMA_BODY + "}", "",
            TRIAGE_RULES, "",
            f'<TargetFunction lines="{start_line}-{end_line}">', target_code, "</TargetFunction>", "",
            "<Context>", context or "No additional context", "</Context>", "",
            "<Knowledge>", knowledge_block, "</Knowledge>",
        ])

    # [HÀM CHÍNH PHASE 2 - một finding] Tra KB theo SWC + mã hàm -> prompt -> LLM -> RepairSuggestion (lỗi -> error).
    def advise_one(self, code: str, vulnerability: dict) -> RepairSuggestion:
        retrieved = self.knowledge_base.repair_knowledge(vulnerability.get("swc") or "Unknown",
                                                         code=str(vulnerability.get("unit_code") or "") or code)
        prompt = self._build_prompt(code, vulnerability, _knowledge_block(vulnerability, retrieved))
        try:
            data = self.llm.generate_json(prompt, ADVISOR_SYSTEM_INSTRUCTION)
        except Exception as error:
            return RepairSuggestion(error=f"{type(error).__name__}: {error}")
        if not isinstance(data, dict):
            return RepairSuggestion(error=f"Advisor returned {type(data).__name__}, expected a JSON object")
        verdict = str(data.get("verdict") or "").strip().lower()
        text = lambda key: str(data.get(key) or "").strip()
        return RepairSuggestion(
            verdict=verdict if verdict in VERDICTS else VERDICT_UNCERTAIN, guard_quote=text("guard_quote"),
            root_cause=text("root_cause"), impact=text("attack_scenario") or text("impact"),
            fix_strategy=text("fix_strategy"),
        )
