"""Phase 4 - Fixer: vá từng hàm lỗi bằng Qwen2.5-Coder-14B (prompt + ca KB của hàm -> LLM -> tách hàm -> ghép)."""

from E2E.common.gpu_client import generate_via_server
from E2E.common.knowledge_base import KnowledgeBase, default_knowledge_base
from E2E.common.regex_patterns import RE_FULL_VERSION, RE_PRAGMA_SPEC
from E2E.phase0_parsing.code_units import CodeUnit, parse_code_units
from E2E.phase0_parsing.contract_context import build_contract_context
from E2E.phase4_fixer.patching import apply_unit_patches, extract_declaration_body, lenient_json_string_field

FIXER_SYSTEM_PROMPT = "You are a Solidity security auditor. Output strictly valid JSON with the requested schema."


# Đoạn "ca thực nghiệm" cho Fixer từ các ca KB của chính hàm đang sửa (KnowledgeBase.cases_for_unit).
def _build_rag_knowledge_section(cases: list[dict]) -> str:
    entries = []
    for ctx in cases:
        swc = ctx.get("swc") or "Security"
        entry = f"- [{swc}] {ctx.get('title') or swc} (similarity: {float(ctx.get('similarity') or 0.0):.2f})"
        if root_cause := (ctx.get("root_cause") or ctx.get("description") or "").strip():
            entry += f"\n  Root cause: {root_cause[:250]}"
        if remediation := (ctx.get("remediation") or "").strip():
            entry += f"\n  Remediation guidance: {remediation[:250]}"
        vulnerable, fixed = str(ctx.get("vulnerable_code") or "").strip(), str(ctx.get("fixed_code") or "").strip()
        if ctx.get("fixed_code") and ctx.get("vulnerable_code") and len(vulnerable) <= 350 and len(fixed) <= 350:
            entry += (f"\n  Vulnerable reference snippet:\n  ```solidity\n{vulnerable}\n  ```\n"
                      f"  Secure reference fix:\n  ```solidity\n{fixed}\n  ```")
        entries.append(entry)
    if not entries:
        return ""
    return "\n\n### RAG Retrieved Knowledge & Defense Context (Empirical Cases):\n" + "\n\n".join(entries)


# Tìm lại một unit trong mã đã parse lại (offset đã đổi do vá hàm khác): theo unit_id, không có thì theo tên duy nhất.
def _find_unit(units: list[CodeUnit], target: CodeUnit) -> CodeUnit | None:
    by_id = next((unit for unit in units if unit.unit_id == target.unit_id), None)
    same_name = [unit for unit in units if unit.name == target.name and unit.contract_name == target.contract_name]
    return by_id or (same_name[0] if len(same_name) == 1 else None)


# Ghép bản thay thế của MỘT hàm vào mã hiện tại (parse lại -> tìm unit -> apply_unit_patches); lỗi -> ValueError.
def _splice_unit(code: str, target: CodeUnit, replacement: str, summary: str) -> str:
    current = _find_unit(parse_code_units(code), target)
    if current is None:
        raise ValueError(f"Unit {target.unit_id} could not be located in the current source")
    updated, outcomes = apply_unit_patches(code, [current], [{"unit_id": current.unit_id, "replacement": replacement,
                                                               "summary": summary}])
    if not any(item.get("status") == "applied" for item in outcomes):
        raise ValueError(str(outcomes[0].get("reason") if outcomes else "patch was not applied"))
    return updated


# Phiên bản solc thấp nhất mà pragma cho phép (bản vá phải compile được với bản này).
def _min_pragma_version(source_code: str) -> tuple[int, int, int] | None:
    match = RE_PRAGMA_SPEC.search(source_code)
    versions = [tuple(map(int, found)) for found in RE_FULL_VERSION.findall(match.group(1))] if match else []
    return min(versions) if versions else None


# Luật cú pháp cho Solidity cũ (< 0.6) chèn vào prompt (vd < 0.4.22 không có constructor, < 0.4.21 không có emit).
def _version_rules(source_code: str) -> str:
    version = _min_pragma_version(source_code)
    if version is None or version >= (0, 6, 0):
        return ""
    spec = RE_PRAGMA_SPEC.search(source_code).group(1).strip()
    rules = [f"   - TARGET SOLIDITY COMPILER: pragma solidity {spec} (legacy {version[0]}.{version[1]}.{version[2]})."]
    if version < (0, 4, 22):
        rules += ["     * DO NOT pass a message string to require(); only `require(condition);` is valid.",
                  "     * DO NOT use the `constructor` keyword; constructors are functions named after the contract.",
                  "     * DO NOT use `abi.encode` / `abi.encodePacked` (not available); pass several arguments "
                  "directly, e.g. `keccak256(a, b)`."]
    else:
        rules.append("     * Keep the declaration kind unchanged: a `constructor(...)` stays a `constructor(...)`.")
    if version < (0, 4, 21):
        rules.append("     * DO NOT use the `emit` keyword; trigger events directly via `EventName(...)`.")
    if version < (0, 5, 0):
        rules.append("     * Keep pre-0.5 syntax (no `address payable`, no `calldata`, `.call.value(x)()` style).")
    rules.append("     * Preserve return semantics: if the original returns a boolean (e.g. `return false;`), "
                 "do NOT change it to `revert()`.")
    return "\n".join(rules) + "\n"


# Prompt Fixer cho MỘT hàm: lỗi + hướng sửa của Advisor (confirmed/uncertain), mã gốc, ngữ cảnh, ca KB, feedback, luật.
def _build_fixer_prompt(unit: CodeUnit, unit_findings: list[dict], unit_suggestions: list[dict], source_code: str,
                        feedback_note: str, knowledge_base: KnowledgeBase) -> str:
    lines = [f"  * [{item.get('swc', '')}] {item.get('title', '')}" for item in unit_findings]
    for suggestion in unit_suggestions:
        if suggestion.get("advisor_verdict") not in {"confirmed", "uncertain"}:
            continue
        if suggestion.get("fix_strategy"):
            lines.append(f"  * Fix Strategy: {suggestion['fix_strategy']}")
        elif suggestion.get("root_cause"):
            lines.append(f"  * Root cause: {suggestion['root_cause']}")
    findings_text = "\n".join(lines) if lines else "Security vulnerability detected."
    context = build_contract_context(unit, source_code, budget_tokens=1500)
    cases = knowledge_base.cases_for_unit(unit.code, {str(item.get("swc")) for item in unit_findings if item.get("swc")})
    rag_section = _build_rag_knowledge_section(cases)
    return (
        "You are an expert Solidity smart contract security researcher.\n"
        f"Target vulnerable function: `{unit.name}`\n"
        f"Identified Vulnerabilities & Diagnoses:\n{findings_text}\n\n"
        f"Original declaration:\n```solidity\n{unit.code}\n```\n\n"
        f"Contract Context & Invariants:\n```solidity\n{context}\n```\n"
        f"{rag_section}\n"
        f"{feedback_note}\n\n"
        "INSTRUCTIONS:\n"
        f"1. Generate a MINIMAL DIFF security repair for `{unit.name}`.\n"
        f"{_version_rules(source_code)}"
        "2. All state write statements and event emissions from the original function MUST be preserved.\n"
        f"3. Only `{unit.name}` is replaced, so the patch must compile against the rest of the contract unchanged: "
        "use only modifiers, functions and state variables that already exist (see Contract Context) and never "
        "introduce a new modifier. To restrict access, add a `require(...)` check on `msg.sender` "
        "inside the function.\n"
        "4. Return ONLY a single valid JSON object with format:\n"
        "{\n"
        f'  "replacement": "full replacement code beginning with function {unit.name} '
        'and ending at its closing brace",\n'
        '  "changed_lines": [1, 2],\n'
        '  "why": "brief rationale"\n'
        "}\n"
        "Do NOT include explanations or text outside the JSON."
    )


# [HÀM CHÍNH PHASE 4] Vá MỘT hàm: prompt -> LLM (temperature 0) -> tách hàm -> ghép; lỗi -> rejected (Phase 5 kiểm chứng).
def patch_unit(code: str, unit: CodeUnit, *, unit_findings: list[dict], unit_suggestions: list[dict], source_code: str,
               feedback_note: str, attempt: int, server_url: str | None,
               knowledge_base: KnowledgeBase | None = None) -> tuple[str, dict]:
    outcome, stage, rationale = {"finding_id": f"LLM:Qwen14B:{unit.unit_id}", "unit_id": unit.unit_id}, "prepare", ""
    try:
        prompt = _build_fixer_prompt(unit, unit_findings, unit_suggestions, source_code, feedback_note,
                                     knowledge_base or default_knowledge_base())
        stage = "generate"
        reply = generate_via_server(prompt, system_prompt=FIXER_SYSTEM_PROMPT, temperature=0.0, server_url=server_url)
        stage = "extract"
        rationale = (lenient_json_string_field(reply, "why") or "").strip()
        candidate = extract_declaration_body(reply, target_name=unit.name)
        if not candidate or unit.name not in candidate:
            raise ValueError(f"Extracted patch does not define expected function {unit.name}")
        stage = "apply"
        updated = _splice_unit(code, unit, candidate, f"Qwen-14B patch for {unit.unit_id} (attempt {attempt})")
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        return code, {**outcome, "status": "rejected", "stage": stage, "attempt": attempt, "error": message,
                      "rationale": rationale, "reason": f"Patch rejected at stage '{stage}': {message}"}
    return updated, {**outcome, "status": "applied", "stage": "applied", "engine": "llm_fixer",
                     "patched_code": candidate, "attempt": attempt, "rationale": rationale,
                     "reason": f"Applied neural unit patch for {unit.unit_id} (attempt {attempt})."}
