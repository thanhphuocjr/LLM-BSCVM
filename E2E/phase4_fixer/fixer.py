"""
Phase 4 - Fixer: repair one vulnerable function at a time with Qwen2.5-Coder-14B.

Prompt = target function + findings and the Advisor's fix strategy + contract context
+ RAG knowledge (similar vulnerable/fixed cases) + compiler-version rules + feedback of
the previous attempt. The reply must be JSON {"replacement", "changed_lines", "why"}.

The replacement is extracted, checked by the behaviour guard and spliced into the
contract by unit id. When any step fails the unit is reported as `rejected` (with the
failing stage); there is no fallback patch source.
"""

from __future__ import annotations

import re
from typing import Any

from E2E.common.gpu_client import generate_via_server
from E2E.phase0_parsing.code_units import CodeUnit, parse_code_units
from E2E.phase0_parsing.contract_context import build_contract_context
from E2E.phase4_fixer.behavior_guard import validate_observable_behavior
from E2E.phase4_fixer.patching import apply_unit_patches, extract_declaration_body, lenient_json_string_field


# Nhiệm vụ: chọn tối đa 3 ca tương tự từ ngữ cảnh RAG (retrieved_context) - ưu tiên ca cùng SWC với lỗi đang sửa,
# bỏ ca similarity < 0.08 - và viết thành đoạn prompt: SWC, root cause, cách sửa, cặp mã lỗi/mã sửa ngắn (<= 350 ký tự).
# Mục đích: cho Fixer "xem ví dụ" sửa lỗi thực tế từ KB 11,869 ca.
def _build_rag_knowledge_section(
    rag_context: list[dict[str, Any]],
    target_swc: str | None = None,
    max_items: int = 3,
) -> str:
    """Xây dựng phần bối cảnh tri thức bảo mật (RAG context) từ 11,869 cases cho Fixer LLM."""
    if not rag_context:
        return ""

    candidates = []
    for ctx in rag_context:
        sim = float(ctx.get("similarity") or 0.0)
        c_swc = ctx.get("swc") or "Security"
        # Ưu tiên case khớp SWC của lỗ hổng mục tiêu
        priority = (2.0 + sim) if (target_swc and c_swc == target_swc) else sim
        if sim >= 0.08 or (target_swc and c_swc == target_swc):
            candidates.append((priority, ctx))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = [c[1] for c in candidates[:max_items]]
    if not selected:
        return ""

    items_text = []
    for ctx in selected:
        c_swc = ctx.get("swc") or "Security"
        c_title = ctx.get("title") or c_swc
        sim = float(ctx.get("similarity") or 0.0)
        c_root = (ctx.get("root_cause") or ctx.get("description") or "").strip()
        c_rem = (ctx.get("remediation") or "").strip()
        entry = f"- [{c_swc}] {c_title} (similarity: {sim:.2f})"
        if c_root:
            entry += f"\n  Root cause: {c_root[:250]}"
        if c_rem:
            entry += f"\n  Remediation guidance: {c_rem[:250]}"
        if ctx.get("fixed_code") and ctx.get("vulnerable_code"):
            v_code = str(ctx["vulnerable_code"]).strip()
            f_code = str(ctx["fixed_code"]).strip()
            if len(v_code) <= 350 and len(f_code) <= 350:
                entry += (
                    f"\n  Vulnerable reference snippet:\n  ```solidity\n{v_code}\n  ```\n"
                    f"  Secure reference fix:\n  ```solidity\n{f_code}\n  ```"
                )
        items_text.append(entry)

    return (
        "\n\n### RAG Retrieved Knowledge & Defense Context (Empirical Cases):\n"
        + "\n\n".join(items_text)
    )


# Tìm lại một unit trong mã đã parse lại (sau khi các hàm khác đã được vá, offset thay đổi): theo unit_id,
# không có thì theo tên hàm duy nhất trong cùng contract.
def _find_unit(units: list[CodeUnit], target: CodeUnit) -> CodeUnit | None:
    """Locate `target` in a re-parsed source: by unit_id, else by a unique name match."""
    by_id = next((unit for unit in units if unit.unit_id == target.unit_id), None)
    if by_id is not None:
        return by_id
    same_name = [
        unit for unit in units
        if unit.name == target.name and unit.contract_name == target.contract_name
    ]
    return same_name[0] if len(same_name) == 1 else None


# Ghép bản thay thế của MỘT hàm vào mã hiện tại (parse lại -> tìm unit -> apply_unit_patches).
# Không ghép được thì raise ValueError. Dùng bởi patch_unit và gpu_pipeline (dựng lại mã từ các bản vá đã đạt).
def _splice_unit(code: str, target: CodeUnit, replacement: str, summary: str) -> str:
    current = _find_unit(parse_code_units(code), target)
    if current is None:
        raise ValueError(f"Unit {target.unit_id} could not be located in the current source")
    updated, outcomes = apply_unit_patches(
        code,
        [current],
        [{"unit_id": current.unit_id, "replacement": replacement, "summary": summary}],
    )
    if not any(item.get("status") == "applied" for item in outcomes):
        reason = outcomes[0].get("reason") if outcomes else "patch was not applied"
        raise ValueError(str(reason))
    return updated


# Phiên bản solc thấp nhất mà pragma cho phép (bản vá phải compile được với bản này).
def _min_pragma_version(source_code: str) -> tuple[int, int, int] | None:
    """Lowest compiler version allowed by the pragma (the patch must compile with it)."""
    match = re.search(r"pragma\s+solidity\s+([^;]+);", source_code)
    if not match:
        return None
    versions = [
        tuple(int(part) for part in found)
        for found in re.findall(r"(\d+)\.(\d+)\.(\d+)", match.group(1))
    ]
    return min(versions) if versions else None


# Sinh luật cú pháp theo phiên bản Solidity cũ (< 0.6) để chèn vào prompt: vd < 0.4.22 không có require(msg),
# không có constructor/abi.encode; < 0.4.21 không có emit; < 0.5 không có address payable/calldata.
# Mục đích: model hay viết cú pháp 0.8 cho contract 0.4 -> compile lỗi; luật này giảm lỗi đó.
def _version_rules(source_code: str) -> str:
    version = _min_pragma_version(source_code)
    if version is None or version >= (0, 6, 0):
        return ""
    spec = re.search(r"pragma\s+solidity\s+([^;]+);", source_code).group(1).strip()
    rules = [f"   - TARGET SOLIDITY COMPILER: pragma solidity {spec} (legacy {version[0]}.{version[1]}.{version[2]})."]
    if version < (0, 4, 22):
        rules.append("     * DO NOT pass a message string to require(); only `require(condition);` is valid.")
        rules.append("     * DO NOT use the `constructor` keyword; constructors are functions named after the contract.")
        rules.append("     * DO NOT use `abi.encode` / `abi.encodePacked` (not available); pass several arguments "
                     "directly, e.g. `keccak256(a, b)`.")
    else:
        rules.append("     * Keep the declaration kind unchanged: a `constructor(...)` stays a `constructor(...)`.")
    if version < (0, 4, 21):
        rules.append("     * DO NOT use the `emit` keyword; trigger events directly via `EventName(...)`.")
    if version < (0, 5, 0):
        rules.append("     * Keep pre-0.5 syntax (no `address payable`, no `calldata`, `.call.value(x)()` style).")
    rules.append(
        "     * Preserve return semantics: if the original returns a boolean (e.g. `return false;`), do NOT change it to `revert()`."
    )
    return "\n".join(rules) + "\n"


# Dựng prompt Fixer cho MỘT hàm: tên hàm, danh sách lỗi + fix_strategy của Advisor (chỉ lấy suggestion đã được
# Advisor xác nhận là true positive), mã gốc, ngữ cảnh contract + bất biến, tri thức RAG, feedback lần trước,
# luật phiên bản, yêu cầu "MINIMAL DIFF", giữ mọi lệnh ghi state/event, chỉ dùng modifier/hàm/biến đã có (bản vá
# chỉ thay một hàm nên không thể khai báo modifier mới), trả JSON {replacement, changed_lines, why}.
def _build_fixer_prompt(
    unit: CodeUnit,
    unit_findings: list[dict[str, Any]],
    unit_suggestions: list[dict[str, Any]],
    source_code: str,
    feedback_note: str,
    rag_section: str,
) -> str:
    finding_lines = [f"  * [{item.get('swc', '')}] {item.get('title', '')}" for item in unit_findings]
    for suggestion in unit_suggestions:
        # Suggestions without `is_true_positive` come from the knowledge-base fallback and describe
        # a different contract's case; feeding them as this function's diagnosis misleads the fixer.
        if not suggestion.get("is_true_positive"):
            continue
        if suggestion.get("fix_strategy"):
            finding_lines.append(f"  * Fix Strategy: {suggestion['fix_strategy']}")
        elif suggestion.get("root_cause"):
            finding_lines.append(f"  * Root cause: {suggestion['root_cause']}")
    findings_text = "\n".join(finding_lines) if finding_lines else "Security vulnerability detected."
    context = build_contract_context(unit, source_code, budget_tokens=1500)
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
        "introduce a new modifier. To restrict access, add a `require(...)` check on `msg.sender` inside the function.\n"
        "4. Return ONLY a single valid JSON object with format:\n"
        "{\n"
        f'  "replacement": "full replacement code beginning with function {unit.name} and ending at its closing brace",\n'
        '  "changed_lines": [1, 2],\n'
        '  "why": "brief rationale"\n'
        "}\n"
        "Do NOT include explanations or text outside the JSON."
    )


# [HÀM CHÍNH PHASE 4] Vá MỘT hàm:
#   prepare (dựng prompt) -> generate (gọi LLM, temperature 0) -> extract (lấy hàm thay thế từ JSON/code fence)
#   -> guard (validate_observable_behavior: không được mất chuyển ETH/ghi state/event/return) -> apply (_splice_unit).
# Bước nào lỗi -> outcome "rejected" kèm stage lỗi (không có nguồn vá dự phòng).
# Trả về (mã mới, outcome {status applied/rejected, engine, stage, rationale...}).
def patch_unit(
    code: str,
    unit: CodeUnit,
    *,
    unit_findings: list[dict[str, Any]],
    unit_suggestions: list[dict[str, Any]],
    source_code: str,
    feedback_note: str,
    rag_section: str,
    attempt: int,
    server_url: str | None,
) -> tuple[str, dict[str, Any]]:
    """Patch one unit with the generative model; any failure is reported as `rejected`."""
    label = f"attempt {attempt}"
    stage = "prepare"
    llm_error = ""
    rationale = ""
    try:
        prompt = _build_fixer_prompt(unit, unit_findings, unit_suggestions, source_code, feedback_note, rag_section)
        stage = "generate"
        reply = generate_via_server(
            prompt,
            system_prompt="You are a Solidity security auditor. Output strictly valid JSON with the requested schema.",
            temperature=0.0,
            server_url=server_url,
        )
        stage = "extract"
        rationale = (lenient_json_string_field(reply, "why") or "").strip()
        candidate = extract_declaration_body(reply, target_name=unit.name)
        if not candidate or unit.name not in candidate:
            raise ValueError(f"Extracted patch does not define expected function {unit.name}")
        stage = "guard"
        validate_observable_behavior(unit.code, candidate)
        stage = "apply"
        updated = _splice_unit(code, unit, candidate, f"Qwen-14B patch for {unit.unit_id} ({label})")
        return updated, {
            "finding_id": f"LLM:Qwen14B:{unit.unit_id}",
            "unit_id": unit.unit_id,
            "status": "applied",
            "stage": "applied",
            "engine": "llm_fixer",
            "patched_code": candidate,
            "attempt": attempt,
            "rationale": rationale,
            "reason": f"Applied neural unit patch for {unit.unit_id} ({label}).",
        }
    except Exception as error:
        llm_error = f"{type(error).__name__}: {error}"

    return code, {
        "finding_id": f"LLM:Qwen14B:{unit.unit_id}",
        "unit_id": unit.unit_id,
        "status": "rejected",
        "stage": stage,
        "attempt": attempt,
        "error": llm_error,
        "rationale": rationale,
        "reason": f"Patch rejected at stage '{stage}': {llm_error}",
    }
