"""
server_gpu_runner.py
══════════════════════════════════════════════════════════════════════════════
E2E Inference Pipeline Runner sử dụng Dedicated GPU Server (RTX 4070 Ti SUPER)
với mô hình Version 13 (Qwen2.5-Coder-14B-Instruct + LoRA Detector v13).

Thay vì chờ Kaggle 15 - 35 phút (xếp hàng, tải container, nạp weights):
  - Phase 1 Detector: Gửi từng function unit lên GPU 0 server (tốc độ ~100ms/unit).
  - Phase 2-5 Generative Agents: Sinh phân tích Advisor, Assessor, và Fixer trực tiếp
    từ Qwen 14B trên GPU server.
  - Phase 1 Re-verification: Đánh giá lại code đã vá để đảm bảo an toàn.
  - Toàn bộ pipeline chạy hoàn tất trong 5 - 15 giây!
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

from E2E.config import DEFAULT_DETECTOR_THRESHOLD, RISK_ORDER
from E2E.local.local_llm_runner import (
    _apply_heuristic_security_patch,
    _generate_advisor_section,
    _generate_assessor_section,
    finding_key,
)
from E2E.local.rag_runner import run_rag
from E2E.local.static_runner import run_static
from E2E.shared.function_parser import (
    CodeUnit,
    apply_unit_patches,
    build_contract_context,
    extract_declaration_body,
    lenient_json_string_field,
    parse_code_units,
    validate_observable_behavior,
)
from E2E.shared.fusion import fuse_findings
from E2E.shared.solc import check_compilation_with_baseline

MAX_VERIFICATION_ATTEMPTS = 3

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_env_file() -> None:
    path = PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value


def get_server_url(server_url: str | None = None) -> str:
    if server_url:
        return server_url.rstrip("/")
    _load_env_file()
    return os.environ.get("SERVER_GPU_URL", "http://100.122.153.35:6868").rstrip("/")


DEFAULT_SERVER_URL = get_server_url()


def _auth_headers() -> dict[str, str]:
    """Optional shared secret for the GPU server (set SERVER_GPU_API_KEY on both sides)."""
    _load_env_file()
    api_key = os.environ.get("SERVER_GPU_API_KEY", "").strip()
    return {"X-API-Key": api_key} if api_key else {}


def _post_json(url: str, payload: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **_auth_headers()},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_server_health(server_url: str | None = None, timeout: int = 3) -> dict[str, Any]:
    """Kiểm tra xem API server v13 trên GPU server có sẵn sàng hay không."""
    actual_url = get_server_url(server_url)
    url = f"{actual_url}/api/v1/health"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LLM-BSCVM-Local", **_auth_headers()})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"status": "offline", "error": str(exc), "url": url}


def normalize_solidity_unit(code_str: str) -> str:
    """Chuẩn hóa cấu trúc thụt lề hàm Solidity về chuẩn 4-spaces của Solidity Style Guide.
    Loại bỏ hoàn toàn lỗi Tokenizer shift trong mô hình neural detector do indentation bất thường."""
    lines = [l.strip() for l in code_str.strip().splitlines() if l.strip()]
    if not lines:
        return code_str.strip()
    formatted = []
    depth = 0
    for l in lines:
        closing = l.startswith("}") or l.startswith(");")
        cur_depth = max(0, depth - 1) if closing else depth
        indent = "    " * cur_depth
        formatted.append(f"{indent}{l}")
        depth += l.count("{") - l.count("}")
        depth = max(0, depth)
    return "\n".join(formatted)


def check_solidity_compilation(code: str) -> dict[str, Any]:
    """Biên dịch kiểm tra cú pháp và tính toàn vẹn của code Solidity bằng solc đúng pragma (B6)."""
    from E2E.shared.solc import compile_solidity_code
    return compile_solidity_code(code)


def detect_via_server(
    code: str,
    threshold: float = DEFAULT_DETECTOR_THRESHOLD,
    server_url: str | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    """Gửi code tới endpoint /api/v1/detect trên GPU server (dùng code gốc để đồng bộ train-serve)."""
    actual_url = get_server_url(server_url)
    url = f"{actual_url}/api/v1/detect"
    payload = {"code": code, "threshold": threshold}
    return _post_json(url, payload, timeout=timeout)


def generate_via_server(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.2,
    max_tokens: int = 1500,
    server_url: str | None = None,
    timeout: int = 120,
) -> str:
    """Gửi prompt tới endpoint /api/v1/generate trên GPU server."""
    actual_url = get_server_url(server_url)
    url = f"{actual_url}/api/v1/generate"
    payload = {
        "prompt": prompt,
        "system_prompt": system_prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    res = _post_json(url, payload, timeout=timeout)
    return str(res.get("response", "")).strip()


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




# ──────────────────────────────────────────────────────────────────────────────
# Unit helpers
# ──────────────────────────────────────────────────────────────────────────────
def _as_code_unit(unit: CodeUnit | dict[str, Any]) -> CodeUnit:
    if isinstance(unit, CodeUnit):
        return unit
    code = str(unit.get("code") or "")
    return CodeUnit(
        unit_id=unit.get("unit_id", "unknown"),
        kind=unit.get("kind", "function"),
        name=unit.get("name", "func"),
        contract_name=unit.get("contract_name", "Contract"),
        start_offset=unit.get("start_offset", 0),
        end_offset=unit.get("end_offset", len(code)),
        start_line=unit.get("start_line", 1),
        end_line=unit.get("end_line", 1),
        code=code,
        has_body=unit.get("has_body", True),
        is_view_or_pure=unit.get("is_view_or_pure", False),
        visibility=unit.get("visibility", "public"),
    )


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


def score_units_via_server(
    units: list[CodeUnit],
    threshold: float,
    server_url: str | None = None,
) -> list[dict[str, Any]]:
    """Score every unit with a body. A failed request is recorded as an error, never as a probability."""
    results: list[dict[str, Any]] = []
    for unit in units:
        if not unit.has_body:
            continue
        base = {
            "unit_id": unit.unit_id,
            "threshold": threshold,
            "start_line": unit.start_line,
            "end_line": unit.end_line,
        }
        try:
            response = detect_via_server(unit.code, threshold=threshold, server_url=server_url)
            probability = float(response["vulnerable_probability"])
        except Exception as error:
            results.append({
                **base,
                "status": "error",
                "label": "Inconclusive",
                "vulnerability_probability": None,
                "error": f"{type(error).__name__}: {error}",
            })
            continue
        results.append({
            **base,
            "status": "ok",
            "label": "Vulnerable" if probability >= threshold else "Safe",
            "vulnerability_probability": round(probability, 4),
            "line_scores": response.get("line_scores"),
        })
    return results


def summarize_detector(unit_results: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """Contract-level detector summary that stays honest about failed units."""
    scored = [item for item in unit_results if item.get("vulnerability_probability") is not None]
    errored = [item for item in unit_results if item.get("status") == "error"]
    if errored and not scored:
        status = "error"
    elif errored:
        status = "partial"
    else:
        status = "ok"
    max_probability = max((float(item["vulnerability_probability"]) for item in scored), default=None)
    if any(item.get("label") == "Vulnerable" for item in scored):
        verdict = "Vulnerable"
    elif status == "ok":
        verdict = "Safe"
    else:
        verdict = "Inconclusive"
    summary = {
        "status": status,
        "verdict": verdict,
        "vulnerability_probability": round(max_probability, 4) if max_probability is not None else None,
        "score": round(max_probability, 4) if max_probability is not None else None,
        "threshold": threshold,
        "unit_results": unit_results,
    }
    if errored:
        summary["error"] = f"{len(errored)}/{len(unit_results)} unit(s) could not be scored by the detector"
    return summary


def _feedback_from_request(repair_feedback: dict[str, Any] | None) -> str:
    """Turn an externally supplied repair_feedback object into prompt text."""
    if not repair_feedback:
        return ""
    parts: list[str] = []
    if repair_feedback.get("compiler_error"):
        parts.append(f"Previous compilation error:\n{repair_feedback['compiler_error']}")
    for channel_name, channel in (repair_feedback.get("components") or {}).items():
        if channel.get("passed", True):
            continue
        channel_findings = channel.get("findings") or channel.get("residual_target_findings") or []
        if channel_findings:
            details = "; ".join(
                str(item.get("message") or item.get("name") or item.get("title") or "")
                for item in channel_findings
                if item
            )
            parts.append(f"Verification channel [{channel_name}] flagged issues: {details}")
        elif channel.get("reason") or channel.get("error"):
            parts.append(f"Verification channel [{channel_name}] reported: {channel.get('reason') or channel.get('error')}")
    return "\n".join(parts)


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


def _version_rules(source_code: str) -> str:
    version = _min_pragma_version(source_code)
    if version is None or version >= (0, 6, 0):
        return ""
    spec = re.search(r"pragma\s+solidity\s+([^;]+);", source_code).group(1).strip()
    rules = [f"   - TARGET SOLIDITY COMPILER: pragma solidity {spec} (legacy {version[0]}.{version[1]}.{version[2]})."]
    if version < (0, 4, 22):
        rules.append("     * DO NOT pass a message string to require(); only `require(condition);` is valid.")
        rules.append("     * DO NOT use the `constructor` keyword; constructors are functions named after the contract.")
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
        "3. Return ONLY a single valid JSON object with format:\n"
        "{\n"
        f'  "replacement": "full replacement code beginning with function {unit.name} and ending at its closing brace",\n'
        '  "changed_lines": [1, 2],\n'
        '  "why": "brief rationale"\n'
        "}\n"
        "Do NOT include explanations or text outside the JSON."
    )


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
    single_pass: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Patch one unit with the generative model, falling back to the Advisor's fixed code.

    single_pass: the fixer's reply is the result. The behaviour guard only annotates it
    (`guard_warning`), an unchanged reply is reported as `unchanged`, and no fallback runs.
    """
    label = f"attempt {attempt}"
    stage = "prepare"
    llm_error = ""
    guard_warning = ""
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
        if single_pass and re.sub(r"\s+", "", candidate) == re.sub(r"\s+", "", unit.code):
            return code, {
                "finding_id": f"LLM:Qwen14B:{unit.unit_id}",
                "unit_id": unit.unit_id,
                "status": "unchanged",
                "stage": "unchanged",
                "engine": "llm_fixer",
                "fixed_code": candidate,
                "attempt": attempt,
                "rationale": rationale,
                "reason": "The fixer returned the function unchanged.",
            }
        stage = "guard"
        try:
            validate_observable_behavior(unit.code, candidate)
        except ValueError as error:
            if not single_pass:
                raise
            guard_warning = str(error)
        stage = "apply"
        updated = _splice_unit(code, unit, candidate, f"Qwen-14B patch for {unit.unit_id} ({label})")
        return updated, {
            "finding_id": f"LLM:Qwen14B:{unit.unit_id}",
            "unit_id": unit.unit_id,
            "status": "applied",
            "stage": "applied",
            "engine": "llm_fixer",
            "fixed_code": candidate,
            "patched_code": candidate,
            "attempt": attempt,
            "guard_warning": guard_warning,
            "rationale": rationale,
            "reason": f"Applied neural unit patch for {unit.unit_id} ({label}).",
        }
    except Exception as error:
        llm_error = f"{type(error).__name__}: {error}"

    advisor_error = ""
    for suggestion in ([] if single_pass else unit_suggestions):
        suggested = str(suggestion.get("fixed_code") or "").strip()
        if not suggested:
            continue
        candidate = extract_declaration_body(suggested, target_name=unit.name)
        if not candidate or unit.name not in candidate:
            continue
        if re.sub(r"\s+", "", candidate) == re.sub(r"\s+", "", unit.code):
            continue
        try:
            validate_observable_behavior(unit.code, candidate)
            updated = _splice_unit(code, unit, candidate, f"Advisor patch for {unit.unit_id}")
        except Exception as error:
            advisor_error = f"{type(error).__name__}: {error}"
            continue
        return updated, {
            "finding_id": suggestion.get("finding_id") or f"Advisor:{unit.unit_id}",
            "unit_id": unit.unit_id,
            "status": "applied",
            "stage": "applied",
            "engine": "advisor_fallback",
            "fixed_code": candidate,
            "patched_code": candidate,
            "attempt": attempt,
            "reason": f"LLM fixer failed at stage '{stage}'; applied the Advisor's patch for {unit.unit_id}.",
        }

    reason = f"Patch rejected at stage '{stage}': {llm_error}"
    if advisor_error:
        reason += f"; advisor fallback rejected: {advisor_error}"
    return code, {
        "finding_id": f"LLM:Qwen14B:{unit.unit_id}",
        "unit_id": unit.unit_id,
        "status": "rejected",
        "stage": stage,
        "attempt": attempt,
        "error": llm_error,
        "rationale": rationale,
        "reason": reason,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Verification of one candidate patch
# ──────────────────────────────────────────────────────────────────────────────
_ERROR_LOCATION_RE = re.compile(r"(-->\s*\S+:\d+:\d+:?|/\S+\.sol:\d+:\d+:?)")


def _compiler_error_signatures(text: str) -> set[str]:
    signatures = set()
    for line in (text or "").splitlines():
        if "Error" not in line:
            continue
        signatures.add(_ERROR_LOCATION_RE.sub("", line).strip())
    return signatures


def evaluate_compilation(original_code: str, fixed_code: str) -> dict[str, Any]:
    """Compile against the original as a baseline.

    - skipped:        no compiler available (not a failure of the patch)
    - not_applicable: the original does not compile standalone (e.g. unresolved imports)
                      and the patch introduced no new compiler error
    """
    result = check_compilation_with_baseline(original_code, fixed_code)
    fixed = result.get("fixed") or {}
    baseline = result.get("baseline") or {}
    if fixed.get("passed") is None:
        state = "skipped"
    elif fixed.get("passed"):
        state = "passed"
    elif result.get("baseline_failed"):
        new_errors = _compiler_error_signatures(fixed.get("error", "")) - _compiler_error_signatures(
            baseline.get("error", "")
        )
        state = "failed" if new_errors else "not_applicable"
    else:
        state = "failed"
    return {
        "state": state,
        "failed": state == "failed",
        "compiler": result.get("compiler"),
        "selection_reason": result.get("selection_reason"),
        "baseline_failed": result.get("baseline_failed", False),
        "error": fixed.get("error", "") if state == "failed" else "",
        "passed": {"passed": True, "failed": False}.get(state),
        "status": state,
    }


_ERROR_LINE_RE = re.compile(r"\S+?\.sol:(\d+):\d+")  # `--> a.sol:12:5` (>=0.5) and `a.sol:12:5: Error` (0.4)


def units_with_compiler_errors(candidate_code: str, error_text: str) -> set[str]:
    """Map solc error locations (`--> file:LINE:COL`) to the units that contain them."""
    lines = {int(value) for value in _ERROR_LINE_RE.findall(error_text or "")}
    return {
        unit.unit_id
        for unit in parse_code_units(candidate_code)
        if any(unit.start_line <= line <= unit.end_line for line in lines)
    }


def _finding_pairs(findings: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {
        (str(item.get("swc")), str(item.get("unit_id")))
        for item in findings
        if item.get("swc") and item.get("unit_id")
    }


def verify_candidate(
    source_code: str,
    candidate_code: str,
    *,
    patched_unit_ids: set[str],
    target_unit_ids: set[str],
    original_unit_results: list[dict[str, Any]],
    target_pairs: set[tuple[str, str]],
    known_pairs: set[tuple[str, str]],
    threshold: float,
    server_url: str | None,
) -> dict[str, Any]:
    """Three-channel verification of a candidate: compiler, neural detector, static + RAG re-run."""
    compile_info = evaluate_compilation(source_code, candidate_code)
    fixed_units = parse_code_units(candidate_code)
    fixed_by_id = {unit.unit_id: unit for unit in fixed_units}

    # Neural detector: re-score patched units only; other units keep their original score.
    rescored: list[dict[str, Any]] = []
    for original in original_unit_results:
        unit_id = original["unit_id"]
        if unit_id not in patched_unit_ids:
            rescored.append(dict(original))
            continue
        fixed_unit = fixed_by_id.get(unit_id)
        if fixed_unit is None:
            rescored.append({**original, "status": "error", "label": "Inconclusive",
                             "vulnerability_probability": None,
                             "error": "patched unit not found in candidate source"})
            continue
        rescored.extend(score_units_via_server([fixed_unit], threshold, server_url))
    detector_failing = sorted(
        item["unit_id"] for item in rescored
        if item["unit_id"] in target_unit_ids and item.get("label") == "Vulnerable"
    )
    unverified = sorted(
        item["unit_id"] for item in rescored
        if item["unit_id"] in target_unit_ids and item.get("vulnerability_probability") is None
    )

    # Static + RAG re-run on the candidate source.
    channel_errors: dict[str, str] = {}
    fixed_local: list[dict[str, Any]] = []
    for name, runner in (("static", run_static), ("rag", run_rag)):
        try:
            fixed_local.extend(runner(candidate_code, fixed_units).get("findings", []))
        except Exception as error:
            channel_errors[name] = f"{type(error).__name__}: {error}"
    residual = [item for item in fixed_local if (str(item.get("swc")), str(item.get("unit_id"))) in target_pairs]
    introduced = [
        item for item in fixed_local
        if str(item.get("unit_id")) in patched_unit_ids
        and (str(item.get("swc")), str(item.get("unit_id"))) not in known_pairs
        and RISK_ORDER.get(str(item.get("severity")), 0) >= RISK_ORDER["High"]
    ]

    compile_culprits = (
        units_with_compiler_errors(candidate_code, compile_info.get("error", ""))
        if compile_info["failed"]
        else set()
    )
    failing_units = set(detector_failing) | set(unverified) | (compile_culprits & patched_unit_ids)
    failing_units.update(str(item.get("unit_id")) for item in residual + introduced if item.get("unit_id"))
    passed = (
        not compile_info["failed"]
        and not failing_units
        and not channel_errors
    )
    scored_targets = [
        float(item["vulnerability_probability"]) for item in rescored
        if item["unit_id"] in target_unit_ids and item.get("vulnerability_probability") is not None
    ]
    return {
        "passed": passed,
        "compilation": compile_info,
        "unit_results": rescored,
        "detector_failing_units": detector_failing,
        "unverified_units": unverified,
        "residual_findings": residual,
        "introduced_findings": introduced,
        "channel_errors": channel_errors,
        "failing_units": sorted(failing_units),
        "compile_culprit_units": sorted(compile_culprits),
        "max_vulnerability_probability": round(max(scored_targets), 4) if scored_targets else None,
    }


def _verification_feedback(attempt: int, max_attempts: int, verification: dict[str, Any]) -> str:
    parts = [f"\n\n[CRITICAL RESIDUAL ERROR FEEDBACK - ATTEMPT {attempt}/{max_attempts}]"]
    compile_info = verification["compilation"]
    if compile_info["failed"]:
        error_lines = [
            line for line in compile_info.get("error", "").splitlines()
            if "Error" in line
        ]
        clean = "\n".join(error_lines[:8]) or compile_info.get("error", "")[:400]
        parts.append(
            f"Solidity compiler ({compile_info.get('compiler')}) failed with error:\n{clean}\n"
            "Your patch broke compilation. You MUST fix this compiler error."
        )
    if verification["detector_failing_units"]:
        parts.append(
            "The neural detector re-evaluated your patch and STILL FLAGGED it as Vulnerable: "
            f"{', '.join(verification['detector_failing_units'])}."
        )
    for label, key in (("still present", "residual_findings"), ("newly introduced", "introduced_findings")):
        for item in verification[key]:
            parts.append(
                f"Static/RAG finding {label}: [{item.get('swc')}] {item.get('title', '')} "
                f"in {item.get('unit_id')} (line {item.get('line')})."
            )
    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────
def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def run_server_gpu_pipeline(
    request: dict[str, Any],
    *,
    server_url: str | None = None,
) -> dict[str, Any]:
    """Thực thi pipeline phase 1C-5 thông qua dịch vụ AI trên GPU Server.

    Vòng sửa lỗi duy nhất của backend server_gpu (tối đa 3 attempt). Mỗi attempt được
    kiểm tra bởi ba kênh: compiler (so với baseline), neural detector trên unit đã vá,
    và static + RAG chạy lại trên code đã vá.
    """
    started_at = time.perf_counter()
    server_url = get_server_url(server_url)
    run_id = request.get("run_id", "server-gpu-run")
    source_code = request.get("source", {}).get("code", "")
    unit_objs = [_as_code_unit(unit) for unit in request.get("code_units", [])]
    units_by_id = {unit.unit_id: unit for unit in unit_objs}
    execution = request.get("execution", {})
    detector_threshold = float(execution.get("detector_threshold") or DEFAULT_DETECTOR_THRESHOLD)
    # single_pass: one fixer call per target unit, no Phase 5; the model's patch is the result.
    single_pass = bool((execution.get("repair") or {}).get("single_pass"))
    max_attempts = 1 if single_pass else min(
        max(1, int((execution.get("repair") or {}).get("max_attempts") or MAX_VERIFICATION_ATTEMPTS)),
        MAX_VERIFICATION_ATTEMPTS,
    )
    local_phase1 = request.get("local_phase1", {})
    static_p1 = local_phase1.get("static", {}) or request.get("static", {})
    rag_p1 = local_phase1.get("rag", {}) or request.get("rag", {})
    rag_context = rag_p1.get("retrieved_context", [])

    health = check_server_health(server_url)
    if health.get("status") != "ready":
        raise ConnectionError(
            f"GPU Server tại {server_url} chưa sẵn sàng (trạng thái: {health.get('status')}). "
            "Vui lòng đảm bảo bạn đã khởi chạy `bash run_api_server.sh` trên server."
        )

    # Phase 1C: neural detector on every unit with a body.
    _log(f"[PHASE 1C] Running Neural Detector v13 on GPU server ({len(unit_objs)} code unit(s))...")
    detector_started = time.perf_counter()
    unit_results = score_units_via_server(unit_objs, detector_threshold, server_url)
    detector_original = summarize_detector(unit_results, detector_threshold)
    detector_elapsed_ms = round((time.perf_counter() - detector_started) * 1000, 2)
    _log(
        f"[PHASE 1C] Completed: {len(unit_results)} unit(s) evaluated, status={detector_original['status']}, "
        f"max_prob={detector_original['vulnerability_probability']}."
    )

    # Phase 1 fused: per-finding fusion decides what reaches the agents.
    _log("[PHASE 1 FUSED] Synchronizing multi-modal detection signals...")
    findings, candidate_findings = fuse_findings(unit_results, static_p1, rag_p1, tau_agent=0.45)
    try:
        from agents.advisor import RepairKnowledgeBase

        findings = RepairKnowledgeBase().enrich_findings(
            findings, code=source_code, code_units=request.get("code_units", [])
        )
    except Exception:
        pass
    # Units flagged only by the neural detector (no SWC from static/RAG) are optional repair
    # targets: `detector_only_min_prob` (execution.repair) raises the bar for them, and a value
    # above 1 keeps them out of repair entirely. They are still reported.
    detector_only_min = (execution.get("repair") or {}).get("detector_only_min_prob")
    detector_only_deferred: list[dict[str, Any]] = []
    if detector_only_min is not None:
        detector_only_deferred = [
            item for item in findings
            if item.get("swc") == "UNMAPPED" and float(item.get("p_detector") or 0.0) < float(detector_only_min)
        ]
        deferred_ids = {id(item) for item in detector_only_deferred}
        findings = [item for item in findings if id(item) not in deferred_ids]
    _log(
        f"[PHASE 1 FUSED] Completed: {len(findings)} confirmed finding(s), {len(candidate_findings)} candidate(s)"
        + (f", {len(detector_only_deferred)} detector-only finding(s) not repaired." if detector_only_deferred else ".")
    )

    # Phase 2: Advisor (triage + remediation). False positives are removed from the repair set.
    _log("[PHASE 2] Running Security Advisor agent (root cause & remediation analysis)...")
    advisor = _generate_advisor_section(findings, source_code, backend="server_gpu", use_llm=True)
    dismissed_ids = set(advisor.get("false_positives", []))
    dismissed = [item for item in findings if finding_key(item) in dismissed_ids]
    findings = [item for item in findings if finding_key(item) not in dismissed_ids]
    _log(
        f"[PHASE 2] Completed: {len(advisor.get('suggestions', []))} suggestion(s), "
        f"{len(dismissed)} finding(s) dismissed as false positive."
    )

    # Phase 3: Assessor (CVSS only feeds the report, so single-pass batch runs skip it).
    if single_pass:
        _log("[PHASE 3] Skipped (single-pass mode).")
        assessor = {"status": "skipped", "assessments": []}
    else:
        _log("[PHASE 3] Running Risk Assessor agent (CVSS v3.1 scoring & priority ranking)...")
        assessor = _generate_assessor_section(
            findings,
            source_code,
            backend="server_gpu",
            use_llm=True,
            advisor_suggestions=advisor.get("suggestions", []),
        )
        _log(f"[PHASE 3] Completed: {len(assessor.get('assessments', []))} item(s) assessed.")

    # Phase 4-5: repair loop.
    finding_units = {item.get("unit_id") for item in findings if item.get("unit_id")}
    target_units = [
        unit for unit in unit_objs
        if unit.unit_id in finding_units and unit.has_body and not unit.is_view_or_pure
    ]
    # Optional budget for very large contracts: keep the units with the strongest fused
    # evidence; the rest are reported as `skipped_targets`, never silently dropped.
    max_target_units = (execution.get("repair") or {}).get("max_target_units")
    skipped_targets: list[str] = []
    if max_target_units and len(target_units) > int(max_target_units):
        strength = {
            unit.unit_id: max(
                float(item.get("p_final") or 0.0) for item in findings if item.get("unit_id") == unit.unit_id
            )
            for unit in target_units
        }
        ranked = sorted(target_units, key=lambda unit: strength[unit.unit_id], reverse=True)
        target_units = ranked[: int(max_target_units)]
        skipped_targets = [unit.unit_id for unit in ranked[int(max_target_units):]]
        _log(f"[PHASE 4] Target budget: repairing {len(target_units)} unit(s), skipping {len(skipped_targets)}.")
    target_unit_ids = {unit.unit_id for unit in target_units}
    target_pairs = {pair for pair in _finding_pairs(findings) if pair[1] in target_unit_ids}
    known_pairs = _finding_pairs(
        [
            *static_p1.get("findings", []),
            *static_p1.get("candidates", []),
            *rag_p1.get("findings", []),
        ]
    )
    suggestions = advisor.get("suggestions", [])
    patch_candidates = list(findings) + suggestions
    first_swc = next((item.get("swc") for item in findings if item.get("swc") not in {None, "UNMAPPED"}), None)
    rag_section = _build_rag_knowledge_section(rag_context, target_swc=first_swc)
    feedback_note = _feedback_from_request(request.get("repair_feedback"))

    fixer_started = time.perf_counter()
    attempts: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    base_code = source_code
    carried_outcomes: list[dict[str, Any]] = []
    active_targets = list(target_units)

    for attempt in range(1, max_attempts + 1 if target_units else 1):
        label = f"Attempt {attempt}/{max_attempts}"
        _log(f"[PHASE 4] Running Fixer agent via Qwen2.5-Coder-14B ({label}, {len(active_targets)} unit(s))...")
        current_code = base_code
        outcomes = list(carried_outcomes)
        for unit in active_targets:
            current_code, outcome = patch_unit(
                current_code,
                unit,
                unit_findings=[item for item in findings if item.get("unit_id") == unit.unit_id],
                unit_suggestions=[item for item in suggestions if item.get("unit_id") == unit.unit_id],
                source_code=source_code,
                feedback_note=feedback_note,
                rag_section=rag_section,
                attempt=attempt,
                server_url=server_url,
                single_pass=single_pass,
            )
            outcomes.append(outcome)

        if (
            not single_pass
            and not any(item.get("status") == "applied" for item in outcomes)
            and base_code == source_code
        ):
            heuristic_code, heuristic_outcomes = _apply_heuristic_security_patch(
                source_code, patch_candidates, code_units=unit_objs
            )
            if heuristic_outcomes:
                current_code = heuristic_code
                outcomes.extend(heuristic_outcomes)

        patched_ids = {
            str(item.get("unit_id")) for item in outcomes
            if item.get("status") == "applied" and item.get("unit_id")
        }
        if single_pass:
            _log("[PHASE 5] Skipped (single-pass mode): the fixer output is taken as the final patch.")
            verification = {
                "passed": False,
                "compilation": {"state": "skipped", "failed": False},
                "unit_results": [],
                "detector_failing_units": [],
                "unverified_units": [],
                "residual_findings": [],
                "introduced_findings": [],
                "channel_errors": {},
                "failing_units": [],
                "compile_culprit_units": [],
                "max_vulnerability_probability": None,
            }
        else:
            _log(f"[PHASE 5] Verifying candidate ({label}): compiler + neural detector + static/RAG re-run...")
            verification = verify_candidate(
                source_code,
                current_code,
                patched_unit_ids=patched_ids,
                target_unit_ids=target_unit_ids,
                original_unit_results=unit_results,
                target_pairs=target_pairs,
                known_pairs=known_pairs,
                threshold=detector_threshold,
                server_url=server_url,
            )
        record = {
            "attempt": attempt,
            "max_attempts": max_attempts,
            "threshold": detector_threshold,
            "verdict": "Unverified" if single_pass else ("Safe" if verification["passed"] else "Vulnerable"),
            "code": current_code,
            "patch_outcomes": outcomes,
            **verification,
        }
        attempts.append(record)
        score = (
            0 if verification["compilation"]["failed"] else 1,
            -len(verification["failing_units"]),
            len(patched_ids),
            -abs(len(current_code) - len(source_code)),
        )
        if best is None or score > best["_score"]:
            best = {**record, "_score": score}
        _log(
            f"[PHASE 5] {label}: {'PASSED' if verification['passed'] else 'FAILED'} "
            f"(compile={verification['compilation']['state']}, failing={verification['failing_units']})."
        )
        if verification["passed"]:
            break
        if attempt == max_attempts:
            break

        record["feedback"] = _verification_feedback(attempt, max_attempts, verification)
        feedback_note = record["feedback"]
        failing = set(verification["failing_units"])
        failing.update(
            unit.unit_id for unit in active_targets
            if unit.unit_id not in patched_ids
        )
        compile_localized = not verification["compilation"]["failed"] or bool(
            set(verification["compile_culprit_units"]) & patched_ids
        )
        base_code, carried_outcomes, active_targets = source_code, [], list(target_units)
        if compile_localized and failing:
            # Next attempt = original source + the patches that already verified; the failing
            # units are re-patched from their original code (never on top of a failed patch).
            keep = [
                item for item in outcomes
                if item.get("status") == "applied" and item.get("unit_id") not in failing
            ]
            rebuilt = source_code
            try:
                for item in keep:
                    unit = units_by_id[item["unit_id"]]
                    rebuilt = _splice_unit(rebuilt, unit, item["patched_code"], str(item.get("reason") or ""))
            except (KeyError, ValueError):
                pass  # cannot rebuild (e.g. rule-based patch without unit code): restart fully
            else:
                base_code, carried_outcomes = rebuilt, keep
                active_targets = [unit for unit in target_units if unit.unit_id in failing]

    fixer_elapsed_ms = round((time.perf_counter() - fixer_started) * 1000, 2)

    if best is None:
        fixed_code, patch_outcomes, final_verification = source_code, [], None
        overall_verdict = "NOT_APPLICABLE"
    else:
        fixed_code = best["code"]
        patch_outcomes = best["patch_outcomes"]
        final_verification = best
        overall_verdict = "UNVERIFIED" if single_pass else ("PASS" if best["passed"] else "WARN")
    _log(f"[PHASE 5] Completed: verdict {overall_verdict} after {len(attempts)} attempt(s).")

    if final_verification is not None and not single_pass:
        fixed_detector = summarize_detector(
            [item for item in final_verification["unit_results"] if item["unit_id"] in target_unit_ids],
            detector_threshold,
        )
    else:
        fixed_detector = {
            "status": "not_run",
            "verdict": "Not run",
            "score": None,
            "vulnerability_probability": None,
            "threshold": detector_threshold,
            "unit_results": [],
            "reason": "Verification skipped (single-pass mode)." if single_pass and final_verification
            else "No function-level repair target.",
        }

    patched_ids = {
        str(item.get("unit_id")) for item in patch_outcomes
        if item.get("status") == "applied" and item.get("unit_id")
    }
    failing_final = set(final_verification["failing_units"]) if final_verification else set()
    verifier_items = []
    for item in findings:
        unit_id = item.get("unit_id")
        if unit_id in failing_final:
            status = "FLAGGED"
        elif unit_id in patched_ids:
            status = "SKIPPED" if single_pass else "PASS"
        else:
            status = "NOT_PATCHED"
        verifier_items.append({
            "finding_id": finding_key(item),
            "unit_id": unit_id,
            "status": status,
            "residual_risk": "None" if status == "PASS" else "Unverified or residual risk remains",
        })

    attempts_public = [{k: v for k, v in record.items() if k != "code"} for record in attempts]
    compilation = final_verification["compilation"] if final_verification else {}
    applied_count = sum(1 for item in patch_outcomes if item.get("status") == "applied")
    phases = {
        "detector": {
            "status": detector_original["status"],
            "original": detector_original,
            "fixed": fixed_detector,
            "candidate_findings": candidate_findings,
            "confirmed_findings": findings,
            "dismissed_findings": dismissed,
            "skipped_targets": skipped_targets,
            "detector_only_deferred": detector_only_deferred,
        },
        "advisor": advisor,
        "assessor": assessor,
        "fixer": {
            "status": "ok",
            "fixed_code": fixed_code,
            "patch_outcomes": patch_outcomes,
            "patches": [
                {"finding_id": item.get("finding_id"), "description": item.get("reason")}
                for item in patch_outcomes
            ],
        },
        "verifier": {
            "status": "ok",
            "verification_model": "solc baseline + Qwen2.5-Coder-14B v13 detector + static/RAG re-run",
            "verdict": overall_verdict,
            "overall_verdict": overall_verdict,
            "verifications": verifier_items,
            "attempts_count": len(attempts),
            "max_attempts": max_attempts,
            "attempts": attempts_public,
            "compilation": compilation,
            "fixed_unit_results": fixed_detector.get("unit_results", []),
            "vulnerability_probability": fixed_detector.get("vulnerability_probability"),
            "threshold": detector_threshold,
            "notes": f"Self-verification over {len(attempts)} attempt(s). Final verdict: {overall_verdict}.",
        },
        "reporter": {
            "status": "ok",
            "executive_summary": (
                f"Audit completed using the GPU server (Qwen2.5-Coder-14B v13) for run {run_id}. "
                f"{len(findings)} confirmed finding(s), {len(dismissed)} dismissed by the Advisor, "
                f"{applied_count} patch(es) applied."
            ),
            "recommendations": advisor["recommendations"],
        },
    }
    return {
        "schema_version": "1.0.0",
        "pipeline_version": "1.0.0",
        "status": "ok" if detector_original["status"] == "ok" else "partial",
        "run_id": run_id,
        "backend": "server_gpu_rtx4070ti",
        "source_sha256": request.get("source", {}).get("sha256", ""),
        "fixed_code": fixed_code,
        "patch_outcomes": patch_outcomes,
        "verification_verdict": overall_verdict,
        "phases": phases,
        "timings": {
            "total_ms": round((time.perf_counter() - started_at) * 1000, 2),
            "detector_ms": detector_elapsed_ms,
            "fixer_ms": fixer_elapsed_ms,
        },
    }
