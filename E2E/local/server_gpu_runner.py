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
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

from E2E.config import DEFAULT_DETECTOR_THRESHOLD
from E2E.local.local_llm_runner import (
    _apply_heuristic_security_patch,
    _derive_findings,
    _generate_advisor_section,
    _generate_assessor_section,
)
from E2E.shared.function_parser import (
    CodeUnit,
    apply_unit_patches,
    extract_declaration_body,
    validate_observable_behavior,
)

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


def _post_json(url: str, payload: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_server_health(server_url: str | None = None, timeout: int = 3) -> dict[str, Any]:
    """Kiểm tra xem API server v13 trên GPU server có sẵn sàng hay không."""
    actual_url = get_server_url(server_url)
    url = f"{actual_url}/api/v1/health"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LLM-BSCVM-Local"})
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


def run_server_gpu_pipeline(
    request: dict[str, Any],
    *,
    server_url: str | None = None,
) -> dict[str, Any]:
    """Thực thi toàn bộ pipeline 6 phase thông qua dịch vụ AI trên GPU Server."""
    started_at = time.perf_counter()
    server_url = get_server_url(server_url)
    run_id = request.get("run_id", "server-gpu-run")
    source_code = request.get("source", {}).get("code", "")
    code_units = request.get("code_units", [])
    findings = _derive_findings(request)
    detector_threshold = float(
        request.get("execution", {}).get("detector_threshold") or DEFAULT_DETECTOR_THRESHOLD
    )

    # Trích xuất RAG context từ Phase 1B (top items từ 11,869 cases)
    rag_context = (
        request.get("local_phase1", {}).get("rag", {}).get("retrieved_context", [])
        or request.get("rag", {}).get("retrieved_context", [])
    )
    if not rag_context and source_code:
        try:
            from E2E.rag.retriever import LocalRAGRetriever
            rag_context = LocalRAGRetriever().retrieve_context(source_code, top_k=5)
        except Exception:
            rag_context = []

    # 1. Kiểm tra kết nối tới Server
    health = check_server_health(server_url)
    if health.get("status") != "ready":
        raise ConnectionError(
            f"GPU Server tại {server_url} chưa sẵn sàng (trạng thái: {health.get('status')}). "
            f"Vui lòng đảm bảo bạn đã khởi chạy `bash run_api_server.sh` trên server."
        )

    # 2. Phase 1: Chạy Detector v13 trên từng function unit và full contract
    unit_results = []
    max_prob = 0.0
    det_start = time.perf_counter()
    print(f"[{time.strftime('%H:%M:%S')}] [PHASE 1C] Running Neural Detector v13 on RTX 4070 Ti SUPER GPU server ({len(code_units)} code unit(s))...", flush=True)

    for unit in code_units:
        # Bỏ qua các khai báo interface hoặc abstract không có thân hàm (Task B8, R-08)
        has_body = unit.get("has_body", True) if isinstance(unit, dict) else getattr(unit, "has_body", True)
        if not has_body:
            continue

        unit_id = unit.get("unit_id", "unknown") if isinstance(unit, dict) else getattr(unit, "unit_id", "unknown")
        unit_code = (unit.get("code") if isinstance(unit, dict) else getattr(unit, "code", "")) or source_code
        try:
            res = detect_via_server(unit_code, threshold=detector_threshold, server_url=server_url)
            prob = float(res.get("vulnerable_probability", 0.0))
            label = "Vulnerable" if prob >= detector_threshold else "Safe"
            unit_results.append({
                "unit_id": unit_id,
                "label": label,
                "vulnerability_probability": round(prob, 4),
                "threshold": detector_threshold,
                "line_scores": res.get("line_scores"),
                "start_line": unit.get("start_line") if isinstance(unit, dict) else getattr(unit, "start_line", 1),
                "end_line": unit.get("end_line") if isinstance(unit, dict) else getattr(unit, "end_line", 1),
            })
            if prob > max_prob:
                max_prob = prob
        except Exception as e:
            # Fallback trung thực khi lỗi mạng/inference: không bịa xác suất 0.85/0.05 (Task A2, R-05)
            unit_results.append({
                "unit_id": unit_id,
                "status": "error",
                "label": "Inconclusive",
                "vulnerability_probability": None,
                "threshold": detector_threshold,
                "error": str(e),
                "start_line": unit.get("start_line") if isinstance(unit, dict) else getattr(unit, "start_line", 1),
                "end_line": unit.get("end_line") if isinstance(unit, dict) else getattr(unit, "end_line", 1),
            })

    detector_elapsed_ms = round((time.perf_counter() - det_start) * 1000, 2)
    is_contract_vuln = max_prob >= detector_threshold or len(findings) > 0
    print(f"[{time.strftime('%H:%M:%S')}] [PHASE 1C] Completed: Neural detection finished ({len(unit_results)} unit(s) evaluated, max_prob: {max_prob:.4f}).", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] [PHASE 1 FUSED] Synchronizing multi-modal detection signals...", flush=True)

    # Task B1: Chạy fuse_findings() trước Phase 2 Agent để lọc false positive và khử trùng lặp
    from E2E.shared.fusion import fuse_findings
    static_p1 = request.get("local_phase1", {}).get("static", {}) or request.get("static", {})
    rag_p1 = request.get("local_phase1", {}).get("rag", {}) or request.get("rag", {})

    confirmed_findings, candidate_findings = fuse_findings(
        unit_results,
        static_p1,
        rag_p1,
        tau_agent=0.45,
    )

    # Chỉ chuyển các finding đã được xác nhận (p_final >= tau_agent) sang Phase 2 Agent (Task B1)
    findings = confirmed_findings

    # Truy xuất cơ sở tri thức lỗ hổng (11,869 cases + 37 SWC registry) theo phương pháp Decompose-Retrieve-Generate của LLM-BSCVM
    try:
        from agents.advisor import RepairKnowledgeBase
        kb = RepairKnowledgeBase()
        findings = kb.enrich_findings(findings, code=source_code, code_units=code_units)
    except Exception:
        pass

    print(f"[{time.strftime('%H:%M:%S')}] [PHASE 1 FUSED] Completed: Multi-modal signals synchronized ({len(findings)} confirmed finding(s), {len(candidate_findings)} candidate(s)).", flush=True)

    # 3. Phase 2-4: Advisor, Assessor, và Fixer
    print(f"[{time.strftime('%H:%M:%S')}] [PHASE 2] Running Security Advisor agent (root cause & remediation analysis)...", flush=True)
    advisor = _generate_advisor_section(findings, source_code, backend="server_gpu", use_llm=True)
    print(f"[{time.strftime('%H:%M:%S')}] [PHASE 2] Completed: Security recommendations generated ({len(advisor.get('suggestions', []))} suggestion(s)).", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] [PHASE 3] Running Risk Assessor agent (CVSS v3.1 scoring & priority ranking)...", flush=True)
    assessor = _generate_assessor_section(
        findings,
        source_code,
        backend="server_gpu",
        use_llm=True,
        advisor_suggestions=advisor.get("suggestions", []),
    )
    print(f"[{time.strftime('%H:%M:%S')}] [PHASE 3] Completed: Risk assessment finished ({len(assessor.get('assessments', []))} item(s) assessed).", flush=True)

    patch_candidates = list(findings) + advisor.get("suggestions", [])
    fixed_code, patch_outcomes = _apply_heuristic_security_patch(source_code, patch_candidates, code_units=code_units)

    # Nâng cấp bản vá bằng Qwen 14B trên GPU server kết hợp vòng lặp kiểm tra tự động (Self-Correction Loop) tối đa 3 lần
    MAX_VERIFICATION_ATTEMPTS = 3
    verification_attempts = []
    overall_verdict = "PASS"
    fixed_unit_results = []
    fixed_max_prob = 0.0
    req_feedback = request.get("repair_feedback") or {}
    feedback_note = ""
    if req_feedback:
        fb_parts = []
        if req_feedback.get("compiler_error"):
            fb_parts.append(f"Previous compilation error:\n{req_feedback['compiler_error']}")
        for comp_name, comp_info in req_feedback.get("components", {}).items():
            if not comp_info.get("passed", True):
                reason = comp_info.get("reason") or comp_info.get("error") or ""
                findings = comp_info.get("findings") or comp_info.get("residual_target_findings") or []
                if findings:
                    f_details = "; ".join(f.get("message") or f.get("name") or f.get("title") or "" for f in findings if f)
                    fb_parts.append(f"Verification channel [{comp_name}] flagged issues: {f_details}")
                elif reason:
                    fb_parts.append(f"Verification channel [{comp_name}] reported: {reason}")
        if fb_parts:
            feedback_note = "\n".join(fb_parts)

    # Trích xuất bối cảnh biến trạng thái, modifiers, events từ hợp đồng bao quanh
    context_lines = []
    for line in source_code.splitlines():
        st = line.strip()
        if (
            st.startswith(("contract ", "pragma ", "import ", "mapping", "address", "uint", "int", "bool", "bytes", "string", "modifier ", "event ", "struct "))
            and not st.startswith("function ")
        ):
            context_lines.append(line)
    contract_context_str = "\n".join(context_lines[:35])
    contract_context_section = f"\n\n### Enclosing Contract Context (State Variables, Modifiers, Events):\n```solidity\n{contract_context_str}\n```" if contract_context_str else ""

    unit_objs = []
    for u in code_units:
        if isinstance(u, CodeUnit):
            unit_objs.append(u)
        elif isinstance(u, dict):
            try:
                unit_objs.append(CodeUnit(**u))
            except Exception:
                unit_objs.append(CodeUnit(
                    unit_id=u.get("unit_id", "unknown"),
                    kind=u.get("kind", "function"),
                    name=u.get("name", "func"),
                    contract_name=u.get("contract_name", "Contract"),
                    start_offset=u.get("start_offset", 0),
                    end_offset=u.get("end_offset", len(u.get("code", ""))),
                    start_line=u.get("start_line", 1),
                    end_line=u.get("end_line", 10),
                    code=u.get("code", ""),
                ))
    unit_objs = []
    for u in code_units:
        if isinstance(u, CodeUnit):
            unit_objs.append(u)
        elif isinstance(u, dict):
            try:
                unit_objs.append(CodeUnit(**u))
            except Exception:
                unit_objs.append(CodeUnit(
                    unit_id=u.get("unit_id", "unknown"),
                    kind=u.get("kind", "function"),
                    name=u.get("name", "func"),
                    contract_name=u.get("contract_name", "Contract"),
                    start_offset=u.get("start_offset", 0),
                    end_offset=u.get("end_offset", len(u.get("code", ""))),
                    start_line=u.get("start_line", 1),
                    end_line=u.get("end_line", 10),
                    code=u.get("code", ""),
                    has_body=u.get("has_body", True),
                    is_view_or_pure=u.get("is_view_or_pure", False),
                ))
    mapped_target_units = [
        u for u in unit_objs
        if any(f.get("unit_id") == u.unit_id and f.get("swc") and f.get("swc") != "UNMAPPED" for f in findings)
    ]
    target_units = mapped_target_units if mapped_target_units else [
        u for u in unit_objs if any(f.get("unit_id") == u.unit_id for f in findings)
    ]
    # Task B8: Bỏ qua các hàm pure / view không ghi state khỏi danh sách mục tiêu sửa lỗi
    target_units = [u for u in target_units if not getattr(u, "is_view_or_pure", False)]

    fixer_loop_start = time.perf_counter()
    best_attempt_code = source_code
    best_attempt_outcomes = []
    best_attempt_score = (-1, -1, -999, -999999)
    active_target_units = list(target_units)

    for attempt in range(1, MAX_VERIFICATION_ATTEMPTS + 1):
        attempt_label = f"Attempt {attempt}/{MAX_VERIFICATION_ATTEMPTS}"
        print(f"[{time.strftime('%H:%M:%S')}] [PHASE 4] Running Fixer agent (synthesizing secure Solidity patch via Qwen2.5-Coder-14B - {attempt_label})...", flush=True)

        current_attempt_code = source_code
        current_patch_outcomes = []

        if is_contract_vuln and findings and active_target_units:
            for t_unit in active_target_units:
                unit_findings = [f for f in findings if f.get("unit_id") == t_unit.unit_id]
                unit_suggestions = [s for s in advisor.get("suggestions", []) if s.get("unit_id") == t_unit.unit_id]

                stage = "prepare"
                try:
                    # Task B3: build_contract_context rút gọn biến state, modifiers, events và invariants (ngân sách 1500 tokens)
                    from E2E.shared.function_parser import build_contract_context
                    contract_context_section = build_contract_context(t_unit, source_code, budget_tokens=1500)

                    finding_context = []
                    for uf in unit_findings:
                        f_title = uf.get("title", "")
                        f_swc = uf.get("swc", "")
                        finding_context.append(f"  * [{f_swc}] {f_title}")
                    for us in unit_suggestions:
                        if us.get("fix_strategy"):
                            finding_context.append(f"  * Fix Strategy: {us['fix_strategy']}")
                        elif us.get("root_cause"):
                            finding_context.append(f"  * Root cause: {us['root_cause']}")

                    findings_str = "\n".join(finding_context) if finding_context else "Security vulnerability detected."

                    sol_match = re.search(r"pragma\s+solidity\s+([^;]+);", source_code)
                    version_rules = ""
                    if sol_match:
                        ver_spec = sol_match.group(1).strip()
                        if any(v in ver_spec for v in ("0.4.", "0.5.")):
                            version_rules = (
                                f"   - TARGET SOLIDITY COMPILER: pragma solidity {ver_spec} (Solidity legacy 0.4.x / 0.5.x).\n"
                                f"     * In Solidity <0.4.22: DO NOT pass a message string to require()! Only single-argument `require(condition);` is valid.\n"
                                f"     * In Solidity <0.4.22: DO NOT use the `constructor` keyword! Use a function named after the contract.\n"
                                f"     * In Solidity <0.4.21: DO NOT use the `emit` keyword! Trigger events directly via `EventName(...)`.\n"
                                f"     * Preserve function return semantics: if the original function returns a boolean (e.g. `return false;`), do NOT alter it to `revert()`.\n"
                            )

                    unit_prompt = (
                        f"You are an expert Solidity smart contract security researcher.\n"
                        f"Target vulnerable function: `{t_unit.name}`\n"
                        f"Identified Vulnerabilities & Diagnoses:\n{findings_str}\n\n"
                        f"Original declaration:\n```solidity\n{t_unit.code}\n```\n\n"
                        f"Contract Context & Invariants:\n```solidity\n{contract_context_section}\n```\n"
                        f"{feedback_note}\n\n"
                        f"INSTRUCTIONS:\n"
                        f"1. Generate a MINIMAL DIFF security repair for `{t_unit.name}`.\n"
                        f"{version_rules}"
                        f"2. All state write statements and event emissions from the original function MUST be preserved.\n"
                        f"3. Return ONLY a single valid JSON object with format:\n"
                        f"{{\n"
                        f'  "replacement": "full replacement code beginning with function {t_unit.name} and ending at its closing brace",\n'
                        f'  "changed_lines": [1, 2],\n'
                        f'  "why": "brief rationale"\n'
                        f"}}\n"
                        f"Do NOT include explanations or text outside the JSON."
                    )

                    stage = "generate"
                    llm_reply = generate_via_server(
                        unit_prompt,
                        system_prompt="You are a Solidity security auditor. Output strictly valid JSON with the requested schema.",
                        temperature=0.0,
                        server_url=server_url,
                    )

                    stage = "extract"
                    candidate_body = extract_declaration_body(llm_reply, target_name=t_unit.name)
                    if not candidate_body or t_unit.name not in candidate_body:
                        raise ValueError(f"Extracted patch does not define expected function {t_unit.name}")

                    stage = "guard"
                    validate_observable_behavior(t_unit.code, candidate_body)

                    stage = "apply"
                    from E2E.shared.function_parser import parse_code_units
                    current_units_in_fixed = parse_code_units(current_attempt_code)
                    active_unit_in_fixed = next(
                        (u for u in current_units_in_fixed if u.unit_id == t_unit.unit_id or u.name == t_unit.name),
                        t_unit,
                    )
                    updated_code, unit_outcomes = apply_unit_patches(
                        current_attempt_code,
                        [active_unit_in_fixed],
                        [{"unit_id": active_unit_in_fixed.unit_id, "replacement": candidate_body, "summary": f"Qwen-14B patch for {t_unit.unit_id} ({attempt_label})"}],
                    )
                    if not any(uo.get("status") == "applied" for uo in unit_outcomes):
                        raise ValueError("Failed to apply patch replacement into contract code")

                    current_attempt_code = updated_code
                    current_patch_outcomes = [po for po in current_patch_outcomes if po.get("unit_id") != t_unit.unit_id]
                    current_patch_outcomes.append({
                        "finding_id": f"LLM:Qwen14B:{t_unit.unit_id}",
                        "unit_id": t_unit.unit_id,
                        "status": "applied",
                        "stage": "applied",
                        "fixed_code": candidate_body,
                        "patched_code": candidate_body,
                        "attempt": attempt,
                        "reason": f"Applied neural unit patch for {t_unit.unit_id} ({attempt_label}).",
                    })
                except Exception as e:
                    # Ghi log lý do lỗi chi tiết theo từng stage (Task A5, R-03)
                    current_patch_outcomes.append({
                        "finding_id": f"LLM:Qwen14B:{t_unit.unit_id}",
                        "unit_id": t_unit.unit_id,
                        "status": "rejected",
                        "stage": stage,
                        "attempt": attempt,
                        "error": str(e),
                        "reason": f"Patch rejected at stage '{stage}': {str(e)}",
                    })

                # Fallback to Advisor's suggestion if LLM patch was not applied
                if not any(po.get("unit_id") == t_unit.unit_id and po.get("status") == "applied" for po in current_patch_outcomes):
                    for us in unit_suggestions:
                        sug_fixed = str(us.get("fixed_code") or "").strip()
                        if sug_fixed and sug_fixed != t_unit.code:
                            candidate_body = extract_declaration_body(sug_fixed, target_name=t_unit.name)
                            if candidate_body and t_unit.name in candidate_body and re.sub(r"\s+", "", candidate_body) != re.sub(r"\s+", "", t_unit.code):
                                try:
                                    validate_observable_behavior(t_unit.code, candidate_body)
                                    from E2E.shared.function_parser import parse_code_units
                                    current_units_in_fixed = parse_code_units(current_attempt_code)
                                    active_unit_in_fixed = next(
                                        (u for u in current_units_in_fixed if u.unit_id == t_unit.unit_id or u.name == t_unit.name),
                                        t_unit,
                                    )
                                    updated_code, unit_outcomes = apply_unit_patches(
                                        current_attempt_code,
                                        [active_unit_in_fixed],
                                        [{"unit_id": active_unit_in_fixed.unit_id, "replacement": candidate_body, "summary": f"Advisor patch for {t_unit.unit_id}"}],
                                    )
                                    if any(uo.get("status") == "applied" for uo in unit_outcomes):
                                        current_attempt_code = updated_code
                                        current_patch_outcomes.append({
                                            "finding_id": us.get("finding_id") or f"Advisor:{t_unit.unit_id}",
                                            "unit_id": t_unit.unit_id,
                                            "status": "applied",
                                            "stage": "applied",
                                            "fixed_code": candidate_body,
                                            "patched_code": candidate_body,
                                            "attempt": attempt,
                                            "reason": f"Applied Advisor security patch for {t_unit.unit_id}.",
                                        })
                                        break
                                except Exception as e:
                                    current_patch_outcomes.append({
                                        "finding_id": us.get("finding_id") or f"Advisor:{t_unit.unit_id}",
                                        "unit_id": t_unit.unit_id,
                                        "status": "rejected",
                                        "stage": "advisor_fallback",
                                        "attempt": attempt,
                                        "error": str(e),
                                        "reason": f"Advisor patch rejected: {str(e)}",
                                    })

        # Fallback to empirical heuristic patch if no unit patch applied in this attempt
        if not any(po.get("status") == "applied" for po in current_patch_outcomes):
            heuristic_code, heuristic_outcomes = _apply_heuristic_security_patch(source_code, patch_candidates, code_units=code_units)
            if heuristic_outcomes:
                current_attempt_code = heuristic_code
                current_patch_outcomes = heuristic_outcomes

        print(f"[{time.strftime('%H:%M:%S')}] [PHASE 4] Completed: Security patch synthesized ({len(current_patch_outcomes)} outcome(s) generated).", flush=True)

        # Giai đoạn 5: Xác minh bản vá bằng Solc Compiler và Neural Detector v13 của Phase 1
        print(f"[{time.strftime('%H:%M:%S')}] [PHASE 5] Running Multi-Tier Self-Verification (Solc + Neural Detector v13) ({attempt_label})...", flush=True)
        attempt_vulnerable = False
        attempt_max_prob = 0.0

        # Step 5A: Solc compilation check on current_attempt_code
        compile_res = check_solidity_compilation(current_attempt_code)
        compile_failed = not compile_res.get("passed", True)
        if compile_failed:
            attempt_vulnerable = True
            print(f"[{time.strftime('%H:%M:%S')}] [PHASE 5] Solc Compilation Check: FAILED with compiler {compile_res.get('compiler')}!", flush=True)
            print(f"[{time.strftime('%H:%M:%S')}] [PHASE 5] Error detail:\n{compile_res.get('error', '').strip()[:400]}", flush=True)
        else:
            print(f"[{time.strftime('%H:%M:%S')}] [PHASE 5] Solc Compilation Check: PASSED with compiler {compile_res.get('compiler')}.", flush=True)

        # Trích xuất các hàm đã được cập nhật trong current_attempt_code
        try:
            from E2E.shared.function_parser import parse_code_units
            patched_units = parse_code_units(current_attempt_code)
            patched_units_map = {u.name: u.code for u in patched_units}
        except Exception:
            patched_units_map = {}

        patched_unit_ids = {po.get("unit_id") for po in current_patch_outcomes if po.get("status") == "applied"}
        confirmed_target_ids = {
            tu.unit_id for tu in target_units
            if any(f.get("unit_id") == tu.unit_id and f.get("swc") and f.get("swc") != "UNMAPPED" for f in findings)
        }
        verification_target_ids = patched_unit_ids or confirmed_target_ids or {tu.unit_id for tu in target_units}

        attempt_unit_results = []
        for u in unit_results:
            u_id = u["unit_id"]
            u_name = u_id.split(".")[-1]
            is_patched = u_id in patched_unit_ids
            is_target = u_id in verification_target_ids

            # Task B8: Chỉ re-scan unit đã bị vá! Unit không bị vá giữ nguyên điểm ban đầu
            if is_patched:
                unit_code_to_test = patched_units_map.get(u_name) or u.get("code") or current_attempt_code
                try:
                    res = detect_via_server(unit_code_to_test, threshold=detector_threshold, server_url=server_url)
                    prob = float(res.get("vulnerable_probability", 0.0))
                    l_scores = res.get("line_scores")
                except Exception:
                    # Task A2: Không bịa xác suất 0.02
                    prob = None
                    l_scores = None
            else:
                prob = u.get("vulnerability_probability")
                l_scores = u.get("line_scores")

            is_vuln = (prob is not None) and (prob >= detector_threshold)
            if is_target and is_vuln:
                attempt_vulnerable = True
            if is_target and prob is not None and prob > attempt_max_prob:
                attempt_max_prob = prob

            attempt_unit_results.append({
                "unit_id": u_id,
                "label": "Vulnerable" if is_vuln else ("Safe" if prob is not None else "Inconclusive"),
                "vulnerability_probability": round(prob, 4) if prob is not None else None,
                "threshold": detector_threshold,
                "line_scores": l_scores,
                "attempt": attempt,
            })

        fixed_max_prob = attempt_max_prob
        attempt_record = {
            "attempt": attempt,
            "max_attempts": MAX_VERIFICATION_ATTEMPTS,
            "max_vulnerability_probability": round(attempt_max_prob, 4),
            "threshold": detector_threshold,
            "compilation": compile_res,
            "verdict": "Vulnerable" if attempt_vulnerable else "Safe",
            "unit_results": attempt_unit_results,
            "patch_outcomes": current_patch_outcomes,
        }

        # Danh sách failing units: loại bỏ hàm view/pure (Task B8)
        failing_units = [
            ur["unit_id"] for ur in attempt_unit_results
            if ur["label"] == "Vulnerable" and ur["unit_id"] in verification_target_ids
            and not any(u.unit_id == ur["unit_id"] and getattr(u, "is_view_or_pure", False) for u in unit_objs)
        ]

        # Chấm điểm lần thử theo bộ (compile_ok, guard_ok, -n_still_vuln, -diff_size) - Task B4
        compile_ok = 1 if not compile_failed else 0
        guard_ok = 1 if not any(po.get("stage") == "guard" for po in current_patch_outcomes) else 0
        n_still_vuln = len(failing_units)
        diff_size = abs(len(current_attempt_code) - len(source_code))
        attempt_score = (compile_ok, guard_ok, -n_still_vuln, -diff_size)

        if attempt_score > best_attempt_score:
            best_attempt_score = attempt_score
            best_attempt_code = current_attempt_code
            best_attempt_outcomes = current_patch_outcomes

        if not attempt_vulnerable and not compile_failed:
            print(f"[{time.strftime('%H:%M:%S')}] [PHASE 5] Verification Succeeded: All units verified Safe by Neural Detector (prob: {attempt_max_prob:.4f} < {detector_threshold}) & Solc compilation passed on {attempt_label}.", flush=True)
            attempt_record["notes"] = f"Verified clean by Phase 1 Neural Detector and Solc compilation on attempt {attempt}."
            verification_attempts.append(attempt_record)
            overall_verdict = "PASS"
            break
        else:
            reasons = []
            if compile_failed:
                reasons.append(f"Solc compilation errors ({compile_res.get('compiler')})")
            if attempt_max_prob >= detector_threshold:
                reasons.append(f"Residual vulnerability flagged by Neural Detector (max prob: {attempt_max_prob:.4f} >= {detector_threshold})")

            print(f"[{time.strftime('%H:%M:%S')}] [PHASE 5] Verification {attempt_label}: Failed due to {', '.join(reasons)}.", flush=True)
            if attempt < MAX_VERIFICATION_ATTEMPTS:
                # Task B4 & B8: Chỉ đưa unit mục tiêu đã thất bại vào lần thử tiếp theo
                active_target_units = [u for u in target_units if u.unit_id in failing_units]
                print(f"[{time.strftime('%H:%M:%S')}] [PHASE 5] Collecting error diagnostics for failing unit(s) {failing_units} (Attempt {attempt + 1}/{MAX_VERIFICATION_ATTEMPTS})...", flush=True)

                feedback_parts = [
                    f"\n\n[CRITICAL RESIDUAL ERROR FEEDBACK - ATTEMPT {attempt}/{MAX_VERIFICATION_ATTEMPTS}]"
                ]
                if compile_failed:
                    # Trích xuất chính xác dòng Error: từ log của solc (Task B4)
                    solc_raw_err = compile_res.get("error", "").strip()
                    err_lines = [el for el in solc_raw_err.splitlines() if "Error:" in el or "ParserError:" in el or "TypeError:" in el]
                    clean_solc_err = "\n".join(err_lines[:8]) if err_lines else solc_raw_err[:400]
                    feedback_parts.append(
                        f"Solidity compiler ({compile_res.get('compiler')}) failed with error:\n"
                        f"{clean_solc_err}\n"
                        f"Your patch broke compilation. You MUST fix this compiler error."
                    )
                if attempt_max_prob >= detector_threshold:
                    feedback_parts.append(
                        f"Phase 1 Neural Detector re-evaluated your patch and STILL FLAGGED it as Vulnerable (max prob: {attempt_max_prob:.1%}).\n"
                        f"Failing unit(s): {', '.join(failing_units)}.\n"
                        f"Ensure state mutations occur before external calls and invariants are preserved."
                    )
                feedback_note = "\n".join(feedback_parts)
                attempt_record["feedback"] = feedback_note
                verification_attempts.append(attempt_record)
            else:
                print(f"[{time.strftime('%H:%M:%S')}] [PHASE 5] Completed: Reached maximum {MAX_VERIFICATION_ATTEMPTS} verification attempts. Retaining best verified attempt.", flush=True)
                attempt_record["notes"] = f"Residual issues remain after {MAX_VERIFICATION_ATTEMPTS} attempts: {', '.join(reasons)}."
                verification_attempts.append(attempt_record)
                overall_verdict = "WARN"

    # Task B4: Pipeline giữ lần thử có điểm tốt nhất (thay vì luôn lấy lần thử cuối)
    fixed_code = best_attempt_code
    patch_outcomes = best_attempt_outcomes
    fixed_unit_results = attempt_unit_results

    print(f"[{time.strftime('%H:%M:%S')}] [PHASE 5] Completed: Multi-Tier Self-Verification finished (Verdict: {overall_verdict}, Total Attempts: {len(verification_attempts)}).", flush=True)

    last_attempt = verification_attempts[-1] if verification_attempts else {}
    last_compile = last_attempt.get("compilation", {})
    compile_passed = last_compile.get("passed", True)

    verifier_items = [
        {
            "finding_id": f.get("finding_id") or f"{f.get('swc', 'Unknown')}:{f.get('unit_id', 'contract')}",
            "status": "PASS" if overall_verdict == "PASS" else "FLAGGED",
            "residual_risk": "None" if overall_verdict == "PASS" else f"Residual probability: {fixed_max_prob:.1%}",
            "verification_notes": f"Multi-Tier Verification (Attempts: {len(verification_attempts)}/{MAX_VERIFICATION_ATTEMPTS}, Prob: {fixed_max_prob:.4f}, Solc: {'Passed' if compile_passed else 'Failed'}).",
            "checklist": [
                {"name": "Solidity Compilation Integrity (solc)", "passed": compile_passed},
                {"name": "Phase 1 Neural Detector Re-scan", "passed": fixed_max_prob < detector_threshold},
                {"name": "Iterative Feedback Loop", "passed": True, "attempts": len(verification_attempts)},
                {"name": "No Residual Logic Vulnerability", "passed": overall_verdict == "PASS"},
            ],
        }
        for f in findings
    ]

    total_elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
    fixer_elapsed_ms = round((time.perf_counter() - fixer_loop_start) * 1000, 2)

    phases = {
        "detector": {
            "status": "ok",
            "original": {
                "status": "ok",
                "verdict": "Vulnerable" if is_contract_vuln else "Safe",
                "vulnerability_probability": round(max_prob, 4),
                "threshold": detector_threshold,
                "unit_results": unit_results,
            },
            "fixed": {
                "status": "ok",
                "verdict": "Safe" if fixed_max_prob < detector_threshold else "Vulnerable",
                "score": round(fixed_max_prob, 4),
                "vulnerability_probability": round(fixed_max_prob, 4),
                "threshold": detector_threshold,
                "unit_results": fixed_unit_results,
            },
            "candidate_findings": candidate_findings,
            "confirmed_findings": findings,
        },
        "advisor": advisor,
        "assessor": assessor,
        "fixer": {
            "status": "ok",
            "fixed_code": fixed_code,
            "patch_outcomes": patch_outcomes,
            "patches": [
                {
                    "finding_id": p["finding_id"],
                    "description": p["reason"],
                }
                for p in patch_outcomes
            ],
        },
        "verifier": {
            "status": "ok",
            "verification_model": "Qwen2.5-Coder-14B v13 Neural Detector & Solc",
            "verdict": overall_verdict,
            "overall_verdict": overall_verdict,
            "verifications": verifier_items,
            "attempts_count": len(verification_attempts),
            "max_attempts": MAX_VERIFICATION_ATTEMPTS,
            "attempts": verification_attempts,
            "compilation": last_compile,
            "fixed_unit_results": fixed_unit_results,
            "vulnerability_probability": round(fixed_max_prob, 4),
            "threshold": detector_threshold,
            "notes": f"Multi-Tier Self-Verification ({len(verification_attempts)} attempt(s)). Final verdict: {overall_verdict}.",
        },
        "reporter": {
            "status": "ok",
            "executive_summary": (
                f"Audit completed using Dedicated GPU Server (v13 Qwen2.5-Coder-14B) for run {run_id}. "
                f"Identified {len(findings)} finding(s). Applied {len(patch_outcomes)} security patch(es)."
            ),
            "recommendations": advisor["recommendations"],
        },
    }

    return {
        "schema_version": "1.0.0",
        "pipeline_version": "1.0.0",
        "status": "ok",
        "run_id": run_id,
        "backend": "server_gpu_rtx4070ti",
        "source_sha256": request.get("source", {}).get("sha256", ""),
        "fixed_code": fixed_code,
        "patch_outcomes": patch_outcomes,
        "verification_verdict": overall_verdict,
        "phases": phases,
        "timings": {
            "total_ms": total_elapsed_ms,
            "detector_ms": detector_elapsed_ms,
            "fixer_ms": fixer_elapsed_ms,
        },
    }
