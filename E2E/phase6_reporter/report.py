"""
Phase 6 - deterministic audit report: one report-data dict (build_report_data + the Phase 5
verdict from apply_verification_to_report) rendered as Markdown, HTML and PDF (reportlab).
"""

from __future__ import annotations

import html
import re
import textwrap
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from E2E.common.io_utils import write_text
from E2E.config import RISK_ORDER
from E2E.phase6_reporter.vulnerability_reference import VULNERABILITY_REFERENCE


DEFAULT_RECOMMENDATIONS = [
    "Apply and manually review every accepted patch before deployment.",
    "Pin a supported Solidity compiler version and run compilation in CI.",
    "Run Slither and the project tests after every security-sensitive change.",
    "Use checks-effects-interactions and established OpenZeppelin guards for external calls.",
    "Repeat the audit after dependency, privilege, or business-logic changes.",
]


# Định dạng lại mã Solidity thành các dòng ổn định (mỗi câu lệnh một dòng, thụt lề theo { }, giữ nguyên chuỗi/comment)
# để bảng so sánh "mã gốc | mã đã vá" dễ đọc, không bị lệch do khác định dạng.
def format_solidity_for_comparison(code: str) -> list[str]:
    """Format Solidity into stable statement lines for a readable visual diff."""
    lines: list[str] = []
    current: list[str] = []
    indent = 0
    paren_depth = 0
    bracket_depth = 0
    expression_brace_depth = 0
    quote = ""
    line_comment = False
    block_comment = False
    pending_space = False
    physical_line_content = False
    index = 0

    # Đẩy dòng đang gom vào kết quả (kèm thụt lề hiện tại).
    def emit(*, blank: bool = False) -> None:
        nonlocal current, pending_space
        text = "".join(current).strip()
        if text:
            lines.append(f"{'    ' * indent}{text}")
        elif blank and lines and lines[-1] != "":
            lines.append("")
        current = []
        pending_space = False

    # Thêm một ký tự vào dòng đang gom (tự chèn dấu cách khi cần).
    def append(char: str) -> None:
        nonlocal pending_space, physical_line_content
        if pending_space and current and not current[-1].endswith((" ", "\t")):
            current.append(" ")
        pending_space = False
        current.append(char)
        physical_line_content = True

    while index < len(code):
        char = code[index]
        pair = code[index : index + 2]

        if line_comment:
            if char == "\n":
                emit()
                line_comment = False
                physical_line_content = False
            else:
                current.append(char)
            index += 1
            continue

        if block_comment:
            current.append(char)
            if pair == "*/":
                current.append("/")
                index += 2
                block_comment = False
                emit()
            else:
                index += 1
            continue

        if quote:
            append(char)
            if char == "\\" and index + 1 < len(code):
                current.append(code[index + 1])
                index += 2
                continue
            if char == quote:
                quote = ""
            index += 1
            continue

        if pair == "//":
            if "".join(current).strip():
                emit()
            current.extend(("/", "/", " "))
            physical_line_content = True
            line_comment = True
            index += 2
            while index < len(code) and code[index] in " \t":
                index += 1
            continue

        if pair == "/*":
            if "".join(current).strip():
                emit()
            current.extend(("/", "*"))
            physical_line_content = True
            block_comment = True
            index += 2
            continue

        if char in {'"', "'"}:
            append(char)
            quote = char
            index += 1
            continue

        if char.isspace():
            if char == "\n":
                if "".join(current).strip():
                    emit()
                elif not physical_line_content:
                    emit(blank=True)
                physical_line_content = False
            else:
                pending_space = True
            index += 1
            continue

        if char == "(":
            append(char)
            paren_depth += 1
        elif char == ")":
            append(char)
            paren_depth = max(0, paren_depth - 1)
        elif char == "[":
            append(char)
            bracket_depth += 1
        elif char == "]":
            append(char)
            bracket_depth = max(0, bracket_depth - 1)
        elif char == "{":
            prefix = "".join(current).rstrip()
            is_call_options = bool(
                paren_depth
                or bracket_depth
                or re.search(r"(?:\.(?:call|delegatecall|staticcall)|\bnew\s+[A-Za-z_]\w*)$", prefix)
            )
            if expression_brace_depth or is_call_options:
                append(char)
                expression_brace_depth += 1
            else:
                append("{")
                emit()
                indent += 1
        elif char == "}":
            if expression_brace_depth:
                append(char)
                expression_brace_depth -= 1
            else:
                emit()
                indent = max(0, indent - 1)
                current.append("}")
                physical_line_content = True
                lookahead = index + 1
                while lookahead < len(code) and code[lookahead] in " \t":
                    lookahead += 1
                if lookahead < len(code) and code[lookahead] == ";":
                    current.append(";")
                    index = lookahead
                emit()
        elif char == ";":
            append(char)
            if paren_depth == 0 and bracket_depth == 0 and expression_brace_depth == 0:
                emit()
        else:
            append(char)
        index += 1

    emit()
    while lines and lines[-1] == "":
        lines.pop()
    return lines


# Dựng các dòng so sánh song song mã gốc / mã đã vá bằng SequenceMatcher; đánh dấu dòng bị thay/thêm ở bản vá.
def build_code_comparison_rows(original_code: str, fixed_code: str) -> list[dict[str, Any]]:
    """Return side-by-side code rows with fixed-side change markers."""
    original_lines = format_solidity_for_comparison(original_code)
    fixed_lines = format_solidity_for_comparison(fixed_code)
    rows: list[dict[str, Any]] = []

    for tag, source_start, source_end, fixed_start, fixed_end in SequenceMatcher(
        None,
        original_lines,
        fixed_lines,
        autojunk=False,
    ).get_opcodes():
        source_chunk = original_lines[source_start:source_end]
        fixed_chunk = fixed_lines[fixed_start:fixed_end]
        changed = tag in {"replace", "insert"}
        row_count = max(len(source_chunk), len(fixed_chunk), 1)
        for index in range(row_count):
            rows.append(
                {
                    "source": source_chunk[index] if index < len(source_chunk) else "",
                    "fixed": fixed_chunk[index] if index < len(fixed_chunk) else "",
                    "fixed_changed": changed and index < len(fixed_chunk),
                }
            )
    return rows or [{"source": "", "fixed": "", "fixed_changed": False}]


# Xuất bảng so sánh mã dạng HTML-trong-Markdown (dòng thay đổi được bọc <mark>).
def _markdown_code_compare(rows: list[dict[str, Any]]) -> str:
    source_lines = [html.escape(row["source"]) or "&nbsp;" for row in rows]
    fixed_lines = []
    for row in rows:
        line = html.escape(row["fixed"]) or "&nbsp;"
        fixed_lines.append(f"<mark>{line}</mark>" if row["fixed_changed"] else line)
    return "\n".join(
        [
            '<table class="code-comparison">',
            "<thead><tr><th>Source Code (Sample)</th><th>Fixed Source Code</th></tr></thead>",
            "<tbody><tr>",
            f"<td><pre><code>{'<br>'.join(source_lines)}</code></pre></td>",
            f"<td><pre><code>{'<br>'.join(fixed_lines)}</code></pre></td>",
            "</tr></tbody>",
            "</table>",
        ]
    )


# Xuất các dòng mã thành <span> cho báo cáo HTML (dòng thay đổi có class "changed").
def _html_code_lines(rows: list[dict[str, Any]], key: str, highlight_key: str | None = None) -> str:
    lines = []
    for row in rows:
        css_class = "code-line"
        if highlight_key and row.get(highlight_key):
            css_class += " changed"
        line = html.escape(row.get(key, "")) or " "
        lines.append(f"<span class='{css_class}'>{line}</span>")
    return "\n".join(lines)


# Lấy kết quả một phase (advisor/assessor/fixer/...) từ server_result.
def _phase(remote: dict[str, Any], name: str) -> dict[str, Any]:
    return remote.get("phases", {}).get(name, {}) if remote else {}


# Tìm mục (suggestion/assessment/verification) ứng với một finding: theo finding_id, không có thì theo SWC + unit.
def _match(items: list[dict[str, Any]], finding: dict[str, Any]) -> dict[str, Any]:
    for item in items:
        if str(item.get("finding_id") or "") == str(finding.get("finding_id") or ""):
            return item
        if item.get("swc") == finding.get("swc") and (
            not item.get("unit_id") or item.get("unit_id") == finding.get("unit_id")
        ):
            return item
    return {}


CHANNEL_LABELS = {
    "slither": "Slither",
    "compile": "Compile",
    "static": "Static Analysis",
    "rag": "RAG",
    "llm": "Neural Detector",
}


# Dựng các dòng bảng "Verification Results" (mục 5.5): mỗi kênh (slither, compile, static, RAG, detector)
# -> PASS/FAIL/ERROR/SKIPPED + chi tiết.
def verification_channel_rows(data: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    channels = data.get("verification_channels", {})
    ordered_keys = [key for key in CHANNEL_LABELS if key in channels]
    ordered_keys.extend(key for key in channels if key not in ordered_keys)
    for key in ordered_keys:
        label = CHANNEL_LABELS.get(key, key.replace("_", " ").title())
        component = channels[key]
        findings = component.get("findings", [])
        detail = str(component.get("reason") or component.get("error") or "").strip()
        if not detail and findings:
            detail = f"{len(findings)} remaining finding(s)"
        if not detail:
            detail = "No remaining vulnerability signal."
        verdict = str(component.get("verdict") or "Not run")
        passed = component.get("passed")
        if passed is None:
            passed = verdict in {"PASS", "WARN"}
        if verdict == "SKIPPED":
            result_str = "SKIPPED"
        elif passed:
            result_str = "PASS"
        elif verdict == "ERROR":
            result_str = "ERROR"
        else:
            result_str = "FAIL"
        rows.append(
            {
                "component": label,
                "result": result_str,
                "verdict": verdict,
                "detail": detail,
            }
        )
    return rows


# Khi bản vá được chấp nhận: cập nhật trạng thái từng finding theo kết luận cuối của Phase 5.
#   - Hàm đã vá (tự kiểm PASS hoặc FLAGGED) -> "Fixed: patch accepted ...". Nếu detector vẫn báo (FLAGGED) thì
#     ghi rõ xác suất so với ngưỡng và gọi đó là detector disagreement (detector hay báo nhầm hàm an toàn).
#   - Hàm không vá -> giữ nguyên "Not patched" (lỗi vẫn mở).
# Thêm một câu kết luận cuối vào executive summary để tóm tắt không mâu thuẫn với từng finding.
def _apply_accepted_verdict(report_data: dict[str, Any], verification: dict[str, Any], verdict: str) -> None:
    probabilities = {
        item.get("unit_id"): item.get("vulnerability_probability")
        for item in verification.get("fixed_unit_results", [])
    }
    threshold = (verification.get("fixed_detection") or {}).get("threshold")
    disagreeing_units = []
    for finding in report_data.get("findings", []):
        self_check = finding.get("self_check_status")
        if self_check not in {"PASS", "FLAGGED"}:
            continue
        finding["verification_status"] = f"Fixed: {verdict}"
        finding["residual_risk"] = "None"
        if self_check == "FLAGGED":
            unit_id = finding.get("unit_id")
            probability = probabilities.get(unit_id)
            detail = (
                f"p={float(probability):.4f} >= threshold {float(threshold):.2f}"
                if probability is not None and threshold is not None
                else "above threshold"
            )
            finding["verification_status"] += (
                f"; the neural detector still flags the patched unit ({detail}) - detector disagreement"
            )
            finding["residual_risk"] = (
                "Low: compiler, behavior checks and Slither passed, but the neural "
                "detector disagrees; manual review is recommended"
            )
            if unit_id not in disagreeing_units:
                disagreeing_units.append(unit_id)
    summary = str(report_data.get("executive_summary") or "").rstrip()
    final_sentence = f" Final verification: {verdict}."
    if disagreeing_units:
        final_sentence += (
            " The neural detector still flags "
            + ", ".join(str(unit) for unit in disagreeing_units)
            + " after the patch (detector disagreement); the deterministic checks decide acceptance."
        )
    if final_sentence.strip() not in summary:
        report_data["executive_summary"] = summary + final_sentence


# Đưa kết quả Phase 5 vào dữ liệu báo cáo: verdict bản vá, các kênh kiểm chứng, số lần sửa, detector_disagreement.
#   - Bản vá được chấp nhận -> _apply_accepted_verdict (trạng thái finding khớp kết luận cuối).
#   - Bản vá bị từ chối/không kết luận được -> viết lại executive summary (KHÔNG được deploy bản vá, lỗi vẫn
#     mở), đánh dấu từng finding và thêm khuyến nghị "Do not deploy the generated patch" lên đầu.
# Mục đích: báo cáo không bao giờ trình bày bản vá thất bại như đã sửa xong, và không ghi một finding là
# FLAGGED khi kết luận cuối là "accepted".
def apply_verification_to_report(
    report_data: dict[str, Any],
    verification: dict[str, Any],
) -> None:
    verdict = str(verification.get("overall_verdict") or "Not complete")
    report_data["patch_verdict"] = verdict
    report_data["detector_disagreement"] = bool(verification.get("detector_disagreement"))
    report_data["verification_channels"] = verification.get("verification_channels", {})
    if verification.get("attempts_count") is not None:
        report_data["attempts_count"] = verification.get("attempts_count")
        report_data["max_attempts"] = verification.get("max_attempts")
    for row in report_data.get("phase_rows", []):
        if row.get("phase") == "5 Verifier":
            row["status"] = verification.get("status", "partial")

    if verdict.startswith("Patch accepted"):
        _apply_accepted_verdict(report_data, verification, verdict)
        return
    if not verdict.startswith(("Patch rejected", "Patch inconclusive", "Inconclusive")):
        return

    finding_count = len(report_data.get("findings", []))
    behavioral = verification.get("behavioral_verification", {})
    regression_count = len(behavioral.get("findings", []))
    detail = (
        f" Deterministic behavioral checks found {regression_count} regression(s)."
        if regression_count
        else ""
    )
    outcome = "rejected" if verdict.startswith("Patch rejected") else "not accepted"
    no_patch = any(
        marker in verdict
        for marker in (
            "no patch was generated", "patch execution incomplete",
            "no function-level repair target", "no effective patch",
        )
    )
    if no_patch:
        report_data["executive_summary"] = (
            f"The audit identified {finding_count} finding(s). No patch was verified: "
            f"{verdict}. The findings remain open and require manual review."
        )
    else:
        report_data["executive_summary"] = (
            f"The audit identified {finding_count} finding(s). A patch was generated but {outcome} "
            f"by deterministic verification: {verdict}.{detail} The patch must not be deployed; "
            "the original finding remains open until a behavior-preserving patch passes compilation, "
            "Slither, and project tests."
        )
    for finding in report_data.get("findings", []):
        # A finding that was never patched keeps its own status ("Not patched"):
        # the rejected patch did not touch it.
        if finding.get("self_check_status") == "NOT_PATCHED":
            continue
        finding["verification_status"] = (
            "Rejected by deterministic verifier"
            if verdict.startswith("Patch rejected")
            else verdict
        )
        finding["residual_risk"] = verdict
    if no_patch:
        return
    rejection_recommendation = (
        "Do not deploy the generated patch; preserve the contract's intended transfers and behavior "
        "while addressing the root cause."
    )
    recommendations = report_data.get("recommendations", [])
    if rejection_recommendation not in recommendations:
        report_data["recommendations"] = [rejection_recommendation, *recommendations]


# Dòng CVSS của một finding (giống nhau ở MD/HTML/PDF): điểm, mức, nguồn điểm, mức lúc phát hiện (nếu khác),
# thứ tự ưu tiên sửa. Rỗng nếu chưa có.
def _cvss_text(finding: dict[str, Any]) -> str:
    if finding.get("cvss_score") is None:
        return ""
    source = {"llm": "LLM-selected metrics", "swc_default": "SWC default vector"}.get(finding.get("cvss_source") or "")
    details = [str(finding.get("severity") or "")]
    if source:
        details.append(source)
    detected = finding.get("detected_severity")
    if detected and detected != finding.get("severity"):
        details.append(f"detected as {detected}")
    text = f"{finding['cvss_score']} ({', '.join(item for item in details if item)})"
    if finding.get("repair_priority"):
        text += f"; repair priority #{finding['repair_priority']}"
    return text


# Dòng kết quả kiểm chứng của một finding (giống nhau ở MD/HTML/PDF): trạng thái + rủi ro còn lại.
def _verification_text(finding: dict[str, Any]) -> str:
    text = str(finding.get("verification_status") or "Not verified")
    if finding.get("residual_risk"):
        text += f"; residual risk: {finding['residual_risk']}"
    return text


# Tên hiển thị của finding: SWC-103 (pragma không cố định) luôn ghi "Floating Pragma".
def _display_title(finding: dict[str, Any]) -> str:
    return "Floating Pragma" if finding.get("swc") == "SWC-103" else str(finding.get("title") or "")


# Trạng thái tự kiểm của vòng sửa phía GPU (gpu_pipeline) -> câu mô tả cho báo cáo.
# Đây CHƯA phải kết luận cuối: apply_verification_to_report ghi đè theo kết luận Phase 5 (overall_verdict).
SELF_CHECK_TEXT = {
    "PASS": "Repair-loop self-check passed (compiler, neural detector, static/RAG re-run)",
    "FLAGGED": "Repair-loop self-check: the neural detector still flags the patched unit",
    "NOT_PATCHED": "Not patched",
}


# Chọn danh sách finding cho báo cáo, trả (findings, dismissed).
#   - Có kết quả GPU (lượt 2): lấy phases.detector.confirmed_findings = finding còn lại SAU KHI Advisor loại
#     báo nhầm.
#     Finding Advisor loại (dismissed_findings) KHÔNG tính là lỗi, chỉ liệt kê ở mục riêng để minh bạch.
#   - Chưa có kết quả GPU (lượt 1 / bản xem trước): dùng finding của fuse_detection (static + RAG).
# Mục đích: số lỗi trong báo cáo, manifest và web app khớp với những gì Advisor xác nhận và Fixer xử lý.
def select_report_findings(
    fused: dict[str, Any],
    remote: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detector = _phase(remote or {}, "detector")
    if "confirmed_findings" not in detector:
        return list(fused.get("findings", [])), []
    return list(detector.get("confirmed_findings", [])), list(detector.get("dismissed_findings", []))


# [HÀM CHÍNH PHASE 6 - dữ liệu] Gom mọi thứ thành một dict báo cáo: thông tin contract, kết luận fusion,
# từng finding (chọn bởi select_report_findings) ghép với gợi ý Advisor + CVSS Assessor + trạng thái tự kiểm
# của vòng sửa, finding bị Advisor loại, finding ứng viên, phân bố mức độ, bảng trạng thái các phase
# (phase_rows), mã gốc/mã vá, bảng tham chiếu lỗ hổng (VULNERABILITY_REFERENCE), khuyến nghị, hạn chế,
# tuyên bố miễn trừ. Lượt 1 (remote=None) dùng để xuất bản xem trước.
def build_report_data(
    request: dict[str, Any],
    fused: dict[str, Any],
    remote: dict[str, Any] | None,
    tool_verification: dict[str, Any],
) -> dict[str, Any]:
    remote = remote or {}
    advisor_items = _phase(remote, "advisor").get("suggestions", [])
    assessor_items = _phase(remote, "assessor").get("assessments", [])
    verifier_items = _phase(remote, "verifier").get("verifications", [])
    fixer = _phase(remote, "fixer")
    selected_findings, dismissed = select_report_findings(fused, remote)
    findings = []

    for finding in selected_findings:
        advice = _match(advisor_items, finding)
        assessment = _match(assessor_items, finding)
        verification = _match(verifier_items, finding)
        self_check = verification.get("status") or ("NOT_PATCHED" if remote else "")
        if self_check:
            status_text = SELF_CHECK_TEXT.get(self_check, self_check)
        else:
            status_text = "Not verified (preview before the GPU phases)"
        findings.append(
            {
                **finding,
                # One severity per finding: the Phase 3 CVSS level when assessed, else the detection
                # severity (preview before the GPU phases). The detection severity is kept alongside.
                "severity": assessment.get("severity") or finding.get("severity", "Unknown"),
                "detected_severity": finding.get("severity", "Unknown"),
                "root_cause": advice.get("root_cause") or finding.get("description") or "",
                # Impact: Advisor's attack scenario > knowledge-base impact of the finding > Assessor's CVSS rationale.
                "impact": advice.get("impact") or finding.get("impact") or assessment.get("justification") or "",
                "cvss_score": assessment.get("cvss_score"),
                "cvss_source": assessment.get("cvss_source"),
                "repair_priority": assessment.get("repair_priority"),
                "repair_steps": advice.get("repair_steps")
                or ([finding["remediation"]] if finding.get("remediation") else []),
                "self_check_status": self_check or None,
                "verification_status": status_text,
                "residual_risk": verification.get("residual_risk") or "",
            }
        )
    dismissed_findings = [
        {
            "finding_id": item.get("finding_id"),
            "swc": item.get("swc"),
            "title": item.get("title"),
            "unit_id": item.get("unit_id"),
            "line": item.get("line"),
            "severity": item.get("severity", "Unknown"),
            "confidence": item.get("confidence"),
            "sources": item.get("sources", []),
            # The Advisor's own explanation (root cause it gave for the false-positive verdict).
            "reason": item.get("dismissal_reason") or "Dismissed by the Advisor as a false positive.",
        }
        for item in dismissed
    ]

    distribution = Counter(item.get("severity", "Unknown") for item in findings)
    original_code = request["source"]["code"]
    fixed_code = fixer.get("fixed_code") or original_code
    compile_status = tool_verification.get("compile", {}).get("status", "not run")
    slither_count = len(tool_verification.get("slither", {}).get("findings", []))
    function_names = ", ".join(unit["unit_id"] for unit in request.get("code_units", [])[:8])
    if len(request.get("code_units", [])) > 8:
        function_names += ", ..."
    if remote:
        applied_count = sum(1 for item in fixer.get("patch_outcomes", []) if item.get("status") == "applied")
        executive_summary = (
            f"Audit completed using the GPU server (Qwen2.5-Coder-14B v13) for run {request['run_id']}. "
            f"{len(findings)} confirmed finding(s), {len(dismissed)} dismissed by the Advisor, "
            f"{applied_count} patch(es) applied."
        )
    else:
        executive_summary = (
            f"The audit result is {fused.get('verdict', 'Inconclusive')} with a fused confidence "
            f"score of {float(fused.get('score') or 0.0):.3f}. The pipeline identified "
            f"{len(findings)} confirmed finding(s). Compilation status: {compile_status}; "
            f"Slither reported {slither_count} detector finding(s) "
            f"(verdict: {tool_verification.get('policy_verdict', 'not run')})."
        )
    # The only backend: model calls go to the GPU server (detector = LoRA v13, agents = base instruct model).
    p1c_title = "1C Qwen2.5-Coder-14B detector (v13)"
    detector_location = agent_location = "GPU Server (RTX 4070 Ti SUPER)"
    detector_engine = "Qwen2.5-Coder-14B-Instruct v13 Quality-First (LoRA r=64, alpha=128)"
    agent_engine = "Qwen2.5-Coder-14B-Instruct v13"

    phase_rows = [
        {
            "phase": "1A Static detector",
            "location": "Local",
            "status": request["local_phase1"]["static"].get("status", "unknown"),
            "engine": request["local_phase1"]["static"].get("engine", "Regex / AST Rule Engine"),
        },
        {
            "phase": "1B RAG detector",
            "location": "Local",
            "status": request["local_phase1"]["rag"].get("status", "unknown"),
            "engine": request["local_phase1"]["rag"].get("engine", "SWC TF-IDF (11,869 cases)"),
        },
        {
            "phase": p1c_title,
            "location": detector_location,
            "status": _phase(remote, "detector").get("status", "pending"),
            "engine": detector_engine,
        },
        {
            "phase": "2 Advisor",
            "location": agent_location,
            "status": _phase(remote, "advisor").get("status", "pending"),
            "engine": agent_engine,
        },
        {
            "phase": "3 Risk assessor",
            "location": agent_location,
            "status": _phase(remote, "assessor").get("status", "pending"),
            "engine": agent_engine,
        },
        {
            "phase": "4 Fixer",
            "location": agent_location,
            "status": fixer.get("status", "pending"),
            "engine": f"{agent_engine} + deterministic patch apply",
        },
        {
            "phase": "5 Verifier",
            "location": "Local",
            "status": tool_verification.get("status", "pending"),
            "engine": "solc compiler, Slither, AST behavioral invariants",
        },
        {
            "phase": "6 Reporter",
            "location": "Local",
            "status": "ok",
            "engine": "deterministic audit report renderer",
        },
    ]

    raw_candidates = (
        fused.get("candidate_findings", [])
        or _phase(remote, "detector").get("candidate_findings", [])
        or []
    )
    for f in findings:
        f["title"] = _display_title(f)
    candidate_findings = [{**cand, "title": _display_title(cand)} for cand in raw_candidates]

    # Highest risk of the findings this report actually lists (not of the findings the Advisor dismissed).
    if findings:
        risk_level = max((f.get("severity", "Unknown") for f in findings), key=lambda sev: RISK_ORDER.get(sev, 0))
    else:
        risk_level = "Unknown" if fused.get("risk_level") == "Unknown" else "None"
    contract_names = sorted({
        unit.get("contract_name") for unit in request.get("code_units", []) if unit.get("contract_name")
    })

    return {
        "title": "Smart Contract Audit Report",
        "run_id": request["run_id"],
        "audit_time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "file_name": request["source"]["file_name"],
        "source_sha256": request["source"]["sha256"],
        "analyzed_object": function_names or request["source"]["file_name"],
        "contract_name": ", ".join(contract_names) or Path(request["source"]["file_name"]).stem,
        "detection_result": fused.get("verdict", "Inconclusive"),
        "analysis_status": fused.get("status", "inconclusive"),
        "score": float(fused.get("score") or 0.0),
        "risk_level": risk_level,
        "executive_summary": executive_summary,
        "findings": findings,
        "candidate_findings": candidate_findings,
        "dismissed_findings": dismissed_findings,
        "distribution": distribution,
        "phase_rows": phase_rows,
        "original_code": original_code,
        "fixed_code": fixed_code,
        "patch_verdict": _phase(remote, "verifier").get("overall_verdict", "Not complete"),
        "reference_table": [
            {
                "name": entry["name"],
                "abbr": entry["abbr"],
                "severity": entry["severity"],
                "detected": any(
                    any(
                        k.lower() in str(f.get("swc") or "").lower()
                        or k.lower() in str(f.get("title") or "").lower()
                        or k.lower() in str(f.get("unit_id") or "").lower()
                        for k in entry.get("keys", [])
                    )
                    for f in findings
                ),
                "impact": entry["impact"],
            }
            for entry in VULNERABILITY_REFERENCE
        ],
        "tool_verification": tool_verification,
        "verification_channels": {},
        "attempts_count": _phase(remote, "verifier").get("attempts_count"),
        "max_attempts": _phase(remote, "verifier").get("max_attempts"),
        "recommendations": list(DEFAULT_RECOMMENDATIONS),
        "limitations": [
            "SWC Registry source material is no longer actively maintained and is not exhaustive.",
            "The fine-tuned classifier is a binary detector; SWC attribution comes from local static/RAG evidence.",
            "Security tools can disagree or time out; manual review and project tests remain required.",
        ],
        "disclaimer": (
            "This report represents a best-effort automated security review. It cannot guarantee that all "
            "vulnerabilities, economic attacks, integration risks, or business-logic defects were identified."
        ),
    }


# Chuỗi "số lần sửa đã chạy / tối đa" cho mục 5.5 (ví dụ "3/3 attempt(s)").
def _attempt_text(data: dict[str, Any]) -> str:
    count = data.get("attempts_count")
    maximum = data.get("max_attempts")
    if not count:
        return "not run"
    return f"{count}/{maximum} attempt(s)" if maximum else f"{count} attempt(s)"


# Cột của bảng phân bố mức độ (mục 4.2): Critical / High / Medium / Low, thêm các mức khác nếu có finding.
def _severity_columns(distribution: Counter) -> list[str]:
    standard = ["Critical", "High", "Medium", "Low"]
    return standard + [level for level in distribution if level not in standard and distribution[level] > 0]


# Xuất báo cáo Markdown theo cấu trúc: 1. Thông tin contract, 2. Tóm tắt, 3. Phương pháp (bảng phase),
# 4. Findings (thống kê, phân bố, finding xác nhận, ứng viên, bị Advisor loại), 5. Phân tích chi tiết (mã gốc,
# gợi ý sửa, mã đã vá, kết quả kiểm chứng), 6. Kết luận & khuyến nghị, 7. Miễn trừ, 8. Phụ lục tham chiếu lỗ hổng.
def render_markdown(data: dict[str, Any]) -> str:
    findings = data["findings"]
    dist = data["distribution"]
    code_comparison = build_code_comparison_rows(data["original_code"], data["fixed_code"])
    channel_rows = verification_channel_rows(data)
    lines = [
        "# Smart Contract Audit Report",
        "",
        "## 1. Contract Information",
        "",
        f"- **Run ID:** `{data['run_id']}`",
        f"- **Analyzed Object:** {data['analyzed_object']}",
        f"- **Source File:** `{data['file_name']}`",
        f"- **Source SHA-256:** `{data['source_sha256']}`",
        f"- **Detection Result:** **{data['detection_result']}**",
        f"- **Analysis Status:** {data['analysis_status']}",
        f"- **Fused Score:** {data['score']:.3f}",
        f"- **Audit Time:** {data['audit_time']}",
        "",
        "## 2. Executive Summary",
        "",
        data["executive_summary"],
        "",
        "## 3. Methodology",
        "",
        (
            "The audit combines static rules, local SWC TF-IDF retrieval, "
            "fine-tuned neural vulnerability detection, generative repair agents, "
            "compiler checks, AST behavioral-preservation verification, and deterministic report generation."
        ),
        "",
        "| Phase | Location | Status | Engine |",
        "| :---- | :------- | :----- | :----- |",
    ]
    lines.extend(
        f"| {row['phase']} | {row['location']} | {row['status']} | {row['engine']} |"
        for row in data["phase_rows"]
    )
    sev_cols = _severity_columns(dist)

    lines.extend(
        [
            "",
            "## 4. Findings",
            "",
            "### 4.1 Vulnerability Statistics",
            "",
            f"- **Detection Result:** {data['detection_result']}",
            f"- **Vulnerability Count:** {len(findings)} confirmed finding(s)",
            "",
            "### 4.2 Vulnerability Severity Distribution",
            "",
            "| " + " | ".join(sev_cols) + " |",
            "| " + " | ".join([":---:"] * len(sev_cols)) + " |",
            "| " + " | ".join(str(dist.get(s, 0)) for s in sev_cols) + " |",
            "",
            "### 4.3 Confirmed Findings Summary",
            "",
            "| ID | Vulnerability | Unit | Line | Severity | Confidence | Evidence |",
            "| :-- | :------------ | :--- | :--: | :------- | :--------: | :------- |",
        ]
    )
    for finding in findings:
        line_str = str(finding.get("line") or "-")
        lines.append(
            f"| {finding['swc']} | {finding['title']} | {finding.get('unit_id') or 'contract'} | "
            f"{line_str} | {finding['severity']} | {float(finding.get('confidence') or 0):.3f} | "
            f"{', '.join(finding.get('sources', []))} |"
        )
    if not findings:
        lines.append("| - | No confirmed findings | - | - | - | - | - |")

    lines.extend(
        [
            "",
            "### 4.4 Candidate Findings",
            "",
        ]
    )
    candidate_findings = data.get("candidate_findings", [])
    if candidate_findings:
        lines.extend(
            [
                "| ID | Vulnerability | Unit | Line | Severity | Confidence | Sources |",
                "| :-- | :------------ | :--- | :--: | :------- | :--------: | :------- |",
            ]
        )
        for cand in candidate_findings:
            line_str = str(cand.get("line") or "-")
            lines.append(
                f"| {cand.get('swc', '-')} | {cand.get('title', '-')} | {cand.get('unit_id') or 'contract'} | "
                f"{line_str} | {cand.get('severity', 'Unknown')} | {float(cand.get('confidence') or 0):.3f} | "
                f"{', '.join(cand.get('sources', []))} |"
            )
    else:
        lines.append("_No unconfirmed candidate findings below threshold._")

    lines.extend(["", "### 4.5 Findings Dismissed by the Advisor", ""])
    dismissed_findings = data.get("dismissed_findings", [])
    if dismissed_findings:
        lines.extend(
            [
                "_Raised by a detection channel, then judged a false positive by the Advisor (Phase 2). "
                "Not counted as vulnerabilities and not repaired._",
                "",
                "| ID | Vulnerability | Unit | Line | Severity | Confidence | Sources | Advisor reason |",
                "| :-- | :------------ | :--- | :--: | :------- | :--------: | :------ | :------------- |",
            ]
        )
        for item in dismissed_findings:
            reason = str(item.get("reason") or "").replace("|", "&#124;").replace("\n", " ")
            lines.append(
                f"| {item.get('swc', '-')} | {item.get('title', '-')} | {item.get('unit_id') or 'contract'} | "
                f"{item.get('line') or '-'} | {item.get('severity', 'Unknown')} | "
                f"{float(item.get('confidence') or 0):.3f} | {', '.join(item.get('sources', []))} | {reason} |"
            )
    else:
        lines.append("_No finding was dismissed by the Advisor._")

    lines.extend(
        [
            "",
            "## 5. Detailed Analysis",
            "",
            "### 5.1 Contract Name",
            "",
            data["contract_name"],
            "",
            "### 5.2 Source Code",
            "",
            "```solidity",
            data["original_code"],
            "```",
            "",
            "### 5.3 Repair Suggestion",
            "",
        ]
    )
    if not findings:
        lines.append("No confirmed vulnerability was available for automated remediation.")
    for index, finding in enumerate(findings, 1):
        lines.extend(
            [
                f"#### {index}. {finding['title']} ({finding['swc']}) - {finding['severity']}",
                "",
                f"- **Unit:** `{finding.get('unit_id') or 'contract'}`",
                f"- **Root cause:** {finding.get('root_cause') or 'Not provided'}",
                f"- **Impact:** {finding.get('impact') or 'Requires manual impact analysis'}",
            ]
        )
        if _cvss_text(finding):
            lines.append(f"- **CVSS:** {_cvss_text(finding)}")
        lines.append("- **Repair steps:**")
        steps = finding.get("repair_steps") or ["Perform manual review and implement the cited remediation."]
        lines.extend(f"  {number}. {step}" for number, step in enumerate(steps, 1))
        lines.append(f"- **Patch verification:** {_verification_text(finding)}")
        lines.append("")
    lines.extend(
        [
            "### 5.4 Fixed Code",
            "",
            f"_Patch verification verdict: **{data['patch_verdict']}**_",
            "",
            _markdown_code_compare(code_comparison),
            "",
            "### 5.5 Verification Results",
            "",
            f"- **Fix attempts:** {_attempt_text(data)}",
            "",
            "| Verification channel | Result | Verdict | Details |",
            "| :------------------- | :----: | :------ | :------ |",
            *[
                f"| {row['component']} | **{row['result']}** | {row['verdict']} | "
                f"{row['detail'].replace('|', '&#124;')} |"
                for row in channel_rows
            ],
            "",
            f"- **Compilation:** {data['tool_verification'].get('compile', {}).get('status', 'not run')}",
            f"- **Slither verdict:** {data['tool_verification'].get('policy_verdict', 'not run')}",
            f"- **Slither verdict reason:** {data['tool_verification'].get('policy_reason', 'not run')}",
            f"- **Slither:** {data['tool_verification'].get('slither', {}).get('status', 'not run')}",
            f"- **Slither findings:** {len(data['tool_verification'].get('slither', {}).get('findings', []))}",
            f"- **Behavior preservation:** {data['tool_verification'].get('behavioral', {}).get('status', 'not run')}",
            "",
        ]
    )
    if behavioral_evidence := [
        item.get("evidence") or item.get("check")
        for item in data["tool_verification"].get("behavioral", {}).get("findings", [])
        if item.get("evidence") or item.get("check")
    ]:
        lines.append("- **Behavior preservation evidence:**")
        lines.extend(f"  - {entry}" for entry in behavioral_evidence)
    lines.extend(["", "## 6. Summary and Recommendations", ""])
    lines.extend(f"- {item}" for item in data["recommendations"])
    lines.extend(["", "### Limitations", ""])
    lines.extend(f"- {item}" for item in data["limitations"])
    lines.extend(["", "## 7. Disclaimer", "", data["disclaimer"], ""])
    lines.extend(
        [
            "",
            "## 8. Appendix: Vulnerability Reference Catalogue",
            "",
            "| Vulnerability Name | Severity | Detected | Impact Scope |",
            "| :----------------- | :------- | :------: | :----------- |",
        ]
    )
    for row in data.get("reference_table", []):
        mark = "✓" if row.get("detected") else ""
        lines.append(f"| {row['name']} ({row['abbr']}) | {row['severity']} | {mark} | {row['impact']} |")
    lines.append("")
    return "\n".join(lines)


# Xuất báo cáo HTML (cùng nội dung với Markdown, có CSS, bảng so sánh mã tô màu dòng thay đổi).
def render_html(data: dict[str, Any]) -> str:
    code_comparison = build_code_comparison_rows(data["original_code"], data["fixed_code"])
    source_code_lines = _html_code_lines(code_comparison, "source")
    fixed_code_lines = _html_code_lines(code_comparison, "fixed", "fixed_changed")
    behavioral = data["tool_verification"].get("behavioral", {})
    behavioral_evidence = "".join(
        f"<li>{html.escape(str(item.get('evidence') or item.get('check') or ''))}</li>"
        for item in behavioral.get("findings", [])
    )
    catalogue_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['name'])} ({html.escape(row['abbr'])})</td>"
        f"<td><span class='severity {row['severity'].lower()}'>{html.escape(row['severity'])}</span></td>"
        f"<td style='text-align:center;'>{'<span style=\"color:#10b981;font-weight:bold;\">✓</span>' if row.get('detected') else '-'}</td>"
        f"<td>{html.escape(row['impact'])}</td>"
        "</tr>"
        for row in data.get("reference_table", [])
    )
    findings_rows = "".join(
        "<tr>"
        f"<td>{html.escape(item['swc'])}</td>"
        f"<td>{html.escape(str(item['title']))}</td>"
        f"<td>{html.escape(str(item.get('unit_id') or 'contract'))}</td>"
        f"<td style='text-align:center;'>{html.escape(str(item.get('line') or '-'))}</td>"
        f"<td><span class='severity {item['severity'].lower()}'>{html.escape(item['severity'])}</span></td>"
        f"<td>{float(item.get('confidence') or 0):.3f}</td>"
        f"<td>{html.escape(', '.join(item.get('sources', [])))}</td>"
        "</tr>"
        for item in data["findings"]
    ) or "<tr><td colspan='7'>No confirmed findings</td></tr>"
    candidates_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(item.get('swc', '-')))}</td>"
        f"<td>{html.escape(str(item.get('title', '-')))}</td>"
        f"<td>{html.escape(str(item.get('unit_id') or 'contract'))}</td>"
        f"<td style='text-align:center;'>{html.escape(str(item.get('line') or '-'))}</td>"
        f"<td><span class='severity {str(item.get('severity', 'Unknown')).lower()}'>{html.escape(str(item.get('severity', 'Unknown')))}</span></td>"
        f"<td>{float(item.get('confidence') or 0):.3f}</td>"
        f"<td>{html.escape(', '.join(item.get('sources', [])))}</td>"
        "</tr>"
        for item in data.get("candidate_findings", [])
    ) or "<tr><td colspan='7'>No unconfirmed candidate findings below threshold</td></tr>"

    dismissed_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(item.get('swc', '-')))}</td>"
        f"<td>{html.escape(str(item.get('title', '-')))}</td>"
        f"<td>{html.escape(str(item.get('unit_id') or 'contract'))}</td>"
        f"<td style='text-align:center;'>{html.escape(str(item.get('line') or '-'))}</td>"
        f"<td>{html.escape(str(item.get('severity', 'Unknown')))}</td>"
        f"<td>{float(item.get('confidence') or 0):.3f}</td>"
        f"<td>{html.escape(', '.join(item.get('sources', [])))}</td>"
        f"<td>{html.escape(str(item.get('reason') or ''))}</td>"
        "</tr>"
        for item in data.get("dismissed_findings", [])
    ) or "<tr><td colspan='8'>No finding was dismissed by the Advisor</td></tr>"

    dist = data["distribution"]
    sev_cols = _severity_columns(dist)
    dist_th = "".join(f"<th>{s}</th>" for s in sev_cols)
    dist_td = "".join(f"<td>{dist.get(s, 0)}</td>" for s in sev_cols)
    distribution_html = f"<table><tr>{dist_th}</tr><tr>{dist_td}</tr></table>"

    phase_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['phase'])}</td><td>{row['location']}</td>"
        f"<td>{html.escape(str(row['status']))}</td><td>{html.escape(str(row['engine']))}</td>"
        "</tr>"
        for row in data["phase_rows"]
    )
    verification_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['component'])}</td>"
        f"<td class='{row['result'].lower()}'>{row['result']}</td>"
        f"<td>{html.escape(row['verdict'])}</td>"
        f"<td>{html.escape(row['detail'])}</td>"
        "</tr>"
        for row in verification_channel_rows(data)
    )
    details = ""
    for index, item in enumerate(data["findings"], 1):
        steps = "".join(f"<li>{html.escape(str(step))}</li>" for step in item.get("repair_steps", []))
        cvss_line = f"<p><b>CVSS:</b> {html.escape(_cvss_text(item))}</p>" if _cvss_text(item) else ""
        details += (
            f"<section><h4>{index}. {html.escape(item['title'])} ({item['swc']})</h4>"
            f"<p><b>Severity:</b> {item['severity']} &nbsp; <b>Unit:</b> "
            f"<code>{html.escape(str(item.get('unit_id') or 'contract'))}</code></p>"
            f"<p><b>Root cause:</b> {html.escape(str(item.get('root_cause') or 'Not provided'))}</p>"
            f"<p><b>Impact:</b> {html.escape(str(item.get('impact') or 'Requires manual analysis'))}</p>"
            f"{cvss_line}"
            f"<ol>{steps}</ol><p><b>Verification:</b> "
            f"{html.escape(_verification_text(item))}</p></section>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{data['title']}</title>
<style>
@page {{ size: A4; margin: 18mm; }}
body {{ font-family: Arial, sans-serif; color: #18202a; margin: 0 auto; max-width: 1050px; line-height: 1.48; }}
header {{ border-bottom: 4px solid #ba2d2d; padding: 30px 0 18px; margin-bottom: 26px; }}
h1 {{ font-size: 30px; margin: 0; }} h2 {{ color: #7b1f1f; border-bottom: 1px solid #d6dbe1; padding-bottom: 5px; }}
h3 {{ color: #26394d; }} h4 {{ font-size: 16px; }}
.meta {{ display: grid; grid-template-columns: 180px 1fr; gap: 5px 14px; }}
.verdict {{ font-weight: bold; color: #9e1d1d; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 22px; font-size: 13px; }}
th {{ background: #26394d; color: white; text-align: left; }} th, td {{ border: 1px solid #bcc5cf; padding: 7px; vertical-align: top; }}
.severity {{ font-weight: bold; }} .critical {{ color: #9e1d1d; }} .high {{ color: #bd4c13; }}
.medium {{ color: #8a6500; }} .low {{ color: #28633b; }}
.pass {{ color: #28633b; font-weight: bold; }} .fail {{ color: #9e1d1d; font-weight: bold; }}
pre {{ background: #f3f5f7; border-left: 4px solid #52687f; padding: 12px; overflow-wrap: anywhere; white-space: pre-wrap; font-size: 11px; }}
.code-compare {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 12px; margin: 12px 0 22px; }}
.code-panel {{ border: 1px solid #bcc5cf; background: #f8fafc; min-width: 0; }}
.code-panel h4 {{ margin: 0; padding: 7px 10px; background: #26394d; color: white; font-size: 12px; }}
.code-panel pre {{ border-left: 0; margin: 0; background: transparent; }}
.code-line {{ display: block; min-height: 1.2em; }}
.code-line.changed {{ background: #fff2a8; outline: 1px solid #e0c448; }}
.notice {{ border-left: 4px solid #a17600; background: #fff8df; padding: 10px 14px; }}
footer {{ border-top: 1px solid #bcc5cf; margin-top: 30px; padding: 12px 0; color: #66717d; font-size: 12px; }}
@media (max-width: 760px) {{ .code-compare {{ grid-template-columns: 1fr; }} }}
</style></head><body>
<header><h1>Smart Contract Audit Report</h1><p>Run <code>{html.escape(data['run_id'])}</code></p></header>
<h2>1. Contract Information</h2><div class="meta">
<b>Analyzed Object</b><span>{html.escape(data['analyzed_object'])}</span>
<b>Source File</b><span>{html.escape(data['file_name'])}</span>
<b>Source SHA-256</b><code>{data['source_sha256']}</code>
<b>Detection Result</b><span class="verdict">{data['detection_result']}</span>
<b>Analysis Status</b><span>{data['analysis_status']}</span>
<b>Fused Score</b><span>{data['score']:.3f}</span>
<b>Audit Time</b><span>{data['audit_time']}</span></div>
<h2>2. Executive Summary</h2><p>{html.escape(data['executive_summary'])}</p>
<h2>3. Methodology</h2><p>The audit combines static rules, local SWC TF-IDF retrieval, fine-tuned neural vulnerability detection,
generative repair agents, compilation, AST behavior-preservation checks, and deterministic reporting.</p>
<table><thead><tr><th>Phase</th><th>Location</th><th>Status</th><th>Engine</th></tr></thead><tbody>{phase_rows}</tbody></table>
<h2>4. Findings</h2><h3>4.1 Vulnerability Statistics</h3>
<p><b>{len(data['findings'])}</b> confirmed finding(s); highest risk: <b>{data['risk_level']}</b>.</p>
<h3>4.2 Vulnerability Severity Distribution</h3>
{distribution_html}
<h3>4.3 Confirmed Findings Summary</h3>
<table><thead><tr><th>ID</th><th>Vulnerability</th><th>Unit</th><th>Line</th><th>Severity</th><th>Confidence</th><th>Evidence</th></tr></thead>
<tbody>{findings_rows}</tbody></table>
<h3>4.4 Candidate Findings</h3>
<table><thead><tr><th>ID</th><th>Vulnerability</th><th>Unit</th><th>Line</th><th>Severity</th><th>Confidence</th><th>Sources</th></tr></thead>
<tbody>{candidates_rows}</tbody></table>
<h3>4.5 Findings Dismissed by the Advisor</h3>
<p>Raised by a detection channel, then judged a false positive by the Advisor (Phase 2). Not counted as vulnerabilities and not repaired.</p>
<table><thead><tr><th>ID</th><th>Vulnerability</th><th>Unit</th><th>Line</th><th>Severity</th><th>Confidence</th><th>Sources</th><th>Advisor reason</th></tr></thead>
<tbody>{dismissed_rows}</tbody></table>
<h2>5. Detailed Analysis</h2><h3>5.1 Contract Name</h3><p>{html.escape(data['contract_name'])}</p>
<h3>5.2 Source Code</h3><pre>{html.escape(data['original_code'])}</pre>
<h3>5.3 Repair Suggestion</h3>{details or '<p>No confirmed vulnerability was available for remediation.</p>'}
<h3>5.4 Fixed Code</h3><p>Patch verification: <b>{html.escape(data['patch_verdict'])}</b></p>
<div class="code-compare">
<section class="code-panel"><h4>Source Code (Sample)</h4><pre>{source_code_lines}</pre></section>
<section class="code-panel"><h4>Fixed Source Code</h4><pre>{fixed_code_lines}</pre></section>
</div>
<h3>5.5 Verification Results</h3>
<p>Fix attempts: <b>{html.escape(_attempt_text(data))}</b></p>
<table><thead><tr><th>Channel</th><th>Result</th><th>Verdict</th><th>Details</th></tr></thead>
<tbody>{verification_rows}</tbody></table>
<p>Compilation: <b>{data['tool_verification'].get('compile', {}).get('status', 'not run')}</b>;
Slither verdict: <b>{data['tool_verification'].get('policy_verdict', 'not run')}</b>;
Slither: <b>{data['tool_verification'].get('slither', {}).get('status', 'not run')}</b>;
Behavior preservation: <b>{behavioral.get('status', 'not run')}</b>.</p>
{f'<ul>{behavioral_evidence}</ul>' if behavioral_evidence else ''}
<h2>6. Summary and Recommendations</h2><ul>{''.join(f'<li>{html.escape(str(item))}</li>' for item in data['recommendations'])}</ul>
<div class="notice"><b>Limitations</b><ul>{''.join(f'<li>{html.escape(str(item))}</li>' for item in data['limitations'])}</ul></div>
<h2>7. Disclaimer</h2><p>{html.escape(data['disclaimer'])}</p>
<h2>8. Appendix: Vulnerability Reference Catalogue</h2>
<table><thead><tr><th>Vulnerability Name</th><th>Severity</th><th>Detected</th><th>Impact Scope</th></tr></thead>
<tbody>{catalogue_rows}</tbody></table>
<footer>Generated by the LLM-BSCVM v13 Smart Contract Security Audit Studio.</footer>
</body></html>"""


# Xuất báo cáo PDF bằng reportlab (A4, bảng, khối mã, chân trang số trang). Không có reportlab -> trả (False, lý do).
def render_pdf(data: dict[str, Any], output_path: Path) -> tuple[bool, str]:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            PageBreak,
            Paragraph,
            Preformatted,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError:
        return False, "reportlab is not installed"

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], textColor=colors.HexColor("#7b1f1f")))
    styles.add(ParagraphStyle(name="CodeSmall", parent=styles["Code"], fontSize=5.8, leading=7.2))
    styles.add(
        ParagraphStyle(
            name="CodeCell",
            parent=styles["Code"],
            fontName="Courier",
            fontSize=5.2,
            leading=6.3,
            leftIndent=0,
            rightIndent=0,
            firstLineIndent=0,
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            name="CodeHeader",
            parent=styles["BodyText"],
            fontSize=7,
            leading=8,
            textColor=colors.white,
            alignment=1,
        )
    )
    story: list[Any] = [Paragraph(data["title"], styles["ReportTitle"]), Spacer(1, 6 * mm)]

    # Kiểu bảng chung của PDF: dòng tiêu đề nền xanh chữ trắng, lưới xám, cỡ chữ font_size.
    def header_table_style(font_size: float, *, valign_top: bool = True) -> list[tuple]:
        style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#26394d")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ]
        if valign_top:
            style.append(("VALIGN", (0, 0), (-1, -1), "TOP"))
        return style

    # Bảng finding (mục 4.3 / 4.4 / 4.5): ID, tên lỗ hổng, unit, dòng, mức độ; rỗng thì một dòng empty_label.
    def findings_table(items: list[dict[str, Any]], empty_label: str) -> Table:
        rows = [["ID", "Vulnerability", "Unit", "Line", "Severity"]]
        rows.extend(
            [
                str(item.get("swc") or "-"),
                str(item.get("title") or "-"),
                item.get("unit_id") or "contract",
                str(item.get("line") or "-"),
                str(item.get("severity") or "Unknown"),
            ]
            for item in items
        )
        if len(rows) == 1:
            rows.append(["-", empty_label, "-", "-", "-"])
        table = Table(rows, repeatRows=1, colWidths=[22 * mm, 60 * mm, 45 * mm, 18 * mm, 25 * mm])
        table.setStyle(TableStyle(header_table_style(7.5)))
        return table

    # Thêm tiêu đề mục vào PDF.
    def heading(text: str, level: int = 1) -> None:
        story.append(Paragraph(html.escape(text), styles[f"Heading{min(level, 3)}"]))

    # Thêm đoạn văn (đã escape) vào PDF.
    def paragraph(text: Any) -> None:
        story.append(Paragraph(html.escape(str(text or "")), styles["BodyText"]))
        story.append(Spacer(1, 2 * mm))

    # Tạo ô chứa mã (Preformatted, tự xuống dòng) cho bảng PDF.
    def code_cell(text: str) -> Paragraph:
        wrapped = textwrap.wrap(
            text,
            width=72,
            replace_whitespace=False,
            drop_whitespace=False,
        ) or [""]
        escaped_parts = [
            (html.escape(part).replace(" ", "&#160;").replace("\t", "&#160;&#160;&#160;&#160;") or "&#160;")
            for part in wrapped
        ]
        return Paragraph("<br/>".join(escaped_parts), styles["CodeCell"])

    # Tạo bảng so sánh mã gốc / mã đã vá trong PDF (tô nền dòng thay đổi).
    def code_compare_table(rows: list[dict[str, Any]]) -> Table:
        table_rows = [
            [
                Paragraph("Source Code (Sample)", styles["CodeHeader"]),
                Paragraph("Fixed Source Code", styles["CodeHeader"]),
            ]
        ]
        table_rows.extend([code_cell(row["source"]), code_cell(row["fixed"])] for row in rows)
        table = Table(table_rows, repeatRows=1, colWidths=[84 * mm, 84 * mm], splitByRow=1)
        table_style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#26394d")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#bcc5cf")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]
        for index, row in enumerate(rows, 1):
            if row["fixed_changed"]:
                table_style.append(("BACKGROUND", (1, index), (1, index), colors.HexColor("#fff2a8")))
        table.setStyle(TableStyle(table_style))
        return table

    heading("1. Contract Information")
    metadata = [
        ["Analyzed Object", data["analyzed_object"]],
        ["Source File", data["file_name"]],
        ["Source SHA-256", data["source_sha256"]],
        ["Detection Result", data["detection_result"]],
        ["Analysis Status", data["analysis_status"]],
        ["Fused Score", f"{data['score']:.3f}"],
        ["Audit Time", data["audit_time"]],
    ]
    table = Table(metadata, colWidths=[40 * mm, 130 * mm])
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.extend([table, Spacer(1, 4 * mm)])
    heading("2. Executive Summary")
    paragraph(data["executive_summary"])
    heading("3. Methodology")
    paragraph(
        "Static analysis, local SWC TF-IDF retrieval, fine-tuned neural vulnerability detection, "
        "generative repair agents, solc compilation, behavior-preservation checks, "
        "and deterministic reporting."
    )
    phase_table = [["Phase", "Location", "Status"], *[[r["phase"], r["location"], r["status"]] for r in data["phase_rows"]]]
    table = Table(phase_table, repeatRows=1, colWidths=[92 * mm, 35 * mm, 43 * mm])
    table.setStyle(TableStyle(header_table_style(8, valign_top=False)))
    story.extend([table, Spacer(1, 4 * mm)])
    heading("4. Findings")
    heading("4.1 Vulnerability Statistics", 2)
    paragraph(f"{len(data['findings'])} confirmed finding(s); highest risk: {data['risk_level']}.")
    heading("4.2 Vulnerability Severity Distribution", 2)
    dist = data["distribution"]
    sev_cols = _severity_columns(dist)
    col_w = 170.0 / len(sev_cols) * mm
    severity_table = [
        sev_cols,
        [dist.get(s, 0) for s in sev_cols],
    ]
    table = Table(severity_table, colWidths=[col_w] * len(sev_cols))
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story.extend([table, Spacer(1, 4 * mm)])

    heading("4.3 Confirmed Findings Summary", 2)
    story.extend([findings_table(data["findings"], "No confirmed findings"), Spacer(1, 4 * mm)])
    heading("4.4 Candidate Findings", 2)
    story.extend([
        findings_table(data.get("candidate_findings", []), "No candidate findings below threshold"),
        Spacer(1, 4 * mm),
    ])
    heading("4.5 Findings Dismissed by the Advisor", 2)
    story.extend([findings_table(data.get("dismissed_findings", []), "No finding was dismissed by the Advisor"), Spacer(1, 2 * mm)])
    for item in data.get("dismissed_findings", []):
        paragraph(f"{item.get('finding_id')}: {item.get('reason')}")
    story.append(PageBreak())
    heading("5. Detailed Analysis")
    heading("5.1 Contract Name", 2)
    paragraph(data["contract_name"])
    heading("5.2 Source Code", 2)
    wrapped_source = "\n".join(textwrap.fill(line, width=118, replace_whitespace=False) for line in data["original_code"].splitlines())
    story.extend([Preformatted(wrapped_source, styles["CodeSmall"]), Spacer(1, 3 * mm)])
    heading("5.3 Repair Suggestion", 2)
    if not data["findings"]:
        paragraph("No confirmed vulnerability was available for automated remediation.")
    for index, item in enumerate(data["findings"], 1):
        heading(f"{index}. {item['title']} ({item['swc']}) - {item['severity']}", 3)
        paragraph(f"Root cause: {item.get('root_cause') or 'Not provided'}")
        paragraph(f"Impact: {item.get('impact') or 'Requires manual analysis'}")
        if _cvss_text(item):
            paragraph(f"CVSS: {_cvss_text(item)}")
        for step_index, step in enumerate(item.get("repair_steps", []), 1):
            paragraph(f"{step_index}. {step}")
        paragraph(f"Patch verification: {_verification_text(item)}")
    heading("5.4 Fixed Code", 2)
    paragraph(f"Patch verification verdict: {data['patch_verdict']}")
    code_comparison = build_code_comparison_rows(data["original_code"], data["fixed_code"])
    story.extend([code_compare_table(code_comparison), Spacer(1, 3 * mm)])
    heading("5.5 Verification Results", 2)
    paragraph(f"Fix attempts: {_attempt_text(data)}")
    verification_table = [["Channel", "Result", "Verdict", "Details"]]
    verification_table.extend(
        [
            row["component"],
            row["result"],
            row["verdict"],
            Paragraph(html.escape(row["detail"]), styles["BodyText"]),
        ]
        for row in verification_channel_rows(data)
    )
    table = Table(
        verification_table,
        repeatRows=1,
        colWidths=[35 * mm, 20 * mm, 30 * mm, 85 * mm],
    )
    table_style = header_table_style(7.5)
    for row_index, row in enumerate(verification_channel_rows(data), 1):
        color = colors.HexColor("#28633b" if row["result"] == "PASS" else "#9e1d1d")
        table_style.append(("TEXTCOLOR", (1, row_index), (1, row_index), color))
    table.setStyle(TableStyle(table_style))
    story.extend([table, Spacer(1, 3 * mm)])
    paragraph(
        f"Compilation: {data['tool_verification'].get('compile', {}).get('status', 'not run')}; "
        f"Slither verdict: {data['tool_verification'].get('policy_verdict', 'not run')}; "
        f"Slither: {data['tool_verification'].get('slither', {}).get('status', 'not run')}; "
        f"Behavior preservation: "
        f"{data['tool_verification'].get('behavioral', {}).get('status', 'not run')}."
    )
    for finding in data["tool_verification"].get("behavioral", {}).get("findings", []):
        paragraph(f"- {finding.get('evidence') or finding.get('check')}")
    heading("6. Summary and Recommendations")
    for item in data["recommendations"]:
        paragraph(f"- {item}")
    heading("Limitations", 2)
    for item in data["limitations"]:
        paragraph(f"- {item}")
    heading("7. Disclaimer")
    paragraph(data["disclaimer"])

    heading("8. Appendix: Vulnerability Reference Catalogue")
    cat_rows = [["Vulnerability", "Severity", "Detected", "Scope"]]
    for row in data.get("reference_table", []):
        cat_rows.append([
            f"{row['name']} ({row['abbr']})",
            row["severity"],
            "✓" if row.get("detected") else "-",
            row["impact"][:70] + "..." if len(row["impact"]) > 70 else row["impact"],
        ])
    table = Table(cat_rows, repeatRows=1, colWidths=[42 * mm, 20 * mm, 18 * mm, 90 * mm])
    table.setStyle(TableStyle(header_table_style(7)))
    story.extend([table, Spacer(1, 4 * mm)])

    # Vẽ chân trang mỗi trang PDF: run_id và số trang.
    def footer(canvas, document) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(20 * mm, 10 * mm, f"Run {data['run_id']}")
        canvas.drawRightString(190 * mm, 10 * mm, f"Page {document.page}")
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return True, ""


# Ghi 3 file báo cáo vào thư mục run: phase6_audit_report.md, .html, .pdf; trả về đường dẫn (pdf=None nếu lỗi).
# Gọi bởi: orchestrator.run_pipeline (lượt 2).
def write_reports(data: dict[str, Any], run_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(run_dir)
    markdown_text = render_markdown(data)
    markdown_path = write_text(output_dir / "phase6_audit_report.md", markdown_text)
    html_path = write_text(output_dir / "phase6_audit_report.html", render_html(data))
    pdf_path = output_dir / "phase6_audit_report.pdf"
    pdf_ok, pdf_error = render_pdf(data, pdf_path)
    return {
        "markdown": str(markdown_path),
        "html": str(html_path),
        "pdf": str(pdf_path) if pdf_ok else None,
        "pdf_error": pdf_error or None,
    }
