"""Phase 6 - báo cáo kiểm toán tất định: dựng một dict dữ liệu báo cáo rồi xuất Markdown, HTML, PDF (reportlab)."""

import html
import textwrap
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

from E2E.common.io_utils import write_text
from E2E.common.regex_patterns import RE_CALL_OPTIONS_PREFIX
from E2E.config import RISK_ORDER
from E2E.phase6_reporter.vulnerability_reference import VULNERABILITY_REFERENCE

DEFAULT_RECOMMENDATIONS = [
    "Apply and manually review every accepted patch before deployment.",
    "Pin a supported Solidity compiler version and run compilation in CI.",
    "Run Slither and the project tests after every security-sensitive change.",
    "Use checks-effects-interactions and established OpenZeppelin guards for external calls.",
    "Repeat the audit after dependency, privilege, or business-logic changes.",
]
CHANNEL_LABELS = {"slither": "Slither", "compile": "Compile", "static": "Static Analysis", "rag": "RAG",
                  "llm": "Neural Detector"}
# Trạng thái tự kiểm của vòng sửa -> câu mô tả (apply_verification_to_report ghi đè theo kết luận Phase 5).
SELF_CHECK_TEXT = {
    "PASS": "Repair-loop self-check passed (compiler, neural detector, static/RAG re-run)",
    "FLAGGED": "Repair-loop self-check: the neural detector still flags the patched unit",
    "NOT_PATCHED": "Not patched",
}
LIMITATIONS = [
    "SWC Registry source material is no longer actively maintained and is not exhaustive.",
    "The fine-tuned classifier is a binary detector; SWC attribution comes from local static/RAG evidence.",
    "Security tools can disagree or time out; manual review and project tests remain required.",
]
DISCLAIMER = ("This report represents a best-effort automated security review. It cannot guarantee that all "
              "vulnerabilities, economic attacks, integration risks, or business-logic defects were identified.")
_DETECTOR_LOCATION = "GPU Server (RTX 4070 Ti SUPER)"
_AGENT_ENGINE = "Qwen2.5-Coder-14B-Instruct v13"


# Định dạng lại mã Solidity thành dòng ổn định (mỗi câu lệnh một dòng, thụt lề theo { }) cho bảng so sánh mã gốc | mã vá.
def format_solidity_for_comparison(code: str) -> list[str]:
    lines, current = [], []
    state = {"indent": 0, "pending_space": False, "content": False}
    depths = {"(": 0, "[": 0, "{": 0}
    quote, line_comment, block_comment, index = "", False, False, 0

    # Đẩy dòng đang gom vào kết quả (kèm thụt lề hiện tại); blank=True thì giữ một dòng trống.
    def emit(blank: bool = False) -> None:
        text = "".join(current).strip()
        if text:
            lines.append("    " * state["indent"] + text)
        elif blank and lines and lines[-1] != "":
            lines.append("")
        current.clear()
        state["pending_space"] = False

    # Thêm một ký tự vào dòng đang gom (chèn một dấu cách nếu trước đó có khoảng trắng).
    def append(char: str) -> None:
        if state["pending_space"] and current and not current[-1].endswith((" ", "\t")):
            current.append(" ")
        state["pending_space"], state["content"] = False, True
        current.append(char)

    while index < len(code):
        char, pair = code[index], code[index:index + 2]
        if line_comment:
            if char == "\n":
                emit()
                line_comment, state["content"] = False, False
            else:
                current.append(char)
        elif block_comment:
            current.append(char)
            if pair == "*/":
                current.append("/")
                index += 1
                block_comment = False
                emit()
        elif quote:
            append(char)
            if char == "\\" and index + 1 < len(code):
                current.append(code[index + 1])
                index += 1
            elif char == quote:
                quote = ""
        elif pair in {"//", "/*"}:
            if "".join(current).strip():
                emit()
            current.extend(("/", "/", " ") if pair == "//" else ("/", "*"))
            state["content"], index = True, index + 1
            line_comment, block_comment = pair == "//", pair == "/*"
            while line_comment and index + 1 < len(code) and code[index + 1] in " \t":
                index += 1
        elif char in "\"'":
            append(char)
            quote = char
        elif char.isspace():
            if char != "\n":
                state["pending_space"] = True
            else:
                if "".join(current).strip():
                    emit()
                elif not state["content"]:
                    emit(blank=True)
                state["content"] = False
        elif char in "([":
            append(char)
            depths[char] += 1
        elif char in ")]":
            append(char)
            opening = "(" if char == ")" else "["
            depths[opening] = max(0, depths[opening] - 1)
        elif char == "{":
            in_expression = depths["{"] or depths["("] or depths["["] \
                or RE_CALL_OPTIONS_PREFIX.search("".join(current).rstrip())
            append(char)
            if in_expression:
                depths["{"] += 1
            else:
                emit()
                state["indent"] += 1
        elif char == "}" and depths["{"]:
            append(char)
            depths["{"] -= 1
        elif char == "}":
            emit()
            state["indent"] = max(0, state["indent"] - 1)
            current.append("}")
            state["content"] = True
            lookahead = index + 1
            while lookahead < len(code) and code[lookahead] in " \t":
                lookahead += 1
            if lookahead < len(code) and code[lookahead] == ";":
                current.append(";")
                index = lookahead
            emit()
        else:
            append(char)
            if char == ";" and not (depths["("] or depths["["] or depths["{"]):
                emit()
        index += 1
    emit()
    while lines and lines[-1] == "":
        lines.pop()
    return lines


# Các dòng so sánh song song mã gốc / mã vá (SequenceMatcher), đánh dấu dòng bị thay / thêm ở bản vá.
def build_code_comparison_rows(original_code: str, fixed_code: str) -> list[dict]:
    source, fixed = format_solidity_for_comparison(original_code), format_solidity_for_comparison(fixed_code)
    rows = []
    for tag, s_start, s_end, f_start, f_end in SequenceMatcher(None, source, fixed, autojunk=False).get_opcodes():
        source_chunk, fixed_chunk = source[s_start:s_end], fixed[f_start:f_end]
        for index in range(max(len(source_chunk), len(fixed_chunk), 1)):
            rows.append({"source": source_chunk[index] if index < len(source_chunk) else "",
                         "fixed": fixed_chunk[index] if index < len(fixed_chunk) else "",
                         "fixed_changed": tag in {"replace", "insert"} and index < len(fixed_chunk)})
    return rows or [{"source": "", "fixed": "", "fixed_changed": False}]


# Kết quả một phase (advisor / assessor / fixer / ...) trong server_result.
def _phase(remote: dict, name: str) -> dict:
    return remote.get("phases", {}).get(name, {}) if remote else {}


# Mục (suggestion / assessment / verification) ứng với một finding: theo finding_id, không có thì theo SWC + unit.
def _match(items: list[dict], finding: dict) -> dict:
    same_id = lambda item: str(item.get("finding_id") or "") == str(finding.get("finding_id") or "")
    same_place = lambda item: item.get("swc") == finding.get("swc") \
        and (not item.get("unit_id") or item.get("unit_id") == finding.get("unit_id"))
    return next((item for item in items if same_id(item) or same_place(item)), {})


# Các dòng bảng "Verification Results" (mục 5.5): mỗi kênh -> PASS / FAIL / ERROR / SKIPPED + chi tiết.
def verification_channel_rows(data: dict) -> list[dict]:
    channels = data.get("verification_channels", {})
    rows = []
    ordered = [key for key in CHANNEL_LABELS if key in channels] + [key for key in channels if key not in CHANNEL_LABELS]
    for key in ordered:
        component = channels[key]
        findings = component.get("findings", [])
        detail = str(component.get("reason") or component.get("error") or "").strip() \
            or (f"{len(findings)} remaining finding(s)" if findings else "No remaining vulnerability signal.")
        verdict = str(component.get("verdict") or "Not run")
        passed = component.get("passed") if component.get("passed") is not None else verdict in {"PASS", "WARN"}
        result = "SKIPPED" if verdict == "SKIPPED" else "PASS" if passed else "ERROR" if verdict == "ERROR" else "FAIL"
        rows.append({"component": CHANNEL_LABELS.get(key, key.replace("_", " ").title()), "result": result,
                     "verdict": verdict, "detail": detail})
    return rows


# Bản vá được chấp nhận: hàm đã vá -> "Fixed"; detector vẫn báo (FLAGGED) -> ghi "detector disagreement"; thêm câu kết luận.
def _apply_accepted_verdict(report_data: dict, verification: dict, verdict: str) -> None:
    probabilities = {item.get("unit_id"): item.get("vulnerability_probability")
                     for item in verification.get("fixed_unit_results", [])}
    threshold = (verification.get("fixed_detection") or {}).get("threshold")
    disagreeing = []
    for finding in report_data.get("findings", []):
        if finding.get("self_check_status") not in {"PASS", "FLAGGED"}:
            continue
        finding["verification_status"], finding["residual_risk"] = f"Fixed: {verdict}", "None"
        if finding.get("self_check_status") == "FLAGGED":
            probability = probabilities.get(finding.get("unit_id"))
            detail = f"p={float(probability):.4f} >= threshold {float(threshold):.2f}" \
                if probability is not None and threshold is not None else "above threshold"
            finding["verification_status"] += \
                f"; the neural detector still flags the patched unit ({detail}) - detector disagreement"
            finding["residual_risk"] = ("Low: compiler, behavior checks and Slither passed, but the neural "
                                        "detector disagrees; manual review is recommended")
            if finding.get("unit_id") not in disagreeing:
                disagreeing.append(finding.get("unit_id"))
    summary = str(report_data.get("executive_summary") or "").rstrip()
    sentence = f" Final verification: {verdict}."
    if disagreeing:
        sentence += (" The neural detector still flags " + ", ".join(str(unit) for unit in disagreeing)
                     + " after the patch (detector disagreement); the deterministic checks decide acceptance.")
    if sentence.strip() not in summary:
        report_data["executive_summary"] = summary + sentence


# Đưa kết quả Phase 5 vào báo cáo; bản vá bị từ chối / không kết luận -> viết lại tóm tắt, không trình bày như đã sửa.
def apply_verification_to_report(report_data: dict, verification: dict) -> None:
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
        return _apply_accepted_verdict(report_data, verification, verdict)
    if not verdict.startswith(("Patch rejected", "Patch inconclusive", "Inconclusive")):
        return
    count = len(report_data.get("findings", []))
    regressions = len(verification.get("behavioral_verification", {}).get("findings", []))
    detail = f" Deterministic behavioral checks found {regressions} regression(s)." if regressions else ""
    no_patch = any(marker in verdict for marker in ("no patch was generated", "patch execution incomplete",
                                                     "no function-level repair target", "no effective patch"))
    if no_patch:
        report_data["executive_summary"] = (f"The audit identified {count} finding(s). No patch was verified: "
                                            f"{verdict}. The findings remain open and require manual review.")
    else:
        outcome = "rejected" if verdict.startswith("Patch rejected") else "not accepted"
        report_data["executive_summary"] = (
            f"The audit identified {count} finding(s). A patch was generated but {outcome} "
            f"by deterministic verification: {verdict}.{detail} The patch must not be deployed; "
            "the original finding remains open until a behavior-preserving patch passes compilation, "
            "Slither, and project tests.")
    for finding in report_data.get("findings", []):
        if finding.get("self_check_status") != "NOT_PATCHED":
            rejected = verdict.startswith("Patch rejected")
            finding["verification_status"] = "Rejected by deterministic verifier" if rejected else verdict
            finding["residual_risk"] = verdict
    rejection = ("Do not deploy the generated patch; preserve the contract's intended transfers and behavior "
                 "while addressing the root cause.")
    if not no_patch and rejection not in report_data.get("recommendations", []):
        report_data["recommendations"] = [rejection, *report_data.get("recommendations", [])]


# Dòng CVSS của một finding: điểm (mức, nguồn điểm, mức lúc phát hiện nếu khác); thứ tự ưu tiên sửa.
def _cvss_text(finding: dict) -> str:
    if finding.get("cvss_score") is None:
        return ""
    source = {"llm": "LLM-selected metrics", "swc_default": "SWC default vector"}.get(finding.get("cvss_source") or "")
    detected = finding.get("detected_severity")
    differs = detected and detected != finding.get("severity")
    details = [str(finding.get("severity") or ""), source, f"detected as {detected}" if differs else ""]
    text = f"{finding['cvss_score']} ({', '.join(item for item in details if item)})"
    return text + (f"; repair priority #{finding['repair_priority']}" if finding.get("repair_priority") else "")


# Dòng kết luận Advisor của finding: Confirmed / Uncertain / Disputed (+ lý do giữ lại); không có kết luận LLM -> "".
def _advisor_verdict_text(finding: dict) -> str:
    if finding.get("advisor_verdict") == "disputed":
        reason = finding.get("advisor_kept_because") or "the dismissal did not pass the check"
        return f"Disputed - judged a false positive, kept because {reason}"
    return {"confirmed": "Confirmed", "uncertain": "Uncertain - depends on code that is not shown; kept for manual review"
            }.get(finding.get("advisor_verdict") or "", "")


# Dòng kết quả kiểm chứng của một finding: trạng thái + rủi ro còn lại.
def _verification_text(finding: dict) -> str:
    text = str(finding.get("verification_status") or "Not verified")
    return text + (f"; residual risk: {finding['residual_risk']}" if finding.get("residual_risk") else "")


# Tên hiển thị của finding (SWC-103 luôn là "Floating Pragma").
def _display_title(finding: dict) -> str:
    return "Floating Pragma" if finding.get("swc") == "SWC-103" else str(finding.get("title") or "")


# Finding của báo cáo -> (findings, dismissed): sau GPU = confirmed_findings (đã bỏ báo nhầm), trước GPU = của fusion.
def select_report_findings(fused: dict, remote: dict | None) -> tuple[list[dict], list[dict]]:
    detector = _phase(remote or {}, "detector")
    if "confirmed_findings" not in detector:
        return list(fused.get("findings", [])), []
    return list(detector.get("confirmed_findings", [])), list(detector.get("dismissed_findings", []))


# Một finding của báo cáo: ghép gợi ý Advisor, CVSS Assessor, trạng thái tự kiểm (mức độ = mức CVSS nếu có).
def _report_finding(finding: dict, advice: dict, assessment: dict, verification: dict, has_remote: bool) -> dict:
    self_check = verification.get("status") or ("NOT_PATCHED" if has_remote else "")
    return {
        **finding,
        "severity": assessment.get("severity") or finding.get("severity", "Unknown"),
        "detected_severity": finding.get("severity", "Unknown"),
        "root_cause": advice.get("root_cause") or finding.get("description") or "",
        "impact": advice.get("impact") or finding.get("impact") or assessment.get("justification") or "",
        "cvss_score": assessment.get("cvss_score"), "cvss_source": assessment.get("cvss_source"),
        "repair_priority": assessment.get("repair_priority"),
        "repair_steps": advice.get("repair_steps") or ([finding["remediation"]] if finding.get("remediation") else []),
        "advisor_verdict": advice.get("advisor_verdict") or "",
        "advisor_kept_because": advice.get("kept_because") or "",
        "self_check_status": self_check or None,
        "verification_status": SELF_CHECK_TEXT.get(self_check, self_check) if self_check
        else "Not verified (preview before the GPU phases)",
        "residual_risk": verification.get("residual_risk") or "",
    }


# Bảng trạng thái 8 phase (mục 3 của báo cáo).
def _phase_rows(request: dict, remote: dict, tool_verification: dict) -> list[dict]:
    static, rag = request["local_phase1"]["static"], request["local_phase1"]["rag"]
    rows = [
        ("1A Static detector", "Local", static.get("status", "unknown"),
         static.get("engine", "Regex / AST Rule Engine")),
        ("1B RAG detector", "Local", rag.get("status", "unknown"), rag.get("engine", "SWC TF-IDF (11,869 cases)")),
        ("1C Qwen2.5-Coder-14B detector (v13)", _DETECTOR_LOCATION, _phase(remote, "detector").get("status", "pending"),
         "Qwen2.5-Coder-14B-Instruct v13 Quality-First (LoRA r=64, alpha=128)"),
        ("2 Advisor", _DETECTOR_LOCATION, _phase(remote, "advisor").get("status", "pending"), _AGENT_ENGINE),
        ("3 Risk assessor", _DETECTOR_LOCATION, _phase(remote, "assessor").get("status", "pending"), _AGENT_ENGINE),
        ("4 Fixer", _DETECTOR_LOCATION, _phase(remote, "fixer").get("status", "pending"),
         f"{_AGENT_ENGINE} + deterministic patch apply"),
        ("5 Verifier", "Local", tool_verification.get("status", "pending"),
         "solc compiler, Slither, AST behavioral invariants"),
        ("6 Reporter", "Local", "ok", "deterministic audit report renderer"),
    ]
    return [{"phase": phase, "location": location, "status": status, "engine": engine}
            for phase, location, status, engine in rows]


# [HÀM CHÍNH PHASE 6 - dữ liệu] Gom thông tin contract, kết luận fusion, finding (+ Advisor/Assessor), phase, mã, tham chiếu.
def build_report_data(request: dict, fused: dict, remote: dict | None, tool_verification: dict) -> dict:
    remote = remote or {}
    advisor = _phase(remote, "advisor").get("suggestions", [])
    assessor = _phase(remote, "assessor").get("assessments", [])
    verifier, fixer = _phase(remote, "verifier").get("verifications", []), _phase(remote, "fixer")
    selected, dismissed = select_report_findings(fused, remote)
    findings = [_report_finding(f, _match(advisor, f), _match(assessor, f), _match(verifier, f), bool(remote))
                for f in selected]
    for finding in findings:
        finding["title"] = _display_title(finding)
    units = request.get("code_units", [])
    names = ", ".join(unit["unit_id"] for unit in units[:8]) + (", ..." if len(units) > 8 else "")
    if remote:
        applied = sum(1 for item in fixer.get("patch_outcomes", []) if item.get("status") == "applied")
        summary = (f"Audit completed using the GPU server (Qwen2.5-Coder-14B v13) for run {request['run_id']}. "
                   f"{len(findings)} confirmed finding(s), {len(dismissed)} dismissed by the Advisor, "
                   f"{applied} patch(es) applied.")
    else:
        summary = (f"The audit result is {fused.get('verdict', 'Inconclusive')} with a fused confidence "
                   f"score of {float(fused.get('score') or 0.0):.3f}. "
                   f"The pipeline identified {len(findings)} confirmed finding(s). "
                   f"Compilation status: {tool_verification.get('compile', {}).get('status', 'not run')}; "
                   f"Slither reported {len(tool_verification.get('slither', {}).get('findings', []))} "
                   "detector finding(s) "
                   f"(verdict: {tool_verification.get('policy_verdict', 'not run')}).")
    candidates = fused.get("candidate_findings", []) or _phase(remote, "detector").get("candidate_findings", []) or []
    contract_names = sorted({unit.get("contract_name") for unit in units if unit.get("contract_name")})
    if findings:
        risk_level = max((f.get("severity", "Unknown") for f in findings),
                         key=lambda severity: RISK_ORDER.get(severity, 0))
    else:
        risk_level = "Unknown" if fused.get("risk_level") == "Unknown" else "None"
    detected = lambda keys: any(key.lower() in str(f.get(field) or "").lower()
                                for f in findings for key in keys for field in ("swc", "title", "unit_id"))
    source = request["source"]
    return {
        "title": "Smart Contract Audit Report", "run_id": request["run_id"],
        "audit_time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "file_name": source["file_name"], "source_sha256": source["sha256"],
        "analyzed_object": names or source["file_name"],
        "contract_name": ", ".join(contract_names) or Path(source["file_name"]).stem,
        "detection_result": fused.get("verdict", "Inconclusive"),
        "analysis_status": fused.get("status", "inconclusive"),
        "score": float(fused.get("score") or 0.0), "risk_level": risk_level, "executive_summary": summary,
        "findings": findings, "candidate_findings": [{**item, "title": _display_title(item)} for item in candidates],
        "dismissed_findings": [{
            "finding_id": item.get("finding_id"), "swc": item.get("swc"), "title": item.get("title"),
            "unit_id": item.get("unit_id"), "line": item.get("line"), "severity": item.get("severity", "Unknown"),
            "confidence": item.get("confidence"), "sources": item.get("sources", []),
            "reason": item.get("dismissal_reason") or "Dismissed by the Advisor as a false positive.",
        } for item in dismissed],
        "distribution": Counter(item.get("severity", "Unknown") for item in findings),
        "phase_rows": _phase_rows(request, remote, tool_verification),
        "original_code": source["code"], "fixed_code": fixer.get("fixed_code") or source["code"],
        "patch_verdict": _phase(remote, "verifier").get("overall_verdict", "Not complete"),
        "reference_table": [{"name": entry["name"], "abbr": entry["abbr"], "severity": entry["severity"],
                             "detected": detected(entry.get("keys", [])), "impact": entry["impact"]}
                            for entry in VULNERABILITY_REFERENCE],
        "tool_verification": tool_verification, "verification_channels": {},
        "attempts_count": _phase(remote, "verifier").get("attempts_count"),
        "max_attempts": _phase(remote, "verifier").get("max_attempts"),
        "recommendations": list(DEFAULT_RECOMMENDATIONS), "limitations": list(LIMITATIONS), "disclaimer": DISCLAIMER,
    }


# Chuỗi "số lần sửa đã chạy / tối đa" cho mục 5.5 (vd "3/3 attempt(s)").
def _attempt_text(data: dict) -> str:
    count, maximum = data.get("attempts_count"), data.get("max_attempts")
    return "not run" if not count else f"{count}/{maximum} attempt(s)" if maximum else f"{count} attempt(s)"


# Cột bảng phân bố mức độ (mục 4.2): Critical / High / Medium / Low + mức khác nếu có finding.
def _severity_columns(distribution: Counter) -> list[str]:
    standard = ["Critical", "High", "Medium", "Low"]
    return standard + [level for level in distribution if level not in standard and distribution[level] > 0]


# Các ô của một dòng bảng finding: ID, tên, unit, dòng, mức độ, confidence, nguồn.
def _row_cells(item: dict) -> list[str]:
    return [str(item.get("swc", "-")), str(item.get("title", "-")), str(item.get("unit_id") or "contract"),
            str(item.get("line") or "-"), str(item.get("severity", "Unknown")),
            f"{float(item.get('confidence') or 0):.3f}",
            ", ".join(item.get("sources", []))]


# Một dòng <tr> của bảng finding trong HTML (badge = tô màu mức độ, reason = thêm cột lý do của Advisor).
def _html_finding_row(item: dict, badge: bool = True, reason: bool = False) -> str:
    swc, title, unit, line, severity, confidence, sources = _row_cells(item)
    severity_html = html.escape(severity)
    if badge:
        severity_html = f"<span class='severity {severity.lower()}'>{severity_html}</span>"
    cells = [f"<td>{html.escape(swc)}</td>", f"<td>{html.escape(title)}</td>", f"<td>{html.escape(unit)}</td>",
             f"<td style='text-align:center;'>{html.escape(line)}</td>", f"<td>{severity_html}</td>",
             f"<td>{confidence}</td>", f"<td>{html.escape(sources)}</td>"]
    if reason:
        cells.append(f"<td>{html.escape(str(item.get('reason') or ''))}</td>")
    return "<tr>" + "".join(cells) + "</tr>"


# Bảng so sánh mã dạng HTML trong Markdown (dòng thay đổi bọc <mark>).
def _markdown_code_compare(rows: list[dict]) -> str:
    source = "<br>".join(html.escape(row["source"]) or "&nbsp;" for row in rows)
    fixed = "<br>".join(f"<mark>{html.escape(row['fixed']) or '&nbsp;'}</mark>" if row["fixed_changed"]
                        else html.escape(row["fixed"]) or "&nbsp;" for row in rows)
    return "\n".join(['<table class="code-comparison">',
                      "<thead><tr><th>Source Code (Sample)</th><th>Fixed Source Code</th></tr></thead>",
                      "<tbody><tr>", f"<td><pre><code>{source}</code></pre></td>",
                      f"<td><pre><code>{fixed}</code></pre></td>", "</tr></tbody>", "</table>"])


# Các dòng mã thành <span> cho HTML (dòng thay đổi có class "changed").
def _html_code_lines(rows: list[dict], key: str, highlight_key: str | None = None) -> str:
    return "\n".join(f"<span class='code-line{' changed' if highlight_key and row.get(highlight_key) else ''}'>"
                     f"{html.escape(row.get(key, '')) or ' '}</span>" for row in rows)


# Báo cáo Markdown: 1 thông tin, 2 tóm tắt, 3 phương pháp, 4 findings, 5 phân tích, 6 khuyến nghị, 7 miễn trừ, 8 tham chiếu.
def render_markdown(data: dict) -> str:
    findings, dist, tool = data["findings"], data["distribution"], data["tool_verification"]
    columns = _severity_columns(dist)
    table_row = lambda cells: "| " + " | ".join(cells) + " |"
    lines = [
        "# Smart Contract Audit Report", "", "## 1. Contract Information", "",
        f"- **Run ID:** `{data['run_id']}`", f"- **Analyzed Object:** {data['analyzed_object']}",
        f"- **Source File:** `{data['file_name']}`", f"- **Source SHA-256:** `{data['source_sha256']}`",
        f"- **Detection Result:** **{data['detection_result']}**", f"- **Analysis Status:** {data['analysis_status']}",
        f"- **Fused Score:** {data['score']:.3f}", f"- **Audit Time:** {data['audit_time']}", "",
        "## 2. Executive Summary", "", data["executive_summary"], "", "## 3. Methodology", "",
        "The audit combines static rules, local SWC TF-IDF retrieval, fine-tuned neural vulnerability detection, "
        "generative repair agents, compiler checks, AST behavioral-preservation verification, "
        "and deterministic report generation.",
        "", "| Phase | Location | Status | Engine |", "| :---- | :------- | :----- | :----- |",
        *(f"| {row['phase']} | {row['location']} | {row['status']} | {row['engine']} |" for row in data["phase_rows"]),
        "", "## 4. Findings", "", "### 4.1 Vulnerability Statistics", "",
        f"- **Detection Result:** {data['detection_result']}",
        f"- **Vulnerability Count:** {len(findings)} confirmed finding(s)",
        "", "### 4.2 Vulnerability Severity Distribution", "",
        "| " + " | ".join(columns) + " |", "| " + " | ".join([":---:"] * len(columns)) + " |",
        "| " + " | ".join(str(dist.get(level, 0)) for level in columns) + " |", "",
        "### 4.3 Confirmed Findings Summary", "",
        "| ID | Vulnerability | Unit | Line | Severity | Confidence | Evidence |",
        "| :-- | :------------ | :--- | :--: | :------- | :--------: | :------- |",
        *([table_row(_row_cells(item)) for item in findings] or ["| - | No confirmed findings | - | - | - | - | - |"]),
        "", "### 4.4 Candidate Findings", "",
    ]
    if data.get("candidate_findings"):
        lines += ["| ID | Vulnerability | Unit | Line | Severity | Confidence | Sources |",
                  "| :-- | :------------ | :--- | :--: | :------- | :--------: | :------- |",
                  *(table_row(_row_cells(item)) for item in data["candidate_findings"])]
    else:
        lines.append("_No unconfirmed candidate findings below threshold._")
    lines += ["", "### 4.5 Findings Dismissed by the Advisor", ""]
    if data.get("dismissed_findings"):
        lines += ["_Raised by a detection channel, then judged a false positive by the Advisor (Phase 2). "
                  "Not counted as vulnerabilities and not repaired._", "",
                  "| ID | Vulnerability | Unit | Line | Severity | Confidence | Sources | Advisor reason |",
                  "| :-- | :------------ | :--- | :--: | :------- | :--------: | :------ | :------------- |",
                  *(table_row([*_row_cells(item), str(item.get("reason") or "").replace("|", "&#124;")
                               .replace("\n", " ")]) for item in data["dismissed_findings"])]
    else:
        lines.append("_No finding was dismissed by the Advisor._")
    lines += ["", "## 5. Detailed Analysis", "", "### 5.1 Contract Name", "", data["contract_name"], "",
              "### 5.2 Source Code", "", "```solidity", data["original_code"], "```", "",
              "### 5.3 Repair Suggestion", ""]
    if not findings:
        lines.append("No confirmed vulnerability was available for automated remediation.")
    for index, finding in enumerate(findings, 1):
        lines += [f"#### {index}. {finding['title']} ({finding['swc']}) - {finding['severity']}", "",
                  f"- **Unit:** `{finding.get('unit_id') or 'contract'}`",
                  f"- **Root cause:** {finding.get('root_cause') or 'Not provided'}",
                  f"- **Impact:** {finding.get('impact') or 'Requires manual impact analysis'}"]
        lines += [f"- **Advisor verdict:** {_advisor_verdict_text(finding)}"] if _advisor_verdict_text(finding) else []
        lines += [f"- **CVSS:** {_cvss_text(finding)}"] if _cvss_text(finding) else []
        steps = finding.get("repair_steps") or ["Perform manual review and implement the cited remediation."]
        lines += ["- **Repair steps:**", *(f"  {number}. {step}" for number, step in enumerate(steps, 1)),
                  f"- **Patch verification:** {_verification_text(finding)}", ""]
    behavioral = tool.get("behavioral", {})
    lines += [
        "### 5.4 Fixed Code", "", f"_Patch verification verdict: **{data['patch_verdict']}**_", "",
        _markdown_code_compare(build_code_comparison_rows(data["original_code"], data["fixed_code"])), "",
        "### 5.5 Verification Results", "", f"- **Fix attempts:** {_attempt_text(data)}", "",
        "| Verification channel | Result | Verdict | Details |",
        "| :------------------- | :----: | :------ | :------ |",
        *(f"| {row['component']} | **{row['result']}** | {row['verdict']} | {row['detail'].replace('|', '&#124;')} |"
          for row in verification_channel_rows(data)), "",
        f"- **Compilation:** {tool.get('compile', {}).get('status', 'not run')}",
        f"- **Slither verdict:** {tool.get('policy_verdict', 'not run')}",
        f"- **Slither verdict reason:** {tool.get('policy_reason', 'not run')}",
        f"- **Slither:** {tool.get('slither', {}).get('status', 'not run')}",
        f"- **Slither findings:** {len(tool.get('slither', {}).get('findings', []))}",
        f"- **Behavior preservation:** {behavioral.get('status', 'not run')}", "",
    ]
    if evidence := [item.get("evidence") or item.get("check") for item in behavioral.get("findings", [])
                    if item.get("evidence") or item.get("check")]:
        lines += ["- **Behavior preservation evidence:**", *(f"  - {entry}" for entry in evidence)]
    lines += ["", "## 6. Summary and Recommendations", "", *(f"- {item}" for item in data["recommendations"]),
              "", "### Limitations", "", *(f"- {item}" for item in data["limitations"]),
              "", "## 7. Disclaimer", "", data["disclaimer"], "", "",
              "## 8. Appendix: Vulnerability Reference Catalogue", "",
              "| Vulnerability Name | Severity | Detected | Impact Scope |",
              "| :----------------- | :------- | :------: | :----------- |",
              *(f"| {row['name']} ({row['abbr']}) | {row['severity']} | {'✓' if row.get('detected') else ''} "
                f"| {row['impact']} |" for row in data.get("reference_table", [])), ""]
    return "\n".join(lines)


# Báo cáo HTML (cùng nội dung với Markdown, có CSS, bảng so sánh mã tô màu dòng thay đổi).
def render_html(data: dict) -> str:
    comparison = build_code_comparison_rows(data["original_code"], data["fixed_code"])
    source_code_lines = _html_code_lines(comparison, "source")
    fixed_code_lines = _html_code_lines(comparison, "fixed", "fixed_changed")
    behavioral = data["tool_verification"].get("behavioral", {})
    behavioral_evidence = "".join(f"<li>{html.escape(str(item.get('evidence') or item.get('check') or ''))}</li>"
                                  for item in behavioral.get("findings", []))
    check_mark = '<span style="color:#10b981;font-weight:bold;">✓</span>'
    catalogue_rows = "".join(
        f"<tr><td>{html.escape(row['name'])} ({html.escape(row['abbr'])})</td>"
        f"<td><span class='severity {row['severity'].lower()}'>{html.escape(row['severity'])}</span></td>"
        f"<td style='text-align:center;'>{check_mark if row.get('detected') else '-'}</td>"
        f"<td>{html.escape(row['impact'])}</td></tr>"
        for row in data.get("reference_table", []))
    findings_rows = "".join(map(_html_finding_row, data["findings"])) \
        or "<tr><td colspan='7'>No confirmed findings</td></tr>"
    candidates_rows = "".join(map(_html_finding_row, data.get("candidate_findings", []))) \
        or "<tr><td colspan='7'>No unconfirmed candidate findings below threshold</td></tr>"
    dismissed_rows = "".join(_html_finding_row(item, badge=False, reason=True)
                             for item in data.get("dismissed_findings", [])) \
        or "<tr><td colspan='8'>No finding was dismissed by the Advisor</td></tr>"
    dist, columns = data["distribution"], _severity_columns(data["distribution"])
    distribution_html = (f"<table><tr>{''.join(f'<th>{level}</th>' for level in columns)}</tr>"
                         f"<tr>{''.join(f'<td>{dist.get(level, 0)}</td>' for level in columns)}</tr></table>")
    phase_rows = "".join(f"<tr><td>{html.escape(row['phase'])}</td><td>{row['location']}</td>"
                         f"<td>{html.escape(str(row['status']))}</td><td>{html.escape(str(row['engine']))}</td></tr>"
                         for row in data["phase_rows"])
    verification_rows = "".join(f"<tr><td>{html.escape(row['component'])}</td>"
                                f"<td class='{row['result'].lower()}'>{row['result']}</td>"
                                f"<td>{html.escape(row['verdict'])}</td><td>{html.escape(row['detail'])}</td></tr>"
                                for row in verification_channel_rows(data))
    details = ""
    for index, item in enumerate(data["findings"], 1):
        advisor = f"<p><b>Advisor verdict:</b> {html.escape(_advisor_verdict_text(item))}</p>" \
            if _advisor_verdict_text(item) else ""
        cvss = f"<p><b>CVSS:</b> {html.escape(_cvss_text(item))}</p>" if _cvss_text(item) else ""
        steps = "".join(f"<li>{html.escape(str(step))}</li>" for step in item.get("repair_steps", []))
        details += (f"<section><h4>{index}. {html.escape(item['title'])} ({item['swc']})</h4>"
                    f"<p><b>Severity:</b> {item['severity']} &nbsp; <b>Unit:</b> "
                    f"<code>{html.escape(str(item.get('unit_id') or 'contract'))}</code></p>"
                    f"<p><b>Root cause:</b> {html.escape(str(item.get('root_cause') or 'Not provided'))}</p>"
                    f"<p><b>Impact:</b> {html.escape(str(item.get('impact') or 'Requires manual analysis'))}</p>"
                    f"{advisor}{cvss}<ol>{steps}</ol>"
                    f"<p><b>Verification:</b> {html.escape(_verification_text(item))}</p></section>")
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


# Báo cáo PDF bằng reportlab (A4, bảng, khối mã, chân trang số trang); không có reportlab -> (False, lý do).
def render_pdf(data: dict, output_path: Path) -> tuple[bool, str]:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError:
        return False, "reportlab is not installed"
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], textColor=colors.HexColor("#7b1f1f")))
    styles.add(ParagraphStyle(name="CodeSmall", parent=styles["Code"], fontSize=5.8, leading=7.2))
    styles.add(ParagraphStyle(name="CodeCell", parent=styles["Code"], fontName="Courier", fontSize=5.2, leading=6.3,
                              leftIndent=0, rightIndent=0, firstLineIndent=0, wordWrap="CJK"))
    styles.add(ParagraphStyle(name="CodeHeader", parent=styles["BodyText"], fontSize=7, leading=8,
                              textColor=colors.white, alignment=1))
    story = [Paragraph(data["title"], styles["ReportTitle"]), Spacer(1, 6 * mm)]

    # Kiểu bảng chung: dòng tiêu đề nền xanh chữ trắng, lưới xám, cỡ chữ font_size (top=True: căn trên).
    def header_style(font_size: float, top: bool = True) -> list:
        return [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#26394d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), font_size)] \
            + ([("VALIGN", (0, 0), (-1, -1), "TOP")] if top else [])

    # Thêm một bảng (kèm style, lặp dòng tiêu đề khi sang trang nếu repeat) và khoảng trống phía sau vào PDF.
    def add_table(rows: list, widths: list, style: list, space: float = 4, repeat: bool = True) -> None:
        table = Table(rows, colWidths=widths, **({"repeatRows": 1} if repeat else {}))
        table.setStyle(TableStyle(style))
        story.extend([table, Spacer(1, space * mm)])

    # Thêm tiêu đề mục (cấp 1-3).
    def heading(text: str, level: int = 1) -> None:
        story.append(Paragraph(html.escape(text), styles[f"Heading{min(level, 3)}"]))

    # Thêm một đoạn văn (đã escape).
    def paragraph(text) -> None:
        story.extend([Paragraph(html.escape(str(text or "")), styles["BodyText"]), Spacer(1, 2 * mm)])

    # Bảng finding (mục 4.3 / 4.4 / 4.5): ID, tên, unit, dòng, mức độ; rỗng thì một dòng empty_label.
    def findings_table(items: list[dict], empty_label: str, space: float = 4) -> None:
        rows = [["ID", "Vulnerability", "Unit", "Line", "Severity"]] + [
            [str(item.get("swc") or "-"), str(item.get("title") or "-"), item.get("unit_id") or "contract",
             str(item.get("line") or "-"), str(item.get("severity") or "Unknown")] for item in items]
        add_table(rows if items else rows + [["-", empty_label, "-", "-", "-"]],
                  [22 * mm, 60 * mm, 45 * mm, 18 * mm, 25 * mm], header_style(7.5), space)

    # Ô chứa mã (tự xuống dòng ở 72 ký tự, giữ khoảng trắng) cho bảng so sánh mã.
    def code_cell(text: str) -> Paragraph:
        parts = textwrap.wrap(text, width=72, replace_whitespace=False, drop_whitespace=False) or [""]
        return Paragraph("<br/>".join(html.escape(part).replace(" ", "&#160;").replace("\t", "&#160;" * 4) or "&#160;"
                                      for part in parts), styles["CodeCell"])

    heading("1. Contract Information")
    metadata = [["Analyzed Object", data["analyzed_object"]], ["Source File", data["file_name"]],
                ["Source SHA-256", data["source_sha256"]], ["Detection Result", data["detection_result"]],
                ["Analysis Status", data["analysis_status"]], ["Fused Score", f"{data['score']:.3f}"],
                ["Audit Time", data["audit_time"]]]
    add_table(metadata, [40 * mm, 130 * mm],
              [("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP")], repeat=False)
    heading("2. Executive Summary")
    paragraph(data["executive_summary"])
    heading("3. Methodology")
    paragraph("Static analysis, local SWC TF-IDF retrieval, fine-tuned neural vulnerability detection, "
              "generative repair agents, solc compilation, behavior-preservation checks, and deterministic reporting.")
    add_table([["Phase", "Location", "Status"],
               *[[row["phase"], row["location"], row["status"]] for row in data["phase_rows"]]],
              [92 * mm, 35 * mm, 43 * mm], header_style(8, top=False))
    heading("4. Findings")
    heading("4.1 Vulnerability Statistics", 2)
    paragraph(f"{len(data['findings'])} confirmed finding(s); highest risk: {data['risk_level']}.")
    heading("4.2 Vulnerability Severity Distribution", 2)
    dist, columns = data["distribution"], _severity_columns(data["distribution"])
    add_table([columns, [dist.get(level, 0) for level in columns]], [170.0 / len(columns) * mm] * len(columns),
              [("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("ALIGN", (0, 0), (-1, -1), "CENTER")], repeat=False)
    heading("4.3 Confirmed Findings Summary", 2)
    findings_table(data["findings"], "No confirmed findings")
    heading("4.4 Candidate Findings", 2)
    findings_table(data.get("candidate_findings", []), "No candidate findings below threshold")
    heading("4.5 Findings Dismissed by the Advisor", 2)
    findings_table(data.get("dismissed_findings", []), "No finding was dismissed by the Advisor", space=2)
    for item in data.get("dismissed_findings", []):
        paragraph(f"{item.get('finding_id')}: {item.get('reason')}")
    story.append(PageBreak())
    heading("5. Detailed Analysis")
    heading("5.1 Contract Name", 2)
    paragraph(data["contract_name"])
    heading("5.2 Source Code", 2)
    source = "\n".join(textwrap.fill(line, width=118, replace_whitespace=False)
                       for line in data["original_code"].splitlines())
    story.extend([Preformatted(source, styles["CodeSmall"]), Spacer(1, 3 * mm)])
    heading("5.3 Repair Suggestion", 2)
    if not data["findings"]:
        paragraph("No confirmed vulnerability was available for automated remediation.")
    for index, item in enumerate(data["findings"], 1):
        heading(f"{index}. {item['title']} ({item['swc']}) - {item['severity']}", 3)
        paragraph(f"Root cause: {item.get('root_cause') or 'Not provided'}")
        paragraph(f"Impact: {item.get('impact') or 'Requires manual analysis'}")
        for label, text in (("Advisor verdict", _advisor_verdict_text(item)), ("CVSS", _cvss_text(item))):
            if text:
                paragraph(f"{label}: {text}")
        for step_index, step in enumerate(item.get("repair_steps", []), 1):
            paragraph(f"{step_index}. {step}")
        paragraph(f"Patch verification: {_verification_text(item)}")
    heading("5.4 Fixed Code", 2)
    paragraph(f"Patch verification verdict: {data['patch_verdict']}")
    comparison = build_code_comparison_rows(data["original_code"], data["fixed_code"])
    compare_rows = [[Paragraph("Source Code (Sample)", styles["CodeHeader"]),
                     Paragraph("Fixed Source Code", styles["CodeHeader"])],
                    *([code_cell(row["source"]), code_cell(row["fixed"])] for row in comparison)]
    compare_style = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#26394d")),
                     ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                     ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#bcc5cf")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                     ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                     ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    compare_style += [("BACKGROUND", (1, index), (1, index), colors.HexColor("#fff2a8"))
                      for index, row in enumerate(comparison, 1) if row["fixed_changed"]]
    table = Table(compare_rows, repeatRows=1, colWidths=[84 * mm, 84 * mm], splitByRow=1)
    table.setStyle(TableStyle(compare_style))
    story.extend([table, Spacer(1, 3 * mm)])
    heading("5.5 Verification Results", 2)
    paragraph(f"Fix attempts: {_attempt_text(data)}")
    channel_rows = verification_channel_rows(data)
    add_table([["Channel", "Result", "Verdict", "Details"],
               *([row["component"], row["result"], row["verdict"],
                  Paragraph(html.escape(row["detail"]), styles["BodyText"])] for row in channel_rows)],
              [35 * mm, 20 * mm, 30 * mm, 85 * mm],
              header_style(7.5) + [("TEXTCOLOR", (1, index), (1, index),
                                    colors.HexColor("#28633b" if row["result"] == "PASS" else "#9e1d1d"))
                                   for index, row in enumerate(channel_rows, 1)], space=3)
    tool = data["tool_verification"]
    paragraph(f"Compilation: {tool.get('compile', {}).get('status', 'not run')}; "
              f"Slither verdict: {tool.get('policy_verdict', 'not run')}; "
              f"Slither: {tool.get('slither', {}).get('status', 'not run')}; "
              f"Behavior preservation: {tool.get('behavioral', {}).get('status', 'not run')}.")
    for finding in tool.get("behavioral", {}).get("findings", []):
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
    add_table([["Vulnerability", "Severity", "Detected", "Scope"],
               *([f"{row['name']} ({row['abbr']})", row["severity"], "✓" if row.get("detected") else "-",
                  row["impact"][:70] + "..." if len(row["impact"]) > 70 else row["impact"]]
                 for row in data.get("reference_table", []))],
              [42 * mm, 20 * mm, 18 * mm, 90 * mm], header_style(7))

    # Chân trang mỗi trang PDF: run_id và số trang.
    def footer(canvas, document) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(20 * mm, 10 * mm, f"Run {data['run_id']}")
        canvas.drawRightString(190 * mm, 10 * mm, f"Page {document.page}")
        canvas.restoreState()

    document = SimpleDocTemplate(str(output_path), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                                 topMargin=18 * mm, bottomMargin=18 * mm)
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return True, ""


# [HÀM CHÍNH PHASE 6] Ghi phase6_audit_report.md / .html / .pdf vào thư mục run, trả về đường dẫn (pdf=None nếu lỗi).
def write_reports(data: dict, run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    markdown_path = write_text(run_dir / "phase6_audit_report.md", render_markdown(data))
    html_path = write_text(run_dir / "phase6_audit_report.html", render_html(data))
    pdf_ok, pdf_error = render_pdf(data, run_dir / "phase6_audit_report.pdf")
    return {"markdown": str(markdown_path), "html": str(html_path),
            "pdf": str(run_dir / "phase6_audit_report.pdf") if pdf_ok else None, "pdf_error": pdf_error or None}
