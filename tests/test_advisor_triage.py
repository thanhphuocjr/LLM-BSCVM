"""Phase 2 Advisor triage (recall-first): three verdicts, the dismissal check (a quoted guard must
exist, corroborated findings are kept), UNMAPPED classification, the prompt, and which Advisor
suggestions the Fixer uses."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from E2E.phase0_parsing.code_units import parse_code_units
from E2E.phase2_advisor.advisor import AdvisorAgent, RepairSuggestion
from E2E.phase2_advisor.advisor_phase import _generate_advisor_section, _quote_in_code
from E2E.phase4_fixer.fixer import _build_fixer_prompt

WITHDRAW = """function withdraw(uint256 amount) external whenNotPaused {
    require(balances[msg.sender] >= amount);
    (bool ok, ) = msg.sender.call{value: amount}("");
    require(ok);
    balances[msg.sender] -= amount;
}"""

SOURCE = "pragma solidity ^0.8.0;\ncontract Bank {\n    mapping(address => uint256) public balances;\n" \
         + "\n".join("    " + line for line in WITHDRAW.splitlines()) + "\n}\n"


# Finding reentrancy của hàm withdraw (static báo tại dòng .call), mặc định không corroborated.
def _finding(**extra) -> dict:
    return {
        "finding_id": "SWC-107:Bank.withdraw", "swc": "SWC-107", "title": "Reentrancy", "severity": "Critical",
        "unit_id": "Bank.withdraw", "unit_code": WITHDRAW, "start_line": 4, "end_line": 9, "line": 4,
        "sources": ["static"], "corroborated": False,
        "evidence": [{"source": "static", "line": 6, "matched_text": ".call{value:",
                      "notes": ["L6: Low-level call sends ETH"]}],
        **extra,
    }


# Finding UNMAPPED (chỉ detector báo) của hàm withdraw.
def _unmapped() -> dict:
    return _finding(finding_id="UNMAPPED:Bank.withdraw", swc="UNMAPPED", title="Qwen2.5-14B neural vulnerability signal",
                    sources=["llm_detector"], evidence=[])


# Chạy _generate_advisor_section với một AdvisorAgent giả luôn trả `suggestion`.
def _run(findings: list, suggestion: RepairSuggestion) -> dict:
    # AdvisorAgent giả.
    class FakeAgent:
        # Bỏ qua tham số khởi tạo.
        def __init__(self, *args, **kwargs) -> None:
            pass

        # Trả cùng một kết luận cho mọi finding.
        def advise_one(self, code, finding, rag_context=None):
            return suggestion

    with patch("E2E.phase2_advisor.advisor_phase.AdvisorAgent", FakeAgent):
        return _generate_advisor_section(findings, SOURCE, use_llm=True)


# Gọi advise_one thật với một LLM giả trả `answer` (đã parse JSON) và một KB rỗng.
def _advise(finding: dict, answer) -> RepairSuggestion:
    # LLM giả: trả nguyên `answer`.
    class FakeLLM:
        # Bỏ qua prompt.
        def generate_json(self, prompt, system_instruction=None):
            return answer

    # KB rỗng.
    class EmptyKB:
        # Không có tri thức nào.
        def retrieve(self, swc, limit=3, code=None):
            return {}

    return AdvisorAgent(llm=FakeLLM(), knowledge_base=EmptyKB()).advise_one(SOURCE, finding)


# Bước loại false positive (advisor_phase).
class DismissalCheckTests(unittest.TestCase):
    # Guard bịa (không có trong mã) -> giữ finding, "disputed", suggestion từ KB (không có fix_strategy của LLM).
    def test_invented_guard_keeps_the_finding_as_disputed(self) -> None:
        result = _run([_finding()], RepairSuggestion(
            verdict="false_positive", guard_quote="onlyOwner", root_cause="Protected by an authorization modifier."))
        self.assertEqual(result["false_positives"], [])
        suggestion = result["suggestions"][0]
        self.assertEqual(suggestion["advisor_verdict"], "disputed")
        self.assertEqual(suggestion["kept_because"], "the guard quoted by the Advisor does not appear in the code")
        self.assertNotIn("fix_strategy", suggestion)

    # Guard có thật trong mã -> finding bị loại, lý do = root_cause của LLM.
    def test_real_guard_dismisses_the_finding(self) -> None:
        result = _run([_finding()], RepairSuggestion(
            verdict="false_positive", guard_quote="balances[msg.sender]  >= amount", root_cause="Balance checked."))
        self.assertEqual(result["false_positives"], ["SWC-107:Bank.withdraw"])
        self.assertEqual(result["false_positive_reasons"], {"SWC-107:Bank.withdraw": "Balance checked."})
        self.assertEqual(result["suggestions"], [])

    # Static + RAG cùng báo (corroborated) -> một kết luận false positive không loại được finding.
    def test_corroborated_finding_is_never_dismissed(self) -> None:
        result = _run([_finding(corroborated=True)], RepairSuggestion(verdict="false_positive", root_cause="No reentrancy."))
        self.assertEqual(result["false_positives"], [])
        self.assertIn("static rules and RAG", result["suggestions"][0]["kept_because"])

    # Uncertain -> giữ lại với root cause + fix_strategy của LLM.
    def test_uncertain_finding_is_kept_with_the_llm_strategy(self) -> None:
        result = _run([_finding()], RepairSuggestion(
            verdict="uncertain", root_cause="Depends on whenNotPaused.", fix_strategy="Apply CEI."))
        suggestion = result["suggestions"][0]
        self.assertEqual(suggestion["advisor_verdict"], "uncertain")
        self.assertEqual(suggestion["repair_steps"], ["Apply CEI."])

    # Finding UNMAPPED được gán lớp lỗi -> suggestion mang advisor_swc.
    def test_unmapped_class_is_carried_by_the_suggestion(self) -> None:
        result = _run([_unmapped()], RepairSuggestion(verdict="confirmed", vulnerability_class="SWC-107"))
        self.assertEqual(result["suggestions"][0]["advisor_swc"], "SWC-107")

    # So guard_quote với mã: bỏ qua khoảng trắng, dấu `, dấu lược "..."; mảnh không có trong mã -> False.
    def test_quote_matching_tolerates_spacing_and_ellipsis(self) -> None:
        self.assertTrue(_quote_in_code("`require(ok);`", SOURCE))
        self.assertTrue(_quote_in_code("require(balances[msg.sender]>=amount) ... require(ok)", SOURCE))
        self.assertFalse(_quote_in_code("require(msg.sender == owner)", SOURCE))
        self.assertFalse(_quote_in_code("...", SOURCE))


# Đọc câu trả lời JSON của LLM (advise_one).
class AdviseOneParsingTests(unittest.TestCase):
    # verdict (không phân biệt hoa/thường) ngoài 3 giá trị hợp lệ -> uncertain (không đọc được thì không loại finding).
    def test_unknown_verdict_becomes_uncertain(self) -> None:
        self.assertEqual(_advise(_finding(), {"verdict": "False_Positive"}).verdict, "false_positive")
        self.assertEqual(_advise(_finding(), {"verdict": "False Positive"}).verdict, "uncertain")
        self.assertEqual(_advise(_finding(), {"root_cause": "x"}).verdict, "uncertain")

    # vulnerability_class chỉ đọc cho finding UNMAPPED; "SWC-107 Reentrancy" -> "SWC-107"; lớp ngoài danh sách -> "".
    def test_vulnerability_class_is_read_for_unmapped_only(self) -> None:
        self.assertEqual(_advise(_unmapped(), {"vulnerability_class": "SWC-107 Reentrancy"}).vulnerability_class, "SWC-107")
        self.assertEqual(_advise(_unmapped(), {"vulnerability_class": "SWC-999"}).vulnerability_class, "")
        self.assertEqual(_advise(_finding(), {"vulnerability_class": "SWC-107"}).vulnerability_class, "")

    # LLM trả JSON không phải object -> error (advisor_phase sẽ dùng gợi ý từ KB).
    def test_non_object_answer_is_an_error(self) -> None:
        self.assertIn("expected a JSON object", _advise(_finding(), [1, 2]).error)


# Nội dung prompt Advisor.
class AdvisorPromptTests(unittest.TestCase):
    # Kiểm soát truy cập chỉ bác được lỗi access control; thiếu mã không phải guard; có bằng chứng Phase 1
    # (dòng .call mà static khớp); không đưa xác suất detector.
    def test_prompt_scopes_access_rule_and_shows_phase1_evidence(self) -> None:
        prompt = AdvisorAgent(llm=object(), knowledge_base=object())._build_prompt(SOURCE, _finding(p_detector=0.97), "")
        self.assertIn("only answer ACCESS-CONTROL findings", prompt)
        self.assertIn("Missing code is not a guard", prompt)
        self.assertIn('static rule at line 6: `(bool ok, ) = msg.sender.call{value: amount}("");`', prompt)
        self.assertNotIn("0.97", prompt)
        self.assertNotIn("may be a false positive", prompt)

    # Finding UNMAPPED: prompt yêu cầu vulnerability_class và có gợi ý ca tương tự của RAG.
    def test_unmapped_prompt_asks_for_a_class_with_rag_hints(self) -> None:
        prompt = AdvisorAgent(llm=object(), knowledge_base=object())._build_prompt(SOURCE, _unmapped(), "", rag_context=[
            {"swc": "SWC-107", "title": "Reentrancy", "similarity": "0.81", "description": "State updated after call."}])
        self.assertIn('"vulnerability_class"', prompt)
        self.assertIn("SWC-135 Business Logic Flaw", prompt)
        self.assertIn("SWC-107 Reentrancy (similarity 0.81): State updated after call.", prompt)
        self.assertNotIn("Qwen2.5-14B neural vulnerability signal", prompt)


# Fixer chỉ dùng chẩn đoán LLM của chính hàm (confirmed / uncertain), không dùng gợi ý KB hay "disputed".
class FixerUsesAdvisorTests(unittest.TestCase):
    # fix_strategy của suggestion uncertain có trong prompt Fixer; của suggestion disputed thì không.
    def test_fixer_prompt_takes_only_llm_diagnoses(self) -> None:
        unit = next(unit for unit in parse_code_units(SOURCE) if unit.name == "withdraw")
        prompt = _build_fixer_prompt(unit, [_finding()], [
            {"advisor_verdict": "uncertain", "fix_strategy": "Update the balance before the call."},
            {"advisor_verdict": "disputed", "fix_strategy": "Knowledge-base text."},
        ], SOURCE, "", "")
        self.assertIn("Fix Strategy: Update the balance before the call.", prompt)
        self.assertNotIn("Knowledge-base text.", prompt)


if __name__ == "__main__":
    unittest.main()
