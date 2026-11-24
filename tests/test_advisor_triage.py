"""Phase 2 Advisor triage: three verdicts, the dismissal check (a quoted guard must exist; the Advisor
also decides on findings that static and RAG both report), the prompt (Phase 1 evidence, including the
Phase 1 classification of detector-only findings), and which Advisor suggestions the Fixer uses."""

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


# Finding chỉ detector báo, đã được Phase 1 phân loại SWC-107 theo ca KB giống nhất.
def _classified() -> dict:
    return _finding(sources=["llm_detector"], evidence=[],
                    classification={"method": "nearest_known_case", "similarity": 0.62, "nearest_case_id": "k1"})


# Chạy _generate_advisor_section với một AdvisorAgent giả luôn trả `suggestion`.
def _run(findings: list, suggestion: RepairSuggestion) -> dict:
    # AdvisorAgent giả.
    class FakeAgent:
        # Bỏ qua tham số khởi tạo.
        def __init__(self, *args, **kwargs) -> None:
            pass

        # Trả cùng một kết luận cho mọi finding.
        def advise_one(self, code, finding):
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
        def repair_knowledge(self, swc, code=None, limit=3):
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

    # Static + RAG cùng báo (corroborated) không tước quyền Advisor: kết luận false positive vẫn loại được finding.
    def test_corroborated_finding_can_be_dismissed_by_the_advisor(self) -> None:
        result = _run([_finding(corroborated=True, sources=["rag", "static"])],
                      RepairSuggestion(verdict="false_positive", root_cause="No reentrancy."))
        self.assertEqual(result["false_positives"], ["SWC-107:Bank.withdraw"])
        self.assertEqual(result["suggestions"], [])

    # Uncertain -> giữ lại với root cause + fix_strategy của LLM.
    def test_uncertain_finding_is_kept_with_the_llm_strategy(self) -> None:
        result = _run([_finding()], RepairSuggestion(
            verdict="uncertain", root_cause="Depends on whenNotPaused.", fix_strategy="Apply CEI."))
        suggestion = result["suggestions"][0]
        self.assertEqual(suggestion["advisor_verdict"], "uncertain")
        self.assertEqual(suggestion["repair_steps"], ["Apply CEI."])

    # Advisor không phân loại lại: suggestion giữ đúng SWC Phase 1 gán.
    def test_classified_finding_keeps_the_phase1_class(self) -> None:
        result = _run([_classified()], RepairSuggestion(verdict="confirmed", root_cause="State updated after call."))
        self.assertEqual(result["suggestions"][0]["swc"], "SWC-107")
        self.assertNotIn("advisor_swc", result["suggestions"][0])

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

    # Finding chỉ detector báo: prompt thẩm định đúng lớp Phase 1 gán (nêu nguồn phân loại), không bắt Advisor phân loại.
    def test_classified_prompt_judges_the_phase1_class(self) -> None:
        prompt = AdvisorAgent(llm=object(), knowledge_base=object())._build_prompt(SOURCE, _classified(), "")
        self.assertIn("Finding: Reentrancy (SWC-107)", prompt)
        self.assertIn("assigned in Phase 1 from the most similar known vulnerable case", prompt)
        self.assertIn("(code similarity 0.62)", prompt)
        self.assertNotIn("vulnerability_class", prompt)


# KB giả cho Fixer: ghi lại SWC được hỏi, trả một ca cố định.
class _CaseKB:
    # Chưa có lần tra nào.
    def __init__(self) -> None:
        self.asked = []

    # Ghi (mã, SWC) được tra, trả một ca reentrancy.
    def cases_for_unit(self, code, swcs, limit=2, min_similarity=0.08):
        self.asked.append((code, set(swcs)))
        return [{"swc": "SWC-107", "title": "Reentrancy", "similarity": 0.5, "root_cause": "Call before update."}]


# Fixer chỉ dùng chẩn đoán LLM của chính hàm (confirmed / uncertain), không dùng gợi ý KB hay "disputed".
class FixerUsesAdvisorTests(unittest.TestCase):
    # fix_strategy của suggestion uncertain có trong prompt Fixer; của suggestion disputed thì không.
    def test_fixer_prompt_takes_only_llm_diagnoses(self) -> None:
        unit = next(unit for unit in parse_code_units(SOURCE) if unit.name == "withdraw")
        prompt = _build_fixer_prompt(unit, [_finding()], [
            {"advisor_verdict": "uncertain", "fix_strategy": "Update the balance before the call."},
            {"advisor_verdict": "disputed", "fix_strategy": "Knowledge-base text."},
        ], SOURCE, "", _CaseKB())
        self.assertIn("Fix Strategy: Update the balance before the call.", prompt)
        self.assertNotIn("Knowledge-base text.", prompt)

    # Ca KB của Fixer được tra lúc sửa, theo mã của CHÍNH hàm đang sửa và SWC của các finding của hàm đó.
    def test_fixer_retrieves_cases_for_the_unit_being_fixed(self) -> None:
        unit = next(unit for unit in parse_code_units(SOURCE) if unit.name == "withdraw")
        kb = _CaseKB()
        prompt = _build_fixer_prompt(unit, [_finding()], [], SOURCE, "", kb)
        self.assertEqual(kb.asked, [(unit.code, {"SWC-107"})])
        self.assertIn("- [SWC-107] Reentrancy (similarity: 0.50)", prompt)


if __name__ == "__main__":
    unittest.main()
