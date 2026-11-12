"""Regression tests for the pipeline logic fixes (fusion fail-closed, corroboration,
RAG exact match, repair-loop carry-over, compile baseline, behaviour guard, CVSS, prompt
context, LLM JSON salvage, Phase 5 behaviour check)."""

from __future__ import annotations

import json
import re
import unittest
from unittest.mock import patch

from E2E.phase0_parsing.code_units import parse_code_units
from E2E.phase1_fusion.fusion import fuse_detection, fuse_findings, merge_findings
from E2E.phase1b_rag.retriever import _is_exact_code_match
from E2E.phase1c_detector import detector
from E2E.phase2_advisor.advisor_phase import _generate_advisor_section
from E2E.phase3_assessor.assessor import compute_cvss_31_base_score
from E2E.phase3_assessor.assessor_phase import _generate_assessor_section, get_cvss_defaults
from E2E.phase4_fixer import fixer
from E2E.phase4_fixer.behavior_guard import validate_observable_behavior
from E2E.phase5_verifier import candidate_check
from E2E.phase5_verifier.tool_matrix import _evaluate_tool_result, _policy_verdict
from E2E.pipeline import gpu_pipeline


OK_EMPTY = {"status": "ok", "score": 0.0, "findings": []}

BANK = """pragma solidity ^0.8.0;
contract Bank {
    mapping(address => uint256) public balances;

    function deposit() external payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw(uint256 amount) external {
        require(balances[msg.sender] >= amount);
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok);
        balances[msg.sender] -= amount;
    }

    function withdrawAll() external {
        uint256 amount = balances[msg.sender];
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok);
        balances[msg.sender] = 0;
    }
}
"""

FIXED_WITHDRAW = """function withdraw(uint256 amount) external {
        require(balances[msg.sender] >= amount);
        balances[msg.sender] -= amount;
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok);
    }"""

FIXED_WITHDRAW_ALL = """function withdrawAll() external {
        uint256 amount = balances[msg.sender];
        balances[msg.sender] = 0;
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok);
    }"""

UNFIXED_WITHDRAW_ALL = """function withdrawAll() external {
        uint256 amount = balances[msg.sender];
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok);
        balances[msg.sender] = 0;
    }"""


# Detector giả lập: trả 0.95 (lỗi) nếu lời gọi chuyển ETH đứng trước lệnh cập nhật số dư, ngược lại 0.05.
def _reentrancy_probability(code: str) -> float:
    """Fake detector: vulnerable iff a value call precedes the balance update."""
    return 0.95 if re.search(r"call\{value[\s\S]*balances\[msg\.sender\]\s*(=|-=)", code) else 0.05


# Fusion không bao giờ kết luận Safe khi detector thiếu dữ liệu.
class FusionFailClosedTests(unittest.TestCase):
    # Mọi hàm detector lỗi -> không bao giờ Safe.
    def test_all_detector_units_errored_is_never_safe(self) -> None:
        detector = {
            "status": "ok",
            "verdict": "Safe",
            "vulnerability_probability": 0.0,
            "threshold": 0.7,
            "unit_results": [
                {"unit_id": "A.f", "status": "error", "label": "Inconclusive", "vulnerability_probability": None}
            ],
        }
        fused = fuse_detection(OK_EMPTY, OK_EMPTY, detector)
        self.assertEqual(fused["verdict"], "Inconclusive")
        self.assertEqual(fused["status"], "partial")
        self.assertIn("detector_incomplete_no_safe_verdict", fused["decision_rules"])

    # Detector chấm thiếu một số hàm -> không bao giờ Safe.
    def test_partially_scored_detector_is_never_safe(self) -> None:
        summary = detector.summarize_detector(
            [
                {"unit_id": "A.f", "status": "ok", "label": "Safe", "vulnerability_probability": 0.1},
                {"unit_id": "A.g", "status": "error", "label": "Inconclusive", "vulnerability_probability": None},
            ],
            0.7,
        )
        self.assertEqual(summary["status"], "partial")
        self.assertEqual(summary["verdict"], "Inconclusive")
        self.assertEqual(fuse_detection(OK_EMPTY, OK_EMPTY, summary)["verdict"], "Inconclusive")

    # Detector chấm đủ và đều an toàn -> vẫn ra Safe được.
    def test_complete_detector_can_still_report_safe(self) -> None:
        summary = detector.summarize_detector(
            [{"unit_id": "A.f", "status": "ok", "label": "Safe", "vulnerability_probability": 0.1}],
            0.7,
        )
        self.assertEqual(fuse_detection(OK_EMPTY, OK_EMPTY, summary)["verdict"], "Safe")


# Quy tắc "hai kênh xác nhận nhau" (corroborated).
class CorroborationTests(unittest.TestCase):
    # Static + Slither cùng báo KHÔNG tính là corroborated (cùng kênh static).
    def test_static_and_slither_are_one_channel(self) -> None:
        static = {
            "status": "ok",
            "findings": [
                {"source": "static", "swc": "SWC-107", "unit_id": "A.w", "severity": "Medium", "confidence": 0.4},
                {"source": "slither", "swc": "SWC-107", "unit_id": "A.w", "severity": "Medium", "confidence": 0.95},
            ],
        }
        merged = merge_findings(static, {"findings": []})
        self.assertFalse(merged[0]["corroborated"])
        fused = fuse_detection(static, OK_EMPTY, None)
        self.assertNotIn("static_rag_corroboration_floor", fused["decision_rules"])

    # Static + RAG cùng báo -> corroborated.
    def test_static_and_rag_are_corroborated(self) -> None:
        static = {"findings": [{"source": "slither", "swc": "SWC-107", "unit_id": "A.w", "confidence": 0.9}]}
        rag = {"findings": [{"source": "rag", "swc": "SWC-107", "unit_id": "A.w", "confidence": 0.7}]}
        self.assertTrue(merge_findings(static, rag)[0]["corroborated"])


# Fusion theo từng finding (fuse_findings).
class FuseFindingsTests(unittest.TestCase):
    # Hàm chỉ detector báo lỗi vẫn thành finding UNMAPPED để được sửa.
    def test_detector_only_unit_reaches_repair_agents(self) -> None:
        units = [{"unit_id": "A.f", "label": "Vulnerable", "vulnerability_probability": 0.9, "start_line": 3}]
        confirmed, _ = fuse_findings(units, {"findings": []}, {"findings": []})
        self.assertEqual([item["unit_id"] for item in confirmed], ["A.f"])
        self.assertEqual(confirmed[0]["swc"], "UNMAPPED")

    # Hàm detector lỗi nhận p_detector = 0.5 (trung lập), không bị coi là an toàn.
    def test_errored_unit_is_neutral_not_safe(self) -> None:
        units = [{"unit_id": "A.w", "status": "error", "vulnerability_probability": None}]
        static = {"findings": [{"source": "static", "swc": "SWC-107", "unit_id": "A.w", "severity": "High", "confidence": 0.9}]}
        confirmed, candidates = fuse_findings(units, static, {"findings": []})
        finding = (confirmed + candidates)[0]
        self.assertEqual(finding["p_detector"], 0.5)
        self.assertTrue(finding["p_detector_missing"])


# Điều kiện "trùng khớp chính xác" của RAG.
class RagExactMatchTests(unittest.TestCase):
    # Getter ngắn không được coi là trùng khớp với ca trong KB.
    def test_short_getter_is_not_an_exact_match(self) -> None:
        getter = "function totalSupply() public view returns (uint256) { return _totalSupply; }"
        self.assertFalse(_is_exact_code_match(getter + " extra body " * 20, getter))
        self.assertFalse(_is_exact_code_match(getter, getter))

    # Hàm giống hệt ca trong KB -> trùng khớp.
    def test_identical_function_is_an_exact_match(self) -> None:
        self.assertTrue(_is_exact_code_match(FIXED_WITHDRAW, FIXED_WITHDRAW))


# Guard hành vi của Phase 4.
class BehaviourGuardTests(unittest.TestCase):
    # Đổi tên biến cục bộ / thay transfer bằng dạng an toàn tương đương -> chấp nhận.
    def test_accepts_local_rename_and_safe_transfer(self) -> None:
        validate_observable_behavior(
            "function f(uint a) public { uint x = a; total += x; emit E(x); }",
            "function f(uint a) public { uint y = a; total += y; emit E(y); }",
        )
        validate_observable_behavior(
            "function g() public { IERC20(t).transfer(a, 1); }",
            "function g() public { SafeERC20.safeTransfer(IERC20(t), a, 1); }",
        )

    # Xoá lời gọi chuyển ETH / lệnh ghi state / event -> từ chối.
    def test_rejects_removed_value_call_state_write_and_event(self) -> None:
        with self.assertRaisesRegex(ValueError, "ETH value call"):
            validate_observable_behavior(UNFIXED_WITHDRAW_ALL, "function withdrawAll() external { balances[msg.sender] = 0; }")
        with self.assertRaisesRegex(ValueError, "state variable"):
            validate_observable_behavior("function s(uint v) public { total = v; }", "function s(uint v) public { }")
        with self.assertRaisesRegex(ValueError, "event"):
            validate_observable_behavior("function t() { Transfer(a, b, 1); }", "function t() { }")


# Tách hàm thay thế từ câu trả lời của model.
class ExtractionTests(unittest.TestCase):
    # JSON hỏng do dấu " không escape vẫn lấy được trường "replacement".
    def test_invalid_json_with_unescaped_quotes_is_decoded(self) -> None:
        from E2E.phase4_fixer.patching import extract_declaration_body

        reply = (
            '{\n  "replacement": "function withdraw() external {\\n    balances[msg.sender] = 0;\\n'
            '    (bool ok, ) = msg.sender.call{value: amount}("");\\n    require(ok, \\"failed\\");\\n}",\n'
            '  "why": "CEI"\n}'
        )
        body = extract_declaration_body(reply, "withdraw")
        self.assertNotIn("\\n", body)
        self.assertIn('call{value: amount}("");\n', body)
        self.assertIn('require(ok, "failed");', body)


# CVSS (Phase 3) và triage của Advisor (Phase 2).
class CvssAndAgentsTests(unittest.TestCase):
    # Điểm CVSS mặc định khớp với công thức tính từ vector.
    def test_fallback_cvss_matches_its_vector(self) -> None:
        for swc in ("SWC-107", "SWC-105", "SWC-115", "SWC-114", "SWC-102", "SWC-999"):
            meta = get_cvss_defaults(swc, "High")
            metrics = dict(part.split(":") for part in meta["cvss_vector"].split("/")[1:])
            expected, _ = compute_cvss_31_base_score(**{k.lower(): v for k, v in metrics.items()})
            self.assertEqual(meta["cvss_score"], expected, swc)

    # Assessor khi không có LLM dùng điểm tính bằng công thức.
    def test_assessor_fallback_uses_computed_score(self) -> None:
        result = _generate_assessor_section([{"swc": "SWC-107", "unit_id": "A.w"}])
        self.assertEqual(result["assessments"][0]["cvss_score"], 9.1)
        self.assertEqual(result["assessments"][0]["severity"], "Critical")
        self.assertEqual(result["assessments"][0]["cvss_source"], "swc_default")

    # repair_priority: điểm CVSS cao nhất sửa trước (Fixer vá theo thứ tự này).
    def test_repair_priority_follows_cvss_score(self) -> None:
        findings = [
            {"swc": "SWC-102", "unit_id": "contract"},
            {"swc": "SWC-107", "unit_id": "A.withdraw"},
            {"swc": "SWC-116", "unit_id": "A.lottery"},
        ]
        result = _generate_assessor_section(findings)
        order = [(item["swc"], item["repair_priority"]) for item in result["assessments"]]
        self.assertEqual(order, [("SWC-107", 1), ("SWC-116", 2), ("SWC-102", 3)])

    # Advisor kết luận false positive -> finding nằm trong false_positives và không có suggestion.
    def test_advisor_false_positive_is_reported_to_caller(self) -> None:
        # Kết quả Advisor giả (is_true_positive=False).
        class FakeSuggestion:
            error = None
            is_true_positive = False
            root_cause = "setOwner is guarded by onlyOwner."

        # AdvisorAgent giả, luôn trả FakeSuggestion.
        class FakeAgent:
            # Bỏ qua tham số khởi tạo.
            def __init__(self, *args, **kwargs) -> None:
                pass

            # Trả FakeSuggestion cho mọi finding.
            def advise_one(self, code, finding):
                return FakeSuggestion()

        findings = [{"finding_id": "SWC-105:A.setOwner", "swc": "SWC-105", "unit_id": "A.setOwner"}]
        with patch("E2E.phase2_advisor.advisor_phase.AdvisorAgent", FakeAgent):
            advisor = _generate_advisor_section(findings, "", use_llm=True)
        self.assertEqual(advisor["false_positives"], ["SWC-105:A.setOwner"])
        self.assertEqual(advisor["false_positive_reasons"], {"SWC-105:A.setOwner": "setOwner is guarded by onlyOwner."})
        self.assertEqual(advisor["suggestions"], [])

    # Kết luận Phase 5 khi không có bản vá: Advisor loại hết -> không cần vá; chỉ còn finding cấp contract -> chưa kết luận.
    def test_verdict_when_no_patch_was_needed_or_possible(self) -> None:
        from E2E.phase5_verifier.final_verification import decide_overall_verdict, skipped_verification

        tool_result, behaviour = skipped_verification()

        # Kết luận cho một kết quả GPU giả (finding còn lại / bị loại, số attempt của vòng sửa).
        def verdict(confirmed: list, dismissed: list, attempts: int) -> str:
            remote = {"status": "ok", "phases": {
                "detector": {"confirmed_findings": confirmed, "dismissed_findings": dismissed},
                "verifier": {"attempts_count": attempts},
            }}
            return decide_overall_verdict(
                remote=remote, fused={"verdict": "Vulnerable"}, code="c", fixed_code="c", patch_applied=False,
                tool_result=tool_result, behavioral_result=behaviour, fixed_static=None, fixed_rag=None,
            )

        pragma = {"finding_id": "SWC-103:contract", "swc": "SWC-103"}
        self.assertEqual(verdict([], [pragma], 0),
                         "No patch required: the Advisor dismissed every finding as a false positive")
        self.assertEqual(verdict([pragma], [], 0),
                         "Inconclusive: no function-level repair target for the confirmed findings")
        self.assertEqual(verdict([pragma], [], 3), "Patch rejected: no effective patch was applied")

    # Luật triage của Advisor nói rõ kiểm tra tx.origin không phải guard (SWC-115).
    def test_advisor_rule_does_not_treat_tx_origin_as_a_guard(self) -> None:
        from E2E.phase2_advisor.advisor import AdvisorAgent

        prompt = AdvisorAgent(llm=object(), knowledge_base=object())._build_prompt(
            BANK, {"swc": "SWC-115", "unit_code": "function f() public { if (tx.origin == owner) { x = 1; } }"}, ""
        )
        self.assertIn("A check on `tx.origin` is NOT a valid guard", prompt)


# Chính sách ma trận tool (Phase 5).
class ToolPolicyTests(unittest.TestCase):
    # Tool không phủ loại lỗi mục tiêu chạy sạch -> inconclusive, không phải passed.
    def test_clean_run_of_non_covering_tool_is_inconclusive(self) -> None:
        result = _evaluate_tool_result("slither", "completed", [], [], {"front_running"})
        self.assertFalse(result["applicable"])
        policy = _policy_verdict({"slither": result}, require_applicable=True)
        self.assertEqual(policy["verdict"], "inconclusive")

    # Finding còn sót chỉ tính khi nằm trong dòng của hàm đã vá.
    def test_residual_is_scoped_to_patched_unit_lines(self) -> None:
        finding = {"tool": "slither", "name": "reentrancy-eth", "category": "reentrancy", "line": [40], "severity": "High"}
        outside = _evaluate_tool_result("slither", "completed", [finding], [finding], {"reentrancy"}, None, {10, 11, 12})
        inside = _evaluate_tool_result("slither", "completed", [finding], [finding], {"reentrancy"}, None, {39, 40})
        self.assertEqual(outside["residual_target_findings"], [])
        self.assertEqual(len(inside["residual_target_findings"]), 1)


# Định danh finding không phụ thuộc vị trí.
class FindingSignatureTests(unittest.TestCase):
    # Đổi đường dẫn tạm / số dòng không làm finding cũ bị coi là mới.
    def test_signature_ignores_paths_and_line_numbers(self) -> None:
        from E2E.phase5_verifier.tool_matrix import _finding_signature

        before = {"name": "reentrancy-eth", "category": "reentrancy",
                  "message": "Reentrancy in Sale.withdraw() (../tmp/e2e-slither-aaaa/C.sol#914-930): call (C.sol#920)"}
        after = {"name": "reentrancy-eth", "category": "reentrancy",
                 "message": "Reentrancy in Sale.withdraw() (../tmp/e2e-slither-bbbb/C.sol#921-937): call (C.sol#927)"}
        self.assertEqual(_finding_signature(before), _finding_signature(after))


# Kiểm tra compile so với bản gốc (candidate_check.evaluate_compilation).
class CompileBaselineTests(unittest.TestCase):
    # Không có solc -> skipped, không phải failed.
    def test_missing_compiler_is_skipped_not_failed(self) -> None:
        skipped = {"passed": None, "status": "skipped", "error": "solc compiler not found"}
        with patch.object(
            candidate_check,
            "check_compilation_with_baseline",
            return_value={"baseline": skipped, "fixed": skipped, "baseline_failed": False},
        ):
            result = candidate_check.evaluate_compilation("a", "b")
        self.assertEqual(result["state"], "skipped")
        self.assertFalse(result["failed"])

    # Bản gốc vốn không compile -> not_applicable, không đổ lỗi bản vá.
    def test_original_that_cannot_compile_does_not_blame_the_patch(self) -> None:
        error = {"passed": False, "status": "failed", "error": 'ParserError: Source "x.sol" not found'}
        with patch.object(
            candidate_check,
            "check_compilation_with_baseline",
            return_value={"baseline": error, "fixed": dict(error), "baseline_failed": True},
        ):
            result = candidate_check.evaluate_compilation("a", "b")
        self.assertEqual(result["state"], "not_applicable")
        self.assertFalse(result["failed"])

    # Bản vá sinh lỗi compile MỚI -> failed.
    def test_new_compiler_error_is_a_failure(self) -> None:
        baseline = {"passed": False, "error": 'ParserError: Source "x.sol" not found'}
        fixed = {"passed": False, "error": baseline["error"] + "\nTypeError: Wrong argument count"}
        with patch.object(
            candidate_check,
            "check_compilation_with_baseline",
            return_value={"baseline": baseline, "fixed": fixed, "baseline_failed": True},
        ):
            self.assertTrue(candidate_check.evaluate_compilation("a", "b")["failed"])


# Các hàm hỗ trợ Fixer / vòng sửa.
class FixerSupportTests(unittest.TestCase):
    # _version_rules sinh luật cú pháp đúng theo pragma.
    def test_version_rules_follow_the_pragma(self) -> None:
        rules_0424 = fixer._version_rules("pragma solidity ^0.4.24;")
        rules_0418 = fixer._version_rules("pragma solidity ^0.4.18;")
        self.assertNotIn("DO NOT use the `constructor` keyword", rules_0424)
        self.assertIn("stays a `constructor(...)`", rules_0424)
        self.assertIn("DO NOT use the `constructor` keyword", rules_0418)
        self.assertEqual(fixer._version_rules("pragma solidity ^0.8.0;"), "")

    # Feedback lỗi compile giữ dòng mã + dấu ^ của solc (biết định danh nào sai), bỏ đường dẫn file tạm.
    def test_compiler_feedback_keeps_the_offending_source_line(self) -> None:
        error = ("Error: Identifier not found or not unique.\n  --> /var/folders/x/T/tmpab.sol:14:56:\n   |\n"
                 "14 |     function mint(address to, uint256 amount) external onlyOwner {\n"
                 "   |                                                        ^^^^^^^^^")
        feedback = candidate_check._compiler_feedback(error)
        self.assertIn("external onlyOwner {", feedback)
        self.assertIn("--> Contract.sol:14:56:", feedback)
        self.assertNotIn("/var/folders", feedback)

    # Lỗi solc được ánh xạ đúng về hàm chứa dòng lỗi.
    def test_compiler_errors_are_mapped_to_units(self) -> None:
        new_style = "Error: Invalid token.\n  --> /tmp/x.sol:11:35:\n"
        old_style = "/tmp/x.sol:17:9: Error: Undeclared identifier."
        self.assertEqual(candidate_check.units_with_compiler_errors(BANK, new_style), {"Bank.withdraw"})
        self.assertEqual(candidate_check.units_with_compiler_errors(BANK, old_style), {"Bank.withdrawAll"})


# Ngữ cảnh contract trong prompt Advisor/Fixer (phase0_parsing/contract_context.py).
class PromptContextTests(unittest.TestCase):
    # Contract 0.4: modifier không ngoặc ở contract cha, chuỗi có ";{}", event cũ không emit.
    LEGACY = """pragma solidity ^0.4.24;
contract Owned {
    address public owner;
    modifier onlyOwner { require(msg.sender == owner); _; }
}
contract Vault is Owned {
    mapping(address => uint) balances;
    string constant NOTE = "a;b{c}";
    event Paid(address to, uint amount);
    function pay(uint amt) public onlyOwner {
        uint local = amt;
        balances[msg.sender] -= local;
        msg.sender.transfer(local);
        Paid(msg.sender, local);
    }
}
"""

    # Biến trạng thái = khai báo cấp contract (có mapping), KHÔNG lấy biến cục bộ trong thân hàm.
    def test_state_variables_are_contract_level_declarations(self) -> None:
        from E2E.phase0_parsing.contract_context import build_contract_context

        withdraw_all = next(unit for unit in parse_code_units(BANK) if unit.name == "withdrawAll")
        context = build_contract_context(withdraw_all, BANK)
        state_section = context.split("\n\n")[0]
        self.assertIn("mapping(address => uint256) public balances;", state_section)
        self.assertNotIn("uint256 amount", state_section)
        self.assertIn("State writes to preserve: balances[msg.sender] =", context)
        self.assertNotIn("amount =", context)

    # Modifier không có dấu ngoặc và modifier của contract cha mà hàm dùng đều được đưa vào ngữ cảnh.
    def test_inherited_modifier_without_parentheses_is_shown(self) -> None:
        from E2E.phase0_parsing.contract_context import build_contract_context

        pay = next(unit for unit in parse_code_units(self.LEGACY) if unit.name == "pay")
        context = build_contract_context(pay, self.LEGACY)
        self.assertIn("modifier onlyOwner { require(msg.sender == owner); _; }", context)
        self.assertIn('string constant NOTE = "a;b{c}";', context)
        self.assertIn("event Paid(address to, uint amount);", context)
        self.assertIn("State writes to preserve: balances[msg.sender] -=", context)
        self.assertNotIn("local", context.split("Observable Invariants:")[1].split("\n")[1])


# Vớt JSON hỏng của model (common/gpu_client.extract_json) và chuẩn hoá metric CVSS (Phase 3).
class LlmOutputTests(unittest.TestCase):
    # JSON hỏng vẫn lấy được đúng các khoá Advisor / Assessor đọc.
    def test_broken_json_salvages_the_keys_the_agents_read(self) -> None:
        from E2E.common.gpu_client import extract_json

        advisor = extract_json('{"is_true_positive": false, "root_cause": "guarded by "onlyOwner"", '
                               '"fix_strategy": "Apply CEI", "evidence_lines": [3,')
        self.assertIs(advisor["is_true_positive"], False)
        self.assertEqual(advisor["fix_strategy"], "Apply CEI")
        assessor = extract_json('{"attack_vector": "N", "privileges_required": "H", "integrity": "H", '
                                '"justification": "owner only" "scope": "U"}')
        self.assertEqual((assessor["attack_vector"], assessor["privileges_required"]), ("N", "H"))

    # Metric viết dạng từ ("High", "network") được đổi về chữ cái: điểm và vector luôn khớp nhau.
    def test_cvss_metric_words_are_normalised(self) -> None:
        self.assertEqual(
            compute_cvss_31_base_score("Network", "low", "None", "None", "Unchanged", "None", "Low", "None"),
            (5.3, "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:L/A:N"),
        )
        self.assertEqual(compute_cvss_31_base_score("?", "", "x", "y", "z", "q", "w", "e"), compute_cvss_31_base_score())


# Advisor: finding cấp contract (không có hàm) được xét trên CHÍNH contract, không phải ca mẫu trong KB.
class AdvisorTargetTests(unittest.TestCase):
    # <TargetFunction> của finding SWC-102 (cấp contract) chứa contract Bank, không chứa vulnerable_code của KB.
    def test_contract_level_finding_targets_the_audited_contract(self) -> None:
        from E2E.phase2_advisor.advisor import AdvisorAgent

        agent = AdvisorAgent(llm=object(), knowledge_base=object())
        finding = {"swc": "SWC-102", "title": "Outdated Compiler Version", "severity": "Low",
                   "vulnerable_code": "contract KnowledgeBaseExample { }"}
        prompt = agent._build_prompt(BANK, finding, "<RetrievedSecurityKnowledge></RetrievedSecurityKnowledge>")
        target = prompt.split("<TargetFunction", 1)[1].split("</TargetFunction>", 1)[0]
        self.assertIn("contract Bank {", target)
        self.assertNotIn("KnowledgeBaseExample", target)


# Phase 5: cảnh báo Slither locked-ether chỉ là hồi quy khi bản vá sinh ra nó.
class LockedEtherTests(unittest.TestCase):
    # Một finding locked-ether của Slither (đã chuẩn hoá).
    LOCKED = {"check": "locked-ether", "name": "locked-ether", "severity": "Medium", "message": "locks Ether"}

    # Chạy verify_patch_behavior với kết quả Slither giả (bản gốc / bản vá).
    def _behaviour(self, original_findings: list) -> dict:
        from E2E.phase5_verifier.behavior_verifier import verify_patch_behavior

        tool_result = {
            "slither": {"findings": [self.LOCKED]},
            "tools": {"slither": {"original_findings": original_findings}},
        }
        withdraw = next(unit for unit in parse_code_units(BANK) if unit.name == "withdraw")
        patched = BANK.replace(withdraw.code, FIXED_WITHDRAW)
        return verify_patch_behavior(BANK, patched, [{"unit_id": "Bank.withdraw", "status": "applied"}], tool_result)

    # Bản gốc đã có locked-ether (cảnh báo sẵn có) -> không phải hồi quy của bản vá.
    def test_pre_existing_locked_ether_is_not_a_regression(self) -> None:
        self.assertNotIn("slither-locked-ether", [item["check"] for item in self._behaviour([self.LOCKED])["findings"]])

    # Bản vá sinh ra locked-ether -> hồi quy hành vi (status failed).
    def test_introduced_locked_ether_is_a_regression(self) -> None:
        behaviour = self._behaviour([])
        self.assertEqual(behaviour["status"], "failed")
        self.assertIn("slither-locked-ether", [item["check"] for item in behaviour["findings"]])


# Chạy thật gpu_pipeline với server GPU GIẢ (detector giả + Fixer giả) để kiểm tra vòng sửa.
class ServerGpuRepairLoopTests(unittest.TestCase):
    """End-to-end run of the GPU runner against a fake server."""

    # The RAG channel is stubbed out so these tests exercise the repair loop itself.
    EMPTY_RAG = {"status": "ok", "findings": [], "retrieved_context": []}

    # Dựng request.json cho contract Bank (RAG để trống).
    def _request(self) -> dict:
        from E2E.phase1a_static.static_runner import run_static

        units = parse_code_units(BANK)
        return {
            "run_id": "fake-gpu",
            "source": {"code": BANK, "sha256": "x", "file_name": "Bank.sol"},
            "code_units": [unit.to_dict() for unit in units],
            "local_phase1": {"static": run_static(BANK, units), "rag": dict(self.EMPTY_RAG)},
            "execution": {"detector_threshold": 0.7, "repair": {"max_attempts": 3}},
        }

    # Chạy run_server_gpu_pipeline với các phụ thuộc được patch (health, detect, generate, Advisor, Assessor...).
    def _run(self, generate, detect=None) -> dict:
        # Detector giả dùng _reentrancy_probability.
        def fake_detect(code, threshold=0.7, server_url=None, timeout=60):
            return {"vulnerable_probability": _reentrancy_probability(code)}

        # Advisor không dùng LLM (dựng từ KB).
        def deterministic_advisor(findings, code="", **kwargs):
            return _generate_advisor_section(findings, code)

        # Assessor không dùng LLM (CVSS mặc định).
        def deterministic_assessor(findings, code="", **kwargs):
            return _generate_assessor_section(findings, code)

        with patch.object(gpu_pipeline, "check_server_health", return_value={"status": "ready"}), \
                patch.object(detector, "detect_via_server", side_effect=detect or fake_detect), \
                patch.object(fixer, "generate_via_server", side_effect=generate), \
                patch.object(gpu_pipeline, "_generate_advisor_section", side_effect=deterministic_advisor), \
                patch.object(gpu_pipeline, "_generate_assessor_section", side_effect=deterministic_assessor), \
                patch.object(candidate_check, "run_rag", return_value=dict(self.EMPTY_RAG)), \
                patch("builtins.print"):
            request = self._request()
            return gpu_pipeline.run_server_gpu_pipeline(request)

    # Vòng sửa: lần 2 giữ bản vá đã đạt của lần 1, chỉ vá lại hàm còn lỗi.
    def test_second_attempt_keeps_first_attempt_patch(self) -> None:
        calls = {"withdrawAll": 0}

        # Fixer giả: lần đầu vá đúng một hàm, sai hàm kia.
        def generate(prompt, **kwargs):
            if "Target vulnerable function: `withdrawAll`" in prompt:
                calls["withdrawAll"] += 1
                body = FIXED_WITHDRAW_ALL if calls["withdrawAll"] >= 2 else UNFIXED_WITHDRAW_ALL
            else:
                body = FIXED_WITHDRAW
            return json.dumps({"replacement": body})

        result = self._run(generate)
        verifier = result["phases"]["verifier"]
        fixed_code = result["phases"]["fixer"]["fixed_code"]
        self.assertEqual(verifier["overall_verdict"], "PASS")
        self.assertEqual(verifier["attempts_count"], 2)
        self.assertEqual(_reentrancy_probability(FIXED_WITHDRAW), 0.05)
        for unit in parse_code_units(fixed_code):
            if unit.name in {"withdraw", "withdrawAll"}:
                self.assertEqual(_reentrancy_probability(unit.code), 0.05, unit.unit_id)
        applied = {item["unit_id"] for item in result["phases"]["fixer"]["patch_outcomes"] if item["status"] == "applied"}
        self.assertEqual(applied, {"Bank.withdraw", "Bank.withdrawAll"})
        self.assertEqual(result["phases"]["detector"]["fixed"]["verdict"], "Safe")

    # Detector không gọi được -> hàm ghi status error (không bịa xác suất).
    def test_unreachable_detector_is_reported_as_error(self) -> None:
        # Detector giả luôn ném lỗi kết nối.
        def broken_detect(*args, **kwargs):
            raise ConnectionError("server down")

        result = self._run(lambda prompt, **kwargs: json.dumps({"replacement": FIXED_WITHDRAW}), detect=broken_detect)
        original = result["phases"]["detector"]["original"]
        self.assertEqual(original["status"], "error")
        self.assertIsNone(original["vulnerability_probability"])
        self.assertEqual(result["status"], "partial")
        self.assertNotEqual(fuse_detection(OK_EMPTY, OK_EMPTY, original)["verdict"], "Safe")


if __name__ == "__main__":
    unittest.main()
