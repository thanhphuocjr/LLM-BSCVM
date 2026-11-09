"""Regression tests for the pipeline logic fixes (fusion fail-closed, corroboration,
RAG exact match, repair-loop carry-over, compile baseline, behaviour guard, CVSS)."""

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

    # Advisor kết luận false positive -> finding nằm trong false_positives và không có suggestion.
    def test_advisor_false_positive_is_reported_to_caller(self) -> None:
        # Kết quả Advisor giả (is_true_positive=False).
        class FakeSuggestion:
            error = None
            is_true_positive = False

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
        self.assertEqual(advisor["suggestions"], [])


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


# Các hàm hỗ trợ batch/Fixer.
class BatchSupportTests(unittest.TestCase):
    # _version_rules sinh luật cú pháp đúng theo pragma.
    def test_version_rules_follow_the_pragma(self) -> None:
        rules_0424 = fixer._version_rules("pragma solidity ^0.4.24;")
        rules_0418 = fixer._version_rules("pragma solidity ^0.4.18;")
        self.assertNotIn("DO NOT use the `constructor` keyword", rules_0424)
        self.assertIn("stays a `constructor(...)`", rules_0424)
        self.assertIn("DO NOT use the `constructor` keyword", rules_0418)
        self.assertEqual(fixer._version_rules("pragma solidity ^0.8.0;"), "")

    # Lỗi solc được ánh xạ đúng về hàm chứa dòng lỗi.
    def test_compiler_errors_are_mapped_to_units(self) -> None:
        new_style = "Error: Invalid token.\n  --> /tmp/x.sol:11:35:\n"
        old_style = "/tmp/x.sol:17:9: Error: Undeclared identifier."
        self.assertEqual(candidate_check.units_with_compiler_errors(BANK, new_style), {"Bank.withdraw"})
        self.assertEqual(candidate_check.units_with_compiler_errors(BANK, old_style), {"Bank.withdrawAll"})


# Khung giờ chạy của batch.
class BatchScheduleTests(unittest.TestCase):
    # Khung "19:00-07:30" vắt qua nửa đêm được xử lý đúng.
    def test_active_hours_window_crossing_midnight(self) -> None:
        import time as _time

        from E2E.experiments.batch_audit import parse_active_hours, within_active_hours

        window = parse_active_hours("19:00-07:30")

        # Tạo thời điểm giả (giờ:phút) để kiểm tra.
        def at(hour, minute):
            return _time.struct_time((2026, 1, 1, hour, minute, 0, 0, 1, -1))

        self.assertTrue(within_active_hours(window, at(23, 0)))
        self.assertTrue(within_active_hours(window, at(7, 29)))
        self.assertFalse(within_active_hours(window, at(7, 30)))
        self.assertFalse(within_active_hours(window, at(12, 0)))
        self.assertTrue(within_active_hours(parse_active_hours("09:00-17:00"), at(9, 0)))
        self.assertTrue(within_active_hours(None, at(12, 0)))


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
    def _run(self, generate, detect=None, repair=None) -> dict:
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
                patch.object(gpu_pipeline, "_apply_heuristic_security_patch", side_effect=lambda code, *a, **k: (code, [])), \
                patch.object(candidate_check, "run_rag", return_value=dict(self.EMPTY_RAG)), \
                patch("builtins.print"):
            request = self._request()
            request["execution"]["repair"].update(repair or {})
            return gpu_pipeline.run_server_gpu_pipeline(request)

    # single_pass: lấy bản vá đầu tiên, gọi Fixer đúng 1 lần, không kiểm chứng.
    def test_single_pass_keeps_first_patch_without_verification(self) -> None:
        prompts: list[str] = []

        # Fixer giả trả bản vá đúng (CEI).
        def generate(prompt, **kwargs):
            prompts.append(prompt)
            if "Target vulnerable function: `withdrawAll`" in prompt:
                return json.dumps({"replacement": FIXED_WITHDRAW_ALL})
            return json.dumps({"replacement": FIXED_WITHDRAW})

        with patch.object(gpu_pipeline, "verify_candidate", side_effect=AssertionError("must not verify")):
            result = self._run(generate, repair={"single_pass": True})
        verifier = result["phases"]["verifier"]
        self.assertEqual(verifier["overall_verdict"], "UNVERIFIED")
        self.assertEqual(verifier["attempts_count"], 1)
        self.assertEqual(len(prompts), 2)
        applied = {item["unit_id"] for item in result["patch_outcomes"] if item["status"] == "applied"}
        self.assertEqual(applied, {"Bank.withdraw", "Bank.withdrawAll"})
        self.assertEqual(
            {item["status"] for item in verifier["verifications"] if item["unit_id"] in applied}, {"SKIPPED"}
        )
        self.assertEqual(result["phases"]["detector"]["fixed"]["unit_results"], [])

    # single_pass: bản vá vi phạm guard vẫn được giữ, chỉ ghi guard_warning.
    def test_single_pass_keeps_guard_violation_as_warning(self) -> None:
        # Fixer giả trả bản vá xoá mất hành vi.
        def generate(prompt, **kwargs):
            if "Target vulnerable function: `withdraw`" in prompt:
                return json.dumps({"replacement": (
                    "function withdraw(uint256 amount) external {\n"
                    "        require(balances[msg.sender] >= amount);\n"
                    "    }"
                )})
            return json.dumps({"replacement": FIXED_WITHDRAW_ALL})

        result = self._run(generate, repair={"single_pass": True})
        outcome = next(item for item in result["patch_outcomes"] if item["unit_id"] == "Bank.withdraw")
        self.assertEqual(outcome["status"], "applied")
        self.assertIn("removed", outcome["guard_warning"])

    # single_pass: model trả nguyên hàm cũ -> status "unchanged" kèm lý do của model.
    def test_single_pass_reports_unchanged_reply_with_rationale(self) -> None:
        original_withdraw_all = next(u.code for u in parse_code_units(BANK) if u.name == "withdrawAll")

        # Fixer giả trả lại hàm gốc.
        def generate(prompt, **kwargs):
            if "Target vulnerable function: `withdrawAll`" in prompt:
                return json.dumps({"replacement": original_withdraw_all, "why": "Not exploitable here."})
            return json.dumps({"replacement": FIXED_WITHDRAW, "why": "Moved the state update before the call."})

        result = self._run(generate, repair={"single_pass": True})
        by_unit = {item["unit_id"]: item for item in result["patch_outcomes"]}
        self.assertEqual(by_unit["Bank.withdrawAll"]["status"], "unchanged")
        self.assertEqual(by_unit["Bank.withdrawAll"]["rationale"], "Not exploitable here.")
        self.assertEqual(by_unit["Bank.withdraw"]["rationale"], "Moved the state update before the call.")
        self.assertEqual({item["engine"] for item in by_unit.values()}, {"llm_fixer"})
        self.assertEqual(result["phases"]["assessor"]["status"], "skipped")

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
        fixed_code = result["fixed_code"]
        self.assertEqual(verifier["overall_verdict"], "PASS")
        self.assertEqual(verifier["attempts_count"], 2)
        self.assertEqual(_reentrancy_probability(FIXED_WITHDRAW), 0.05)
        for unit in parse_code_units(fixed_code):
            if unit.name in {"withdraw", "withdrawAll"}:
                self.assertEqual(_reentrancy_probability(unit.code), 0.05, unit.unit_id)
        applied = {item["unit_id"] for item in result["patch_outcomes"] if item["status"] == "applied"}
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
