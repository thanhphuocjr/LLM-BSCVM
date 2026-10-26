"""Regression tests for the pipeline logic fixes (fusion fail-closed, corroboration,
RAG exact match, repair-loop carry-over, compile baseline, behaviour guard, CVSS)."""

from __future__ import annotations

import json
import re
import unittest
from unittest.mock import patch

from E2E.local import server_gpu_runner
from E2E.local.local_llm_runner import (
    _generate_advisor_section,
    _generate_assessor_section,
    get_cvss_defaults,
)
from E2E.local.tool_verifier import _evaluate_tool_result, _policy_verdict
from E2E.rag.retriever import _is_exact_code_match
from E2E.shared.function_parser import parse_code_units, validate_observable_behavior
from E2E.shared.fusion import fuse_detection, fuse_findings, merge_findings
from E2E.agents.assessor import compute_cvss_31_base_score


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


def _reentrancy_probability(code: str) -> float:
    """Fake detector: vulnerable iff a value call precedes the balance update."""
    return 0.95 if re.search(r"call\{value[\s\S]*balances\[msg\.sender\]\s*(=|-=)", code) else 0.05


class FusionFailClosedTests(unittest.TestCase):
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

    def test_partially_scored_detector_is_never_safe(self) -> None:
        summary = server_gpu_runner.summarize_detector(
            [
                {"unit_id": "A.f", "status": "ok", "label": "Safe", "vulnerability_probability": 0.1},
                {"unit_id": "A.g", "status": "error", "label": "Inconclusive", "vulnerability_probability": None},
            ],
            0.7,
        )
        self.assertEqual(summary["status"], "partial")
        self.assertEqual(summary["verdict"], "Inconclusive")
        self.assertEqual(fuse_detection(OK_EMPTY, OK_EMPTY, summary)["verdict"], "Inconclusive")

    def test_complete_detector_can_still_report_safe(self) -> None:
        summary = server_gpu_runner.summarize_detector(
            [{"unit_id": "A.f", "status": "ok", "label": "Safe", "vulnerability_probability": 0.1}],
            0.7,
        )
        self.assertEqual(fuse_detection(OK_EMPTY, OK_EMPTY, summary)["verdict"], "Safe")


class CorroborationTests(unittest.TestCase):
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

    def test_static_and_rag_are_corroborated(self) -> None:
        static = {"findings": [{"source": "slither", "swc": "SWC-107", "unit_id": "A.w", "confidence": 0.9}]}
        rag = {"findings": [{"source": "rag", "swc": "SWC-107", "unit_id": "A.w", "confidence": 0.7}]}
        self.assertTrue(merge_findings(static, rag)[0]["corroborated"])


class FuseFindingsTests(unittest.TestCase):
    def test_detector_only_unit_reaches_repair_agents(self) -> None:
        units = [{"unit_id": "A.f", "label": "Vulnerable", "vulnerability_probability": 0.9, "start_line": 3}]
        confirmed, _ = fuse_findings(units, {"findings": []}, {"findings": []})
        self.assertEqual([item["unit_id"] for item in confirmed], ["A.f"])
        self.assertEqual(confirmed[0]["swc"], "UNMAPPED")

    def test_errored_unit_is_neutral_not_safe(self) -> None:
        units = [{"unit_id": "A.w", "status": "error", "vulnerability_probability": None}]
        static = {"findings": [{"source": "static", "swc": "SWC-107", "unit_id": "A.w", "severity": "High", "confidence": 0.9}]}
        confirmed, candidates = fuse_findings(units, static, {"findings": []})
        finding = (confirmed + candidates)[0]
        self.assertEqual(finding["p_detector"], 0.5)
        self.assertTrue(finding["p_detector_missing"])


class RagExactMatchTests(unittest.TestCase):
    def test_short_getter_is_not_an_exact_match(self) -> None:
        getter = "function totalSupply() public view returns (uint256) { return _totalSupply; }"
        self.assertFalse(_is_exact_code_match(getter + " extra body " * 20, getter))
        self.assertFalse(_is_exact_code_match(getter, getter))

    def test_identical_function_is_an_exact_match(self) -> None:
        self.assertTrue(_is_exact_code_match(FIXED_WITHDRAW, FIXED_WITHDRAW))


class BehaviourGuardTests(unittest.TestCase):
    def test_accepts_local_rename_and_safe_transfer(self) -> None:
        validate_observable_behavior(
            "function f(uint a) public { uint x = a; total += x; emit E(x); }",
            "function f(uint a) public { uint y = a; total += y; emit E(y); }",
        )
        validate_observable_behavior(
            "function g() public { IERC20(t).transfer(a, 1); }",
            "function g() public { SafeERC20.safeTransfer(IERC20(t), a, 1); }",
        )

    def test_rejects_removed_value_call_state_write_and_event(self) -> None:
        with self.assertRaisesRegex(ValueError, "ETH value call"):
            validate_observable_behavior(UNFIXED_WITHDRAW_ALL, "function withdrawAll() external { balances[msg.sender] = 0; }")
        with self.assertRaisesRegex(ValueError, "state variable"):
            validate_observable_behavior("function s(uint v) public { total = v; }", "function s(uint v) public { }")
        with self.assertRaisesRegex(ValueError, "event"):
            validate_observable_behavior("function t() { Transfer(a, b, 1); }", "function t() { }")


class ExtractionTests(unittest.TestCase):
    def test_invalid_json_with_unescaped_quotes_is_decoded(self) -> None:
        from E2E.shared.function_parser import extract_declaration_body

        reply = (
            '{\n  "replacement": "function withdraw() external {\\n    balances[msg.sender] = 0;\\n'
            '    (bool ok, ) = msg.sender.call{value: amount}("");\\n    require(ok, \\"failed\\");\\n}",\n'
            '  "why": "CEI"\n}'
        )
        body = extract_declaration_body(reply, "withdraw")
        self.assertNotIn("\\n", body)
        self.assertIn('call{value: amount}("");\n', body)
        self.assertIn('require(ok, "failed");', body)


class CvssAndAgentsTests(unittest.TestCase):
    def test_fallback_cvss_matches_its_vector(self) -> None:
        for swc in ("SWC-107", "SWC-105", "SWC-115", "SWC-114", "SWC-102", "SWC-999"):
            meta = get_cvss_defaults(swc, "High")
            metrics = dict(part.split(":") for part in meta["cvss_vector"].split("/")[1:])
            expected, _ = compute_cvss_31_base_score(**{k.lower(): v for k, v in metrics.items()})
            self.assertEqual(meta["cvss_score"], expected, swc)

    def test_assessor_fallback_uses_computed_score(self) -> None:
        result = _generate_assessor_section([{"swc": "SWC-107", "unit_id": "A.w"}])
        self.assertEqual(result["assessments"][0]["cvss_score"], 9.1)
        self.assertEqual(result["assessments"][0]["severity"], "Critical")

    def test_advisor_false_positive_is_reported_to_caller(self) -> None:
        class FakeSuggestion:
            error = None
            is_true_positive = False

        class FakeAgent:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def advise_one(self, code, finding):
                return FakeSuggestion()

        findings = [{"finding_id": "SWC-105:A.setOwner", "swc": "SWC-105", "unit_id": "A.setOwner"}]
        with patch("E2E.local.local_llm_runner.AdvisorAgent", FakeAgent):
            advisor = _generate_advisor_section(findings, "", backend="server_gpu", use_llm=True)
        self.assertEqual(advisor["false_positives"], ["SWC-105:A.setOwner"])
        self.assertEqual(advisor["suggestions"], [])


class ToolPolicyTests(unittest.TestCase):
    def test_clean_run_of_non_covering_tool_is_inconclusive(self) -> None:
        result = _evaluate_tool_result("slither", "completed", [], [], {"front_running"})
        self.assertFalse(result["applicable"])
        policy = _policy_verdict({"slither": result}, require_applicable=True)
        self.assertEqual(policy["verdict"], "inconclusive")

    def test_residual_is_scoped_to_patched_unit_lines(self) -> None:
        finding = {"tool": "slither", "name": "reentrancy-eth", "category": "reentrancy", "line": [40], "severity": "High"}
        outside = _evaluate_tool_result("slither", "completed", [finding], [finding], {"reentrancy"}, None, {10, 11, 12})
        inside = _evaluate_tool_result("slither", "completed", [finding], [finding], {"reentrancy"}, None, {39, 40})
        self.assertEqual(outside["residual_target_findings"], [])
        self.assertEqual(len(inside["residual_target_findings"]), 1)


class FindingSignatureTests(unittest.TestCase):
    def test_signature_ignores_paths_and_line_numbers(self) -> None:
        from E2E.local.tool_verifier import _finding_signature

        before = {"name": "reentrancy-eth", "category": "reentrancy",
                  "message": "Reentrancy in Sale.withdraw() (../tmp/e2e-slither-aaaa/C.sol#914-930): call (C.sol#920)"}
        after = {"name": "reentrancy-eth", "category": "reentrancy",
                 "message": "Reentrancy in Sale.withdraw() (../tmp/e2e-slither-bbbb/C.sol#921-937): call (C.sol#927)"}
        self.assertEqual(_finding_signature(before), _finding_signature(after))


class CompileBaselineTests(unittest.TestCase):
    def test_missing_compiler_is_skipped_not_failed(self) -> None:
        skipped = {"passed": None, "status": "skipped", "error": "solc compiler not found"}
        with patch.object(
            server_gpu_runner,
            "check_compilation_with_baseline",
            return_value={"baseline": skipped, "fixed": skipped, "baseline_failed": False},
        ):
            result = server_gpu_runner.evaluate_compilation("a", "b")
        self.assertEqual(result["state"], "skipped")
        self.assertFalse(result["failed"])

    def test_original_that_cannot_compile_does_not_blame_the_patch(self) -> None:
        error = {"passed": False, "status": "failed", "error": 'ParserError: Source "x.sol" not found'}
        with patch.object(
            server_gpu_runner,
            "check_compilation_with_baseline",
            return_value={"baseline": error, "fixed": dict(error), "baseline_failed": True},
        ):
            result = server_gpu_runner.evaluate_compilation("a", "b")
        self.assertEqual(result["state"], "not_applicable")
        self.assertFalse(result["failed"])

    def test_new_compiler_error_is_a_failure(self) -> None:
        baseline = {"passed": False, "error": 'ParserError: Source "x.sol" not found'}
        fixed = {"passed": False, "error": baseline["error"] + "\nTypeError: Wrong argument count"}
        with patch.object(
            server_gpu_runner,
            "check_compilation_with_baseline",
            return_value={"baseline": baseline, "fixed": fixed, "baseline_failed": True},
        ):
            self.assertTrue(server_gpu_runner.evaluate_compilation("a", "b")["failed"])


class BatchSupportTests(unittest.TestCase):
    def test_version_rules_follow_the_pragma(self) -> None:
        rules_0424 = server_gpu_runner._version_rules("pragma solidity ^0.4.24;")
        rules_0418 = server_gpu_runner._version_rules("pragma solidity ^0.4.18;")
        self.assertNotIn("DO NOT use the `constructor` keyword", rules_0424)
        self.assertIn("stays a `constructor(...)`", rules_0424)
        self.assertIn("DO NOT use the `constructor` keyword", rules_0418)
        self.assertEqual(server_gpu_runner._version_rules("pragma solidity ^0.8.0;"), "")

    def test_compiler_errors_are_mapped_to_units(self) -> None:
        new_style = "Error: Invalid token.\n  --> /tmp/x.sol:11:35:\n"
        old_style = "/tmp/x.sol:17:9: Error: Undeclared identifier."
        self.assertEqual(server_gpu_runner.units_with_compiler_errors(BANK, new_style), {"Bank.withdraw"})
        self.assertEqual(server_gpu_runner.units_with_compiler_errors(BANK, old_style), {"Bank.withdrawAll"})


class BatchScheduleTests(unittest.TestCase):
    def test_active_hours_window_crossing_midnight(self) -> None:
        import time as _time

        from E2E.tools.batch_audit import parse_active_hours, within_active_hours

        window = parse_active_hours("19:00-07:30")

        def at(hour, minute):
            return _time.struct_time((2026, 1, 1, hour, minute, 0, 0, 1, -1))

        self.assertTrue(within_active_hours(window, at(23, 0)))
        self.assertTrue(within_active_hours(window, at(7, 29)))
        self.assertFalse(within_active_hours(window, at(7, 30)))
        self.assertFalse(within_active_hours(window, at(12, 0)))
        self.assertTrue(within_active_hours(parse_active_hours("09:00-17:00"), at(9, 0)))
        self.assertTrue(within_active_hours(None, at(12, 0)))


class ServerGpuRepairLoopTests(unittest.TestCase):
    """End-to-end run of the GPU runner against a fake server."""

    # The RAG channel is stubbed out so these tests exercise the repair loop itself.
    EMPTY_RAG = {"status": "ok", "findings": [], "retrieved_context": []}

    def _request(self) -> dict:
        from E2E.local.static_runner import run_static

        units = parse_code_units(BANK)
        return {
            "run_id": "fake-gpu",
            "source": {"code": BANK, "sha256": "x", "file_name": "Bank.sol"},
            "code_units": [unit.to_dict() for unit in units],
            "local_phase1": {"static": run_static(BANK, units), "rag": dict(self.EMPTY_RAG)},
            "execution": {"detector_threshold": 0.7, "repair": {"max_attempts": 3}},
        }

    def _run(self, generate, detect=None, repair=None) -> dict:
        def fake_detect(code, threshold=0.7, server_url=None, timeout=60):
            return {"vulnerable_probability": _reentrancy_probability(code)}

        def deterministic_advisor(findings, code="", **kwargs):
            return _generate_advisor_section(findings, code)

        def deterministic_assessor(findings, code="", **kwargs):
            return _generate_assessor_section(findings, code)

        with patch.object(server_gpu_runner, "check_server_health", return_value={"status": "ready"}), \
                patch.object(server_gpu_runner, "detect_via_server", side_effect=detect or fake_detect), \
                patch.object(server_gpu_runner, "generate_via_server", side_effect=generate), \
                patch.object(server_gpu_runner, "_generate_advisor_section", side_effect=deterministic_advisor), \
                patch.object(server_gpu_runner, "_generate_assessor_section", side_effect=deterministic_assessor), \
                patch.object(server_gpu_runner, "_apply_heuristic_security_patch", side_effect=lambda code, *a, **k: (code, [])), \
                patch.object(server_gpu_runner, "run_rag", return_value=dict(self.EMPTY_RAG)), \
                patch("builtins.print"):
            request = self._request()
            request["execution"]["repair"].update(repair or {})
            return server_gpu_runner.run_server_gpu_pipeline(request)

    def test_single_pass_keeps_first_patch_without_verification(self) -> None:
        prompts: list[str] = []

        def generate(prompt, **kwargs):
            prompts.append(prompt)
            if "Target vulnerable function: `withdrawAll`" in prompt:
                return json.dumps({"replacement": FIXED_WITHDRAW_ALL})
            return json.dumps({"replacement": FIXED_WITHDRAW})

        with patch.object(server_gpu_runner, "verify_candidate", side_effect=AssertionError("must not verify")):
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

    def test_single_pass_keeps_guard_violation_as_warning(self) -> None:
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

    def test_single_pass_reports_unchanged_reply_with_rationale(self) -> None:
        original_withdraw_all = next(u.code for u in parse_code_units(BANK) if u.name == "withdrawAll")

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

    def test_second_attempt_keeps_first_attempt_patch(self) -> None:
        calls = {"withdrawAll": 0}

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

    def test_unreachable_detector_is_reported_as_error(self) -> None:
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
