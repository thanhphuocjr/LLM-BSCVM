from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from E2E.common.io_utils import read_json, sha256_text
from E2E.config import DEMO_CONTRACTS_DIR, DETECTOR_ADAPTER_DIR
from E2E.phase0_parsing.code_units import parse_code_units
from E2E.phase1_fusion.fusion import fuse_detection
from E2E.phase1a_static.static_runner import run_static
from E2E.phase1b_rag.rag_runner import run_rag
from E2E.phase4_fixer.patching import apply_unit_patches
from E2E.phase5_verifier.behavior_verifier import verify_patch_behavior
from E2E.phase5_verifier.final_verification import build_verification_channels
from E2E.phase6_reporter.report import (
    DEFAULT_RECOMMENDATIONS,
    apply_verification_to_report,
    build_code_comparison_rows,
    build_report_data,
    render_html,
    render_markdown,
)
from E2E.pipeline.orchestrator import run_pipeline


EXAMPLES = DEMO_CONTRACTS_DIR


# Kiểm tra Phase 0 (tách hàm) và việc ghép bản vá theo unit_id.
class ParserTests(unittest.TestCase):
    # parse_code_units tách đúng hàm khi có ngoặc lồng nhau và dấu { } trong chuỗi.
    def test_parser_handles_nested_braces_and_strings(self) -> None:
        code = """
        contract C {
            // function fake() public {}
            function a() public {
                string memory value = "}";
                if (true) { value = "{"; }
            }
            function b() external {}
        }
        """
        units = parse_code_units(code)
        self.assertEqual([unit.unit_id for unit in units], ["C.a", "C.b"])
        self.assertIn('value = "{"', units[0].code)

    # apply_unit_patches ghép bản vá đúng hàm theo unit_id.
    def test_patch_is_applied_by_stable_unit_id(self) -> None:
        code = "contract C { function a() public { uint x = 1; } }"
        units = parse_code_units(code)
        fixed, outcomes = apply_unit_patches(
            code,
            units,
            [{"unit_id": "C.a", "replacement": "function a() public { uint x = 2; }"}],
        )
        self.assertIn("x = 2", fixed)
        self.assertEqual(outcomes[0]["status"], "applied")


# Kiểm tra các kênh phát hiện (static, RAG), fusion và helper tri thức RAG trên contract demo.
class DetectorTests(unittest.TestCase):
    # Đọc VulnerableBank.sol và SafeBank.sol trong demo_contracts/.
    @classmethod
    def setUpClass(cls) -> None:
        cls.vulnerable = (EXAMPLES / "VulnerableBank.sol").read_text(encoding="utf-8")
        cls.safe = (EXAMPLES / "SafeBank.sol").read_text(encoding="utf-8")

    # Static thấy SWC-107 ở VulnerableBank, không báo nhầm SWC-104, và SafeBank không có finding.
    def test_static_reentrancy_and_safe_control(self) -> None:
        vulnerable = run_static(self.vulnerable)
        safe = run_static(self.safe)
        self.assertTrue(any(item["swc"] == "SWC-107" for item in vulnerable["findings"]))
        self.assertFalse(any(item["swc"] == "SWC-104" for item in vulnerable["findings"]))
        self.assertEqual(safe["findings"], [])

    # RAG: finding đầu tiên của VulnerableBank là SWC-107; SafeBank không có finding.
    def test_rag_reentrancy_and_safe_control(self) -> None:
        vulnerable = run_rag(self.vulnerable)
        safe = run_rag(self.safe)
        self.assertEqual(vulnerable["findings"][0]["swc"], "SWC-107")
        self.assertEqual(safe["findings"], [])

    # Fusion fail-closed: mọi kênh lỗi -> Inconclusive; thiếu detector -> Inconclusive; đủ kênh và an toàn -> Safe.
    def test_fusion_does_not_fail_open(self) -> None:
        error = {"status": "error", "score": 0.0, "verdict": "Error", "findings": []}
        result = fuse_detection(error, error, error)
        self.assertEqual(result["verdict"], "Inconclusive")

        local_safe = fuse_detection(run_static(self.safe), run_rag(self.safe), None)
        self.assertEqual(local_safe["verdict"], "Inconclusive")

        llm_safe = {"status": "ok", "score": 0.05, "threshold": 0.25, "verdict": "Safe"}
        complete_safe = fuse_detection(run_static(self.safe), run_rag(self.safe), llm_safe)
        self.assertEqual(complete_safe["verdict"], "Safe")

        llm_only_positive = {
            "status": "ok",
            "score": 0.30,
            "threshold": 0.25,
            "verdict": "Vulnerable",
            "unit_results": [
                {
                    "unit_id": "SafeBank.withdraw",
                    "label": "Vulnerable",
                    "vulnerability_probability": 0.30,
                }
            ],
        }
        positive = fuse_detection(
            {"status": "ok", "score": 0.0, "verdict": "Safe", "findings": []},
            {"status": "ok", "score": 0.0, "verdict": "Safe", "findings": []},
            llm_only_positive,
        )
        self.assertEqual(positive["verdict"], "Vulnerable")
        self.assertEqual(positive["findings"][0]["swc"], "UNMAPPED")

    # Detector offline: trọng số chia 50/50 cho static/RAG và sàn bằng chứng vẫn hoạt động.
    def test_adaptive_fusion_when_detector_offline(self) -> None:
        """Kiểm tra khi detector offline/pending, RAG và Static được chia đều 50/50 và floor hoạt động."""
        rag_vuln = {
            "status": "ok",
            "score": 0.85,
            "verdict": "Vulnerable",
            "findings": [
                {
                    "swc": "SWC-107",
                    "unit_id": "Bank.withdraw",
                    "title": "Reentrancy",
                    "severity": "High",
                    "confidence": 0.85,
                    "source": "rag",
                }
            ],
        }
        static_empty = {"status": "ok", "score": 0.0, "verdict": "Safe", "findings": []}
        detector_pending = {"component": "llm_detector", "status": "pending", "score": 0.0, "verdict": "Pending"}

        result = fuse_detection(static_empty, rag_vuln, detector_pending)
        self.assertEqual(result["verdict"], "Vulnerable")
        self.assertGreaterEqual(result["score"], 0.65)
        self.assertEqual(result["configured_weights"]["rag"], 0.50)
        self.assertEqual(result["configured_weights"]["static"], 0.50)

    # Detector cho điểm thấp (0.05) không được "dập tắt" phát hiện mạnh của RAG/static.
    def test_adaptive_fusion_prevents_detector_false_negative_suppression(self) -> None:
        """Kiểm tra khi detector trả về score thấp (0.05), phát hiện có độ tin cậy cao của RAG không bị dập tắt."""
        rag_vuln = {
            "status": "ok",
            "score": 0.88,
            "verdict": "Vulnerable",
            "findings": [
                {
                    "swc": "SWC-107",
                    "unit_id": "Vault.withdraw",
                    "title": "Reentrancy",
                    "severity": "High",
                    "confidence": 0.88,
                    "source": "rag",
                }
            ],
        }
        static_empty = {"status": "ok", "score": 0.0, "verdict": "Safe", "findings": []}
        detector_false_negative = {
            "component": "llm_detector",
            "status": "ok",
            "score": 0.05,
            "threshold": 0.25,
            "verdict": "Safe",
            "unit_results": [],
        }

        result = fuse_detection(static_empty, rag_vuln, detector_false_negative)
        self.assertEqual(result["verdict"], "Vulnerable")
        self.assertGreaterEqual(result["score"], 0.50)
        self.assertEqual(result["configured_weights"]["rag"], 0.40)

    # _build_rag_knowledge_section ưu tiên đúng ca cùng SWC và định dạng đúng.
    def test_rag_knowledge_section_builder(self) -> None:
        """Kiểm tra helper _build_rag_knowledge_section định dạng và ưu tiên đúng case."""
        from E2E.phase4_fixer.fixer import _build_rag_knowledge_section

        self.assertEqual(_build_rag_knowledge_section([]), "")

        rag_ctx = [
            {
                "swc": "SWC-115",
                "title": "tx.origin",
                "similarity": 0.45,
                "root_cause": "tx.origin used for auth",
                "remediation": "use msg.sender",
                "vulnerable_code": "require(tx.origin == owner);",
                "fixed_code": "require(msg.sender == owner);",
            },
            {
                "swc": "SWC-107",
                "title": "Reentrancy",
                "similarity": 0.82,
                "root_cause": "State updated after call",
                "remediation": "Apply CEI pattern",
                "vulnerable_code": "msg.sender.call{value: bal}(); bal = 0;",
                "fixed_code": "bal = 0; msg.sender.call{value: bal}();",
            },
        ]

        section = _build_rag_knowledge_section(rag_ctx, target_swc="SWC-107", max_items=2)
        self.assertIn("RAG Retrieved Knowledge", section)
        self.assertIn("SWC-107", section)
        self.assertIn("SWC-115", section)
        self.assertIn("Apply CEI pattern", section)


# Kiểm tra tích hợp Phase 5 + Phase 6: feedback, hồi quy hành vi, báo cáo, ma trận tool.
class IntegrationTests(unittest.TestCase):
    # build_verification_channels ghi rõ lý do thất bại của từng kênh static / RAG / detector trên bản vá.
    def test_verification_channels_contain_failed_static_rag_and_llm_reasons(self) -> None:
        static_result = {
            "status": "ok",
            "verdict": "Vulnerable",
            "findings": [
                {
                    "swc": "SWC-107",
                    "unit_id": "C.withdraw",
                    "title": "Reentrancy",
                    "description": "State is updated after the external call.",
                }
            ],
        }
        rag_result = {
            "status": "ok",
            "verdict": "Vulnerable",
            "findings": [
                {
                    "swc": "SWC-107",
                    "unit_id": "C.withdraw",
                    "remediation": "Apply checks-effects-interactions.",
                }
            ],
        }
        detector = {
            "status": "ok",
            "verdict": "Vulnerable",
            "unit_results": [{"unit_id": "C.withdraw", "label": "Vulnerable", "vulnerability_probability": 0.91}],
        }
        channels = build_verification_channels(
            {"compile": {"status": "passed"}},
            static_result=static_result,
            rag_result=rag_result,
            detector_fixed=detector,
        )
        self.assertFalse(any(channels[name]["passed"] for name in ("static", "rag", "llm")))
        self.assertTrue(channels["compile"]["passed"])
        self.assertEqual(channels["static"]["findings"][0]["swc"], "SWC-107")
        self.assertIn("checks-effects-interactions", channels["rag"]["findings"][0]["remediation"])
        self.assertEqual(channels["llm"]["reason"], "The neural detector still flags: C.withdraw (0.910).")

    # Bản vá làm mất hành vi -> verify_patch_behavior failed và báo cáo ghi "must not be deployed" dù model nói đã sửa.
    def test_behavioral_regression_overrides_model_narrative(self) -> None:
        original = """
        contract C {
            mapping(address => uint256) balance;
            function withdraw() external {
                uint256 amount = balance[msg.sender];
                (bool ok,) = msg.sender.call{value: amount}("");
                require(ok);
                balance[msg.sender] = 0;
            }
        }
        """
        fixed = """
        contract C {
            mapping(address => uint256) balance;
            function withdraw() external {
                balance[msg.sender] = 0;
            }
        }
        """
        behavior = verify_patch_behavior(
            original,
            fixed,
            [{"unit_id": "C.withdraw", "status": "applied"}],
            {
                "slither": {
                    "findings": [
                        {
                            "check": "locked-ether",
                            "impact": "Medium",
                            "description": "Contract cannot withdraw Ether.",
                        }
                    ]
                }
            },
        )
        self.assertEqual(behavior["status"], "failed")
        self.assertTrue(
            any("ETH value call" in item["evidence"] for item in behavior["findings"])
        )

        report_data = {
            "executive_summary": "The model says the patch is Fixed.",
            "findings": [{"verification_status": "Fixed", "residual_risk": "None"}],
            "phase_rows": [{"phase": "5 Verifier", "status": "ok"}],
            "recommendations": [],
        }
        apply_verification_to_report(
            report_data,
            {
                "status": "complete",
                "overall_verdict": "Patch rejected: behavioral regression detected",
                "behavioral_verification": behavior,
            },
        )
        self.assertIn("must not be deployed", report_data["executive_summary"])
        self.assertEqual(
            report_data["findings"][0]["verification_status"],
            "Rejected by deterministic verifier",
        )
        self.assertEqual(report_data["phase_rows"][0]["status"], "complete")

    # Bản vá bị reject: chỉ finding đã vá bị ghi "Rejected"; finding chưa vá giữ nguyên trạng thái.
    def test_rejected_patch_keeps_status_of_unpatched_findings(self) -> None:
        report_data = {
            "executive_summary": "",
            "findings": [
                {"self_check_status": "PASS", "verification_status": "self-check passed"},
                {"self_check_status": "NOT_PATCHED", "verification_status": "Not patched"},
            ],
            "phase_rows": [],
            "recommendations": [],
        }
        apply_verification_to_report(
            report_data,
            {"status": "complete", "overall_verdict": "Patch rejected: fixed code does not compile"},
        )
        statuses = [item["verification_status"] for item in report_data["findings"]]
        self.assertEqual(statuses[0], "Rejected by deterministic verifier")
        self.assertEqual(statuses[1], "Not patched")

    # Phase 6 tự dựng tóm tắt (số finding, số bị Advisor loại, số bản vá) và dùng bộ khuyến nghị mặc định.
    def test_report_summary_and_recommendations_come_from_phase_6(self) -> None:
        request = {
            "run_id": "test-summary",
            "source": {"file_name": "Bank.sol", "sha256": "0" * 64, "code": "contract Bank {}"},
            "code_units": [],
            "local_phase1": {"static": {"findings": []}, "rag": {"findings": []}},
        }
        finding = {"finding_id": "SWC-107:Bank.withdraw", "swc": "SWC-107", "unit_id": "Bank.withdraw",
                   "title": "Reentrancy", "severity": "Critical"}
        remote = {"phases": {
            "detector": {"confirmed_findings": [finding], "dismissed_findings": [dict(finding, swc="SWC-102")]},
            "fixer": {"patch_outcomes": [{"unit_id": "Bank.withdraw", "status": "applied"}]},
        }}
        data = build_report_data(request, {"verdict": "Vulnerable", "score": 0.9}, remote, {})
        self.assertEqual(
            data["executive_summary"],
            "Audit completed using the GPU server (Qwen2.5-Coder-14B v13) for run test-summary. "
            "1 confirmed finding(s), 1 dismissed by the Advisor, 1 patch(es) applied.",
        )
        self.assertEqual(data["recommendations"], DEFAULT_RECOMMENDATIONS)

    # Báo cáo có bảng so sánh mã gốc / mã vá và đánh dấu dòng thay đổi.
    def test_fixed_code_report_renders_side_by_side_diff(self) -> None:
        original = "contract C {\n    uint x;\n    function f() public {}\n}"
        fixed = "contract C {\n    uint256 x;\n    function f() public {}\n}"
        rows = build_code_comparison_rows(original, fixed)
        self.assertTrue(any(row["fixed_changed"] and "uint256" in row["fixed"] for row in rows))

        report_data = {
            "title": "Smart Contract Audit Report",
            "run_id": "diff-test",
            "analyzed_object": "C",
            "contract_name": "C",
            "file_name": "C.sol",
            "source_sha256": "abc",
            "detection_result": "Vulnerable",
            "analysis_status": "complete",
            "score": 0.9,
            "risk_level": "High",
            "audit_time": "2026-08-11T00:00:00+07:00",
            "executive_summary": "Summary.",
            "findings": [],
            "distribution": {},
            "phase_rows": [],
            "original_code": original,
            "fixed_code": fixed,
            "patch_verdict": "Fixed",
            "tool_verification": {},
            "recommendations": [],
            "limitations": [],
            "disclaimer": "Disclaimer.",
        }
        markdown = render_markdown(report_data)
        html_report = render_html(report_data)
        self.assertIn("Source Code (Sample)", markdown)
        self.assertIn("Fixed Source Code", markdown)
        self.assertIn("<mark>    uint256 x;</mark>", markdown)
        self.assertIn("code-line changed", html_report)

    # Mã viết trên một dòng được tách thành từng câu lệnh để diff dễ đọc.
    def test_fixed_code_diff_expands_one_line_patch_into_statements(self) -> None:
        original = """contract C {
    function transfer(address to, uint256 amount) external {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
        balances[to] += amount;
    }
}"""
        fixed = (
            "contract C { function transfer(address to, uint256 amount) external { "
            'require(to != address(0), "Invalid recipient"); '
            'require(balances[msg.sender] >= amount, "Insufficient balance"); '
            "balances[msg.sender] -= amount; balances[to] += amount; } }"
        )

        rows = build_code_comparison_rows(original, fixed)
        fixed_lines = [row["fixed"].strip() for row in rows if row["fixed"].strip()]
        self.assertIn('require(to != address(0), "Invalid recipient");', fixed_lines)
        self.assertIn("balances[msg.sender] -= amount;", fixed_lines)
        self.assertIn("balances[to] += amount;", fixed_lines)
        self.assertFalse(any("; balances" in line for line in fixed_lines))

        call_rows = build_code_comparison_rows(
            "contract C { function f() external {} }",
            'contract C { function f() external { target.call{value: 1}(""); } }',
        )
        self.assertTrue(
            any('target.call{value: 1}("");' in row["fixed"] for row in call_rows)
        )

    # Chạy run_pipeline lượt 2 với kết quả GPU giả lập -> đủ artifact, manifest, báo cáo MD/HTML.
    def test_remote_result_completes_report(self) -> None:
        code = (EXAMPLES / "VulnerableBank.sol").read_text(encoding="utf-8")
        run_id = "mock-remote"
        fixed_function = """function withdraw() external {
        uint256 amount = balances[msg.sender];
        require(amount > 0, "No balance");
        balances[msg.sender] = 0;
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
    }"""
        remote = {
            "schema_version": "1.0.0",
            "run_id": run_id,
            "source_sha256": sha256_text(code),
            "status": "ok",
            "models": {
                "detector": "Qwen2.5-Coder-14B-Instruct + LoRA v13",
                "generator": "Qwen2.5-Coder-14B-Instruct",
            },
            "timings": {"total_ms": 1},
            "errors": [],
            "phases": {
                "detector": {
                    "status": "ok",
                    "original": {
                        "status": "ok",
                        "score": 0.95,
                        "threshold": 0.25,
                        "verdict": "Vulnerable",
                    },
                    "fixed": {
                        "status": "ok",
                        "score": 0.05,
                        "threshold": 0.25,
                        "verdict": "Safe",
                    },
                },
                "advisor": {
                    "status": "ok",
                    "suggestions": [
                        {
                            "finding_id": "SWC-107:VulnerableBank.withdraw",
                            "swc": "SWC-107",
                            "unit_id": "VulnerableBank.withdraw",
                            "root_cause": "State is updated after the external value call.",
                            "impact": "An attacker can withdraw repeatedly.",
                            "repair_steps": ["Move the state update before the call."],
                        }
                    ],
                },
                "assessor": {
                    "status": "ok",
                    "assessments": [
                        {
                            "finding_id": "SWC-107:VulnerableBank.withdraw",
                            "swc": "SWC-107",
                            "unit_id": "VulnerableBank.withdraw",
                            "cvss_score": 9.0,
                            "impact": "Loss of contract funds.",
                            "repair_priority": "Immediate",
                        }
                    ],
                },
                "fixer": {
                    "status": "ok",
                    "fixed_code": code.replace(
                        parse_code_units(code)[1].code,
                        fixed_function,
                    ),
                    "patch_outcomes": [
                        {"unit_id": "VulnerableBank.withdraw", "status": "applied"}
                    ],
                },
                "verifier": {
                    "status": "ok",
                    "overall_verdict": "Fixed with manual review required",
                    "verifications": [
                        {
                            "finding_id": "SWC-107:VulnerableBank.withdraw",
                            "swc": "SWC-107",
                            "unit_id": "VulnerableBank.withdraw",
                            "status": "Fixed",
                            "residual_risk": "Low",
                        }
                    ],
                },
            },
        }
        tool_result = {
            "component": "tool_verification",
            "status": "ok",
            "policy_verdict": "passed",
            "policy_reason": "1/1 tools completed without residual target findings.",
            "compile": {"status": "passed"},
            "slither": {"status": "passed", "findings": []},
            "tools": {
                "slither": {
                    "run_status": "completed",
                    "verdict": "PASS",
                    "findings": [],
                    "residual_target_findings": [],
                    "introduced_findings": [],
                    "detail": "No findings after fix.",
                }
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("E2E.phase5_verifier.final_verification.verify_security_tool_matrix", return_value=tool_result):
                manifest = run_pipeline(
                    code,
                    file_name="VulnerableBank.sol",
                    run_id=run_id,
                    runs_dir=temp_dir,
                    remote_result=remote,
                )
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["detection"]["verdict"], "Vulnerable")
            self.assertTrue(Path(manifest["reports"]["pdf"]).exists())
            verification = json.loads(
                (Path(temp_dir) / run_id / "phase5_verification.json").read_text(encoding="utf-8")
            )
            self.assertEqual(verification["fixed_detection"]["verdict"], "Safe")

        disagreement_run_id = "mock-detector-disagreement"
        remote["run_id"] = disagreement_run_id
        remote["status"] = "ok"
        remote["phases"]["detector"]["fixed"] = {
            "status": "ok",
            "score": 0.95,
            "threshold": 0.25,
            "verdict": "Vulnerable",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("E2E.phase5_verifier.final_verification.verify_security_tool_matrix", return_value=tool_result):
                disagreement = run_pipeline(
                    code,
                    file_name="VulnerableBank.sol",
                    run_id=disagreement_run_id,
                    runs_dir=temp_dir,
                    remote_result=remote,
            )
            self.assertEqual(
                disagreement["verification"]["overall_verdict"],
                "Patch accepted by Slither",
            )
            disagreement_report = json.loads(
                (Path(temp_dir) / disagreement_run_id / "phase6_report_data.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertNotIn("not accepted", disagreement_report["executive_summary"])

        rejected_run_id = "mock-false-fixed"
        remote["run_id"] = rejected_run_id
        remote["status"] = "partial"
        remote["phases"]["detector"]["fixed"] = {
            "status": "ok",
            "score": 0.95,
            "threshold": 0.25,
            "verdict": "Vulnerable",
        }
        remote["phases"]["fixer"] = {
            "status": "error",
            "fixed_code": code,
            "patch_outcomes": [],
        }
        remote["phases"]["verifier"] = {
            "status": "ok",
            "overall_verdict": "Fixed",
            "verifications": [],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("E2E.phase5_verifier.final_verification.verify_security_tool_matrix", return_value=tool_result):
                rejected = run_pipeline(
                    code,
                    file_name="VulnerableBank.sol",
                    run_id=rejected_run_id,
                    runs_dir=temp_dir,
                    remote_result=remote,
            )
            self.assertEqual(
                rejected["verification"]["overall_verdict"],
                "Patch rejected: no effective patch was applied",
            )
            report_data = json.loads(
                (Path(temp_dir) / rejected_run_id / "phase6_report_data.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                report_data["patch_verdict"],
                "Patch rejected: no effective patch was applied",
            )
            # No patch exists, so the summary says so and no "do not deploy the patch" warning is added.
            self.assertIn("No patch was verified: Patch rejected: no effective patch was applied",
                          report_data["executive_summary"])
            self.assertEqual(report_data["recommendations"], DEFAULT_RECOMMENDATIONS)

    # Slither không chạy được -> SKIPPED và kết luận inconclusive; chạy sạch -> passed.
    def test_tool_policy_for_skipped_and_clean_slither(self) -> None:
        from E2E.phase5_verifier.tool_matrix import _evaluate_tool_result, _policy_verdict

        skipped_eval = _evaluate_tool_result(
            "slither",
            "unavailable",
            [],
            [],
            {"reentrancy"},
            error="slither not installed",
        )
        self.assertEqual(skipped_eval["verdict"], "SKIPPED")
        self.assertEqual(_policy_verdict({"slither": skipped_eval})["verdict"], "inconclusive")

        policy = _policy_verdict({"slither": {"run_status": "completed", "verdict": "PASS"}})
        self.assertEqual(policy["verdict"], "passed")

    # Slither FAIL thì báo cáo Markdown/HTML phải ghi "failed" (không bao giờ "passed").
    def test_slither_status_consistency_when_failed(self) -> None:
        from E2E.phase5_verifier.tool_matrix import _evaluate_tool_result
        evaluated_slither = _evaluate_tool_result(
            "slither",
            "completed",
            [
                {
                    "tool": "slither",
                    "name": "erc20-interface",
                    "check": "erc20-interface",
                    "category": "access_control",
                    "severity": "Medium",
                    "description": "Incorrect interface",
                }
            ],
            [],
            {"access_control"},
        )
        self.assertEqual(evaluated_slither["verdict"], "FAIL")

        request = {
            "run_id": "test-slither-fail",
            "source": {"file_name": "Test.sol", "sha256": "0" * 64, "code": "contract Test {}"},
            "code_units": [],
            "local_phase1": {
                "static": {"findings": []},
                "rag": {"findings": []},
            },
        }
        fused = {"verdict": "Vulnerable", "score": 0.8, "findings": []}
        tool_verification = {
            "component": "tool_verification",
            "status": "partial",
            "policy_verdict": "failed",
            "policy_reason": "Security tools still report residual findings: slither",
            "compile": {"status": "passed"},
            "tools": {"slither": evaluated_slither},
            "slither": {
                "status": "failed",
                "findings": evaluated_slither["findings"],
                "verdict": "FAIL",
            },
            "behavioral": {"status": "passed", "findings": []},
        }

        report_data = build_report_data(request, fused, None, tool_verification)
        self.assertEqual(report_data["tool_verification"]["slither"]["status"], "failed")

        md = render_markdown(report_data)
        self.assertIn("- **Slither:** failed", md)
        self.assertNotIn("- **Slither:** passed", md)

        html = render_html(report_data)
        self.assertIn("Slither: <b>failed</b>", html)
        self.assertNotIn("Slither: <b>passed</b>", html)

    # Một mức độ duy nhất: báo cáo dùng mức CVSS của Phase 3 (phân bố, highest risk, tiêu đề), giữ mức lúc phát hiện.
    def test_report_severity_is_the_cvss_level(self) -> None:
        finding = {
            "finding_id": "SWC-107:Bank.withdraw", "swc": "SWC-107", "unit_id": "Bank.withdraw",
            "title": "Reentrancy", "severity": "Critical", "confidence": 0.99, "sources": ["static"],
        }
        request = {
            "run_id": "test-severity",
            "source": {"file_name": "Bank.sol", "sha256": "0" * 64, "code": "contract Bank {}"},
            "code_units": [],
            "local_phase1": {"static": {"findings": []}, "rag": {"findings": []}},
        }
        remote = {
            "phases": {
                "detector": {"confirmed_findings": [finding], "dismissed_findings": []},
                "assessor": {"assessments": [{
                    "finding_id": "SWC-107:Bank.withdraw", "swc": "SWC-107", "unit_id": "Bank.withdraw",
                    "severity": "High", "cvss_score": 8.2, "cvss_source": "llm", "repair_priority": 1,
                }]},
            }
        }
        data = build_report_data(request, {"verdict": "Vulnerable", "score": 0.9}, remote, {})
        self.assertEqual(data["findings"][0]["severity"], "High")
        self.assertEqual(data["findings"][0]["detected_severity"], "Critical")
        self.assertEqual(data["risk_level"], "High")
        self.assertEqual(dict(data["distribution"]), {"High": 1})
        md = render_markdown(data)
        self.assertIn("#### 1. Reentrancy (SWC-107) - High", md)
        self.assertIn("8.2 (High, LLM-selected metrics, detected as Critical); repair priority #1", md)

    # Lượt 2: báo cáo chỉ liệt kê finding Advisor xác nhận; finding bị loại nằm ở mục 4.5.
    # Bản vá accepted nhưng detector vẫn báo (FLAGGED) -> "Fixed ... detector disagreement", không còn FLAGGED.
    def test_report_uses_advisor_confirmed_findings_and_final_verdict(self) -> None:
        reentrancy = {
            "finding_id": "SWC-107:Bank.withdraw", "swc": "SWC-107", "unit_id": "Bank.withdraw",
            "title": "Reentrancy", "severity": "Critical", "confidence": 0.99, "sources": ["static"],
        }
        pragma = {
            "finding_id": "SWC-102:contract", "swc": "SWC-102", "unit_id": None,
            "title": "Outdated Compiler Version", "severity": "Low", "confidence": 0.9, "sources": ["slither"],
        }
        request = {
            "run_id": "test-report-findings",
            "source": {"file_name": "Bank.sol", "sha256": "0" * 64, "code": "contract Bank {}"},
            "code_units": [],
            "local_phase1": {"static": {"findings": []}, "rag": {"findings": []}},
        }
        fused = {"verdict": "Vulnerable", "score": 0.97, "findings": [reentrancy, pragma]}
        remote = {
            "phases": {
                "detector": {"confirmed_findings": [reentrancy], "dismissed_findings": [pragma]},
                "advisor": {"suggestions": [], "false_positives": ["SWC-102:contract"]},
                "verifier": {
                    "attempts_count": 3,
                    "max_attempts": 3,
                    "verifications": [{"finding_id": "SWC-107:Bank.withdraw", "status": "FLAGGED"}],
                },
                "fixer": {"fixed_code": "contract Bank { }"},
            }
        }
        report_data = build_report_data(request, fused, remote, {"behavioral": {"status": "passed"}})
        self.assertEqual([f["swc"] for f in report_data["findings"]], ["SWC-107"])
        self.assertEqual([f["swc"] for f in report_data["dismissed_findings"]], ["SWC-102"])

        apply_verification_to_report(
            report_data,
            {
                "status": "complete",
                "overall_verdict": "Patch accepted by Slither",
                "detector_disagreement": True,
                "attempts_count": 3,
                "max_attempts": 3,
                "fixed_detection": {"threshold": 0.7},
                "fixed_unit_results": [{"unit_id": "Bank.withdraw", "vulnerability_probability": 0.8962}],
            },
        )
        status = report_data["findings"][0]["verification_status"]
        self.assertTrue(status.startswith("Fixed: Patch accepted"))
        self.assertIn("p=0.8962 >= threshold 0.70", status)
        self.assertIn("Final verification: Patch accepted", report_data["executive_summary"])
        md = render_markdown(report_data)
        self.assertIn("- **Fix attempts:** 3/3 attempt(s)", md)
        self.assertIn("### 4.5 Findings Dismissed by the Advisor", md)
        self.assertNotIn("FLAGGED", md)


# Kiểm tra cấu hình detector v13 có trên máy (cần thư mục models/).
class TestV13DetectorIntegration(unittest.TestCase):
    # threshold_config.json: base model Qwen2.5-Coder-14B-Instruct, ngưỡng 0.700, max_len 1024.
    def test_v13_threshold_config_values(self):
        config_path = DETECTOR_ADAPTER_DIR / "threshold_config.json"
        self.assertTrue(config_path.exists(), "threshold_config.json must exist in DETECTOR_ADAPTER_DIR")
        data = read_json(config_path)
        self.assertEqual(data.get("base_model"), "Qwen/Qwen2.5-Coder-14B-Instruct")
        self.assertEqual(data.get("best_threshold"), 0.700)
        self.assertEqual(data.get("max_len"), 1024)

    # Không truyền ngưỡng -> lấy 0.700 từ threshold_config.json.
    def test_resolve_detector_threshold_default(self):
        from E2E.pipeline.orchestrator import resolve_detector_threshold
        threshold = resolve_detector_threshold(None)
        self.assertEqual(threshold, 0.700)

    # Truyền ngưỡng 0.85 -> dùng đúng 0.85.
    def test_resolve_detector_threshold_explicit(self):
        from E2E.pipeline.orchestrator import resolve_detector_threshold
        self.assertEqual(resolve_detector_threshold(0.85), 0.85)

    # Các file adapter v13 bắt buộc tồn tại và không rỗng.
    def test_v13_adapter_files_present(self):
        for expected in ("adapter_model.safetensors", "adapter_config.json", "threshold_config.json", "line_head.pt"):
            path = DETECTOR_ADAPTER_DIR / expected
            self.assertTrue(path.exists(), f"Missing required v13 file: {expected}")
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

