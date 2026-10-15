from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from E2E.local.behavior_verifier import verify_patch_behavior
from E2E.local.kaggle_job_client import inject_request
from E2E.local.orchestrator import (
    build_repair_feedback,
    run_kaggle_repair_loop,
    run_pipeline,
)
from E2E.local.rag_runner import run_rag
from E2E.local.report import (
    apply_verification_to_report,
    build_code_comparison_rows,
    normalize_recommendations,
    render_html,
    render_markdown,
)
from E2E.local.static_runner import run_static
from E2E.config import DETECTOR_ADAPTER_DIR
from E2E.shared.function_parser import apply_unit_patches, parse_code_units
from E2E.shared.fusion import fuse_detection
from E2E.shared.io_utils import read_json, sha256_text


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def load_kaggle_runtime_namespace() -> dict:
    runtime_path = Path(__file__).resolve().parents[1] / "kaggle" / "kaggle_runtime.py"
    source = runtime_path.read_text(encoding="utf-8")
    parts = re.split(r"\n(?:if __name__|\s*main\(\))", source)
    if len(parts) < 2:
        raise AssertionError("Kaggle runtime no longer has an executable main call")
    definitions = parts[0]
    namespace = {"__name__": "kaggle_runtime_test"}
    exec(compile(definitions, str(runtime_path), "exec"), namespace)
    return namespace


class ParserTests(unittest.TestCase):
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


class DetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vulnerable = (EXAMPLES / "VulnerableBank.sol").read_text(encoding="utf-8")
        cls.safe = (EXAMPLES / "SafeBank.sol").read_text(encoding="utf-8")

    def test_static_reentrancy_and_safe_control(self) -> None:
        vulnerable = run_static(self.vulnerable)
        safe = run_static(self.safe)
        self.assertTrue(any(item["swc"] == "SWC-107" for item in vulnerable["findings"]))
        self.assertFalse(any(item["swc"] == "SWC-104" for item in vulnerable["findings"]))
        self.assertEqual(safe["findings"], [])

    def test_rag_reentrancy_and_safe_control(self) -> None:
        vulnerable = run_rag(self.vulnerable)
        safe = run_rag(self.safe)
        self.assertEqual(vulnerable["findings"][0]["swc"], "SWC-107")
        self.assertEqual(safe["findings"], [])

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

    def test_rag_knowledge_section_builder(self) -> None:
        """Kiểm tra helper _build_rag_knowledge_section định dạng và ưu tiên đúng case."""
        from E2E.local.server_gpu_runner import _build_rag_knowledge_section

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


class IntegrationTests(unittest.TestCase):
    def test_repair_feedback_contains_failed_static_rag_and_llm_reasons(self) -> None:
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
            "vulnerability_reason": "The external call still occurs before the balance update.",
            "unit_results": [{"unit_id": "C.withdraw", "label": "Vulnerable"}],
        }
        feedback = build_repair_feedback(
            static_result,
            rag_result,
            detector,
            {},
            attempt=1,
        )
        self.assertFalse(feedback["passed"])
        self.assertEqual(feedback["components"]["static"]["findings"][0]["swc"], "SWC-107")
        self.assertIn("checks-effects-interactions", feedback["components"]["rag"]["findings"][0]["remediation"])
        self.assertIn("external call", feedback["components"]["llm"]["reason"])

        runtime = load_kaggle_runtime_namespace()
        retry_findings = runtime["combined_findings"](
            {
                "local_phase1": {"provisional_fusion": {"findings": []}},
                "repair_feedback": feedback,
            },
            detector,
        )
        self.assertEqual(retry_findings[0]["swc"], "SWC-107")
        self.assertEqual(set(retry_findings[0]["sources"]), {"static", "rag"})

    def test_kaggle_repair_loop_runs_one_attempt_and_records_tool_feedback(self) -> None:
        code = "contract C { function f() external {} }"
        units = parse_code_units(code)
        safe_local = {"status": "ok", "verdict": "Safe", "findings": [], "score": 0.0}
        request = {
            "schema_version": "1.0.0",
            "pipeline_version": "1.0.0",
            "run_id": "repair-loop-test",
            "created_at": "2026-08-11T00:00:00Z",
            "source": {"file_name": "C.sol", "sha256": sha256_text(code), "code": code},
            "code_units": [unit.to_dict() for unit in units],
            "local_phase1": {
                "static": safe_local,
                "rag": safe_local,
                "provisional_fusion": {"findings": []},
            },
            "execution": {"mode": "full", "detector_threshold": 0.25},
        }

        class FakeClient:
            def __init__(self, vulnerable_attempts: int = 3) -> None:
                self.requests = []
                self.vulnerable_attempts = vulnerable_attempts

            def submit_and_wait(self, request_path, _run_dir):
                submitted = json.loads(Path(request_path).read_text(encoding="utf-8"))
                self.requests.append(submitted)
                attempt = len(self.requests)
                vulnerable = attempt <= self.vulnerable_attempts
                detector_fixed = {
                    "status": "ok",
                    "score": 0.9 if vulnerable else 0.05,
                    "threshold": 0.25,
                    "verdict": "Vulnerable" if vulnerable else "Safe",
                    "vulnerability_reason": f"attempt {attempt} still has an unsafe call",
                    "unit_results": [
                        {
                            "unit_id": "C.f",
                            "label": "Vulnerable" if vulnerable else "Safe",
                            "vulnerability_probability": 0.9 if vulnerable else 0.05,
                        }
                    ],
                }
                return {
                    "result": {
                        "schema_version": "1.0.0",
                        "run_id": submitted["run_id"],
                        "source_sha256": submitted["source"]["sha256"],
                        "status": "ok",
                        "phases": {
                            "detector": {
                                "status": "ok",
                                "original": detector_fixed,
                                "fixed": detector_fixed,
                            },
                            "fixer": {
                                "status": "ok",
                                "fixed_code": submitted["source"]["code"],
                                "patches": [],
                                "patch_outcomes": [],
                            },
                            "verifier": {
                                "status": "ok",
                                "vulnerability_reason": detector_fixed["vulnerability_reason"],
                            },
                        },
                        "models": {},
                        "timings": {"total_ms": 1},
                        "errors": [],
                    }
                }

        client = FakeClient()
        matrix_call_count = {"count": 0}

        def failing_matrix(*_args, **_kwargs):
            matrix_call_count["count"] += 1
            return {
                "component": "tool_verification",
                "status": "partial",
                "policy_verdict": "failed",
                "policy_reason": f"attempt {matrix_call_count['count']} still has an unsafe call",
                "summary": {"pass": 8, "fail": 1, "warn": 0, "error": 0},
                "compile": {"status": "passed"},
                "slither": {"status": "failed", "findings": []},
                "tools": {
                    "slither": {
                        "run_status": "completed",
                        "verdict": "FAIL",
                        "findings": [],
                        "residual_target_findings": [
                            {
                                "tool": "slither",
                                "name": "reentrancy-eth",
                                "category": "reentrancy",
                                "severity": "High",
                                "message": "external call before state update",
                            }
                        ],
                        "introduced_findings": [],
                        "detail": f"attempt {matrix_call_count['count']} still has an unsafe call",
                    }
                },
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            request_path = Path(temp_dir) / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            with patch("E2E.local.orchestrator.verify_security_tool_matrix", side_effect=failing_matrix):
                result = run_kaggle_repair_loop(
                    request_path,
                    temp_dir,
                    client,
                    max_attempts=99,
                )

        self.assertEqual(len(client.requests), 1)
        self.assertEqual(result["result"]["timings"]["repair_attempts"], 1)
        self.assertNotIn("repair_feedback", client.requests[0])
        self.assertIn("attempt 1", result["attempts"][0]["components"]["slither"]["reason"])
        self.assertFalse(result["attempts"][-1]["passed"])

        passing_client = FakeClient(vulnerable_attempts=1)
        passing_matrix_result = {
            "component": "tool_verification",
            "status": "ok",
            "policy_verdict": "passed",
            "policy_reason": "9/9 tools completed without residual target findings.",
            "summary": {"pass": 9, "fail": 0, "warn": 0, "error": 0},
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
            request_path = Path(temp_dir) / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            with patch(
                "E2E.local.orchestrator.verify_security_tool_matrix",
                return_value=passing_matrix_result,
            ):
                passing_result = run_kaggle_repair_loop(
                    request_path,
                    temp_dir,
                    passing_client,
                )
        self.assertEqual(len(passing_client.requests), 1)
        self.assertTrue(passing_result["attempts"][-1]["passed"])

    def test_kaggle_repair_loop_retries_on_compiler_error_only(self) -> None:
        source_code = "contract Test { function f() public {} }"
        request = {
            "schema_version": "1.0.0",
            "pipeline_version": "1.0.0",
            "run_id": "test-compile-retry",
            "source": {
                "file_name": "Test.sol",
                "sha256": "abc",
                "code": source_code,
            },
            "code_units": [],
            "local_phase1": {
                "provisional_fusion": {"findings": []}
            },
        }

        class CompileRetryClient:
            def __init__(self):
                self.requests = []

            def submit_and_wait(self, request_path, _attempt_dir):
                req = json.loads(Path(request_path).read_text(encoding="utf-8"))
                self.requests.append(req)
                attempt_num = len(self.requests)
                return {
                    "result": {
                        "schema_version": "1.0.0",
                        "run_id": req["run_id"],
                        "source_sha256": req["source"]["sha256"],
                        "status": "ok",
                        "phases": {
                            "detector": {
                                "status": "ok",
                                "original": {"verdict": "Vulnerable", "score": 0.9, "unit_results": []},
                                "fixed": {"verdict": "Vulnerable", "score": 0.9, "unit_results": []},
                            },
                            "fixer": {
                                "status": "ok",
                                "fixed_code": f"// Attempt {attempt_num}\n" + req["source"]["code"],
                                "patches": [],
                                "patch_outcomes": [],
                            },
                            "verifier": {
                                "status": "ok",
                                "vulnerability_reason": "Still vulnerable",
                            },
                        },
                        "models": {},
                        "timings": {"total_ms": 10},
                        "errors": [],
                    }
                }

        client = CompileRetryClient()
        attempt_tracker = {"count": 0}

        def matrix_with_compile_error(*_args, **_kwargs):
            attempt_tracker["count"] += 1
            if attempt_tracker["count"] == 1:
                # Attempt 1: Compilation fails with wrong argument count
                return {
                    "component": "tool_verification",
                    "status": "partial",
                    "policy_verdict": "failed",
                    "policy_reason": "Fixed source does not compile.",
                    "summary": {"pass": 0, "fail": 0, "warn": 0, "error": 0},
                    "compile": {
                        "status": "failed",
                        "stderr": "Error: Wrong argument count for function call: 2 arguments given but expected 4.",
                    },
                    "slither": {"status": "skipped", "findings": []},
                    "tools": {
                        "slither": {
                            "run_status": "skipped",
                            "verdict": "SKIPPED",
                            "findings": [],
                            "residual_target_findings": [],
                            "introduced_findings": [],
                            "detail": "fixed source did not compile",
                            "error": "fixed source did not compile",
                        }
                    },
                }
            else:
                # Attempt 2: Compilation succeeds, but tool matrix has residual findings (should NOT retry for tool findings)
                return {
                    "component": "tool_verification",
                    "status": "partial",
                    "policy_verdict": "failed",
                    "policy_reason": "Tool matrix found residual issues.",
                    "summary": {"pass": 8, "fail": 1, "warn": 0, "error": 0},
                    "compile": {"status": "passed"},
                    "slither": {"status": "failed", "findings": []},
                    "tools": {
                        "slither": {
                            "run_status": "completed",
                            "verdict": "FAIL",
                            "findings": [],
                            "residual_target_findings": [{"tool": "slither", "message": "unsafe call"}],
                            "introduced_findings": [],
                            "detail": "unsafe call",
                        }
                    },
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            request_path = Path(temp_dir) / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            with patch(
                "E2E.local.orchestrator.verify_security_tool_matrix",
                side_effect=matrix_with_compile_error,
            ):
                loop_result = run_kaggle_repair_loop(
                    request_path,
                    temp_dir,
                    client,
                    max_attempts=3,
                )

        # Verified: exactly 2 attempts took place (retried attempt 1 on compiler error, did NOT retry attempt 2 on tool failure)
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(len(loop_result["attempts"]), 2)
        # Attempt 1 failed compilation
        self.assertEqual(loop_result["attempts"][0]["compile_status"], "failed")
        self.assertIn("Wrong argument count", loop_result["attempts"][0]["compiler_error"])
        # Attempt 2 received the compiler error feedback from attempt 1
        self.assertIn("repair_feedback", client.requests[1])
        self.assertIn(
            "Wrong argument count",
            client.requests[1]["repair_feedback"]["compiler_error"],
        )
        # Attempt 2 passed compilation
        self.assertEqual(loop_result["attempts"][1]["compile_status"], "passed")

    def test_kaggle_json_contracts_and_no_patch_guard(self) -> None:
        runtime = load_kaggle_runtime_namespace()
        generator = runtime["JSONGenerator"](None, None, {"max_retries": 0})
        generator._generate = lambda _prompt: json.dumps(
            [
                {
                    "unit_id": "C.withdraw",
                    "replacement": "function withdraw() external {}",
                    "summary": "Replace the vulnerable function.",
                }
            ]
        )
        fixer = generator.ask(
            phase="fixer",
            system="test",
            user="test",
            required={"patches": list, "notes": list},
            item_requirements=(
                "patches",
                {"unit_id": str, "replacement": str, "summary": str},
            ),
        )
        self.assertEqual(fixer["status"], "ok")
        self.assertEqual(fixer["notes"], [])
        self.assertEqual(fixer["patches"][0]["unit_id"], "C.withdraw")

        generator._generate = lambda _prompt: json.dumps(
            {"assessments": [{"finding_id": "F-1", "swc": "SWC-107"}]}
        )
        assessor = generator.ask(
            phase="assessor",
            system="test",
            user="test",
            required={"assessments": list},
            item_requirements=(
                "assessments",
                {
                    "finding_id": str,
                    "swc": str,
                    "cvss_score": (int, float),
                },
            ),
        )
        self.assertEqual(assessor["status"], "error")
        self.assertIn("cvss_score", assessor["errors"][0])

        guarded = runtime["no_patch_verifier"](
            [{"finding_id": "F-1", "swc": "SWC-107", "unit_id": "C.withdraw"}],
            [],
            source_changed=False,
        )
        self.assertEqual(guarded["verifications"][0]["status"], "Not Fixed")
        self.assertIn("No patch was applied", guarded["overall_verdict"])

        fallback_verifier = runtime["inconclusive_verifier_fallback"](
            [{"finding_id": "F-1", "swc": "SWC-107", "unit_id": "C.withdraw"}],
            [{"unit_id": "C.withdraw", "status": "applied"}],
            {"status": "error", "errors": ["invalid JSON"]},
        )
        self.assertEqual(fallback_verifier["status"], "ok")
        self.assertEqual(
            fallback_verifier["verifications"][0]["status"],
            "Inconclusive",
        )
        self.assertNotIn("Fixed", fallback_verifier["overall_verdict"])

        original = {
            "unit_id": "C.withdraw",
            "kind": "function",
            "name": "withdraw",
            "code": (
                "function withdraw() external { "
                '(bool ok,) = msg.sender.call{value: 1}(""); require(ok); balance = 0; }'
            ),
        }
        generator._generate = lambda _prompt: """```solidity
function withdraw() external {
    balance = 0;
    (bool ok,) = msg.sender.call{value: 1}("");
    require(ok);
}
```"""
        patch_value, patch_errors = runtime["generate_function_patch"](
            generator,
            original,
            {
                "finding_id": "F-1",
                "swc": "SWC-107",
                "unit_id": "C.withdraw",
                "description": "State is changed after an external call.",
            },
        )
        self.assertEqual(patch_errors, [])
        self.assertEqual(patch_value["unit_id"], "C.withdraw")
        self.assertLess(
            patch_value["replacement"].find("balance = 0"),
            patch_value["replacement"].find(".call"),
        )
        with self.assertRaisesRegex(ValueError, "removed observable behavior"):
            runtime["validate_observable_behavior"](
                original["code"],
                "function withdraw() external { balance = 0; }",
            )

        assessment = runtime["grounded_assessor_fallback"](
            [
                {
                    "finding_id": "F-1",
                    "swc": "SWC-107",
                    "unit_id": "C.withdraw",
                    "severity": "Critical",
                    "description": "Reentrant withdrawal can drain funds.",
                }
            ],
            {"errors": ["missing cvss_score"]},
        )
        self.assertEqual(assessment["status"], "ok")
        self.assertEqual(assessment["assessments"][0]["cvss_score"], 9.0)

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

    def test_report_normalizes_object_recommendations(self) -> None:
        result = normalize_recommendations(
            [{"title": "Run tests", "description": "Execute the full project suite."}]
        )
        self.assertEqual(result, ["Run tests: Execute the full project suite."])

    def test_fixed_code_report_renders_side_by_side_diff(self) -> None:
        original = "contract C {\n    uint x;\n    function f() public {}\n}"
        fixed = "contract C {\n    uint256 x;\n    function f() public {}\n}"
        rows = build_code_comparison_rows(original, fixed)
        self.assertTrue(any(row["fixed_changed"] and "uint256" in row["fixed"] for row in rows))

        report_data = {
            "title": "Smart Contract Audit Report",
            "run_id": "diff-test",
            "analyzed_object": "C",
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
        html_report = render_html(report_data, markdown)
        self.assertIn("Source Code (Sample)", markdown)
        self.assertIn("Fixed Source Code", markdown)
        self.assertIn("<mark>    uint256 x;</mark>", markdown)
        self.assertIn("code-line changed", html_report)

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

    def test_notebook_request_injection(self) -> None:
        request = {"run_id": "test", "source": {"code": "contract C {}"}}
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "job.ipynb"
            inject_request(
                Path(__file__).resolve().parents[1] / "kaggle" / "codellama_e2e.ipynb",
                request,
                output,
            )
            notebook = json.loads(output.read_text(encoding="utf-8"))
            parameter_source = "".join(notebook["cells"][1]["source"])
            self.assertIn("REQUEST_B64", parameter_source)
            self.assertNotIn('REQUEST_B64 = ""', parameter_source)
            namespace: dict = {}
            exec(compile(parameter_source, "parameters-cell", "exec"), namespace)
            self.assertTrue(namespace["REQUEST_B64"])

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
                "detector": "CodeLlama-7b-hf + adapter",
                "generator": "CodeLlama-7b-Instruct-hf",
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
                "reporter": {
                    "status": "ok",
                    "executive_summary": "One critical reentrancy issue was identified and patched.",
                    "recommendations": ["Run the complete project test suite."],
                },
            },
        }
        tool_result = {
            "component": "tool_verification",
            "status": "ok",
            "policy_verdict": "passed",
            "policy_reason": "9/9 tools completed without residual target findings.",
            "summary": {"pass": 9, "fail": 0, "warn": 0, "error": 0},
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
            with patch("E2E.local.orchestrator.verify_security_tool_matrix", return_value=tool_result):
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
            with patch("E2E.local.orchestrator.verify_security_tool_matrix", return_value=tool_result):
                disagreement = run_pipeline(
                    code,
                    file_name="VulnerableBank.sol",
                    run_id=disagreement_run_id,
                    runs_dir=temp_dir,
                    remote_result=remote,
            )
            self.assertEqual(
                disagreement["verification"]["overall_verdict"],
                "Patch accepted by security tool matrix",
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
            "patches": [],
            "patch_outcomes": [],
        }
        remote["phases"]["verifier"] = {
            "status": "ok",
            "overall_verdict": "Fixed",
            "verifications": [],
        }
        remote["phases"]["reporter"]["recommendations"] = [
            {
                "title": "Run tests",
                "description": "Execute the complete project test suite.",
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("E2E.local.orchestrator.verify_security_tool_matrix", return_value=tool_result):
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
            self.assertIn(
                "Run tests: Execute the complete project test suite.",
                report_data["recommendations"],
            )

    def test_docker_availability_and_tool_policy(self) -> None:
        from E2E.local.tool_verifier import _evaluate_tool_result, _is_docker_available, _policy_verdict

        self.assertIsInstance(_is_docker_available(), bool)

        skipped_eval = _evaluate_tool_result(
            "osiris",
            "unavailable",
            [],
            [],
            {"reentrancy"},
            error="Docker daemon is not running",
        )
        self.assertEqual(skipped_eval["verdict"], "SKIPPED")

        policy = _policy_verdict({
            "slither": {"run_status": "completed", "verdict": "PASS"},
            "osiris": {"run_status": "unavailable", "verdict": "SKIPPED"},
        })
        self.assertEqual(policy["verdict"], "passed")

    def test_slither_status_consistency_when_failed(self) -> None:
        from E2E.local.report import build_report_data, render_markdown, render_html
        from E2E.local.tool_verifier import _evaluate_tool_result

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

        html = render_html(report_data, md)
        self.assertIn("Slither: <b>failed</b>", html)
        self.assertNotIn("Slither: <b>passed</b>", html)


class TestV13DetectorIntegration(unittest.TestCase):
    def test_v13_threshold_config_values(self):
        config_path = DETECTOR_ADAPTER_DIR / "threshold_config.json"
        self.assertTrue(config_path.exists(), "threshold_config.json must exist in DETECTOR_ADAPTER_DIR")
        data = read_json(config_path)
        self.assertEqual(data.get("base_model"), "Qwen/Qwen2.5-Coder-14B-Instruct")
        self.assertEqual(data.get("best_threshold"), 0.700)
        self.assertEqual(data.get("max_len"), 1024)

    def test_resolve_detector_threshold_default(self):
        from E2E.local.orchestrator import resolve_detector_threshold
        threshold = resolve_detector_threshold(None)
        self.assertEqual(threshold, 0.700)

    def test_resolve_detector_threshold_explicit(self):
        from E2E.local.orchestrator import resolve_detector_threshold
        self.assertEqual(resolve_detector_threshold(0.85), 0.85)

    def test_v13_adapter_files_present(self):
        for expected in ("adapter_model.safetensors", "adapter_config.json", "threshold_config.json", "line_head.pt"):
            path = DETECTOR_ADAPTER_DIR / expected
            self.assertTrue(path.exists(), f"Missing required v13 file: {expected}")
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

