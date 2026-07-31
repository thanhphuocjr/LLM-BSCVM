from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from E2E.local.behavior_verifier import verify_patch_behavior
from E2E.local.kaggle_job_client import inject_request
from E2E.local.orchestrator import run_pipeline
from E2E.local.rag_runner import run_rag
from E2E.local.report import apply_verification_to_report, normalize_recommendations
from E2E.local.static_runner import run_static
from E2E.shared.function_parser import apply_unit_patches, parse_code_units
from E2E.shared.fusion import fuse_detection
from E2E.shared.io_utils import sha256_text


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def load_kaggle_runtime_namespace() -> dict:
    runtime_path = Path(__file__).resolve().parents[1] / "kaggle" / "kaggle_runtime.py"
    source = runtime_path.read_text(encoding="utf-8")
    definitions, separator, _ = source.rpartition("\nmain()")
    if not separator:
        raise AssertionError("Kaggle runtime no longer has an executable main call")
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


class IntegrationTests(unittest.TestCase):
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
            "compile": {"status": "passed"},
            "slither": {"status": "passed", "findings": []},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("E2E.local.orchestrator.verify_tools", return_value=tool_result):
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
            with patch("E2E.local.orchestrator.verify_tools", return_value=tool_result):
                disagreement = run_pipeline(
                    code,
                    file_name="VulnerableBank.sol",
                    run_id=disagreement_run_id,
                    runs_dir=temp_dir,
                    remote_result=remote,
                )
            self.assertEqual(
                disagreement["verification"]["overall_verdict"],
                "Patch inconclusive: CodeLlama detector remains positive while deterministic checks pass",
            )
            disagreement_report = json.loads(
                (Path(temp_dir) / disagreement_run_id / "phase6_report_data.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIn("not accepted", disagreement_report["executive_summary"])

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
            with patch("E2E.local.orchestrator.verify_tools", return_value=tool_result):
                rejected = run_pipeline(
                    code,
                    file_name="VulnerableBank.sol",
                    run_id=rejected_run_id,
                    runs_dir=temp_dir,
                    remote_result=remote,
                )
            self.assertEqual(
                rejected["verification"]["overall_verdict"],
                "Patch rejected: local static/RAG findings remain after redetection",
            )
            report_data = json.loads(
                (Path(temp_dir) / rejected_run_id / "phase6_report_data.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                report_data["patch_verdict"],
                "Patch rejected: local static/RAG findings remain after redetection",
            )
            self.assertIn(
                "Run tests: Execute the complete project test suite.",
                report_data["recommendations"],
            )


if __name__ == "__main__":
    unittest.main()
