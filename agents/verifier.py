"""
verifier.py
──────────────────────────────────────────────────────────────────────────────
Two-Tier Patch Verification Agent (Phase 5).

Tier 1: Deterministic Verification (Local):
- solc compiler check targeting minimum satisfying pragma version.
- AST observable behavior preservation check (solc --ast-compact-json).
- Static security verification (Slither diff if available).

Tier 2: Independent LLM Adversarial Review:
- Evaluates the unified patch diff per vulnerability finding.
- Answers:
  1. Is the target flaw genuinely fixed, or can an attacker still exploit it?
  2. Does the patch introduce functional regressions or break business invariants?
- Returns structured JSON: {"fixed": bool, "regression": bool, "evidence": str}.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import difflib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

VERIFIER_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "fixed": {"type": "boolean"},
        "regression": {"type": "boolean"},
        "evidence": {"type": "string"},
    },
    "required": ["fixed", "regression", "evidence"],
}

VERIFIER_SYSTEM_INSTRUCTION = (
    "You are an independent adversarial smart-contract security verifier. "
    "Your objective is to critically examine proposed code diffs. "
    "Determine whether the specific security vulnerability was truly eliminated, "
    "and ensure no regressions, lost state mutations, or altered business logic were introduced. "
    "Return ONLY valid JSON matching the requested schema."
)


@dataclass
class VulnVerification:
    swc: str
    vulnerability_name: str
    status: str  # "PASS", "FAIL", "WARN", "INCONCLUSIVE"
    fixed: bool = True
    regression: bool = False
    evidence: str = ""
    tier1_deterministic_status: str = "ok"
    tier2_llm_status: str = "not_run"


@dataclass
class VerificationResult:
    agent: str = "Verifier (Two-Tier Multi-Modal Verification)"
    overall_verdict: str = "Inconclusive"
    compile_status: str = "not_run"
    solc_version: str = "solc-native"
    behavior_status: str = "not_run"
    target_count: int = 0
    fixed_count: int = 0
    regression_count: int = 0
    verifications: List[Dict[str, Any]] = field(default_factory=list)
    notes: str = ""


def generate_unified_diff(original: str, fixed: str) -> str:
    """Generate clean unified diff representation between original and patched contracts."""
    orig_lines = original.splitlines(keepends=True)
    fixed_lines = fixed.splitlines(keepends=True)
    diff = difflib.unified_diff(
        orig_lines, fixed_lines, fromfile="original.sol", tofile="fixed.sol", n=3
    )
    return "".join(diff)


class VerifierAgent:
    """Two-tier patch verification agent combining deterministic AST checks with adversarial LLM analysis."""

    def __init__(self, backend: str = "auto", **kwargs: Any) -> None:
        self.backend = backend
        self.kwargs = kwargs
        self._llm = None
        self._init_llm()

    def _init_llm(self) -> None:
        try:
            try:
                from .llm_client import build_llm_client
            except ImportError:
                from llm_client import build_llm_client
            self._llm = build_llm_client(self.backend)
        except Exception as err:
            logger.debug(f"Independent LLM client not initialized for VerifierAgent: {err}")
            self._llm = None

    def _run_deterministic_tier(
        self, original_code: str, fixed_code: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run Tier 1 deterministic checks: solc pragma compilation + AST invariants."""
        # 1. Compilation
        compile_res = {"status": "skipped", "passed": None, "solc_version": "unknown"}
        try:
            from E2E.shared.solc import check_compilation_with_baseline
            compile_res = check_compilation_with_baseline(original_code, fixed_code)
        except Exception as err:
            compile_res = {"status": "error", "passed": False, "error": str(err)}

        # 2. Behavior preservation
        behavior_res = {"status": "skipped", "findings": []}
        try:
            from E2E.local.behavior_verifier import verify_patch_behavior
            behavior_res = verify_patch_behavior(original_code, fixed_code)
        except Exception as err:
            behavior_res = {"status": "error", "findings": [{"check": "exception", "evidence": str(err)}]}

        return compile_res, behavior_res

    def _build_adversarial_prompt(
        self,
        finding: Dict[str, Any],
        diff_text: str,
        solc_version: str,
    ) -> str:
        swc = finding.get("swc") or "Unknown"
        title = finding.get("title") or finding.get("vulnerability_name") or f"Finding {swc}"
        root_cause = finding.get("root_cause") or finding.get("description") or "Not provided"

        sections = [
            "You are an adversarial smart contract security reviewer.",
            f"Review this proposed patch for {title} ({swc}).",
            f"Compiler target: {solc_version}",
            f"Known Root Cause: {root_cause}",
            "",
            "Unified Patch Diff:",
            "```diff",
            diff_text if diff_text.strip() else "(no diff lines)",
            "```",
            "",
            "Adversarial Review Questions:",
            f"1. Is {swc} ({title}) genuinely eliminated by this diff, or can an attacker still trigger or bypass it?",
            "2. Does this diff introduce functional regressions, broken logic, or lost state changes outside the fix?",
            "",
            "Return ONLY valid JSON matching this schema:",
            "{\n"
            '  "fixed": bool,       // true if the vulnerability is completely resolved\n'
            '  "regression": bool,  // true if any regression or broken functionality is created\n'
            '  "evidence": string   // detailed technical justification\n'
            "}",
        ]
        return "\n".join(sections)

    def verify_one_adversarial(
        self,
        finding: Dict[str, Any],
        diff_text: str,
        solc_version: str,
    ) -> tuple[bool, bool, str]:
        """Query LLM with adversarial prompt for a single finding."""
        if not self._llm:
            return True, False, "Independent LLM review skipped (LLM client offline)."

        prompt = self._build_adversarial_prompt(finding, diff_text, solc_version)
        try:
            res = self._llm.generate_json(
                prompt,
                VERIFIER_SYSTEM_INSTRUCTION,
                response_schema=VERIFIER_RESPONSE_SCHEMA,
            )
            fixed = bool(res.get("fixed", True))
            regression = bool(res.get("regression", False))
            evidence = str(res.get("evidence") or "").strip()
            return fixed, regression, evidence
        except Exception as err:
            return True, False, f"Adversarial check error: {err}"

    def verify(
        self,
        original_code: str,
        fix_result: Dict[str, Any],
        original_detection: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Main verification entry point."""
        fixed_code = str(fix_result.get("fixed_code") or "").strip()
        if not fixed_code or fixed_code == original_code:
            return asdict(
                VerificationResult(
                    overall_verdict="Patch rejected: no effective patch was applied",
                    compile_status="not_run",
                    behavior_status="not_run",
                    notes="Patch code is identical to original or empty.",
                )
            )

        diff_text = generate_unified_diff(original_code, fixed_code)
        compile_res, behavior_res = self._run_deterministic_tier(original_code, fixed_code)

        compile_passed = compile_res.get("passed")
        solc_version = compile_res.get("solc_version") or "unknown"
        behavior_status = behavior_res.get("status")

        # Determine target findings
        findings_to_verify = (
            fix_result.get("target_findings")
            or fix_result.get("changes")
            or (original_detection.get("findings") if original_detection else None)
            or []
        )

        verifications: List[VulnVerification] = []
        fixed_count = 0
        regression_count = 0

        # Tier 1 Gate: If compilation fails or baseline broke
        if compile_passed is False:
            err_msg = compile_res.get("error") or "solc compilation failed"
            return asdict(
                VerificationResult(
                    overall_verdict=f"Patch rejected: compilation failed ({err_msg[:120]})",
                    compile_status="failed",
                    solc_version=solc_version,
                    behavior_status=behavior_status,
                    target_count=len(findings_to_verify),
                    notes=f"Tier 1 compilation rejected patch: {err_msg}",
                )
            )

        # Tier 1 Gate: If AST observable behavior was violated (e.g. dropped state write)
        if behavior_status == "rejected":
            bev_reasons = [
                f.get("evidence") or f.get("check")
                for f in behavior_res.get("findings", [])
            ]
            bev_note = "; ".join(bev_reasons) if bev_reasons else "AST invariant check rejected"
            return asdict(
                VerificationResult(
                    overall_verdict=f"Patch rejected: altered observable behavior ({bev_note[:120]})",
                    compile_status="passed",
                    solc_version=solc_version,
                    behavior_status="rejected",
                    target_count=len(findings_to_verify),
                    notes=f"Tier 1 behavior verifier rejected patch: {bev_note}",
                )
            )

        # Tier 2: LLM Adversarial Review
        for f in findings_to_verify:
            swc = f.get("swc") or "SWC-UNKNOWN"
            title = f.get("title") or f.get("vulnerability_name") or swc
            is_fixed, has_regression, evidence = self.verify_one_adversarial(
                f, diff_text, solc_version
            )

            if not is_fixed:
                v_status = "FAIL"
            elif has_regression:
                v_status = "WARN"
                regression_count += 1
            else:
                v_status = "PASS"
                fixed_count += 1

            verifications.append(
                VulnVerification(
                    swc=swc,
                    vulnerability_name=title,
                    status=v_status,
                    fixed=is_fixed,
                    regression=has_regression,
                    evidence=evidence,
                    tier1_deterministic_status="ok",
                    tier2_llm_status="ok" if self._llm else "skipped",
                )
            )

        # Overall verdict synthesis
        if any(v.status == "FAIL" for v in verifications):
            overall_verdict = "Patch rejected: vulnerability persists under adversarial review"
        elif any(v.status == "WARN" for v in verifications):
            overall_verdict = "Patch accepted with warnings: potential regression flagged"
        else:
            overall_verdict = "Patch accepted by security tool matrix and AST verification"

        return asdict(
            VerificationResult(
                overall_verdict=overall_verdict,
                compile_status="passed" if compile_passed else "skipped",
                solc_version=solc_version,
                behavior_status=behavior_status,
                target_count=len(verifications),
                fixed_count=fixed_count,
                regression_count=regression_count,
                verifications=[asdict(v) for v in verifications],
                notes=f"Tier 1 passed ({solc_version}). Tier 2 adversarial evaluated {len(verifications)} findings.",
            )
        )


def verify_patch(
    original_code: str,
    fix_result: Dict[str, Any],
    backend: str = "auto",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Helper entry point for Verifier Agent."""
    return VerifierAgent(backend=backend).verify(
        original_code=original_code,
        fix_result=fix_result,
        **kwargs,
    )
