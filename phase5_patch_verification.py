#!/usr/bin/env python3
"""
Phase 5 — Patch Verification (Verifier agent) of the LLM-BSCVM framework.

Closes the loop on the repair pipeline:

    Detection (1) -> Advisor (2) -> Assessor (3) -> Fixer (4) -> Verifier (5)

The Verifier takes the Fixer's repaired contract and confirms the patch is
trustworthy by combining two signals:

  * Static re-detection — the phase-1 detector is re-run on the FIXED contract;
    a vulnerability that no longer fires is strong evidence it was removed, and
    any vulnerability that appears only after the patch is flagged as introduced.
  * Adversarial LLM review — for each targeted vulnerability the agent tries to
    bypass the fix, and a separate check confirms the public interface and
    functionality were preserved.

It emits a per-vulnerability status and an overall verdict (PASS / NEEDS_REVIEW
/ FAIL).

Like the earlier phases, running with no arguments uses the inline CODE_TO_TEST
sample and runs the whole chain end-to-end:

    python3 phase5_patch_verification.py

Useful options (every upstream output can be reused to save time / API calls):

  * Fast static-only detection:                 --fast-detection
  * Skip the per-vuln Assessor:                  --no-assessor
  * Analyze your own contract:                   --code-file C.sol
  * Reuse a saved Advisor output:                --repair-file repair.json
  * Reuse a saved Assessor output:               --risk-file risk.json
  * Reuse a saved Fixer output:                  --fix-file fix.json
  * Skip re-detection (LLM-only verification):   --no-redetection
  * Embed every upstream agent output:           --include-upstream
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.advisor import AdvisorAgent  # noqa: E402
from agents.assessor import AssessorAgent  # noqa: E402
from agents.fixer import FixerAgent  # noqa: E402
from agents.verifier import VerifierAgent  # noqa: E402
from phase2_repair_suggestion import load_detection_result, read_code, run_phase1_detection  # noqa: E402


def get_repair_result(args: argparse.Namespace, code: str, detection_result: dict[str, Any]) -> dict[str, Any]:
    if args.repair_file:
        saved = json.loads(Path(args.repair_file).read_text(encoding="utf-8"))
        return saved.get("repair", saved)
    return AdvisorAgent(knowledge_dir=args.knowledge_dir, backend=args.backend).advise(
        code, detection_result
    )


def get_risk_result(
    args: argparse.Namespace, code: str, repair_result: dict[str, Any]
) -> dict[str, Any] | None:
    if args.risk_file:
        saved = json.loads(Path(args.risk_file).read_text(encoding="utf-8"))
        return saved.get("risk_assessment", saved)
    if args.no_assessor:
        return None
    return AssessorAgent(backend=args.backend).assess(code, repair_result)


def get_fix_result(
    args: argparse.Namespace,
    code: str,
    repair_result: dict[str, Any],
    risk_result: dict[str, Any] | None,
) -> dict[str, Any]:
    if args.fix_file:
        saved = json.loads(Path(args.fix_file).read_text(encoding="utf-8"))
        return saved.get("repair_patch", saved)
    return FixerAgent(backend=args.backend).fix(code, repair_result, risk_result)


def redetect_fixed(args: argparse.Namespace, fixed_code: str) -> dict[str, Any] | None:
    """Re-run the phase-1 detector on the repaired contract (the regression check)."""
    if args.no_redetection or not fixed_code.strip():
        return None
    try:
        return run_phase1_detection(args, fixed_code)
    except Exception as error:  # noqa: BLE001 - verification still runs LLM-only
        print(f"Warning: re-detection on the fixed contract failed: {error}", file=sys.stderr)
        return None


def write_or_print(output: dict[str, Any], output_path: str | None) -> None:
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    if output_path:
        Path(output_path).write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote verification result to {output_path}")
    else:
        print(rendered)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the repaired contract (Verifier agent).")
    parser.add_argument("--code-file", default=None, help="Path to the Solidity contract. Highest priority.")
    parser.add_argument("--code", default=None, help="Raw Solidity code string.")
    parser.add_argument("--repair-file", default=None, help="Saved phase-2 (Advisor) JSON to reuse.")
    parser.add_argument("--risk-file", default=None, help="Saved phase-3 (Assessor) JSON to reuse.")
    parser.add_argument("--fix-file", default=None, help="Saved phase-4 (Fixer) JSON to reuse.")
    parser.add_argument("--detection-file", default=None, help="Saved phase-1 detection JSON to reuse.")
    parser.add_argument("--no-assessor", action="store_true", help="Skip phase 3; order fixes by severity.")
    parser.add_argument(
        "--no-redetection",
        action="store_true",
        help="Skip re-running detection on the fixed contract (LLM-only verification).",
    )
    parser.add_argument(
        "--fast-detection",
        action="store_true",
        help="Run detection in static-only mode (skip the heavy CodeBERT + RAG components).",
    )
    parser.add_argument("--device", default="cpu", help="Device for phase-1 detection: auto | cpu | cuda | mps.")
    parser.add_argument("--backend", default="auto", help="LLM backend: auto (.env LLM_BACKEND) | gemini | ollama.")
    parser.add_argument("--knowledge-dir", default=str(PROJECT_ROOT / "rag" / "knowledge_store"))
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument("--include-upstream", action="store_true", help="Embed every upstream agent output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    code = read_code(args)

    # Detection on the ORIGINAL contract (reused across the chain + for attribution).
    detection_result = load_detection_result(args, code)
    repair_result = get_repair_result(args, code, detection_result)
    risk_result = get_risk_result(args, code, repair_result)
    fix_result = get_fix_result(args, code, repair_result, risk_result)

    redetection_result = redetect_fixed(args, str(fix_result.get("fixed_code") or ""))

    verification = VerifierAgent(backend=args.backend).verify(
        original_code=code,
        fix_result=fix_result,
        redetection_result=redetection_result,
        original_detection=detection_result,
    )

    output: dict[str, Any] = {"verification": verification}
    if args.include_upstream:
        output["detection"] = detection_result
        output["advisor"] = repair_result
        output["assessor"] = risk_result
        output["fixer"] = fix_result
        output["redetection"] = redetection_result
    else:
        output["summary"] = {
            "overall_verdict": verification.get("overall_verdict"),
            "fixed_count": verification.get("fixed_count"),
            "target_count": verification.get("target_count"),
            "resolution_rate": verification.get("resolution_rate"),
            "introduced_vulnerabilities": len(verification.get("introduced_vulnerabilities") or []),
        }

    write_or_print(output, args.output)


if __name__ == "__main__":
    main()
