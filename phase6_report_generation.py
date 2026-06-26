#!/usr/bin/env python3
"""
Phase 6 — Report Generation (Reporter agent), the final stage of LLM-BSCVM.

Runs the complete six-agent pipeline and integrates every upstream result into
one smart-contract audit report:

    Detection (1) -> Advisor (2) -> Assessor (3) -> Fixer (4) -> Verifier (5) -> Reporter (6)

The report follows the project's reference structure with seven sections:
contract information, executive summary, methodology, findings (statistics +
severity distribution + vulnerability reference table), detailed analysis
(source code, repair suggestion, fixed code), summary & recommendations, and a
disclaimer. It is written as Markdown (and optionally JSON).

Like the earlier phases, running with no arguments uses the inline CODE_TO_TEST
sample and runs the whole chain end-to-end:

    python3 phase6_report_generation.py --output-md report.md

Useful options (every upstream output can be reused to save time / API calls):

  * Fast static-only detection:        --fast-detection
  * Skip the per-vuln Assessor:        --no-assessor
  * Skip the Fixer + Verifier:         --no-repair      (report on detection only)
  * Analyze your own contract:         --code-file C.sol
  * Reuse saved upstream outputs:      --detection-file / --repair-file /
                                       --risk-file / --fix-file / --verify-file
  * Write the rendered report:         --output-md report.md
  * Write the full JSON data:          --output report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.advisor import AdvisorAgent  # noqa: E402
from agents.assessor import AssessorAgent  # noqa: E402
from agents.fixer import FixerAgent  # noqa: E402
from agents.reporter import ReporterAgent  # noqa: E402
from agents.verifier import VerifierAgent  # noqa: E402
from phase2_repair_suggestion import load_detection_result, read_code, run_phase1_detection  # noqa: E402


def get_repair_result(args, code, detection_result):
    if args.repair_file:
        return json.loads(Path(args.repair_file).read_text(encoding="utf-8")).get("repair") or json.loads(
            Path(args.repair_file).read_text(encoding="utf-8")
        )
    return AdvisorAgent(knowledge_dir=args.knowledge_dir, backend=args.backend).advise(code, detection_result)


def get_risk_result(args, code, repair_result):
    if args.risk_file:
        saved = json.loads(Path(args.risk_file).read_text(encoding="utf-8"))
        return saved.get("risk_assessment", saved)
    if args.no_assessor:
        return None
    return AssessorAgent(backend=args.backend).assess(code, repair_result)


def get_fix_result(args, code, repair_result, risk_result):
    if args.fix_file:
        saved = json.loads(Path(args.fix_file).read_text(encoding="utf-8"))
        return saved.get("repair_patch", saved)
    return FixerAgent(backend=args.backend).fix(code, repair_result, risk_result)


def get_verification_result(args, code, fix_result, detection_result):
    if args.verify_file:
        saved = json.loads(Path(args.verify_file).read_text(encoding="utf-8"))
        return saved.get("verification", saved)
    redetection = None
    if not args.no_redetection:
        try:
            redetection = run_phase1_detection(args, str(fix_result.get("fixed_code") or ""))
        except Exception as error:  # noqa: BLE001
            print(f"Warning: re-detection on the fixed contract failed: {error}", file=sys.stderr)
    return VerifierAgent(backend=args.backend).verify(
        original_code=code,
        fix_result=fix_result,
        redetection_result=redetection,
        original_detection=detection_result,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the final audit report (Reporter agent).")
    parser.add_argument("--code-file", default=None, help="Path to the Solidity contract. Highest priority.")
    parser.add_argument("--code", default=None, help="Raw Solidity code string.")
    parser.add_argument("--detection-file", default=None, help="Saved phase-1 detection JSON to reuse.")
    parser.add_argument("--repair-file", default=None, help="Saved phase-2 (Advisor) JSON to reuse.")
    parser.add_argument("--risk-file", default=None, help="Saved phase-3 (Assessor) JSON to reuse.")
    parser.add_argument("--fix-file", default=None, help="Saved phase-4 (Fixer) JSON to reuse.")
    parser.add_argument("--verify-file", default=None, help="Saved phase-5 (Verifier) JSON to reuse.")
    parser.add_argument("--no-assessor", action="store_true", help="Skip phase 3; order fixes by severity.")
    parser.add_argument("--no-repair", action="store_true", help="Skip Fixer + Verifier (report detection only).")
    parser.add_argument("--no-redetection", action="store_true", help="Skip re-detection in the Verifier.")
    parser.add_argument(
        "--fast-detection", action="store_true",
        help="Run detection in static-only mode (skip the heavy CodeBERT + RAG components).",
    )
    parser.add_argument("--device", default="cpu", help="Device for phase-1 detection: auto | cpu | cuda | mps.")
    parser.add_argument("--backend", default="auto", help="LLM backend: auto (.env LLM_BACKEND) | gemini | ollama.")
    parser.add_argument("--knowledge-dir", default=str(PROJECT_ROOT / "rag" / "knowledge_store"))
    parser.add_argument("--audit-time", default=None, help="Audit timestamp (default: now).")
    parser.add_argument("--output", default=None, help="Optional full-report JSON output path.")
    parser.add_argument("--output-md", default=None, help="Optional rendered Markdown report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    code = read_code(args)
    audit_time = args.audit_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    detection_result = load_detection_result(args, code)

    repair_result = risk_result = fix_result = verification_result = None
    is_vulnerable = bool((detection_result.get("final") or {}).get("is_vulnerable"))

    # Only run the analysis/repair chain when the contract is flagged vulnerable.
    if is_vulnerable:
        repair_result = get_repair_result(args, code, detection_result)
        risk_result = get_risk_result(args, code, repair_result)
        if not args.no_repair:
            fix_result = get_fix_result(args, code, repair_result, risk_result)
            verification_result = get_verification_result(args, code, fix_result, detection_result)

    report = ReporterAgent(backend=args.backend).report(
        code=code,
        detection_result=detection_result,
        repair_result=repair_result,
        risk_result=risk_result,
        fix_result=fix_result,
        verification_result=verification_result,
        audit_time=audit_time,
    )

    markdown = report.get("markdown", "")
    if args.output_md:
        Path(args.output_md).write_text(markdown + "\n", encoding="utf-8")
        print(f"Wrote audit report to {args.output_md}")
    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote report data to {args.output}")
    if not args.output_md and not args.output:
        print(markdown)


if __name__ == "__main__":
    main()
