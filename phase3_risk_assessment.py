#!/usr/bin/env python3
"""
Phase 3 — Risk Assessment (Assessor agent) of the LLM-BSCVM framework.

Chains the full upstream pipeline and then assesses the risk of every detected
vulnerability with the CVSS standard + a four-level system (Critical / High /
Medium / Low), producing a risk distribution and a repair-priority ordering.

    Detection (phase 1) -> Advisor (phase 2) -> Assessor (phase 3)

Just like the earlier phases, running with no arguments uses the inline
CODE_TO_TEST sample and runs the whole chain end-to-end:

    python3 phase3_risk_assessment.py

Other options:

  * Static-only detection (fast, skips CodeBERT + RAG):
        python3 phase3_risk_assessment.py --fast-detection

  * Analyze your own contract:
        python3 phase3_risk_assessment.py --code-file C.sol

  * Reuse a saved phase-2 (Advisor) output and only run the Assessor:
        python3 phase3_risk_assessment.py --repair-file repair.json

  * Reuse a saved phase-1 (detection) output:
        python3 phase3_risk_assessment.py --detection-file det.json
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
from phase2_repair_suggestion import load_detection_result, read_code  # noqa: E402


def get_repair_result(args: argparse.Namespace, code: str) -> dict[str, Any]:
    """Return the Advisor (phase 2) output, reusing a saved file if provided."""
    if args.repair_file:
        saved = json.loads(Path(args.repair_file).read_text(encoding="utf-8"))
        # Accept both the bare Advisor dict and the phase-2 wrapper {"repair": {...}}.
        return saved.get("repair", saved)

    detection_result = load_detection_result(args, code)
    return AdvisorAgent(knowledge_dir=args.knowledge_dir, backend=args.backend).advise(
        code, detection_result
    )


def write_or_print(output: dict[str, Any], output_path: str | None) -> None:
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    if output_path:
        Path(output_path).write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote risk assessment to {output_path}")
    else:
        print(rendered)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assess vulnerability risk (CVSS + four-level system).")
    parser.add_argument("--code-file", default=None, help="Path to the Solidity contract. Highest priority.")
    parser.add_argument("--code", default=None, help="Raw Solidity code string.")
    parser.add_argument("--repair-file", default=None, help="Saved phase-2 (Advisor) JSON to reuse.")
    parser.add_argument("--detection-file", default=None, help="Saved phase-1 detection JSON to reuse.")
    parser.add_argument(
        "--fast-detection",
        action="store_true",
        help="Run detection in static-only mode (skip the heavy CodeBERT + RAG components).",
    )
    parser.add_argument("--device", default="cpu", help="Device for phase-1 detection: auto | cpu | cuda | mps.")
    parser.add_argument("--backend", default="auto", help="LLM backend: auto (from .env LLM_BACKEND) | gemini | ollama for the agents.")
    parser.add_argument("--knowledge-dir", default=str(PROJECT_ROOT / "rag" / "knowledge_store"))
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument("--include-repair", action="store_true", help="Embed the full Advisor output in the result.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    code = read_code(args)

    repair_result = get_repair_result(args, code)
    risk_result = AssessorAgent(backend=args.backend).assess(code, repair_result)

    output: dict[str, Any] = {"risk_assessment": risk_result}
    if args.include_repair:
        output["repair"] = repair_result
    else:
        output["repair_summary"] = {
            "input_verdict": repair_result.get("input_verdict"),
            "suggestion_count": repair_result.get("suggestion_count"),
        }

    write_or_print(output, args.output)


if __name__ == "__main__":
    main()
