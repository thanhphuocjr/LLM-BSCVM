#!/usr/bin/env python3
"""
Phase 2 — Repair Suggestion (Advisor agent) of the LLM-BSCVM framework.

Takes the result of the detection phase (phase1_integrated_detector.py) and
generates a structured repair suggestion for every detected vulnerability,
following the paper's "Decompose-Retrieve-Generate" approach.

Just like phase 1, running with no arguments uses the inline CODE_TO_TEST
sample (imported from phase 1) and runs the full pipeline end-to-end:

    python3 phase2_repair_suggestion.py

Other options:

  * Static-only detection (fast, skips CodeBERT + RAG):
        python3 phase2_repair_suggestion.py --fast-detection

  * Analyze your own contract:
        python3 phase2_repair_suggestion.py --code-file C.sol

  * Reuse a saved phase-1 JSON instead of re-running detection:
        python3 phase2_repair_suggestion.py --detection-file det.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.advisor import AdvisorAgent  # noqa: E402
from phase1_integrated_detector import CODE_TO_TEST  # noqa: E402


def read_code(args: argparse.Namespace) -> str:
    if args.code_file:
        return Path(args.code_file).read_text(encoding="utf-8")
    if args.code:
        return args.code
    return CODE_TO_TEST


def load_detection_result(args: argparse.Namespace, code: str) -> dict[str, Any]:
    if args.detection_file:
        return json.loads(Path(args.detection_file).read_text(encoding="utf-8"))
    return run_phase1_detection(args, code)


def run_phase1_detection(args: argparse.Namespace, code: str) -> dict[str, Any]:
    """Run phase1_integrated_detector.py as a subprocess and return its JSON."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sol", prefix="phase2_input_", encoding="utf-8", delete=False
    ) as handle:
        handle.write(code)
        code_file = Path(handle.name)

    output_file = Path(tempfile.mktemp(suffix=".json", prefix="phase2_detection_"))
    command = [
        sys.executable,
        str(PROJECT_ROOT / "phase1_integrated_detector.py"),
        "--code-file",
        str(code_file),
        "--output",
        str(output_file),
        "--device",
        args.device,
    ]
    if args.fast_detection:
        # Static-only: skips the heavy CodeBERT + embedding-RAG components.
        command += ["--skip-llm", "--skip-rag"]

    try:
        completed = subprocess.run(command, cwd=str(PROJECT_ROOT), text=True, capture_output=True)
        if completed.returncode != 0:
            raise RuntimeError(
                "Phase-1 detection failed:\n" + (completed.stderr or completed.stdout or "").strip()
            )
        return json.loads(output_file.read_text(encoding="utf-8"))
    finally:
        code_file.unlink(missing_ok=True)
        output_file.unlink(missing_ok=True)


def write_or_print(output: dict[str, Any], output_path: str | None) -> None:
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    if output_path:
        Path(output_path).write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote repair suggestions to {output_path}")
    else:
        print(rendered)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate repair suggestions for detected vulnerabilities.")
    parser.add_argument("--code-file", default=None, help="Path to the Solidity contract. Highest priority.")
    parser.add_argument("--code", default=None, help="Raw Solidity code string.")
    parser.add_argument("--detection-file", default=None, help="Saved phase-1 detection JSON to reuse.")
    parser.add_argument(
        "--fast-detection",
        action="store_true",
        help="Run detection in static-only mode (skip the heavy CodeBERT + RAG components).",
    )
    parser.add_argument("--device", default="cpu", help="Device for phase-1 detection: auto | cpu | cuda | mps.")
    parser.add_argument("--backend", default="gemini", help="Generative LLM backend for the Advisor.")
    parser.add_argument("--knowledge-dir", default=str(PROJECT_ROOT / "rag" / "knowledge_store"))
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--include-detection",
        action="store_true",
        help="Embed the full detection result in the output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    code = read_code(args)
    detection_result = load_detection_result(args, code)

    agent = AdvisorAgent(knowledge_dir=args.knowledge_dir, backend=args.backend)
    repair_result = agent.advise(code, detection_result)

    output: dict[str, Any] = {"repair": repair_result}
    if args.include_detection:
        output["detection"] = detection_result
    else:
        output["detection_summary"] = detection_result.get("final", {})

    write_or_print(output, args.output)


if __name__ == "__main__":
    main()
