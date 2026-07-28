#!/usr/bin/env python3
"""Run the full phase-1 -> phase-6 pipeline with the configured CodeLlama detector."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADAPTER = PROJECT_ROOT / "models" / "codellama-vuln-detector"
DEFAULT_WORK_DIR = PROJECT_ROOT / "E2E"


def load_env_file(path: Path = PROJECT_ROOT / ".env") -> dict[str, str]:
    env = os.environ.copy()
    if not path.exists():
        return env
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    return env


def run(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command))
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), env=env, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all LLM-BSCVM phases end-to-end.")
    parser.add_argument("--code-file", required=True, help="Solidity contract to audit.")
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR), help="Directory for intermediate outputs.")
    parser.add_argument("--output-md", default=None, help="Markdown report path.")
    parser.add_argument("--output-json", default=None, help="Full report JSON path.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps.")
    parser.add_argument("--backend", default="auto", help="Agent LLM backend: auto, gemini, or ollama.")
    parser.add_argument("--detector-adapter", default=None, help="Override CodeLlama detector adapter path.")
    parser.add_argument("--detector-base-model", default=None, help="Override base model id/path; default reads adapter config.")
    parser.add_argument("--detector-input-mode", default=None, choices=["auto", "pair", "code", "function_tag"])
    parser.add_argument("--detector-load-in-4bit", default=None, help="auto/true/false.")
    parser.add_argument("--allow-model-download", action="store_true", help="Allow Hugging Face base-model downloads.")
    parser.add_argument("--skip-rag", action="store_true", help="Skip RAG if the local knowledge store is unavailable.")
    parser.add_argument("--fast-detection", action="store_true", help="Skip detector + RAG and use static analysis only.")
    parser.add_argument("--no-repair", action="store_true", help="Generate report without Fixer/Verifier.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir)
    if not work_dir.is_absolute():
        work_dir = PROJECT_ROOT / work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    env = load_env_file()
    adapter = Path(args.detector_adapter or env.get("DETECTOR_ADAPTER_PATH", str(DEFAULT_ADAPTER)))
    if not adapter.is_absolute():
        adapter = PROJECT_ROOT / adapter

    env["DETECTOR_ADAPTER_PATH"] = str(adapter)
    env["DETECTOR_BASE_MODEL"] = args.detector_base_model or env.get("DETECTOR_BASE_MODEL", "auto")
    env["DETECTOR_INPUT_MODE"] = args.detector_input_mode or env.get("DETECTOR_INPUT_MODE", "auto")
    env["DETECTOR_LOAD_IN_4BIT"] = args.detector_load_in_4bit or env.get("DETECTOR_LOAD_IN_4BIT", "auto")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    detection_json = work_dir / "phase1_detection.json"
    phase1 = [
        sys.executable,
        str(PROJECT_ROOT / "phase1_integrated_detector.py"),
        "--code-file",
        args.code_file,
        "--output",
        str(detection_json),
        "--device",
        args.device,
        "--detector-adapter",
        str(adapter),
        "--detector-base-model",
        env["DETECTOR_BASE_MODEL"],
        "--detector-input-mode",
        env["DETECTOR_INPUT_MODE"],
        "--detector-load-in-4bit",
        env["DETECTOR_LOAD_IN_4BIT"],
    ]
    if args.allow_model_download:
        phase1.append("--allow-model-download")
    if args.skip_rag:
        phase1.append("--skip-rag")
    if args.fast_detection:
        phase1.extend(["--skip-llm", "--skip-rag"])
    run(phase1, env)

    output_md = Path(args.output_md) if args.output_md else work_dir / "audit_report.md"
    output_json = Path(args.output_json) if args.output_json else work_dir / "audit_report.json"
    phase6 = [
        sys.executable,
        str(PROJECT_ROOT / "phase6_report_generation.py"),
        "--code-file",
        args.code_file,
        "--detection-file",
        str(detection_json),
        "--backend",
        args.backend,
        "--device",
        args.device,
        "--output-md",
        str(output_md),
        "--output",
        str(output_json),
    ]
    if args.no_repair:
        phase6.append("--no-repair")
    run(phase6, env)

    print(f"Report Markdown: {output_md}")
    print(f"Report JSON: {output_json}")


if __name__ == "__main__":
    main()
