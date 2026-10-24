#!/usr/bin/env python3
"""
phase5_patch_verification.py
──────────────────────────────────────────────────────────────────────────────
Phase 5 — Patch Verification (Verifier agent) của framework LLM-BSCVM.

Đã làm sạch toàn bộ mã nguồn kiểm tra cũ (Slither, SmartBugs, heuristic rules).
Sẵn sàng tiếp nhận phương pháp xác minh mới.
──────────────────────────────────────────────────────────────────────────────
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

from agents.verifier import VerifierAgent  # noqa: E402
from phase2_repair_suggestion import load_detection_result, read_code  # noqa: E402


def write_or_print(output: dict[str, Any], output_path: str | None) -> None:
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    if output_path:
        Path(output_path).write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote verification result to {output_path}")
    else:
        print(rendered)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify repaired contract (Phase 5).")
    parser.add_argument("--code-file", default=None, help="Path to Solidity contract.")
    parser.add_argument("--code", default=None, help="Raw Solidity code string.")
    parser.add_argument("--fix-file", default=None, help="Saved phase-4 (Fixer) JSON to reuse.")
    parser.add_argument("--detection-file", default=None, help="Saved phase-1 detection JSON to reuse.")
    parser.add_argument("--backend", default="auto", help="LLM backend: auto | gemini | ollama.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    code = read_code(args)
    detection_result = load_detection_result(args, code)

    fix_result = {}
    if args.fix_file:
        saved = json.loads(Path(args.fix_file).read_text(encoding="utf-8"))
        fix_result = saved.get("repair_patch", saved)

    # Chạy Verifier Agent (đã làm sạch)
    verification = VerifierAgent(backend=args.backend).verify(
        original_code=code,
        fix_result=fix_result,
        original_detection=detection_result,
    )

    output: dict[str, Any] = {
        "status": "ready_for_new_verification",
        "verification": verification,
    }
    write_or_print(output, args.output)


if __name__ == "__main__":
    main()
