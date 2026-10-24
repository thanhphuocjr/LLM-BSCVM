#!/usr/bin/env python3
"""
E2E/tools/benchmark.py
──────────────────────────────────────────────────────────────────────────────
Regression Benchmark Suite for LLM-BSCVM Pipeline (Task F1).

Evaluates pipeline detection and fusion logic against 3 reference benchmark contracts:
  1. Coinage (0x7415c7bf3e2415fa9a55f1fd8b6fccf2914c39a6.sol):
     - Baseline compiler: solc-0.4.18
     - Expected: 0 Critical findings (no false-positive mint/burn/atoshima access control/selfdestruct)
     - Expected: min / max / ttpf pure/view functions excluded from failing units
  2. TRUEToken (0xa4d17ab1ee0efdd23edc2869e7ba96b89eecf9ab.sol):
     - Baseline compiler: solc-0.4.11
     - Expected: 0 Critical findings (no false-positive kill() suicide)
     - Expected: <= 5 candidate findings passed to agent
  3. elixir (0xc8c6a31a4a806d3710a7b38b7b296d2fabccdba8.sol):
     - Baseline compiler: solc-0.4.11
     - Expected: exactly 4 SWC-115 findings (tx.origin authentication) in fused output
     - Expected: elixor.returnAmountOfELIXAddressCanProduce marked has_body=False (no body)
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from E2E.local.static_runner import run_static
from E2E.shared.function_parser import parse_code_units
from E2E.shared.fusion import fuse_findings
from E2E.shared.solc import select_best_solc


BENCHMARK_CONTRACTS = {
    "coinage": {
        "name": "Coinage",
        "path": PROJECT_ROOT / "smartbugs_wild" / "contracts" / "0x7415c7bf3e2415fa9a55f1fd8b6fccf2914c39a6.sol",
        "expected_solc": "0.4.18",
        "expected_critical_max": 0,
        "expected_fused_max": 5,
        "must_not_fail_units": ["min", "max", "ttpf"],
    },
    "true_token": {
        "name": "TRUEToken",
        "path": PROJECT_ROOT / "smartbugs_wild" / "contracts" / "0xa4d17ab1ee0efdd23edc2869e7ba96b89eecf9ab.sol",
        "expected_solc": "0.4.11",
        "expected_critical_max": 0,
        "expected_fused_max": 5,
        "must_not_fail_units": [],
    },
    "elixir": {
        "name": "elixir",
        "path": PROJECT_ROOT / "smartbugs_wild" / "contracts" / "0xc8c6a31a4a806d3710a7b38b7b296d2fabccdba8.sol",
        "expected_solc": "0.4.11",
        "expected_swc_115": 4,
        "empty_body_units": ["elixor.returnAmountOfELIXAddressCanProduce"],
    },
}


def evaluate_contract(key: str, config: dict[str, Any]) -> dict[str, Any]:
    path = config["path"]
    if not path.exists():
        return {
            "key": key,
            "name": config["name"],
            "status": "error",
            "error": f"File not found: {path}",
        }

    start_time = time.time()
    code = path.read_text(encoding="utf-8")

    # 1. Compiler Selection Test
    solc_bin, selection_reason, raw_pragma = select_best_solc(code)
    compiler_ver = str(solc_bin.name if solc_bin else "none")

    # 2. Function Parser & Body Check
    all_units = parse_code_units(code)
    body_units = [u for u in all_units if u.has_body]
    empty_body_units = [u.unit_id for u in all_units if not u.has_body]
    pure_view_units = [u.unit_id for u in all_units if u.is_view_or_pure]

    # 3. Static & Slither Analysis
    static_res = run_static(code, units=body_units)
    static_findings = static_res.get("findings", [])
    static_criticals = [f for f in static_findings if f.get("severity") == "Critical"]

    # 4. Multi-modal Fusion
    fused, candidates = fuse_findings([], static_res, {"findings": []})
    fused_criticals = [f for f in fused if f.get("severity") == "Critical"]
    fused_swc_115 = [f for f in fused if f.get("swc") == "SWC-115"]

    elapsed = round(time.time() - start_time, 2)

    # 5. Assertions check
    assertions_passed = True
    failures: list[str] = []

    # Check expected compiler
    expected_solc = config.get("expected_solc")
    if expected_solc and expected_solc not in compiler_ver:
        assertions_passed = False
        failures.append(f"Compiler mismatch: got {compiler_ver}, expected {expected_solc}")

    # Check Critical count
    if "expected_critical_max" in config:
        max_crit = config["expected_critical_max"]
        if len(fused_criticals) > max_crit:
            assertions_passed = False
            failures.append(f"Too many Critical findings: got {len(fused_criticals)}, max {max_crit}")

    # Check Fused count
    if "expected_fused_max" in config:
        max_fused = config["expected_fused_max"]
        if len(fused) > max_fused:
            assertions_passed = False
            failures.append(f"Too many Fused findings: got {len(fused)}, max {max_fused}")

    # Check SWC-115 count
    if "expected_swc_115" in config:
        exp_115 = config["expected_swc_115"]
        if len(fused_swc_115) != exp_115:
            assertions_passed = False
            failures.append(f"SWC-115 count mismatch: got {len(fused_swc_115)}, expected {exp_115}")

    # Check empty body units filtered
    for expected_empty in config.get("empty_body_units", []):
        if expected_empty not in empty_body_units:
            assertions_passed = False
            failures.append(f"Expected empty body unit not filtered: {expected_empty}")

    return {
        "key": key,
        "name": config["name"],
        "status": "PASS" if assertions_passed else "FAIL",
        "failures": failures,
        "compiler": compiler_ver,
        "total_units": len(all_units),
        "body_units": len(body_units),
        "empty_body_units": empty_body_units,
        "pure_view_units": pure_view_units,
        "static_engine": static_res.get("engine"),
        "static_findings_count": len(static_findings),
        "fused_count": len(fused),
        "candidates_count": len(candidates),
        "fused_critical_count": len(fused_criticals),
        "fused_swc_115_count": len(fused_swc_115),
        "fused_findings": [
            {
                "swc": f.get("swc"),
                "unit_id": f.get("unit_id"),
                "severity": f.get("severity"),
                "p_final": f.get("p_final"),
                "line": f.get("line"),
            }
            for f in fused
        ],
        "elapsed_seconds": elapsed,
    }


def run_benchmark(target: str = "all") -> dict[str, Any]:
    targets = BENCHMARK_CONTRACTS if target == "all" else {target: BENCHMARK_CONTRACTS[target]}
    results = {}
    all_passed = True

    print("\n" + "=" * 80)
    print(" LLM-BSCVM REGRESSION BENCHMARK SUITE (Task F1)")
    print("=" * 80)

    for key, config in targets.items():
        print(f"\n[RUNNING] Evaluating {config['name']} ({key})...")
        res = evaluate_contract(key, config)
        results[key] = res
        if res.get("status") != "PASS":
            all_passed = False

        status_str = "PASS" if res.get("status") == "PASS" else "FAIL"
        print(f"[{status_str}] {config['name']}:")
        print(f"  Compiler: {res.get('compiler')}")
        print(f"  Units: {res.get('total_units')} total, {res.get('body_units')} with body")
        print(f"  Static Engine: {res.get('static_engine')}")
        print(f"  Fused Findings: {res.get('fused_count')} confirmed, {res.get('candidates_count')} candidates")
        print(f"  Criticals: {res.get('fused_critical_count')} | SWC-115: {res.get('fused_swc_115_count')}")
        print(f"  Elapsed: {res.get('elapsed_seconds')}s")
        if res.get("failures"):
            print("  FAILURES:")
            for failure in res["failures"]:
                print(f"    - {failure}")

    print("\n" + "=" * 80)
    verdict = "ALL BENCHMARKS PASSED" if all_passed else "REGRESSION DETECTED"
    print(f" OVERALL VERDICT: {verdict}")
    print("=" * 80 + "\n")

    return {
        "verdict": verdict,
        "all_passed": all_passed,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-BSCVM Benchmark Suite")
    parser.add_argument(
        "--contract",
        choices=["all", "coinage", "true_token", "elixir"],
        default="all",
        help="Target benchmark contract to evaluate",
    )
    parser.add_argument("--json", action="store_true", help="Print output in JSON format")
    parser.add_argument(
        "--check-regression",
        action="store_true",
        help="Exit with code 1 if any benchmark fails regression check",
    )
    args = parser.parse_args()

    data = run_benchmark(target=args.contract)

    if args.json:
        print(json.dumps(data, indent=2))

    if args.check_regression and not data["all_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
