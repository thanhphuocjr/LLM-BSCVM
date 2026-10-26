#!/usr/bin/env python3
"""
Benchmark 9 Docker Security Tools on Solodit / iAudit Dataset (iaudit_solodit.jsonl).

This script:
1. Loads samples from E2E/data/eval/iaudit_solodit.jsonl
2. Wraps function snippets into compilable Solidity contracts with pragma heuristics
3. Evaluates each sample against all 9 Docker security tools (Slither, Mythril, Osiris,
   HoneyBadger, Manticore, Maian, Oyente, Securify, SmartCheck)
4. Saves incremental results to CSV so progress is never lost
5. Computes comprehensive metrics per tool and ensemble (TP, FP, TN, FN, Accuracy,
   Precision, Recall, F1, Rate of all tools reporting safe on vulnerable samples)
6. Exports both detailed CSV and summary Markdown/CSV reports.

Usage:
  # Quick test on first 5 samples:
  python3 -m E2E.tools.benchmark_solodit_9tools --limit 5

  # Run on test split (650 samples) with 4 worker threads:
  python3 -m E2E.tools.benchmark_solodit_9tools --split test --workers 4

  # Run only fast tools (Slither, SmartCheck, Mythril) on 20 samples:
  python3 -m E2E.tools.benchmark_solodit_9tools --tools slither smartcheck mythril --limit 20
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "E2E" / "data" / "eval" / "iaudit_solodit.jsonl"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "E2E" / "data" / "eval" / "results"

SECURITY_TOOL_NAMES = [
    "slither",
    "mythril",
    "smartcheck",
    "honeybadger",
    "maian",
    "manticore",
    "osiris",
    "oyente",
    "securify",
]

# Informational/style findings that should not be counted as actual security vulnerabilities
INFORMATIONAL_NAMES = {
    "solc-version",
    "naming-convention",
    "pragma",
    "SOLIDITY_PRAGMAS_VERSION",
    "SOLIDITY_VISIBILITY",
    "SOLIDITY_DEPRECATED_CONSTRUCTIONS",
    "SOLIDITY_SAFEMATH",
}


def prepare_contract_code(raw_code: str) -> str:
    """Intelligently wrap raw function snippets into valid Solidity contracts."""
    code = raw_code.strip()

    # Check if already a full contract, interface, or library
    has_contract_decl = bool(
        re.search(r"\b(?:contract|library|interface)\s+\w+", code)
    )
    if has_contract_decl:
        if not re.search(r"pragma\s+solidity", code):
            code = "// SPDX-License-Identifier: MIT\npragma solidity >=0.5.0 <0.9.0;\n" + code
        return code

    # Heuristic for pragma version
    pragma = ">=0.5.0 <0.9.0"
    if re.search(r"\bnow\b", code) or re.search(r"\bthrow\b", code) or re.search(r"\bconstant\s+returns\b", code):
        pragma = "0.5.17"
    elif "SafeERC20" in code or "IERC20" in code:
        pragma = "^0.8.0"

    # Minimal stub types commonly used in Solodit snippets to aid AST parser
    stubs = """
    // Minimal stubs for external types
    struct Receipt { uint80 amount; uint80 balance; }
    struct Vest { uint256 amount; uint256 vestPeriodInSeconds; uint256 creationTimestamp; uint256 withdrawnAmount; }
    struct PoolInfo { address lpToken; uint256 allocPoint; uint256 lastRewardBlock; uint256 accITokenPerShare; bool enableBonus; uint256 totalAmount; }
    struct UserInfo { uint256 amount; uint256 rewardDebt; uint256 rewardReceived; uint256 lastReceived; }
    struct DepositOrder { uint256 orderTime; uint256 amount; }
    struct CrowdfundOptions { string name; string symbol; address splitRecipient; uint96 splitBps; address initialContributor; address initialDelegate; address gateKeeper; uint256 gateKeeperId; }
    """

    wrapped = f"""// SPDX-License-Identifier: MIT
pragma solidity {pragma};

contract EvaluationTarget {{
{stubs}
    {code}
}}
"""
    return wrapped


def run_single_tool_on_code(
    smartbugs_bin: Path,
    tool: str,
    code: str,
    sample_id: str,
    timeout: int = 120,
) -> dict[str, Any]:
    """Execute a single SmartBugs tool via Docker container."""
    from E2E.local.tool_verifier import _read_smartbugs_results, _run

    with tempfile.TemporaryDirectory(prefix=f"eval-{tool}-{sample_id}-") as temp_dir:
        temp_root = Path(temp_dir)
        source_path = temp_root / f"{sample_id}.sol"
        source_path.write_text(code, encoding="utf-8")

        result_dir = temp_root / "results"
        result_dir.mkdir(parents=True, exist_ok=True)

        command = [
            str(smartbugs_bin),
            "-t",
            tool,
            "-f",
            str(source_path),
            "--processes",
            "1",
            "--timeout",
            str(timeout),
            "--mem-limit",
            "4g",
            "--json",
            "--results",
            str(result_dir),
            "--overwrite",
            "--quiet",
        ]

        t0 = time.perf_counter()
        run_res = _run(command, timeout=timeout + 60, cwd=smartbugs_bin.parent)
        elapsed_sec = round(time.perf_counter() - t0, 2)

        has_result = any(result_dir.rglob("result.json"))
        findings, errors, fails = _read_smartbugs_results(tool, result_dir)

        # Filter out purely benign style / pragma noise
        security_findings = [
            f for f in findings
            if str(f.get("name") or f.get("check") or "") not in INFORMATIONAL_NAMES
            and str(f.get("severity") or "").lower() not in {"informational", "info", "optimization"}
        ]
        # If no severe findings, but non-empty raw findings, check if any meaningful issue exists
        if not security_findings and findings:
            security_findings = [
                f for f in findings
                if str(f.get("name") or "") not in INFORMATIONAL_NAMES
            ]

        is_completed = has_result or run_res["status"] == "passed"
        if not is_completed:
            status = "error"
            verdict = "Error"
            error_msg = run_res.get("error") or run_res.get("stderr") or "SmartBugs execution failed"
        else:
            status = "completed"
            verdict = "Vulnerable" if len(security_findings) > 0 else "Safe"
            error_msg = ""

        return {
            "tool": tool,
            "status": status,
            "verdict": verdict,
            "is_vulnerable": verdict == "Vulnerable",
            "finding_count": len(security_findings),
            "raw_finding_count": len(findings),
            "finding_names": "; ".join(str(f.get("name") or "") for f in security_findings[:5]),
            "elapsed_sec": elapsed_sec,
            "error": error_msg[:200] if error_msg else "",
        }


def evaluate_sample(
    sample: dict[str, Any],
    smartbugs_bin: Path,
    tools: list[str],
    timeout: int = 120,
) -> dict[str, Any]:
    """Evaluate one sample across all selected tools."""
    sample_id = sample.get("id", "unknown")
    raw_code = sample.get("code", "")
    ground_truth = sample.get("label", "Unknown")  # "Safe" or "Vulnerable"
    gt_is_vuln = (ground_truth == "Vulnerable")
    split = sample.get("split", "unknown")

    contract_code = prepare_contract_code(raw_code)

    row_data: dict[str, Any] = {
        "sample_id": sample_id,
        "split": split,
        "ground_truth": ground_truth,
        "gt_is_vuln": 1 if gt_is_vuln else 0,
        "code_length": len(raw_code),
        "code_lines": raw_code.count("\n") + 1,
        "code_preview": raw_code.replace("\n", " ")[:100],
    }

    tool_results: dict[str, dict[str, Any]] = {}
    vulnerable_tools = []
    completed_tools = []
    error_tools = []

    for tool in tools:
        t_res = run_single_tool_on_code(
            smartbugs_bin=smartbugs_bin,
            tool=tool,
            code=contract_code,
            sample_id=sample_id,
            timeout=timeout,
        )
        tool_results[tool] = t_res

        row_data[f"{tool}_status"] = t_res["status"]
        row_data[f"{tool}_verdict"] = t_res["verdict"]
        row_data[f"{tool}_is_vuln"] = 1 if t_res["is_vulnerable"] else 0
        row_data[f"{tool}_finding_count"] = t_res["finding_count"]
        row_data[f"{tool}_finding_names"] = t_res["finding_names"]
        row_data[f"{tool}_elapsed_s"] = t_res["elapsed_sec"]

        if t_res["status"] == "completed":
            completed_tools.append(tool)
            if t_res["is_vulnerable"]:
                vulnerable_tools.append(tool)
        else:
            error_tools.append(tool)

    # Ensemble statistics
    tools_vuln_count = len(vulnerable_tools)
    tools_completed_count = len(completed_tools)
    any_tool_vuln = (tools_vuln_count > 0)
    majority_vuln = (tools_vuln_count >= max(1, len(tools) // 2 + 1))
    all_completed_safe = (tools_completed_count > 0 and tools_vuln_count == 0)

    # False Negative of full suite: GT is Vulnerable, but ALL tools report Safe
    fn_consensus = (gt_is_vuln and all_completed_safe)
    # False Positive of suite: GT is Safe, but at least 1 tool flags Vulnerable
    fp_any = (not gt_is_vuln and any_tool_vuln)

    row_data["tools_completed_count"] = tools_completed_count
    row_data["tools_vuln_count"] = tools_vuln_count
    row_data["tools_vuln_list"] = "; ".join(vulnerable_tools)
    row_data["tools_error_count"] = len(error_tools)
    row_data["tools_error_list"] = "; ".join(error_tools)
    row_data["ensemble_any_vuln"] = 1 if any_tool_vuln else 0
    row_data["ensemble_majority_vuln"] = 1 if majority_vuln else 0
    row_data["all_tools_safe"] = 1 if all_completed_safe else 0
    row_data["false_negative_consensus"] = 1 if fn_consensus else 0
    row_data["false_positive_any"] = 1 if fp_any else 0

    return row_data


def compute_metrics(y_true: list[int], y_pred: list[int | None]) -> dict[str, Any]:
    """Compute binary classification metrics ignoring None/Error predictions."""
    valid_pairs = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
    if not valid_pairs:
        return {
            "total_evaluated": 0,
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "detection_rate": 0.0, "fpr": 0.0,
        }

    tp = sum(1 for t, p in valid_pairs if t == 1 and p == 1)
    fp = sum(1 for t, p in valid_pairs if t == 0 and p == 1)
    tn = sum(1 for t, p in valid_pairs if t == 0 and p == 0)
    fn = sum(1 for t, p in valid_pairs if t == 1 and p == 0)
    total = len(valid_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "total_evaluated": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "detection_rate": round(recall * 100, 2),  # Percentage
        "fpr": round(fpr * 100, 2),                # Percentage
    }


def generate_summary_report(
    rows: list[dict[str, Any]],
    tools: list[str],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Generate both CSV and Markdown summary reports."""
    summary_rows = []
    y_true = [row["gt_is_vuln"] for row in rows]
    total_samples = len(rows)
    total_vuln = sum(y_true)
    total_safe = total_samples - total_vuln

    # Per-tool metrics
    for tool in tools:
        y_pred = []
        errors = 0
        total_time = 0.0
        for row in rows:
            status = row.get(f"{tool}_status", "error")
            total_time += float(row.get(f"{tool}_elapsed_s") or 0.0)
            if status == "completed":
                y_pred.append(row.get(f"{tool}_is_vuln", 0))
            else:
                y_pred.append(None)
                errors += 1

        m = compute_metrics(y_true, y_pred)
        avg_time = round(total_time / total_samples, 2) if total_samples else 0.0
        err_rate = round(errors / total_samples * 100, 2) if total_samples else 0.0

        summary_rows.append({
            "model_or_tool": tool.upper(),
            "category": "Single Tool",
            "evaluated_samples": m["total_evaluated"],
            "error_count": errors,
            "error_rate_pct": err_rate,
            "tp": m["tp"],
            "fp": m["fp"],
            "tn": m["tn"],
            "fn": m["fn"],
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1_score": m["f1"],
            "detection_rate_pct": m["detection_rate"],
            "fpr_pct": m["fpr"],
            "avg_time_sec": avg_time,
        })

    # Ensemble metrics
    for ens_name, col_name, cat in [
        ("Ensemble (Any Tool >= 1)", "ensemble_any_vuln", "Ensemble"),
        ("Ensemble (Majority >= 5)", "ensemble_majority_vuln", "Ensemble"),
    ]:
        y_pred_ens = [row.get(col_name, 0) for row in rows]
        m = compute_metrics(y_true, y_pred_ens)
        summary_rows.append({
            "model_or_tool": ens_name,
            "category": cat,
            "evaluated_samples": total_samples,
            "error_count": 0,
            "error_rate_pct": 0.0,
            "tp": m["tp"],
            "fp": m["fp"],
            "tn": m["tn"],
            "fn": m["fn"],
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1_score": m["f1"],
            "detection_rate_pct": m["detection_rate"],
            "fpr_pct": m["fpr"],
            "avg_time_sec": "-",
        })

    # Suite False Negative Rate: All tools said Safe, but GT was Vulnerable
    fn_consensus_count = sum(row.get("false_negative_consensus", 0) for row in rows)
    fn_consensus_rate = round(fn_consensus_count / total_vuln * 100, 2) if total_vuln else 0.0

    # Write summary CSV
    summary_csv_path = output_dir / "solodit_9tools_benchmark_summary.csv"
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    # Write Markdown report
    report_md_path = output_dir / "solodit_9tools_benchmark_report.md"
    md_content = f"""# 📊 BÁO CÁO ĐÁNH GIÁ 9 CÔNG CỤ BẢO MẬT TRÊN TẬP DỮ LIỆU SOLODIT

**Ngày thực hiện:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Tổng số mẫu kiểm thử:** {total_samples} mẫu (🔴 {total_vuln} Vulnerable / 🟢 {total_safe} Safe)  
**Tập dữ liệu:** `{DATASET_PATH.name}`  

---

## 🎯 I. CHỈ SỐ QUAN TRỌNG NHẤT THEO YÊU CẦU

| Chỉ số tổng hợp cốt lõi | Giá trị | Ý nghĩa |
| :--- | :---: | :--- |
| **Tỉ lệ TẤT CẢ 9 Tool báo KHÔNG LỖI nhưng sample THỰC SỰ CÓ LỖI (Suite False Negative)** | **{fn_consensus_rate}%** ({fn_consensus_count}/{total_vuln} mẫu) | Tỉ lệ lỗi nguy hiểm bị cả 9 công cụ bỏ sót hoàn toàn |
| **Độ bao phủ phát hiện lỗi (Ensemble Any-Tool Recall)** | **{summary_rows[-2]['detection_rate_pct']}%** | Tỉ lệ phát hiện ít nhất bởi 1 trong 9 công cụ |
| **Tỉ lệ báo động giả tổng hợp (Ensemble False Positive Rate)** | **{summary_rows[-2]['fpr_pct']}%** | Tỉ lệ công cụ báo có lỗi trên các hàm an toàn |

---

## 📈 II. BẢNG TỔNG HỢP HIỆU NĂNG TỪNG CÔNG CỤ VÀ ENSEMBLE

| Công cụ / Phương pháp | TP | FP | TN | FN | Accuracy | Precision | Recall (Phát hiện) | F1-Score | Tỉ lệ Lỗi (Error %) | Thời gian TB |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for sr in summary_rows:
        md_content += f"| **{sr['model_or_tool']}** | {sr['tp']} | {sr['fp']} | {sr['tn']} | {sr['fn']} | {sr['accuracy']:.4f} | {sr['precision']:.4f} | **{sr['recall']:.4f}** ({sr['detection_rate_pct']}%) | **{sr['f1_score']:.4f}** | {sr['error_rate_pct']}% | {sr['avg_time_sec']}s |\n"

    md_content += f"""
---

## 📂 III. CÁC TỆP ĐẦU RA CHI TIẾT
- **Bảng chi tiết từng mẫu:** [`{output_dir.name}/solodit_9tools_benchmark_details.csv`](file://{output_dir.resolve()}/solodit_9tools_benchmark_details.csv)
- **Bảng tổng hợp chỉ số:** [`{output_dir.name}/solodit_9tools_benchmark_summary.csv`](file://{output_dir.resolve()}/solodit_9tools_benchmark_summary.csv)

"""
    report_md_path.write_text(md_content, encoding="utf-8")
    return summary_csv_path, report_md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 9 Docker security tools benchmark on Solodit dataset.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="Path to dataset JSONL (default: iaudit_solodit.jsonl)",
    )
    parser.add_argument(
        "--split",
        choices=("all", "test", "val", "train"),
        default="test",
        help="Dataset split to evaluate (default: test - 650 samples)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process (for testing/debugging)",
    )
    parser.add_argument(
        "--tools",
        nargs="*",
        default=SECURITY_TOOL_NAMES,
        help=f"List of tools to evaluate (default: all 9 tools: {', '.join(SECURITY_TOOL_NAMES)})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent worker threads for sample evaluation (default: 4)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout in seconds per tool execution (default: 120)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory to save output CSV and report files",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume execution from existing details CSV without re-evaluating completed samples",
    )
    return parser.parse_args()


def main() -> None:
    from E2E.local.tool_verifier import _is_docker_available, _smartbugs_executable

    args = parse_args()

    # 1. Check Docker & SmartBugs
    if not _is_docker_available():
        print("❌ Lỗi: Docker daemon không hoạt động! Vui lòng khởi động Docker Desktop trước.", file=sys.stderr)
        sys.exit(1)

    smartbugs_bin, sb_reason = _smartbugs_executable()
    if smartbugs_bin is None:
        print(f"❌ Lỗi: Không tìm thấy SmartBugs executable ({sb_reason})", file=sys.stderr)
        sys.exit(1)

    print(f"✅ Docker & SmartBugs đã sẵn sàng: {smartbugs_bin}")
    print(f"🛠️ Danh sách tools ({len(args.tools)}): {', '.join(args.tools)}")

    # 2. Load dataset
    if not args.dataset.exists():
        print(f"❌ Lỗi: Không tìm thấy file dataset: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    all_samples = []
    with args.dataset.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if args.split == "all" or item.get("split") == args.split:
                all_samples.append(item)

    if args.limit:
        all_samples = all_samples[:args.limit]

    total_samples = len(all_samples)
    print(f"📦 Đã tải {total_samples} mẫu từ split '{args.split}' (Limit: {args.limit or 'Không giới hạn'})")

    if total_samples == 0:
        print("⚠️ Không có mẫu nào để đánh giá!")
        return

    # 3. Setup output files & resume capability
    args.results_dir.mkdir(parents=True, exist_ok=True)
    details_csv_path = args.results_dir / "solodit_9tools_benchmark_details.csv"

    completed_ids: set[str] = set()
    evaluated_rows: list[dict[str, Any]] = []

    if args.resume and details_csv_path.exists():
        with details_csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                completed_ids.add(r["sample_id"])
                # cast numeric fields
                r["gt_is_vuln"] = int(r["gt_is_vuln"])
                r["ensemble_any_vuln"] = int(r.get("ensemble_any_vuln", 0))
                r["ensemble_majority_vuln"] = int(r.get("ensemble_majority_vuln", 0))
                r["all_tools_safe"] = int(r.get("all_tools_safe", 0))
                r["false_negative_consensus"] = int(r.get("false_negative_consensus", 0))
                r["false_positive_any"] = int(r.get("false_positive_any", 0))
                for t in args.tools:
                    if f"{t}_is_vuln" in r:
                        r[f"{t}_is_vuln"] = int(r[f"{t}_is_vuln"])
                    if f"{t}_finding_count" in r:
                        r[f"{t}_finding_count"] = int(r[f"{t}_finding_count"])
                evaluated_rows.append(r)
        print(f"🔄 Tiếp tục (Resume): Đã bỏ qua {len(completed_ids)} mẫu đã hoàn thành trước đó.")

    samples_to_run = [s for s in all_samples if s.get("id") not in completed_ids]
    print(f"🚀 Bắt đầu đánh giá {len(samples_to_run)} mẫu còn lại với {args.workers} luồng song song...\n")

    # Determine CSV fieldnames from a dummy evaluation or schema
    sample_fieldnames = [
        "sample_id", "split", "ground_truth", "gt_is_vuln", "code_length", "code_lines", "code_preview",
    ]
    for t in args.tools:
        sample_fieldnames.extend([
            f"{t}_status", f"{t}_verdict", f"{t}_is_vuln", f"{t}_finding_count", f"{t}_finding_names", f"{t}_elapsed_s",
        ])
    sample_fieldnames.extend([
        "tools_completed_count", "tools_vuln_count", "tools_vuln_list", "tools_error_count", "tools_error_list",
        "ensemble_any_vuln", "ensemble_majority_vuln", "all_tools_safe", "false_negative_consensus", "false_positive_any",
    ])

    # Open CSV for incremental writing
    csv_mode = "a" if (args.resume and details_csv_path.exists()) else "w"
    csv_file = details_csv_path.open(csv_mode, newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=sample_fieldnames)
    if csv_mode == "w":
        csv_writer.writeheader()
        csv_file.flush()

    start_time = time.time()
    processed_count = 0

    try:
        if args.workers <= 1:
            for i, sample in enumerate(samples_to_run, 1):
                row = evaluate_sample(sample, smartbugs_bin, args.tools, timeout=args.timeout)
                evaluated_rows.append(row)
                csv_writer.writerow(row)
                csv_file.flush()
                processed_count += 1
                vuln_str = f"🔴 {row['tools_vuln_count']}/{len(args.tools)} tools báo lỗi" if row['tools_vuln_count'] else "🟢 Safe"
                print(f"[{processed_count}/{len(samples_to_run)}] Sample {sample.get('id')} ({row['ground_truth']}) -> {vuln_str} ({row['tools_vuln_list'] or 'None'})")
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                future_to_sample = {
                    executor.submit(evaluate_sample, s, smartbugs_bin, args.tools, args.timeout): s
                    for s in samples_to_run
                }
                for future in as_completed(future_to_sample):
                    sample = future_to_sample[future]
                    try:
                        row = future.result()
                        evaluated_rows.append(row)
                        csv_writer.writerow(row)
                        csv_file.flush()
                        processed_count += 1
                        vuln_str = f"🔴 {row['tools_vuln_count']}/{len(args.tools)} tools báo lỗi" if row['tools_vuln_count'] else "🟢 Safe"
                        print(f"[{processed_count}/{len(samples_to_run)}] Sample {sample.get('id')} ({row['ground_truth']}) -> {vuln_str} ({row['tools_vuln_list'] or 'None'})")
                    except Exception as e:
                        print(f"❌ Lỗi khi xử lý sample {sample.get('id')}: {e}", file=sys.stderr)
    finally:
        csv_file.close()

    total_time = round(time.time() - start_time, 2)
    print(f"\n🎉 Đã hoàn tất đánh giá toàn bộ {len(evaluated_rows)} mẫu trong {total_time}s!")

    # 4. Generate Summary Reports
    summary_csv, report_md = generate_summary_report(evaluated_rows, args.tools, args.results_dir)
    print(f"📄 File chi tiết CSV: {details_csv_path}")
    print(f"📊 File tổng hợp CSV: {summary_csv}")
    print(f"📝 Báo cáo Markdown: {report_md}")


if __name__ == "__main__":
    main()
