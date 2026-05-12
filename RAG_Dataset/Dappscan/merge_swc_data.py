"""
merge_swc_data.py
-----------------
Đọc training_data_functions_only_cleaned.json và swc_registry.json,
ghép thông tin SWC (title, description, remediation) vào các sample Vulnerable,
rồi xuất ra file JSON kết quả.

Cách chạy (dùng path mặc định đã cấu hình sẵn):
    python3 merge_swc_data.py

Hoặc override path bằng tham số:
    python3 merge_swc_data.py \
        --training  /path/to/training.json \
        --swc       /path/to/swc_registry.json \
        --output    /path/to/output.json
"""

import json
import re
import argparse
import sys
from pathlib import Path

# ──────────────────────────────────────────────
# ★ CẤU HÌNH PATH MẶC ĐỊNH
# ──────────────────────────────────────────────
DEFAULT_TRAINING = "/Users/phuocthanh/Documents/RAG/RAG_Dataset/Dappscan/training_data_functions_only_cleaned.json"
DEFAULT_SWC      = "/Users/phuocthanh/Documents/RAG/Pull SWC/swc_registry.json"
DEFAULT_OUTPUT   = "/Users/phuocthanh/Documents/RAG/RAG_Dataset/Dappscan/merged_output.json"


# ──────────────────────────────────────────────
# 1. Parse arguments
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Merge SWC info into training data.")
    parser.add_argument(
        "--training",
        default=DEFAULT_TRAINING,
        help=f"Path to training data JSON (default: {DEFAULT_TRAINING})",
    )
    parser.add_argument(
        "--swc",
        default=DEFAULT_SWC,
        help=f"Path to SWC registry JSON (default: {DEFAULT_SWC})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output file path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────
# 2. Load files
# ──────────────────────────────────────────────
def load_json(path: str) -> any:
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────
# 3. Build SWC lookup dict  { "SWC-107": {title, description, remediation} }
# ──────────────────────────────────────────────
def build_swc_lookup(swc_list: list) -> dict:
    lookup = {}
    for entry in swc_list:
        swc_id = entry.get("id", "").strip()          # e.g. "SWC-107"
        if swc_id:
            lookup[swc_id] = {
                "title":       entry.get("title", ""),
                "description": entry.get("description", ""),
                "remediation": entry.get("remediation", ""),
            }
    return lookup


# ──────────────────────────────────────────────
# 4. Extract SWC-ID from category string
#    "SWC-114-Transaction Order Dependence" → "SWC-114"
# ──────────────────────────────────────────────
def extract_swc_id(category: str) -> str:
    m = re.match(r"(SWC-\d+)", category.strip())
    return m.group(1) if m else ""


# ──────────────────────────────────────────────
# 5. Transform one sample
# ──────────────────────────────────────────────
def transform_sample(sample: dict, swc_lookup: dict) -> dict:
    output = sample.get("Output", "")        # "Safe" | "Vulnerable"
    code   = sample.get("Input", "")         # source code string

    result = {
        "Code":   code,
        "Output": output,
    }

    if output.lower() == "vulnerable":
        meta     = sample.get("Metadata", {})
        category = meta.get("category", "")
        swc_id   = extract_swc_id(category)
        swc_info = swc_lookup.get(swc_id, {})

        result["Metadata"] = {
            "SWC":         swc_id,
            "Title":       swc_info.get("title",       "N/A – SWC not found in registry"),
            "Description": swc_info.get("description", ""),
            "Remediation": swc_info.get("remediation", ""),
        }

    return result


# ──────────────────────────────────────────────
# 6. Main
# ──────────────────────────────────────────────
def main():
    args = parse_args()

    print(f"[INFO] Loading training data : {args.training}")
    training_data = load_json(args.training)
    if not isinstance(training_data, list):
        print("[ERROR] training data must be a JSON array.", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Loading SWC registry  : {args.swc}")
    swc_list = load_json(args.swc)
    if not isinstance(swc_list, list):
        print("[ERROR] SWC registry must be a JSON array.", file=sys.stderr)
        sys.exit(1)

    swc_lookup = build_swc_lookup(swc_list)
    print(f"[INFO] SWC entries loaded     : {len(swc_lookup)}")
    print(f"[INFO] Training samples total : {len(training_data)}")

    # Stats
    safe_count  = sum(1 for s in training_data if s.get("Output", "").lower() == "safe")
    vuln_count  = sum(1 for s in training_data if s.get("Output", "").lower() == "vulnerable")
    print(f"[INFO]   Safe       : {safe_count}")
    print(f"[INFO]   Vulnerable : {vuln_count}")

    # Transform
    output_data = [transform_sample(s, swc_lookup) for s in training_data]

    # Check for unmatched SWC IDs
    unmatched = set()
    for s in training_data:
        if s.get("Output", "").lower() == "vulnerable":
            cat = s.get("Metadata", {}).get("category", "")
            swc_id = extract_swc_id(cat)
            if swc_id and swc_id not in swc_lookup:
                unmatched.add(swc_id)
    if unmatched:
        print(f"[WARN] SWC IDs in training data but NOT in registry: {sorted(unmatched)}")

    # Write output
    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Output written to      : {out_path.resolve()}")
    print(f"[INFO] Total records          : {len(output_data)}")


if __name__ == "__main__":
    main()