#!/usr/bin/env python3
"""Build Full Contract Level Dataset from DAppSCAN.

Creates:
1. data/training/dappscan_contracts_vuln.jsonl (948 full contracts with confirmed SWC vulnerabilities)
2. data/training/dappscan_contracts_safe.jsonl (948 balanced substantive safe full contracts)
3. data/training/dappscan_contracts_full.jsonl (combined 1,896 balanced full contracts)
"""

import os
import re
import json
import random
from pathlib import Path

# Paths
E2E_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = E2E_ROOT / "data" / "source" / "raw" / "DAppSCAN"
SWC_DIR = RAW_ROOT / "DAppSCAN-source" / "SWCsource"
CONTRACTS_DIR = RAW_ROOT / "DAppSCAN-source" / "contracts"
OUT_DIR = E2E_ROOT / "data" / "training"

def clean_swc_comments(code: str) -> str:
    """Strip SWC annotation comments (e.g. // SWC-101) to prevent label leakage."""
    code = re.sub(r"/\*+\s*SWC-\d+.*?\*/", "", code, flags=re.DOTALL)
    out = []
    for line in code.split("\n"):
        stripped = line.lstrip()
        if re.match(r"^//+\s*SWC-\d+", stripped):
            continue
        m = re.search(r"//+\s*SWC-\d+[^\n]*$", line)
        if m:
            line = line[:m.start()].rstrip()
        out.append(line)
    return "\n".join(out)

def extract_project_name(rel_path: str) -> str:
    parts = Path(rel_path).parts
    for i, part in enumerate(parts):
        if part == "contracts" and i + 1 < len(parts):
            return parts[i + 1]
    return "Unknown"

def build_datasets():
    print(f"Scanning DAppSCAN files at: {RAW_ROOT}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Extract Vulnerable Contracts
    swc_files = sorted(list(SWC_DIR.rglob("*.json")))
    print(f"Found {len(swc_files)} SWC annotation files...")
    
    vuln_records = []
    vuln_rel_paths = set()
    
    for idx, sf in enumerate(swc_files):
        try:
            meta = json.loads(sf.read_text(encoding="utf-8", errors="ignore"))
            rel_fp = meta.get("filePath")
            if not rel_fp:
                continue
                
            sol_path = RAW_ROOT / rel_fp if not os.path.isabs(rel_fp) else Path(rel_fp)
            if not sol_path.exists():
                continue
                
            vuln_rel_paths.add(rel_fp)
            raw_code = sol_path.read_text(encoding="utf-8", errors="ignore")
            clean_code = clean_swc_comments(raw_code)
            
            swcs_meta = meta.get("SWCs", [])
            swc_ids = sorted(list(set(
                m.group(1) for s in swcs_meta
                if (m := re.match(r"(SWC-\d+)", str(s.get("category", ""))))
            )))
            
            project = extract_project_name(rel_fp)
            record = {
                "id": f"dappscan_contract_v_{idx}",
                "project": project,
                "file_name": sol_path.name,
                "file_path": rel_fp,
                "label": "Vulnerable",
                "swc_ids": swc_ids,
                "swc_details": swcs_meta,
                "n_lines": len(raw_code.splitlines()),
                "n_chars": len(raw_code),
                "code": clean_code,
                "raw_code": raw_code,
            }
            vuln_records.append(record)
        except Exception as e:
            print(f"Error processing {sf}: {e}")
            
    print(f"Extracted {len(vuln_records)} vulnerable full contracts.")
    
    # 2. Extract Substantive Safe Contracts
    all_sol_files = sorted(list(CONTRACTS_DIR.rglob("*.sol")))
    print(f"Scanning {len(all_sol_files)} total .sol files for safe candidates...")
    
    safe_candidates = []
    for sol_path in all_sol_files:
        rel_fp = str(sol_path.relative_to(RAW_ROOT))
        if rel_fp in vuln_rel_paths:
            continue
            
        try:
            raw_code = sol_path.read_text(encoding="utf-8", errors="ignore")
            # Quality check: contains actual contract or library
            if not re.search(r"\b(contract|library)\s+[A-Za-z0-9_]+", raw_code):
                continue
            # Quality check: non-trivial length (25 <= lines <= 900)
            lines = raw_code.splitlines()
            if 25 <= len(lines) <= 900:
                safe_candidates.append((sol_path, rel_fp, raw_code))
        except:
            continue
            
    print(f"Found {len(safe_candidates)} high-quality safe contract candidates.")
    
    # Sample balanced number of safe contracts to match vulnerable count (948)
    random.seed(42)
    selected_safe = random.sample(safe_candidates, min(len(vuln_records), len(safe_candidates)))
    
    safe_records = []
    for idx, (sol_path, rel_fp, raw_code) in enumerate(selected_safe):
        project = extract_project_name(rel_fp)
        record = {
            "id": f"dappscan_contract_s_{idx}",
            "project": project,
            "file_name": sol_path.name,
            "file_path": rel_fp,
            "label": "Safe",
            "swc_ids": [],
            "swc_details": [],
            "n_lines": len(raw_code.splitlines()),
            "n_chars": len(raw_code),
            "code": raw_code,
            "raw_code": raw_code,
        }
        safe_records.append(record)
        
    print(f"Sampled {len(safe_records)} safe full contracts for 1:1 balance.")
    
    # 3. Write Datasets
    vuln_out = OUT_DIR / "dappscan_contracts_vuln.jsonl"
    with open(vuln_out, "w", encoding="utf-8") as f:
        for r in vuln_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved: {vuln_out} ({len(vuln_records)} contracts)")
    
    safe_out = OUT_DIR / "dappscan_contracts_safe.jsonl"
    with open(safe_out, "w", encoding="utf-8") as f:
        for r in safe_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved: {safe_out} ({len(safe_records)} contracts)")
    
    # 4. Combined Balanced Dataset
    combined = vuln_records + safe_records
    random.seed(42)
    random.shuffle(combined)
    
    full_out = OUT_DIR / "dappscan_contracts_full.jsonl"
    with open(full_out, "w", encoding="utf-8") as f:
        for r in combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved: {full_out} ({len(combined)} contracts: {len(vuln_records)} Vulnerable + {len(safe_records)} Safe)")

if __name__ == "__main__":
    build_datasets()
