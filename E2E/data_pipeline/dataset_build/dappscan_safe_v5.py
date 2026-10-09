#!/usr/bin/env python3
"""Extract high-quality Safe functions from DAppSCAN contracts for v5.

Enhancements:
1. Rejects trivial 1-line getters (e.g. `return _decimals`, `return _name`).
2. Rejects boilerplate address dumps (>4 hex addresses).
3. Requires substantive functions (100 <= len(code) <= 3800).
4. Structural deduplication to prevent OpenZeppelin template dominance.
5. Caps selection per file to avoid imbalance.
"""
import json, re, os, collections, sys
from pathlib import Path

E2E_ROOT = Path(__file__).resolve().parents[2]
ROOT = str(E2E_ROOT / "data" / "source" / "raw" / "DAppSCAN")
OUT = str(E2E_ROOT / "data" / "training")

sys.path.insert(0, os.path.dirname(__file__))
from dappscan_extract_v5 import find_functions, strip_swc_markers

def is_trivial_getter_or_boilerplate(code):
    # Strip comments and whitespace
    lines = [ln.strip() for ln in code.split("\n") if ln.strip() and not ln.strip().startswith("//")]
    joined = " ".join(lines)
    
    # Check trivial return
    if re.search(r"\{\s*return\s+(_?[A-Za-z0-9_]+|0x0|\d+|true|false)\s*;\s*\}", joined):
        return True, "trivial_getter"
        
    # Check trivial empty constructor / body
    if re.search(r"\{\s*\}", joined) and len(lines) <= 3:
        return True, "empty_body"
        
    # Check hardcoded addresses dump
    hex_addrs = re.findall(r"0x[a-fA-F0-9]{40}", code)
    if len(hex_addrs) > 4:
        return True, "boilerplate_addresses"
        
    return False, None

def main():
    print("=== Extracting High-Quality Safe Functions from DAppSCAN (v5) ===")
    vuln_path = f"{OUT}/dappscan_vuln_v5.jsonl"
    if not os.path.exists(vuln_path):
        raise FileNotFoundError(f"Vulnerable dataset {vuln_path} not found. Run dappscan_extract_v5.py first.")
        
    vuln = [json.loads(l) for l in open(vuln_path, encoding="utf-8")]
    flagged = collections.defaultdict(list)
    for r in vuln:
        flagged[r["file"]].append((r.get("decl_start", r.get("line", 0)), r.get("decl_end", r.get("line", 0))))
        
    print(f"Total files with verified vulnerabilities: {len(flagged)}")
    
    def overlaps(span, spans):
        s, e = span
        return any(not (e < a or s > b) for a, b in spans)
        
    def structural_sig(c):
        return re.sub(r"\s+", "", re.sub(r"\b[A-Za-z_]\w*\b", "X", c))
        
    CAP_PER_FILE = 8
    safe = []
    seen = set()
    per_file = collections.Counter()
    skip = collections.Counter()
    
    for frel, spans in flagged.items():
        solpath = os.path.join(ROOT, frel)
        if not os.path.isfile(solpath):
            skip["file_missing"] += 1
            continue
            
        try:
            src = open(solpath, encoding="utf-8", errors="replace").read()
            funcs = find_functions(src)
        except Exception:
            skip["file_read_error"] += 1
            continue
            
        for f in funcs:
            if per_file[frel] >= CAP_PER_FILE:
                break
                
            span = (f["start_line"], f["end_line"])
            if overlaps(span, spans):
                skip["is_flagged"] += 1
                continue
                
            code_raw = src[f["start_idx"]:f["end_idx"]]
            if re.search(r"SWC-\d+", code_raw):
                skip["has_marker->flagged"] += 1
                continue
                
            code, _ = strip_swc_markers(code_raw)
            if len(code) < 100:
                skip["too_short(<100)"] += 1
                continue
            if len(code) > 3800:
                skip["too_long(>3800)"] += 1
                continue
                
            is_trivial, reason = is_trivial_getter_or_boilerplate(code)
            if is_trivial:
                skip[reason] += 1
                continue
                
            s = structural_sig(code)
            if s in seen:
                skip["dup_structural"] += 1
                continue
                
            seen.add(s)
            per_file[frel] += 1
            
            project = frel.split("/")[2] if frel.count("/") >= 2 else ""
            safe.append({
                "id": f"dappscan_safe_v5_{len(safe)}",
                "source": "dappscan",
                "project": project,
                "file": frel,
                "function": f["name"] or f["kind"],
                "swc_id": None,
                "swc_category": None,
                "label": "Safe",
                "code": code,
                "n_chars": len(code)
            })
            
    print("Skip statistics:")
    for k, v in sorted(skip.items(), key=lambda x: -x[1]):
        print(f"  {k:30s}: {v}")
        
    print(f"\n=> Tổng số hàm Safe chất lượng cao trích được: {len(safe)}")
    print(f"  Từ {len(set(r['file'] for r in safe))} files across {len(set(r['project'] for r in safe))} projects.")
    
    out_path = f"{OUT}/dappscan_safe_v5.jsonl"
    with open(out_path, "w", encoding="utf-8") as fo:
        for r in safe:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✅ Ghi thành công vào {out_path}")

if __name__ == "__main__":
    main()
