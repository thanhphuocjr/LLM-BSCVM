#!/usr/bin/env python3
"""Trích Safe negatives từ DAppSCAN: hàm KHÔNG bị gắn cờ trong chính các file đã có vuln.
Negative khớp (cùng project/file/style) -> không tạo confound nguồn/độ dài/era.
Dedup cấu trúc để boilerplate OZ (transfer/approve...) không áp đảo."""
import json, re, os, collections, sys
sys.path.insert(0, os.path.dirname(__file__))
from dappscan_extract import find_functions, strip_swc_markers, extract_contract_context, ROOT, OUT

vuln = [json.loads(l) for l in open(f"{OUT}/dappscan_vuln_functions.jsonl")]
flagged = collections.defaultdict(list)
for r in vuln:
    flagged[r["file"]].append((r["decl_start"], r["decl_end"]))
print(f"File có vuln: {len(flagged)}")

def overlaps(span, spans):
    s, e = span
    return any(not (e < a or s > b) for a, b in spans)
def sig(code):
    return re.sub(r"\s+", "", re.sub(r"\b[A-Za-z_]\w*\b", "X", code))

CAP_PER_FILE = 8
safe = []; seen = set(); per_file = collections.Counter()
skip = collections.Counter()
for frel, spans in flagged.items():
    solpath = os.path.join(ROOT, frel)
    if not os.path.isfile(solpath):
        skip["file_missing"] += 1; continue
    src = open(solpath, encoding="utf-8", errors="replace").read()
    for f in find_functions(src):
        if per_file[frel] >= CAP_PER_FILE: break
        span = (f["start_line"], f["end_line"])
        if overlaps(span, spans): skip["is_flagged"] += 1; continue
        code_raw = src[f["start_idx"]:f["end_idx"]]
        if re.search(r"SWC-\d+", code_raw): skip["has_marker->flagged"] += 1; continue
        code, _ = strip_swc_markers(code_raw)
        if len(code) < 80: skip["too_short"] += 1; continue
        s = sig(code)
        if s in seen: skip["dup_structural"] += 1; continue
        seen.add(s); per_file[frel] += 1
        context = extract_contract_context(src, f)
        safe.append({"source": "dappscan",
                     "project": frel.split("/")[2] if frel.count("/") >= 2 else "",
                     "file": frel, "function": f["name"] or f["kind"],
                     "swc_id": None, "swc_category": None, "label": "Safe", "code": code,
                     **context})

print("skip:", dict(skip))
print(f"Safe candidates (đã dedup): {len(safe)} | từ {len(set(r['file'] for r in safe))} file / {len(set(r['project'] for r in safe))} project")
lens = [len(r["code"]) for r in safe]
def percentile(values, q):
    values = sorted(values)
    if not values:
        return 0
    idx = round((len(values) - 1) * q / 100)
    return values[idx]
print(f"n_chars safe: p50={int(percentile(lens,50))} p90={int(percentile(lens,90))} max={max(lens)}")
with open(f"{OUT}/dappscan_safe_pool.jsonl", "w", encoding="utf-8") as fo:
    for r in safe: fo.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"✅ Ghi {len(safe)} -> {OUT}/dappscan_safe_pool.jsonl")
