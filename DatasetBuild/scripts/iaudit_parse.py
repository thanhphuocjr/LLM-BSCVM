#!/usr/bin/env python3
"""Parse iAudit/TrustLLM (Solodit) -> (code, label) function-level.
Code nằm trong block ```Solidiy ...```; label trong completion; dedup theo id (nhiều biến thể prompt/1 mẫu)."""
import json, re, os, ast, collections
BASE = "/Users/phuocthanh/Documents/RAG/AllDataCrawl/iAudit-TrustLLM/dataset"
OUT  = "/Users/phuocthanh/Documents/RAG/DatasetBuild/output"
CODE_RE = re.compile(r"```[A-Za-z]*\s*(.*?)```", re.DOTALL)

def extract_code(text):
    m = CODE_RE.search(text or "")
    return m.group(1).strip() if m else None
def label_of(c):
    c = (c or "").lower()
    return "Vulnerable" if "vulnerable" in c else ("Safe" if "safe" in c else None)

rows = {}  # id -> record (giữ 1 bản/id)
def load_class(path, split):
    seen = 0
    for line in open(path, encoding="utf-8"):
        if not line.strip(): continue
        r = json.loads(line); code = extract_code(r["prompt"]); lab = label_of(r.get("completion"))
        if not code or not lab: continue
        seen += 1
        i = r["id"]
        rows.setdefault(i, {"id": i, "code": code, "label": lab, "split": split,
                            "source": "solodit_iaudit", "swc_id": None, "swc_category": None})
    return seen

n_tr = load_class(f"{BASE}/train/train_data_class_public.jsonl", "train")
n_va = load_class(f"{BASE}/validation/val_data_class_public.jsonl", "val")

# test (dict; meta là repr Python-dict)
test = json.load(open(f"{BASE}/test/test_public.json"))
test_rows = []
for i, rec in test.items():
    lab = "Vulnerable" if str(rec.get("ground_truth_label", "")).lower().startswith("vul") else "Safe"
    code = None
    meta = rec.get("meta")
    if isinstance(meta, str):
        try: meta = ast.literal_eval(meta)
        except Exception: meta = None
    if isinstance(meta, dict): code = (meta.get("context") or "").strip() or None
    if not code:
        ip = rec.get("input_prompts_list")
        if isinstance(ip, list) and ip: code = extract_code(ip[0])
    if code:
        test_rows.append({"id": f"test_{i}", "code": code, "label": lab, "split": "test",
                          "source": "solodit_iaudit", "swc_id": None, "swc_category": None})

trainval = list(rows.values())
print(f"class lines đọc: train={n_tr} val={n_va}")
print(f"Unique theo id (train+val): {len(trainval)}")
print(f"  nhãn:", dict(collections.Counter(r['label'] for r in trainval)))
print(f"  split:", dict(collections.Counter(r['split'] for r in trainval)))
print(f"Test: {len(test_rows)}  nhãn:", dict(collections.Counter(r['label'] for r in test_rows)))

import numpy as np, re as _re
allr = trainval + test_rows
# dedup theo code (cùng function xuất hiện lại)
def sig(c): return _re.sub(r"\s+", "", _re.sub(r"\b[A-Za-z_]\w*\b", "X", c))
seen = set(); dedup = []
for r in allr:
    s = sig(r["code"])
    if s in seen: continue
    seen.add(s); dedup.append(r)
print(f"\nSau dedup cấu trúc: {len(dedup)} (bỏ {len(allr)-len(dedup)})")
print("  nhãn:", dict(collections.Counter(r['label'] for r in dedup)))
lens = [len(r["code"]) for r in dedup]
print(f"  n_chars: p50={int(np.percentile(lens,50))} p90={int(np.percentile(lens,90))} max={max(lens)}")
# rò rỉ?
leak = sum(1 for r in dedup if _re.search(r"//[^\n]*(vulnerab|SWC-\d+|@audit)", r["code"], _re.I))
print(f"  rò rỉ comment (vulnerab/SWC/@audit): {leak}")

with open(f"{OUT}/iaudit_solodit.jsonl", "w", encoding="utf-8") as f:
    for r in dedup: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"✅ Ghi {len(dedup)} -> {OUT}/iaudit_solodit.jsonl")
