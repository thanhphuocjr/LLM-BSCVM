#!/usr/bin/env python3
"""Clean & parse iAudit/TrustLLM (Solodit) dataset for v5.

Enhancements:
1. Rejects interface declarations without body (ending with ';' without '{...}').
2. Uses robust brace-matching parser to detect and cleanly split multi-function snippets.
3. Strips leakage comments (@audit, // vulnerable, // SWC-*, etc.).
4. Filters out extreme outliers (<60 chars, >3800 chars).
5. Performs structural deduplication.
"""
import json, re, os, ast, collections
from pathlib import Path

E2E_ROOT = Path(__file__).resolve().parents[2]
BASE = E2E_ROOT / "data" / "source" / "raw" / "iAudit-TrustLLM" / "dataset"
OUT = E2E_ROOT / "data" / "training"

CODE_RE = re.compile(r"```(?:Solidity|solidity)?\s*(.*?)```", re.DOTALL)
KW = re.compile(r"\b(function|constructor|fallback|receive|modifier)\b")
IDENT = re.compile(r"[A-Za-z_]\w*")

def extract_code(text):
    if not text:
        return None
    m = CODE_RE.search(text)
    return m.group(1).strip() if m else None

def label_of(c):
    c = (c or "").lower()
    return "Vulnerable" if "vulnerable" in c else ("Safe" if "safe" in c else None)

def blank_noncode(s):
    out = list(s); i, n, st = 0, len(s), "CODE"
    while i < n:
        c = s[i]; nxt = s[i+1] if i+1 < n else ""
        if st == "CODE":
            if c == "/" and nxt == "/": out[i]=out[i+1]=" "; i+=2; st="LINE"; continue
            if c == "/" and nxt == "*": out[i]=out[i+1]=" "; i+=2; st="BLOCK"; continue
            if c == '"': out[i]=" "; i+=1; st="DQ"; continue
            if c == "'": out[i]=" "; i+=1; st="SQ"; continue
            i+=1; continue
        if st == "LINE":
            if c == "\n": st="CODE"
            else: out[i]=" "
            i+=1; continue
        if st == "BLOCK":
            if c == "*" and nxt == "/": out[i]=out[i+1]=" "; i+=2; st="CODE"; continue
            if c != "\n": out[i]=" "
            i+=1; continue
        if st == "DQ":
            if c == "\\": out[i]=" "; out[i+1]=" " if i+1<n else out[i]; i+=2; continue
            if c == '"': out[i]=" "; i+=1; st="CODE"; continue
            if c != "\n": out[i]=" "
            i+=1; continue
        if st == "SQ":
            if c == "\\": out[i]=" "; out[i+1]=" " if i+1<n else out[i]; i+=2; continue
            if c == "'": out[i]=" "; i+=1; st="CODE"; continue
            if c != "\n": out[i]=" "
            i+=1; continue
    return "".join(out)

def find_functions(src):
    code = blank_noncode(src)
    funcs = []
    for m in KW.finditer(code):
        kw = m.group(1); p = m.end()
        if kw in ("function", "modifier"):
            q = p
            while q < len(code) and code[q].isspace(): q += 1
            nm = IDENT.match(code, q)
            name = nm.group(0) if nm and (code[q] not in "(") else ""
        else:
            name = kw
        depth = 0; body_open = None; j = m.end()
        while j < len(code):
            c = code[j]
            if c == "(": depth += 1
            elif c == ")": depth -= 1
            elif c == "{" and depth <= 0: body_open = j; break
            elif c == ";" and depth <= 0: break
            j += 1
        if body_open is None:
            continue
        d = 0; k = body_open; end = None
        while k < len(code):
            if code[k] == "{": d += 1
            elif code[k] == "}":
                d -= 1
                if d == 0: end = k; break
            k += 1
        if end is None:
            continue
        funcs.append({
            "kind": kw,
            "name": name,
            "start_idx": m.start(),
            "end_idx": end + 1,
            "code": src[m.start():end+1].strip()
        })
    return funcs

def clean_leak_comments(code):
    lines = code.split("\n")
    cleaned = []
    for line in lines:
        if re.search(r"//[^\n]*(vulnerab|SWC-\d+|@audit|exploit|attacker|hack)", line, re.I):
            line = re.sub(r"//[^\n]*(vulnerab|SWC-\d+|@audit|exploit|attacker|hack)[^\n]*", "", line, flags=re.I)
        cleaned.append(line)
    return "\n".join(cleaned).strip()

def is_valid_body(code):
    if not code or len(code.strip()) < 60:
        return False
    # Check if ends with semicolon without a body
    stripped = code.strip()
    if stripped.endswith(";") and "{" not in stripped:
        return False
    if "{" not in stripped or "}" not in stripped:
        return False
    return True

def process_raw_sample(raw_id, raw_code, label, split):
    raw_code = clean_leak_comments(raw_code)
    if not is_valid_body(raw_code):
        return []
    
    funcs = find_functions(raw_code)
    if not funcs:
        # If brace-matching didn't parse full keyword, but has valid body and function keyword
        if is_valid_body(raw_code) and any(k in raw_code for k in ("function", "constructor", "modifier", "fallback", "receive")):
            if len(raw_code) <= 3800:
                return [{
                    "id": f"solodit_{raw_id}",
                    "source": "solodit",
                    "label": label,
                    "swc_id": None,
                    "swc_category": None,
                    "function": None,
                    "project": None,
                    "file": None,
                    "code": raw_code,
                    "n_chars": len(raw_code),
                    "split": split
                }]
        return []
    
    if len(funcs) == 1:
        fn_code = funcs[0]["code"]
        if is_valid_body(fn_code) and len(fn_code) <= 3800:
            return [{
                "id": f"solodit_{raw_id}",
                "source": "solodit",
                "label": label,
                "swc_id": None,
                "swc_category": None,
                "function": funcs[0]["name"] or None,
                "project": None,
                "file": None,
                "code": fn_code,
                "n_chars": len(fn_code),
                "split": split
            }]
        return []
    
    # If multiple functions:
    # CRITICAL FIX: If labeled 'Vulnerable', do NOT smear 'Vulnerable' to all sub-functions!
    # Drop ambiguous multi-function Vulnerable blocks to prevent false positives.
    if label == "Vulnerable":
        return []
        
    # If labeled 'Safe', all functions inside a verified safe block are safe
    results = []
    for idx, fn in enumerate(funcs):
        fn_code = fn["code"]
        if not is_valid_body(fn_code) or len(fn_code) > 3800:
            continue
        results.append({
            "id": f"solodit_{raw_id}_fn{idx}",
            "source": "solodit",
            "label": "Safe",
            "swc_id": None,
            "swc_category": None,
            "function": fn["name"] or None,
            "project": None,
            "file": None,
            "code": fn_code,
            "n_chars": len(fn_code),
            "split": split
        })
    return results

def main():
    print("=== Processing iAudit / Solodit dataset (v5) ===")
    raw_samples = {}
    
    def load_class_file(file_path, split):
        count = 0
        for line in open(file_path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            code = extract_code(r.get("prompt"))
            lab = label_of(r.get("completion"))
            if not code or not lab:
                continue
            sample_id = r["id"]
            raw_samples.setdefault(sample_id, {"code": code, "label": lab, "split": split})
            count += 1
        return count

    n_train = load_class_file(BASE / "train" / "train_data_class_public.jsonl", "train")
    n_val = load_class_file(BASE / "validation" / "val_data_class_public.jsonl", "val")
    print(f"Loaded class lines: train={n_train}, val={n_val}. Unique IDs: {len(raw_samples)}")
    
    test_json_path = BASE / "test" / "test_public.json"
    test_data = json.load(open(test_json_path, encoding="utf-8"))
    for i, rec in test_data.items():
        lab = "Vulnerable" if str(rec.get("ground_truth_label", "")).lower().startswith("vul") else "Safe"
        code = None
        meta = rec.get("meta")
        if isinstance(meta, str):
            try:
                meta = ast.literal_eval(meta)
            except Exception:
                meta = None
        if isinstance(meta, dict):
            code = (meta.get("context") or "").strip() or None
        if not code:
            ip = rec.get("input_prompts_list")
            if isinstance(ip, list) and ip:
                code = extract_code(ip[0])
        if code:
            raw_samples.setdefault(f"test_{i}", {"code": code, "label": lab, "split": "test"})
    print(f"Total raw samples (including test): {len(raw_samples)}")
    
    processed = []
    rejected_count = 0
    multi_split_count = 0
    
    for s_id, s_data in raw_samples.items():
        res = process_raw_sample(s_id, s_data["code"], s_data["label"], s_data["split"])
        if not res:
            rejected_count += 1
        else:
            if len(res) > 1:
                multi_split_count += 1
            processed.extend(res)
            
    print(f"Extracted {len(processed)} individual functions. Rejected {rejected_count} noisy/empty samples. Multi-splits: {multi_split_count}.")
    
    # Structural deduplication
    def structural_sig(c):
        return re.sub(r"\s+", "", re.sub(r"\b[A-Za-z_]\w*\b", "X", c))
        
    seen = set()
    dedup = []
    for r in processed:
        s = structural_sig(r["code"])
        if s in seen:
            continue
        seen.add(s)
        dedup.append(r)
        
    print(f"After structural dedup: {len(dedup)} samples.")
    print("Class distribution:", dict(collections.Counter(r["label"] for r in dedup)))
    print("Split distribution:", dict(collections.Counter(r["split"] for r in dedup)))
    
    out_file = OUT / "iaudit_solodit_v5.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for r in dedup:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✅ Successfully written to {out_file}")

if __name__ == "__main__":
    main()
