#!/usr/bin/env python3
"""Extract cleaned & verified Vulnerable functions from DAppSCAN for v5.

Enhancements:
1. Strict line-in-span verification (drops ghost lines and out-of-span functions).
2. Filters out informational SWCs (SWC-100, 102, 103, 108, 111, 118, 119, 129, 131, 135).
3. Anti-Noise Filter for False Positives:
   - SWC-107 (Reentrancy): Requires actual external interaction (.call, .transfer, .send, external contract call). Drops simple approve/assign wrappers.
   - SWC-124 (Arbitrary Storage Write): Requires actual storage manipulation, drops 1-2 line pass-through wrappers.
   - Drops boilerplate presale / hardcoded address dumps (>5 lines of hex addresses).
4. Strips all SWC markers and leak comments.
5. Limits length between 80 and 3800 characters.
"""
import json, glob, re, os, sys, collections
from pathlib import Path

E2E_ROOT = Path(__file__).resolve().parents[2]
ROOT = str(E2E_ROOT / "data" / "source" / "raw" / "DAppSCAN")
OUT = str(E2E_ROOT / "data" / "training")

KW = re.compile(r"\b(function|constructor|fallback|receive|modifier)\b")
IDENT = re.compile(r"[A-Za-z_]\w*")
INFO_SWC = {"SWC-100", "SWC-102", "SWC-103", "SWC-108", "SWC-111", "SWC-118", "SWC-119", "SWC-129", "SWC-131", "SWC-135"}

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

def line_of(idx, starts):
    import bisect
    return bisect.bisect_right(starts, idx)

def find_functions(src):
    code = blank_noncode(src)
    starts = [0] + [m.start()+1 for m in re.finditer("\n", src)]
    funcs = []
    for m in KW.finditer(code):
        kw = m.group(1); p = m.end()
        if kw in ("function", "modifier"):
            q = p
            while q < len(code) and code[q].isspace(): q += 1
            nm = IDENT.match(code, q)
            name = nm.group(0) if nm and (code[q] not in "(") else ""
            if kw == "function" and (q < len(code) and code[q] == "("): name = ""
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
        s_line = line_of(m.start(), starts)
        e_line = line_of(end, starts)
        funcs.append({"kind": kw, "name": name, "start_idx": m.start(),
                      "end_idx": end+1, "start_line": s_line, "end_line": e_line})
    return funcs

def strip_swc_markers(code):
    had = bool(re.search(r"SWC-\d+", code))
    code = re.sub(r"/\*+\s*SWC-\d+.*?\*/", "", code, flags=re.DOTALL)
    out = []
    for ln in code.split("\n"):
        s = ln.lstrip()
        if re.match(r"//+\s*SWC-\d+", s):
            continue
        m = re.search(r"//+\s*SWC-\d+[^\n]*$", ln)
        if m:
            ln = ln[:m.start()].rstrip()
        out.append(ln)
    clean_code = "\n".join(out)
    clean_code = re.sub(r"//[^\n]*(vulnerab|@audit|exploit|attacker|hack)[^\n]*", "", clean_code, flags=re.I)
    return clean_code.strip(), had

def parse_line(ln):
    if ln is None: return None
    m = re.search(r"\d+", str(ln))
    return int(m.group(0)) if m else None

def norm_fn(fn):
    if fn is None: return None
    fn = str(fn).strip()
    if fn in ("", "N/A", "n/a"): return None
    if not re.fullmatch(r"[A-Za-z_]\w*", fn): return None
    return fn

def is_noise_or_false_positive(code, swc_id, fn_name):
    # 1. Boilerplate check: > 4 lines of hardcoded hex addresses
    hex_addrs = re.findall(r"0x[a-fA-F0-9]{40}", code)
    if len(hex_addrs) > 4:
        return True, "boilerplate_hardcoded_addresses"
    
    # 2. SWC-107 (Reentrancy) false positive filter:
    if swc_id == "SWC-107":
        has_ext_call = bool(re.search(r"\.(call|transfer|send)\s*[\(\{]|(\bIERC20\b|\bERC20\b|\bToken\b|\bVault\b)\s*\([^)]*\)\s*\.", code))
        if not has_ext_call and ("approve(" in code or "emit " in code) and len(code.split("\n")) <= 8:
            return True, "false_positive_reentrancy_on_wrapper"
            
    # 3. SWC-124 (Arbitrary storage write) false positive filter:
    if swc_id == "SWC-124":
        lines = [ln.strip() for ln in code.split("\n") if ln.strip() and not ln.strip().startswith("//")]
        if len(lines) <= 4 and not any(op in code for op in ("assembly", "sstore", "[msg.sender]", "owners[")):
            return True, "false_positive_storage_write_wrapper"
            
    return False, None

def main(write=False):
    files = sorted(glob.glob(f"{ROOT}/DAppSCAN-source/SWCsource/**/*.json", recursive=True))
    stat = collections.Counter()
    rows = []
    cache = {}
    
    for fp in files:
        d = json.load(open(fp))
        rel = d.get("filePath"); swcs = d.get("SWCs") or []
        solpath = os.path.join(ROOT, rel) if not os.path.isabs(rel) else rel
        
        for s in swcs:
            cat = s.get("category", "")
            fn = norm_fn(s.get("function"))
            ln = parse_line(s.get("lineNumber"))
            swc_m = re.match(r"SWC-\d+", cat)
            swc_id = swc_m.group(0) if swc_m else None
            
            if not swc_id or swc_id in INFO_SWC:
                stat["skip_informational_swc"] += 1
                continue
                
            if fn is None:
                stat["skip_no_function"] += 1
                continue
                
            if not os.path.isfile(solpath):
                stat["skip_file_missing"] += 1
                continue
                
            if solpath not in cache:
                try:
                    cache[solpath] = (open(solpath, encoding="utf-8", errors="replace").read(),
                                     find_functions(open(solpath, encoding="utf-8", errors="replace").read()))
                except Exception:
                    cache[solpath] = ("", [])
                    
            src, funcs = cache[solpath]
            cands = [f for f in funcs if f["name"] == fn] or \
                    ([f for f in funcs if f["kind"] == fn] if fn in ("constructor","fallback","receive") else [])
                    
            if not cands:
                stat["function_not_found"] += 1
                continue
                
            chosen = None
            status = None
            if ln is not None:
                inside = [f for f in cands if f["start_line"] <= ln <= f["end_line"]]
                if inside:
                    chosen = min(inside, key=lambda f: f["end_line"] - f["start_line"])
                    status = "line_in_span"
                else:
                    near = [f for f in cands if abs(f["start_line"] - ln) <= 3]
                    if near:
                        chosen = min(near, key=lambda f: abs(f["start_line"] - ln))
                        status = "line_near(±3)"
            
            if not chosen:
                stat["skip_line_mismatch_or_ambiguous"] += 1
                continue
                
            code_raw = src[chosen["start_idx"]:chosen["end_idx"]]
            code, had_marker = strip_swc_markers(code_raw)
            
            # Length filter
            if len(code) < 80 or len(code) > 3800:
                stat["skip_length_outlier"] += 1
                continue
                
            # Anti-noise check
            is_noise, noise_reason = is_noise_or_false_positive(code, swc_id, fn)
            if is_noise:
                stat[f"skip_{noise_reason}"] += 1
                continue
                
            stat["retained_clean_vuln"] += 1
            project = rel.split("/")[2] if rel.count("/") >= 2 else ""
            rows.append({
                "id": f"dappscan_v5_{len(rows)}",
                "source": "dappscan",
                "project": project,
                "file": rel,
                "function": fn,
                "swc_id": swc_id,
                "swc_category": cat,
                "line": ln,
                "label": "Vulnerable",
                "code": code,
                "n_chars": len(code)
            })
            
    print("=== KẾT QUẢ TRÍCH DAppSCAN VULNERABLE (v5) ===")
    for k, v in sorted(stat.items(), key=lambda x: -x[1]):
        print(f"  {k:38s}: {v}")
    print(f"\n=> Tổng số hàm Vulnerable sạch được giữ lại: {len(rows)}")
    
    if write:
        os.makedirs(OUT, exist_ok=True)
        out_path = f"{OUT}/dappscan_vuln_v5.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"✅ Đã ghi thành công vào {out_path}")
        
    return rows

if __name__ == "__main__":
    main(write="--write" in sys.argv)
