#!/usr/bin/env python3
"""Trích hàm-level từ DAppSCAN-source với KIỂM CHỨNG lineNumber.

Ý tưởng: label cho biết (filePath, function, lineNumber, SWC category).
Ta tìm ĐÚNG hàm mang tên đó và xác nhận lineNumber nằm TRONG span của hàm.
Nếu lineNumber "ảo" (rác / N/A / ngoài span mọi hàm cùng tên) -> đánh dấu, KHÔNG lấy bừa.

Chạy:  python3 dappscan_extract.py --analyze   # chỉ báo cáo tỉ lệ + ví dụ
       python3 dappscan_extract.py --write     # ghi output jsonl
"""
import json, glob, re, os, sys, collections

ROOT = "/Users/phuocthanh/Documents/RAG/AllDataCrawl/DAppSCAN"
OUT  = "/Users/phuocthanh/Documents/RAG/DatasetBuild/output"

KW = re.compile(r"\b(function|constructor|fallback|receive|modifier)\b")
IDENT = re.compile(r"[A-Za-z_]\w*")

# ---------- 1) blank comment/string (giữ độ dài + newline) để đếm brace chuẩn ----------
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
            if c == "\n": st="CODE"          # giữ newline
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
    # starts = danh sách index bắt đầu mỗi dòng; trả về số dòng 1-index
    import bisect
    return bisect.bisect_right(starts, idx)

# ---------- 2) tìm mọi hàm/constructor/modifier + span ----------
def find_functions(src):
    code = blank_noncode(src)
    starts = [0] + [m.start()+1 for m in re.finditer("\n", src)]
    funcs = []
    for m in KW.finditer(code):
        kw = m.group(1); p = m.end()
        # tên
        if kw in ("function", "modifier"):
            mm = IDENT.match(code, p + (len(code[p:]) - len(code[p:].lstrip())))
            # bỏ khoảng trắng thủ công
            q = p
            while q < len(code) and code[q].isspace(): q += 1
            nm = IDENT.match(code, q)
            name = nm.group(0) if nm and (code[q] not in "(") else ""  # function() -> anon
            if kw == "function" and (q < len(code) and code[q] == "("): name = ""  # anonymous fallback
        else:
            name = kw  # constructor/fallback/receive
        # tìm body: quét từ m.start, theo dõi paren depth; '{' ở depth 0 = body, ';' = không body
        depth = 0; body_open = None; j = m.end()
        while j < len(code):
            c = code[j]
            if c == "(": depth += 1
            elif c == ")": depth -= 1
            elif c == "{" and depth <= 0: body_open = j; break
            elif c == ";" and depth <= 0: break   # abstract/interface, không body
            j += 1
        if body_open is None:
            continue
        # brace match
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

CONTRACT_RE = re.compile(r"\b(contract|library|interface)\s+([A-Za-z_]\w*)\b")

def find_contracts(src):
    code = blank_noncode(src)
    contracts = []
    for m in CONTRACT_RE.finditer(code):
        open_idx = code.find("{", m.end())
        semi_idx = code.find(";", m.end())
        if open_idx < 0 or (semi_idx >= 0 and semi_idx < open_idx):
            continue
        d = 0; end = None; k = open_idx
        while k < len(code):
            if code[k] == "{": d += 1
            elif code[k] == "}":
                d -= 1
                if d == 0:
                    end = k
                    break
            k += 1
        if end is None:
            continue
        header = src[m.start():open_idx].strip()
        inheritance = []
        inh = re.search(r"\bis\s+(.+)$", header, flags=re.S)
        if inh:
            inheritance = [x.strip() for x in inh.group(1).split(",") if x.strip()]
        contracts.append({
            "kind": m.group(1),
            "name": m.group(2),
            "header": re.sub(r"\s+", " ", header),
            "inheritance": inheritance,
            "start_idx": m.start(),
            "body_open": open_idx,
            "end_idx": end + 1,
        })
    return contracts

def _blank_spans(src, spans):
    out = list(src)
    for s, e in spans:
        for i in range(max(0, s), min(len(out), e)):
            if out[i] != "\n":
                out[i] = " "
    return "".join(out)

def _strip_comments_text(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//.*", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def _top_level_semicolon_statements(text):
    code = blank_noncode(text)
    out = []
    depth = 0
    start = 0
    for i, ch in enumerate(code):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
            if depth == 0:
                start = i + 1
        elif ch == ";" and depth == 0:
            stmt = _strip_comments_text(text[start:i+1])
            start = i + 1
            if stmt:
                out.append(stmt)
    return out

def extract_contract_context(src, func):
    """Trich context quanh function ma khong dua body function khac vao input.

    Context nay giup model thay state vars/modifier/inheritance, nhung van tranh
    leak marker SWC va tranh nhom ca contract day du vao sample.
    """
    contracts = [c for c in find_contracts(src) if c["body_open"] <= func["start_idx"] <= c["end_idx"]]
    contract = min(contracts, key=lambda c: c["end_idx"] - c["start_idx"]) if contracts else None
    if contract is None:
        return {
            "contract_context": None,
            "state_vars": None,
            "modifiers": None,
            "modifier_signatures": None,
            "inheritance": None,
        }

    funcs = [
        f for f in find_functions(src)
        if contract["body_open"] <= f["start_idx"] <= contract["end_idx"]
    ]
    masked = _blank_spans(src, [(f["start_idx"], f["end_idx"]) for f in funcs])
    body = masked[contract["body_open"] + 1:contract["end_idx"] - 1]
    state_vars = []
    for stmt in _top_level_semicolon_statements(body):
        if re.match(r"^(using|event|error|enum|struct|function|modifier|constructor|fallback|receive)\b", stmt):
            continue
        if len(stmt) > 500:
            continue
        state_vars.append(stmt)

    modifier_sigs = []
    code = blank_noncode(src)
    for f in funcs:
        if f["kind"] != "modifier":
            continue
        open_idx = code.find("{", f["start_idx"], f["end_idx"])
        sig = src[f["start_idx"]:(open_idx if open_idx >= 0 else f["end_idx"])].strip()
        sig = _strip_comments_text(sig)
        if sig:
            modifier_sigs.append(sig + (";" if not sig.endswith(";") else ""))

    state_vars_text = "\n".join(state_vars[:24])[:2500] or None
    modifier_text = "\n".join(modifier_sigs[:16])[:1800] or None
    inheritance_text = ", ".join(contract["inheritance"]) or None
    contract_context = contract["header"][:500]
    return {
        "contract_context": contract_context,
        "state_vars": state_vars_text,
        "modifiers": modifier_text,
        "modifier_signatures": modifier_text,
        "inheritance": inheritance_text,
    }

def strip_swc_markers(code):
    """Bỏ comment đánh dấu SWC do DAppSCAN chèn (chống rò rỉ nhãn).
    Trả (code_sạch, had_marker). had_marker=True -> đã trích đúng hàm bị gắn cờ."""
    had = bool(re.search(r"SWC-\d+", code))
    # block /* SWC-... */
    code = re.sub(r"/\*+\s*SWC-\d+.*?\*/", "", code, flags=re.DOTALL)
    out = []
    for ln in code.split("\n"):
        s = ln.lstrip()
        if re.match(r"//+\s*SWC-\d+", s):      # dòng chỉ là comment SWC -> bỏ hẳn
            continue
        m = re.search(r"//+\s*SWC-\d+[^\n]*$", ln)   # comment SWC đuôi dòng -> cắt
        if m:
            ln = ln[:m.start()].rstrip()
        out.append(ln)
    return "\n".join(out), had

def parse_line(ln):
    if ln is None: return None
    m = re.search(r"\d+", str(ln))
    return int(m.group(0)) if m else None

def norm_fn(fn):
    if fn is None: return None
    fn = str(fn).strip()
    if fn in ("", "N/A", "n/a"): return None
    if not re.fullmatch(r"[A-Za-z_]\w*", fn): return None   # tên rác (có text/space)
    return fn

# ---------- 3) chạy ----------
def main(write=False):
    files = glob.glob(f"{ROOT}/DAppSCAN-source/SWCsource/**/*.json", recursive=True)
    stat = collections.Counter(); examples = collections.defaultdict(list)
    rows = []
    cache = {}
    for fp in files:
        d = json.load(open(fp))
        rel = d.get("filePath"); swcs = d.get("SWCs") or []
        solpath = os.path.join(ROOT, rel) if not os.path.isabs(rel) else rel
        if not os.path.isfile(solpath):
            # thử path tương đối từ ROOT
            solpath = os.path.join(ROOT, rel)
        for s in swcs:
            cat = s.get("category", ""); fn = norm_fn(s.get("function")); ln = parse_line(s.get("lineNumber"))
            swc_id = (re.match(r"SWC-\d+", cat) or [None])
            swc_id = swc_id.group(0) if hasattr(swc_id, "group") else None
            if fn is None:
                stat["skip_no_function(N/A/rác)"] += 1
                if len(examples["skip_no_function"]) < 4: examples["skip_no_function"].append((rel, s.get("function"), s.get("lineNumber"), cat))
                continue
            if not os.path.isfile(solpath):
                stat["skip_file_missing"] += 1; continue
            if solpath not in cache:
                try: cache[solpath] = find_functions(open(solpath, encoding="utf-8", errors="replace").read())
                except Exception: cache[solpath] = []
            funcs = cache[solpath]
            cands = [f for f in funcs if f["name"] == fn] or \
                    ([f for f in funcs if f["kind"] == fn] if fn in ("constructor","fallback","receive") else [])
            if not cands:
                stat["function_not_found"] += 1
                if len(examples["not_found"]) < 5: examples["not_found"].append((os.path.basename(rel), fn, ln, cat))
                continue
            # chọn theo lineNumber
            status = None; chosen = None
            if ln is not None:
                inside = [f for f in cands if f["start_line"] <= ln <= f["end_line"]]
                if inside:
                    chosen = min(inside, key=lambda f: f["end_line"]-f["start_line"]); status = "line_in_span"
                else:
                    near = [f for f in cands if abs(f["start_line"]-ln) <= 3]
                    if near:
                        chosen = min(near, key=lambda f: abs(f["start_line"]-ln)); status = "line_near(±3)"
                    elif len(cands) == 1:
                        chosen = cands[0]; status = "name_only_line_mismatch"
                    else:
                        chosen = min(cands, key=lambda f: abs(f["start_line"]-ln)); status = "name_multi_line_mismatch"
            else:
                chosen = cands[0] if len(cands)==1 else None
                status = "name_only_noline" if chosen else "name_multi_noline_ambiguous"
                if chosen is None:
                    stat[status]+=1
                    if len(examples[status])<4: examples[status].append((os.path.basename(rel),fn,ln,cat))
                    continue
            stat[status] += 1
            if status in ("name_only_line_mismatch","name_multi_line_mismatch") and len(examples[status])<5:
                examples[status].append((os.path.basename(rel), fn, ln,
                                          f"decl@{chosen['start_line']}-{chosen['end_line']}", cat))
            src = open(solpath, encoding="utf-8", errors="replace").read()
            code_raw = src[chosen["start_idx"]:chosen["end_idx"]]
            code, had_marker = strip_swc_markers(code_raw)
            context = extract_contract_context(src, chosen)
            stat["_marker_confirmed" if had_marker else "_marker_absent"] += 1
            rows.append({"source":"dappscan","project":rel.split("/")[2] if rel.count("/")>=2 else "",
                         "file":rel,"function":fn,"swc_id":swc_id,"swc_category":cat,
                         "line":ln,"decl_start":chosen["start_line"],"decl_end":chosen["end_line"],
                         "match_status":status,"swc_marker_confirmed":had_marker,
                         "label":"Vulnerable","code":code, **context})
    # báo cáo
    print("=== KẾT QUẢ TRÍCH DAppSCAN (vuln) ===")
    for k,v in sorted(stat.items(), key=lambda x:-x[1]): print(f"  {k:34s} {v}")
    print(f"\n  => Rows trích được (mọi status match): {len(rows)}")
    hi = [r for r in rows if r["match_status"] in ("line_in_span","line_near(±3)")]
    print(f"  => Độ tin cậy CAO (line khớp span/near): {len(hi)}")
    print("\n--- Ví dụ SKIP no_function (N/A/rác) ---")
    for e in examples["skip_no_function"]: print("   ", e)
    print("--- Ví dụ function_not_found ---")
    for e in examples["not_found"]: print("   ", e)
    print("--- Ví dụ name_only_line_mismatch (line ảo, tên duy nhất) ---")
    for e in examples["name_only_line_mismatch"]: print("   ", e)
    print("--- Ví dụ name_multi_line_mismatch (nhiều hàm trùng tên, line ngoài span) ---")
    for e in examples["name_multi_line_mismatch"]: print("   ", e)
    if write:
        os.makedirs(OUT, exist_ok=True)
        with open(f"{OUT}/dappscan_vuln_functions.jsonl","w",encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")
        print(f"\n✅ Ghi {len(rows)} -> {OUT}/dappscan_vuln_functions.jsonl")
    # trả để kiểm token-length
    return rows

if __name__ == "__main__":
    main(write="--write" in sys.argv)
