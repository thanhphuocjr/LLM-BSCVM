#!/usr/bin/env python3
"""
clean_detect_dataset_v2.py — bản v2 cho bộ PHÂN LOẠI NHỊ PHÂN.

Khác v1 (clean_detect_dataset.py) ở 3 điểm (Nhóm 3 trong review):
  (#6) GIỮ swc_ids + LỌC nhãn dappscan mờ: bỏ mẫu Vuln mà TẤT CẢ finding chỉ là
       loại informational/low-signal (SWC-100/102/103/108/111/118/129/131/135).
       -> phần Vuln còn lại là lỗ hổng THẬT, phân biệt được với Safe.
  (#7) CÂN BẰNG ĐỘ DÀI THEO TỪNG NGUỒN (không phải toàn cục): trong mỗi nguồn,
       mỗi bin độ dài -> #Safe = #Vuln. Phá confound "len ngược chiều giữa các nguồn".
  (#8) Hard-negative theo-cặp-đã-vá cho dappscan: KHÔNG làm được (data không có bản
       vá của chính contract). Đã ghi rõ; giữ dappscan_safe làm hard-negative gần nhất.

Sinh:
  detect_dataset_clean_v2.jsonl     - làm sạch, giữ swc_ids (KHÔNG cân bằng)
  detect_dataset_balanced_v2.jsonl  - lọc SWC mờ + cân bằng độ dài theo nguồn

In report: length/source/granularity shortcut-AUC + tfidf túi-từ, TRƯỚC/SAU,
và median độ dài Safe-vs-Vuln THEO TỪNG NGUỒN (kỳ vọng: hội tụ sau cân bằng).
"""
import json, re, collections, random, statistics as st
import numpy as np

SRC  = "detect_dataset.jsonl"
RAG  = "rag_knowledge.jsonl"
SEED = 42
random.seed(SEED); np.random.seed(SEED)

# ── (#6) SWC informational/low-signal: mẫu chỉ-toàn các SWC này -> loại khỏi Vuln ──
#   (không loại nếu mẫu còn ÍT NHẤT một SWC "thật" khác)
FUZZY_SWC = {
    "SWC-100",  # Function Default Visibility (informational)
    "SWC-102",  # Outdated Compiler Version (informational)
    "SWC-103",  # Floating Pragma (informational)
    "SWC-108",  # State Variable Default Visibility (informational)
    "SWC-111",  # Use of Deprecated Solidity Functions (low)
    "SWC-118",  # Incorrect Constructor Name (legacy)
    "SWC-129",  # Typographical Error (low)
    "SWC-131",  # Presence of unused variables (informational)
    "SWC-135",  # Code With No Effects (informational)
}
N_LEN_BINS = 8   # số bin độ dài khi cân bằng trong-nguồn (nhỏ hơn v1 để nguồn nhỏ không rỗng bin)

# ---------- nạp + join source & swc_ids từ rag_knowledge ----------
def norm(c): return re.sub(r"\s+", " ", c).strip()
det = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
src_by, swc_by = {}, {}
for l in open(RAG, encoding="utf-8"):
    if not l.strip(): continue
    r = json.loads(l)
    if r.get("code"):
        k = norm(r["code"])
        src_by[k] = r.get("source", "?")
        swc_by[k] = r.get("swc_ids") or []
for r in det:
    k = norm(r["code"])
    r["source"] = src_by.get(k, "unknown")
    r["swc_ids"] = swc_by.get(k, [])
    r["origin"] = "raw"
print(f"Nạp detect_dataset: {len(det)} | nhãn {dict(collections.Counter(r['label'] for r in det))}")

# ---------- nạp thêm nguồn BỔ SUNG (giữ swc_ids nếu file có) ----------
extra = []
DAPP_SAFE = "../raw_dataset/dappscan_safe_clean.json"
try:
    for r in json.load(open(DAPP_SAFE, encoding="utf-8")):
        if r.get("code"):
            extra.append({"code": r["code"], "label": r["label"],
                          "source": r.get("source", "dappscan"),
                          "swc_ids": r.get("swc_ids") or [], "origin": "dappscan_safe"})
except FileNotFoundError:
    print("  (bỏ qua dappscan_safe: không thấy file)")
try:
    for l in open("synthetic_samples.jsonl", encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            extra.append({"code": r["code"], "label": r["label"], "source": r["source"],
                          "swc_ids": r.get("swc_ids") or [], "origin": "synthetic"})
except FileNotFoundError:
    print("  (bỏ qua synthetic: chạy gen_synthetic_samples.py trước)")
det = det + extra
print(f"+ bổ sung {len(extra)} -> tổng {len(det)} | nhãn {dict(collections.Counter(r['label'] for r in det))}")

# ---------- 1) UNWRAP bundle JSON đa-file -> từng contract (giữ swc_ids) ----------
def is_json_wrapped(c): return c.lstrip().startswith('{"') and '"content"' in c[:300]
def looks_solidity(c): return re.search(r"\b(contract|library|interface)\b", c) and len(c) > 200
unwrapped, kept, n_bundle = [], [], 0
for r in det:
    if is_json_wrapped(r["code"]):
        try:
            obj = json.loads(r["code"]); n_bundle += 1
            for fn, inner in obj.items():
                content = inner.get("content", "") if isinstance(inner, dict) else str(inner)
                if fn.endswith(".sol") and looks_solidity(content):
                    unwrapped.append({"code": content, "label": r["label"], "source": r["source"],
                                      "swc_ids": r.get("swc_ids") or [], "origin": "unwrapped_bundle"})
        except Exception:
            kept.append(r)
    else:
        r.setdefault("origin", "raw"); kept.append(r)
print(f"[1] Unwrap {n_bundle} bundle JSON -> {len(unwrapped)} contract")

# ---------- 2) lọc mẫu hỏng / rác ----------
pool = kept + unwrapped
def is_bad(c):
    if len(c) < 120: return "too_short"
    if len(c) > 60000: return "too_long"
    if "�" in c: return "corrupt_char"
    if is_json_wrapped(c): return "still_wrapped"
    return None
clean, dropped = [], collections.Counter()
for r in pool:
    why = is_bad(r["code"])
    if why: dropped[why] += 1; continue
    clean.append(r)
print(f"[2] Lọc rác: bỏ {sum(dropped.values())} {dict(dropped)} -> còn {len(clean)}")

# ---------- 3) dedup exact theo chữ ký cấu trúc ----------
def structural_sig(c):
    s = re.sub(r"\b[A-Za-z_]\w*\b", "X", c)
    return re.sub(r"\s+", "", s)
seen, deduped = set(), []
for r in clean:
    k = structural_sig(r["code"])
    if k in seen: continue
    seen.add(k); deduped.append(r)
print(f"[3] Dedup exact-structural: bỏ {len(clean)-len(deduped)} -> còn {len(deduped)}")

# ---------- 4) metadata ----------
def granularity(c):
    if re.search(r"\b(contract|library|interface)\b", c): return "contract"
    if re.search(r"\b(function|constructor|modifier)\b", c): return "snippet"
    return "other"
for r in deduped:
    r["n_chars"] = len(r["code"]); r["n_lines"] = r["code"].count("\n") + 1
    r["granularity"] = granularity(r["code"]); r.setdefault("origin", "raw")

# ---------- (#6) LỌC nhãn Vuln mờ: chỉ-toàn FUZZY_SWC -> bỏ ----------
def only_fuzzy(r):
    sw = r.get("swc_ids") or []
    return len(sw) > 0 and all(s in FUZZY_SWC for s in sw)   # có swc & tất cả đều mờ
before = len(deduped)
fuzzy_drop = collections.Counter()
kept2 = []
for r in deduped:
    if r["label"] == "Vulnerable" and only_fuzzy(r):
        fuzzy_drop[r["source"]] += 1; continue
    kept2.append(r)
deduped = kept2
print(f"[#6] Bỏ Vuln chỉ-toàn-SWC-mờ: {before-len(deduped)} mẫu {dict(fuzzy_drop)} -> còn {len(deduped)}")

# ---------- REPORT shortcut-AUC ----------
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def shortcut_report(rows, tag):
    y = np.array([1 if r["label"] == "Vulnerable" else 0 for r in rows])
    if len(set(y)) < 2: print(f"  [{tag}] 1 lớp, bỏ qua"); return
    Xlen = np.array([[r["n_chars"], r["n_lines"]] for r in rows], float)
    srcs = sorted({r["source"] for r in rows}); s2i = {s: i for i, s in enumerate(srcs)}
    Xsrc = np.zeros((len(rows), len(srcs)))
    for i, r in enumerate(rows): Xsrc[i, s2i[r["source"]]] = 1
    gs = sorted({r["granularity"] for r in rows}); g2i = {g: i for i, g in enumerate(gs)}
    Xgr = np.zeros((len(rows), len(gs)))
    for i, r in enumerate(rows): Xgr[i, g2i[r["granularity"]]] = 1
    def auc_of(X):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
        return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3,
                          max_features=30000, sublinear_tf=True)
    Xb = vec.fit_transform([r["code"] for r in rows])
    Xtr, Xte, ytr, yte = train_test_split(Xb, y, test_size=0.2, random_state=SEED, stratify=y)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
    tfidf_auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
    print(f"  [{tag}] n={len(rows)} %Vuln={100*y.mean():.1f}")
    print(f"     shortcut-AUC  length={auc_of(Xlen):.3f}  source={auc_of(Xsrc):.3f}  "
          f"granularity={auc_of(Xgr):.3f}  | tfidf túi-từ={tfidf_auc:.3f}")

def per_source_len(rows, tag):
    by = collections.defaultdict(lambda: {"Safe": [], "Vulnerable": []})
    for r in rows: by[r["source"]][r["label"]].append(r["n_chars"])
    print(f"  [{tag}] median len Safe vs Vuln THEO NGUỒN (kỳ vọng hội tụ):")
    for s in sorted(by):
        sf, vu = by[s]["Safe"], by[s]["Vulnerable"]
        ms = int(st.median(sf)) if sf else -1; mv = int(st.median(vu)) if vu else -1
        print(f"     {s:18s} Safe={ms:6d}(n={len(sf)})  Vuln={mv:6d}(n={len(vu)})")

print("\n=== REPORT: TRƯỚC cân bằng (đã làm sạch + lọc SWC) ===")
shortcut_report(deduped, "clean_v2"); per_source_len(deduped, "clean_v2")

# ---------- (#7) CÂN BẰNG ĐỘ DÀI THEO TỪNG NGUỒN ----------
by_src = collections.defaultdict(list)
for r in deduped: by_src[r["source"]].append(r)
balanced = []
print(f"\n[#7] Cân bằng độ dài TRONG TỪNG NGUỒN ({N_LEN_BINS} bin/nguồn, giữ min(Safe,Vuln)/bin):")
for s in sorted(by_src):
    rows_s = by_src[s]
    lens = np.array([r["n_chars"] for r in rows_s])
    if len(set(r["label"] for r in rows_s)) < 2:
        print(f"   {s:18s}: 1 lớp -> giữ nguyên {len(rows_s)} (không cân bằng được)")
        balanced += rows_s; continue
    edges = np.unique(np.percentile(lens, np.linspace(0, 100, N_LEN_BINS + 1)))
    bins = np.digitize(lens, edges[1:-1])
    by_bin = collections.defaultdict(lambda: {"Safe": [], "Vulnerable": []})
    for r, b in zip(rows_s, bins): by_bin[b][r["label"]].append(r)
    kept_s = 0
    for b in sorted(by_bin):
        sf, vu = by_bin[b]["Safe"], by_bin[b]["Vulnerable"]
        k = min(len(sf), len(vu))
        random.shuffle(sf); random.shuffle(vu)
        balanced += sf[:k] + vu[:k]; kept_s += 2 * k
    print(f"   {s:18s}: {len(rows_s):5d} -> giữ {kept_s}")
random.shuffle(balanced)

print("\n=== REPORT: SAU cân bằng (theo nguồn) ===")
shortcut_report(balanced, "balanced_v2"); per_source_len(balanced, "balanced_v2")
print("  nhãn tổng:", dict(collections.Counter(r["label"] for r in balanced)))
print("  nguồn x nhãn:", {s: dict(collections.Counter(r["label"] for r in by_src_after))
                          for s, by_src_after in
                          ((s, [r for r in balanced if r["source"] == s]) for s in sorted(by_src))})

# ---------- ghi file (GIỮ swc_ids) ----------
def dump(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"code": r["code"], "label": r["label"], "source": r["source"],
                                "swc_ids": r.get("swc_ids") or [], "granularity": r["granularity"],
                                "n_chars": r["n_chars"], "origin": r.get("origin", "raw")},
                               ensure_ascii=False) + "\n")
dump(deduped, "detect_dataset_clean_v2.jsonl")
dump(balanced, "detect_dataset_balanced_v2.jsonl")
print(f"\n✅ Ghi detect_dataset_clean_v2.jsonl ({len(deduped)}) + detect_dataset_balanced_v2.jsonl ({len(balanced)})")
print("   (#8 hard-negative đã-vá cho dappscan: KHÔNG khả thi — data thiếu bản vá của chính contract.)")
