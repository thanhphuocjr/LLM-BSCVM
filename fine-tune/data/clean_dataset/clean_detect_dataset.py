#!/usr/bin/env python3
"""
Làm sạch + cân bằng detect_dataset.jsonl cho bộ PHÂN LOẠI NHỊ PHÂN.

Sinh 2 file:
  detect_dataset_clean.jsonl     - đã sửa hỏng + thêm metadata (KHÔNG bỏ mẫu vì confound)
  detect_dataset_balanced.jsonl  - cân bằng độ dài/nhãn để phá shortcut length & granularity

Và in report before/after: length-only AUC, source-only AUC, granularity-only AUC,
TF-IDF túi-từ AUC (trần lối tắt). Mục tiêu: kéo các shortcut-AUC về ~0.5.
"""
import json, re, collections, random, statistics as st
import numpy as np

SRC = "detect_dataset.jsonl"
RAG = "rag_knowledge.jsonl"
SEED = 42
random.seed(SEED); np.random.seed(SEED)

# ---------- nạp + join source ----------
def norm(c): return re.sub(r"\s+", " ", c).strip()
det = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
src_by = {}
for l in open(RAG, encoding="utf-8"):
    if not l.strip(): continue
    r = json.loads(l)
    if r.get("code"): src_by[norm(r["code"])] = r.get("source", "?")
for r in det:
    r["source"] = src_by.get(norm(r["code"]), "unknown"); r["origin"] = "raw"
print(f"Nạp detect_dataset: {len(det)} | nhãn {dict(collections.Counter(r['label'] for r in det))}")

# ---------- nạp thêm nguồn BỔ SUNG ----------
extra = []
# (i) dappscan Safe -> phá 'dappscan 100% Vulnerable'
DAPP_SAFE = "../raw_dataset/dappscan_safe_clean.json"
try:
    for r in json.load(open(DAPP_SAFE, encoding="utf-8")):
        if r.get("code"):
            extra.append({"code": r["code"], "label": r["label"],
                          "source": r.get("source", "dappscan"), "origin": "dappscan_safe"})
except FileNotFoundError:
    print("  (bỏ qua dappscan_safe: không thấy file)")
# (ii) synthetic hard-negative (cặp Vulnerable<->Fixed)
try:
    for l in open("synthetic_samples.jsonl", encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            extra.append({"code": r["code"], "label": r["label"],
                          "source": r["source"], "origin": "synthetic"})
except FileNotFoundError:
    print("  (bỏ qua synthetic: chạy gen_synthetic_samples.py trước)")

det = det + extra
print(f"+ bổ sung {len(extra)} (dappscan_safe + synthetic) -> tổng {len(det)} "
      f"| nhãn {dict(collections.Counter(r['label'] for r in det))}")

# ---------- 1) UNWRAP các bundle JSON đa-file -> từng contract riêng ----------
def is_json_wrapped(c): return c.lstrip().startswith('{"') and '"content"' in c[:300]
def looks_solidity(c):
    return re.search(r"\b(contract|library|interface)\b", c) and len(c) > 200

unwrapped, kept = [], []
n_bundle = 0
for r in det:
    if is_json_wrapped(r["code"]):
        try:
            obj = json.loads(r["code"]); n_bundle += 1
            for fn, inner in obj.items():
                content = inner.get("content", "") if isinstance(inner, dict) else str(inner)
                if fn.endswith(".sol") and looks_solidity(content):
                    unwrapped.append({"code": content, "label": r["label"],
                                      "source": r["source"], "origin": "unwrapped_bundle"})
        except Exception:
            kept.append(r)  # giải mã lỗi -> giữ nguyên, sẽ bị lọc sau
    else:
        r.setdefault("origin", "raw"); kept.append(r)
print(f"\n[1] Unwrap {n_bundle} bundle JSON -> {len(unwrapped)} contract (trước dedup)")

# ---------- 2) hợp nhất + LỌC mẫu hỏng / rác ----------
pool = kept + unwrapped
def is_bad(c):
    if len(c) < 120: return "too_short"          # < ~120 ký tự: không đủ ngữ cảnh
    if len(c) > 60000: return "too_long"          # bị chunking bỏ phần lớn dù sao
    if "�" in c: return "corrupt_char"
    if is_json_wrapped(c): return "still_wrapped"
    return None
clean, dropped = [], collections.Counter()
for r in pool:
    why = is_bad(r["code"])
    if why: dropped[why] += 1; continue
    clean.append(r)
print(f"[2] Lọc rác: bỏ {sum(dropped.values())} {dict(dropped)} -> còn {len(clean)}")

# ---------- 3) dedup exact theo chữ ký cấu trúc (đổi định danh -> X) ----------
def structural_sig(c):
    s = re.sub(r"\b[A-Za-z_]\w*\b", "X", c)
    return re.sub(r"\s+", "", s)
seen, deduped = set(), []
for r in clean:
    k = structural_sig(r["code"])
    if k in seen: continue
    seen.add(k); deduped.append(r)
print(f"[3] Dedup exact-structural: bỏ {len(clean)-len(deduped)} -> còn {len(deduped)}")

# ---------- 4) gắn metadata ----------
def granularity(c):
    if re.search(r"\b(contract|library|interface)\b", c): return "contract"
    if re.search(r"\b(function|constructor|modifier)\b", c): return "snippet"
    return "other"
for r in deduped:
    r["n_chars"] = len(r["code"])
    r["n_lines"] = r["code"].count("\n") + 1
    r["granularity"] = granularity(r["code"])
    r.setdefault("origin", "raw")

# ---------- REPORT: các shortcut-AUC ----------
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def shortcut_report(rows, tag):
    y = np.array([1 if r["label"] == "Vulnerable" else 0 for r in rows])
    if len(set(y)) < 2: print(f"  [{tag}] 1 lớp, bỏ qua"); return
    # length-only
    Xlen = np.array([[r["n_chars"], r["n_lines"]] for r in rows], float)
    # source-only
    srcs = sorted({r["source"] for r in rows}); s2i = {s: i for i, s in enumerate(srcs)}
    Xsrc = np.zeros((len(rows), len(srcs)))
    for i, r in enumerate(rows): Xsrc[i, s2i[r["source"]]] = 1
    # granularity-only
    gs = sorted({r["granularity"] for r in rows}); g2i = {g: i for i, g in enumerate(gs)}
    Xgr = np.zeros((len(rows), len(gs)))
    for i, r in enumerate(rows): Xgr[i, g2i[r["granularity"]]] = 1
    def auc_of(X):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
        return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
    # tfidf túi-từ (trần lối tắt)
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3,
                          max_features=30000, sublinear_tf=True)
    Xb = vec.fit_transform([r["code"] for r in rows])
    Xtr, Xte, ytr, yte = train_test_split(Xb, y, test_size=0.2, random_state=SEED, stratify=y)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
    tfidf_auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
    lv = [r["n_chars"] for r in rows if r["label"] == "Vulnerable"]
    ls = [r["n_chars"] for r in rows if r["label"] == "Safe"]
    print(f"  [{tag}] n={len(rows)} %Vuln={100*y.mean():.1f}")
    print(f"     len median Vuln={int(st.median(lv))} vs Safe={int(st.median(ls))}")
    print(f"     shortcut-AUC  length={auc_of(Xlen):.3f}  source={auc_of(Xsrc):.3f}  "
          f"granularity={auc_of(Xgr):.3f}  | tfidf túi-từ={tfidf_auc:.3f}")

print("\n=== REPORT: TRƯỚC cân bằng (đã làm sạch) ===")
shortcut_report(deduped, "clean")

# ---------- 5) CÂN BẰNG ĐỘ DÀI: mỗi bin độ dài -> 50/50 nhãn ----------
# Phá shortcut length & granularity: trong từng khoảng độ dài, số Safe = số Vuln.
lens = np.array([r["n_chars"] for r in deduped])
edges = np.unique(np.percentile(lens, np.linspace(0, 100, 11)))  # 10 bin theo decile
bins = np.digitize(lens, edges[1:-1])
by_bin = collections.defaultdict(lambda: {"Safe": [], "Vulnerable": []})
for r, b in zip(deduped, bins): by_bin[b][r["label"]].append(r)
balanced = []
print("\n[5] Cân bằng độ dài theo decile (giữ min(Safe,Vuln) mỗi bin):")
for b in sorted(by_bin):
    sf, vu = by_bin[b]["Safe"], by_bin[b]["Vulnerable"]
    k = min(len(sf), len(vu))
    random.shuffle(sf); random.shuffle(vu)
    balanced += sf[:k] + vu[:k]
    print(f"   bin{b}: Safe {len(sf)} / Vuln {len(vu)} -> giữ {k}+{k}")
random.shuffle(balanced)

print("\n=== REPORT: SAU cân bằng ===")
shortcut_report(balanced, "balanced")

# ---------- ghi file ----------
def dump(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"code": r["code"], "label": r["label"],
                                "source": r["source"], "granularity": r["granularity"],
                                "n_chars": r["n_chars"], "origin": r.get("origin", "raw")},
                               ensure_ascii=False) + "\n")
dump(deduped, "detect_dataset_clean.jsonl")
dump(balanced, "detect_dataset_balanced.jsonl")
print(f"\n✅ Ghi detect_dataset_clean.jsonl ({len(deduped)}) + detect_dataset_balanced.jsonl ({len(balanced)})")
print("   Gợi ý: train trên _balanced để đo năng lực THẬT; giữ _clean làm superset đầy đủ.")
