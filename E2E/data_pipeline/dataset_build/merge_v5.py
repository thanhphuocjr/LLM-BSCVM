#!/usr/bin/env python3
"""Merge DAppSCAN (v5) + Solodit (v5) -> detect_v5_functionlevel.jsonl & .csv.

Features:
1. Structural cross-source deduplication (keeping Vulnerable on overlap).
2. Per-source length-matched 1:1 Safe/Vulnerable balancing (eliminating length/source shortcuts).
3. Leakage-proof Train/Val/Test splitting using near-duplicate graph clustering (TF-IDF cosine >= 0.85).
4. Automated verification & Shortcut-AUC evaluation.
5. Outputs JSONL and CSV versions.
"""
import json, re, os, random, collections, csv
from pathlib import Path
import numpy as np

random.seed(42)
np.random.seed(42)

E2E_ROOT = Path(__file__).resolve().parents[2]
OUT = E2E_ROOT / "data" / "training"

def load_jsonl(p):
    return [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]

def structural_sig(c):
    return re.sub(r"\s+", "", re.sub(r"\b[A-Za-z_]\w*\b", "X", c))

def length_balance(rows, nbins=8):
    if len({r["label"] for r in rows}) < 2:
        return []
    lens = np.array([r["n_chars"] for r in rows])
    edges = np.unique(np.percentile(lens, np.linspace(0, 100, nbins + 1)))
    b = np.digitize(lens, edges[1:-1] if len(edges) > 2 else edges)
    cell = collections.defaultdict(lambda: {"Safe": [], "Vulnerable": []})
    for r, bi in zip(rows, b):
        cell[bi][r["label"]].append(r)
    out = []
    for bi, c in cell.items():
        k = min(len(c["Safe"]), len(c["Vulnerable"]))
        random.shuffle(c["Safe"])
        random.shuffle(c["Vulnerable"])
        out += c["Safe"][:k] + c["Vulnerable"][:k]
    return out

def main():
    print("=== Merging Dataset v5 (Function-Level) ===")
    dv = load_jsonl(OUT / "dappscan_vuln_v5.jsonl")
    ds = load_jsonl(OUT / "dappscan_safe_v5.jsonl")
    ia = load_jsonl(OUT / "iaudit_solodit_v5.jsonl")
    
    for r in dv + ds:
        r["source"] = "dappscan"
    for r in ia:
        r["source"] = "solodit"
        
    pool = dv + ds + ia
    for r in pool:
        r["n_chars"] = len(r["code"])
        r.setdefault("function", None)
        r.setdefault("project", None)
        r.setdefault("file", None)
        r.setdefault("swc_id", None)
        r.setdefault("swc_category", None)
        
    print(f"Loaded raw pools: DAppSCAN Vuln={len(dv)}, DAppSCAN Safe={len(ds)}, Solodit={len(ia)} (Total={len(pool)})")
    
    # 1. Cross-source structural deduplication (prioritize Vulnerable)
    pool.sort(key=lambda r: 0 if r["label"] == "Vulnerable" else 1)
    seen = set()
    dedup = []
    for r in pool:
        s = structural_sig(r["code"])
        if s in seen:
            continue
        seen.add(s)
        r["_sig"] = s
        dedup.append(r)
        
    print(f"After cross-source structural dedup: {len(dedup)} (removed {len(pool) - len(dedup)})")
    
    # 2. Length-match & balance per source (1:1)
    balanced = []
    for src in ("dappscan", "solodit"):
        rs = [r for r in dedup if r["source"] == src]
        bal = length_balance(rs, nbins=8)
        v = sum(x["label"] == "Vulnerable" for x in bal)
        print(f"  {src:10s}: {len(rs)} -> Balanced {len(bal)} (Vuln {v} / Safe {len(bal) - v})")
        balanced += bal
        
    random.shuffle(balanced)
    n = len(balanced)
    print(f"\nTotal balanced dataset size: {n}")
    
    # 3. Leakage-proof Train/Val/Test Split (TF-IDF cosine >= 0.85 + Structural Clustering)
    from sklearn.feature_extraction.text import TfidfVectorizer
    codes = [r["code"] for r in balanced]
    
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Structural clusters
    sg = collections.defaultdict(list)
    for i, r in enumerate(balanced):
        sg[r["_sig"]].append(i)
    for g in sg.values():
        for j in g[1:]:
            union(g[0], j)
            
    # TF-IDF Near-dup clusters
    vec = TfidfVectorizer(analyzer="word", ngram_range=(4, 6), min_df=2, max_features=120000, dtype=np.float32)
    X = vec.fit_transform(codes)
    npair = 0
    for s in range(0, n, 512):
        sim = X[s:s + 512] @ X.T
        sim.data[sim.data < 0.85] = 0
        sim.eliminate_zeros()
        sim = sim.tocoo()
        for li, j in zip(sim.row, sim.col):
            gi = s + int(li)
            if gi < int(j):
                union(gi, int(j))
                npair += 1

    comp = collections.defaultdict(list)
    for i in range(n):
        comp[find(i)].append(i)
        
    clusters = list(comp.values())
    random.Random(42).shuffle(clusters)
    
    ntest = nval = int(0.10 * n)
    test_i, val_i, train_i = [], [], []
    for cl in clusters:
        if len(test_i) + len(cl) <= ntest:
            test_i += cl
        elif len(val_i) + len(cl) <= nval:
            val_i += cl
        else:
            train_i += cl
            
    for idx, sp in ((train_i, "train"), (val_i, "val"), (test_i, "test")):
        for i in idx:
            balanced[i]["split"] = sp
            
    print(f"Clustering: {npair} near-dup pairs -> {len(clusters)} disjoint clusters.")
    print(f"Split distribution: train={len(train_i)}, val={len(val_i)}, test={len(test_i)}")
    
    # Verify residual leakage
    Xtr = X[train_i]
    def check_residual(name, idx):
        if not idx:
            return
        mx = np.zeros(len(idx))
        for s in range(0, len(idx), 512):
            sub_idx = [idx[k] for k in range(s, min(s + 512, len(idx)))]
            mx[s:s + 512] = (X[sub_idx] @ Xtr.T).max(axis=1).toarray().ravel()
        leaks = int((mx >= 0.85).sum())
        print(f"  {name}: {leaks}/{len(idx)} cosine >= 0.85 to train (median similarity: {np.median(mx):.2f})")
    print("Residual leakage check:")
    check_residual("val ", val_i)
    check_residual("test", test_i)
    
    # 4. Shortcut-AUC evaluation
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    
    y = np.array([1 if r["label"] == "Vulnerable" else 0 for r in balanced])
    def calc_auc(X_features):
        Xtr_sub, Xte_sub, ytr_sub, yte_sub = train_test_split(X_features, y, test_size=0.25, random_state=42, stratify=y)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr_sub, ytr_sub)
        return roc_auc_score(yte_sub, clf.predict_proba(Xte_sub)[:, 1])
        
    Xlen = np.array([[r["n_chars"]] for r in balanced], float)
    srcs = sorted({r["source"] for r in balanced})
    s2i = {s: i for i, s in enumerate(srcs)}
    Xsrc = np.zeros((n, len(srcs)))
    for i, r in enumerate(balanced):
        Xsrc[i, s2i[r["source"]]] = 1
    Xbow = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, max_features=40000, sublinear_tf=True).fit_transform(codes)
    
    print("\n=== SHORTCUT-AUC EVALUATION (Target: length & source ~0.50) ===")
    auc_len = calc_auc(Xlen)
    auc_src = calc_auc(Xsrc)
    auc_bow = calc_auc(Xbow)
    print(f"  Length-only AUC : {auc_len:.3f} (Ideal: 0.500)")
    print(f"  Source-only AUC : {auc_src:.3f} (Ideal: 0.500)")
    print(f"  Lexical Bag-of-Words AUC : {auc_bow:.3f} (Baseline to beat)")
    
    # 5. Output files
    for i, r in enumerate(balanced):
        r.pop("_sig", None)
        r["id"] = f"{r['source']}_v5_{i}"
        
    fields = [
        "id", "source", "label", "swc_id", "swc_category", "function", "project", "file", "n_chars", "split", "code"
    ]
    
    out_jsonl = OUT / "detect_v5_functionlevel.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in balanced:
            f.write(json.dumps({k: r.get(k) for k in fields}, ensure_ascii=False) + "\n")
    print(f"\n✅ Successfully written {len(balanced)} rows to {out_jsonl}")
    
    out_csv = OUT / "detect_v5_functionlevel.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in balanced:
            writer.writerow({k: r.get(k) for k in fields})
    print(f"✅ Successfully written CSV to {out_csv}")
    
    # Preview CSV (shortened code for easy viewing)
    out_preview = OUT / "detect_v5_preview.csv"
    with open(out_preview, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in balanced:
            row = dict(r)
            row["code"] = (row["code"][:140] + "...") if len(row["code"]) > 140 else row["code"]
            writer.writerow({k: row.get(k) for k in fields})
    print(f"✅ Successfully written Preview CSV to {out_preview}")
    
    # Final statistics
    n_vuln = sum(r["label"] == "Vulnerable" for r in balanced)
    print(f"\nFinal Summary: Total={len(balanced)}, Vulnerable={n_vuln}, Safe={len(balanced) - n_vuln}")
    print("Split counts:", dict(collections.Counter(r["split"] for r in balanced)))
    for src in srcs:
        s_count = sum(1 for r in balanced if r["source"] == src and r["label"] == "Safe")
        v_count = sum(1 for r in balanced if r["source"] == src and r["label"] == "Vulnerable")
        print(f"  Source '{src}': Safe={s_count}, Vulnerable={v_count}")

if __name__ == "__main__":
    main()
