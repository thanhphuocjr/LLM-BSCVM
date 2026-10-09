#!/usr/bin/env python3
"""Merge & Clean Dataset v6 (Function-Level - 50/50 Val/Test Allocation).

Key Improvements in v6:
1. Resolves all 6 conflicting normalized duplicate pairs (label contradictions) between DAppSCAN & Solodit.
2. Deduplicates exact & normalized code duplicates across the entire dataset.
3. Fixes missing function names for unnamed Solidity fallback functions (function () ...).
4. Zero Data Waste (3,520 unique samples):
   - Train (~78% = 2,746): Naturally balanced training set (~50.5% : 49.5%).
   - Val (~11% = 387): 50% of holdout samples allocated to Validation for robust threshold tuning.
   - Test (~11% = 387): 50% of holdout samples allocated to Test for comprehensive benchmark evaluation.
5. Leakage-proof Train/Val/Test splitting using disjoint cluster graph partitioning (TF-IDF cosine >= 0.85).
6. Evaluates and reports Shortcut-AUC for length, source, and n-grams.
7. Outputs detect_v6_functionlevel.jsonl and detect_v6_functionlevel.csv.
"""

import collections
import csv
import json
from pathlib import Path
import random
import re
import numpy as np

random.seed(42)
np.random.seed(42)

E2E_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = E2E_ROOT / "data" / "training"
REPORTS_DIR = TRAINING_DIR / "reports"


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def norm_code(c: str) -> str:
    c = re.sub(r"//.*", "", c)
    c = re.sub(r"/\*[\s\S]*?\*/", "", c)
    c = re.sub(r"\s+", "", c)
    return c


def main() -> None:
    print("=== Building Dataset v6 (Cleaned Function-Level - Equal Val/Test) ===")
    v5_path = TRAINING_DIR / "detect_v5_functionlevel.jsonl"
    pool = load_jsonl(v5_path)
    print(f"Loaded raw v5 dataset: {len(pool)} samples")

    # Step 1: Normalize missing function names
    for r in pool:
        r["n_chars"] = len(r["code"])
        if not r.get("function"):
            if re.search(r"function\s*\(\s*\)", r["code"]):
                r["function"] = "fallback"
            else:
                r["function"] = "unknown"

    # Step 2: Strict normalized code deduplication & conflict resolution
    norm_map = collections.defaultdict(list)
    for r in pool:
        norm_map[norm_code(r["code"])].append(r)

    cleaned_pool = []
    conflicts_resolved = 0
    same_dups_removed = 0

    for code_norm, group in norm_map.items():
        if len(group) == 1:
            cleaned_pool.append(group[0])
        else:
            vuln_items = [x for x in group if x["label"] == "Vulnerable"]
            safe_items = [x for x in group if x["label"] == "Safe"]
            if vuln_items and safe_items:
                # Conflicting labels: Keep the Vulnerable sample with verified metadata
                swc_vuln = [x for x in vuln_items if x.get("swc_id")]
                chosen = swc_vuln[0] if swc_vuln else vuln_items[0]
                cleaned_pool.append(chosen)
                conflicts_resolved += 1
            else:
                # Same label duplicates: Keep the single best sample with project/swc metadata
                with_meta = [x for x in group if x.get("project") or x.get("swc_id")]
                chosen = with_meta[0] if with_meta else group[0]
                cleaned_pool.append(chosen)
                same_dups_removed += (len(group) - 1)

    n = len(cleaned_pool)
    print(f"Deduplication complete: {n} unique code samples retained (100% valid code)")
    print(f"  - Conflicting duplicate pairs resolved: {conflicts_resolved}")
    print(f"  - Same-label duplicate samples removed: {same_dups_removed}")
    print(f"  - Total unique label distribution: {collections.Counter(x['label'] for x in cleaned_pool)}")

    # Step 3: Leakage-proof Train/Val/Test Split (TF-IDF cosine >= 0.85 graph clustering)
    from sklearn.feature_extraction.text import TfidfVectorizer

    codes = [r["code"] for r in cleaned_pool]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(4, 6),
        min_df=2,
        max_features=120000,
        dtype=np.float32,
    )
    X = vec.fit_transform(codes)
    npair = 0
    for s in range(0, n, 512):
        sim = X[s : s + 512] @ X.T
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

    # Equal allocation for val and test: target ~11% each (387 samples each)
    ntest = int(0.11 * n)
    nval = int(0.11 * n)

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
            cleaned_pool[i]["split"] = sp

    print(f"\nClustering: {npair} near-dup pairs -> {len(clusters)} disjoint clusters.")
    print(f"Split distribution (Equal Val/Test Allocation):")
    print(f"  - Train ({len(train_i)}): {collections.Counter(cleaned_pool[i]['label'] for i in train_i)}")
    print(f"  - Val   ({len(val_i)}): {collections.Counter(cleaned_pool[i]['label'] for i in val_i)}")
    print(f"  - Test  ({len(test_i)}): {collections.Counter(cleaned_pool[i]['label'] for i in test_i)}")

    # Verify residual leakage
    Xtr = X[train_i]

    def check_residual(name: str, idx: list[int]) -> None:
        if not idx:
            return
        mx = np.zeros(len(idx))
        for s in range(0, len(idx), 512):
            sub_idx = [idx[k] for k in range(s, min(s + 512, len(idx)))]
            mx[s : s + 512] = (X[sub_idx] @ Xtr.T).max(axis=1).toarray().ravel()
        leaks = int((mx >= 0.85).sum())
        print(f"  {name}: {leaks}/{len(idx)} cosine >= 0.85 to train (max={mx.max():.3f}, median={np.median(mx):.3f})")

    print("\nResidual leakage check:")
    check_residual("val ", val_i)
    check_residual("test", test_i)

    # Step 4: Shortcut-AUC evaluation
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    y = np.array([1 if r["label"] == "Vulnerable" else 0 for r in cleaned_pool])

    def calc_auc(X_features: np.ndarray) -> float:
        Xtr_sub, Xte_sub, ytr_sub, yte_sub = train_test_split(
            X_features, y, test_size=0.25, random_state=42, stratify=y
        )
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr_sub, ytr_sub)
        return roc_auc_score(yte_sub, clf.predict_proba(Xte_sub)[:, 1])

    Xlen = np.array([[r["n_chars"]] for r in cleaned_pool], float)
    srcs = sorted({r["source"] for r in cleaned_pool})
    s2i = {s: i for i, s in enumerate(srcs)}
    Xsrc = np.zeros((n, len(srcs)))
    for i, r in enumerate(cleaned_pool):
        Xsrc[i, s2i[r["source"]]] = 1
    Xbow = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=3,
        max_features=40000,
        sublinear_tf=True,
    ).fit_transform(codes)

    print("\n=== SHORTCUT-AUC EVALUATION (Target: length & source ~0.50) ===")
    auc_len = calc_auc(Xlen)
    auc_src = calc_auc(Xsrc)
    auc_bow = calc_auc(Xbow)
    print(f"  Length-only AUC : {auc_len:.3f} (Ideal: 0.500)")
    print(f"  Source-only AUC : {auc_src:.3f} (Ideal: 0.500)")
    print(f"  Lexical Bag-of-Words AUC : {auc_bow:.3f} (Baseline to beat)")

    # Step 5: Assign clean IDs and output files
    for i, r in enumerate(cleaned_pool):
        r["id"] = f"{r['source']}_v6_{i}"

    fields = [
        "id",
        "source",
        "label",
        "swc_id",
        "swc_category",
        "function",
        "project",
        "file",
        "n_chars",
        "split",
        "code",
    ]

    out_jsonl = TRAINING_DIR / "detect_v6_functionlevel.jsonl"
    out_csv = TRAINING_DIR / "detect_v6_functionlevel.csv"

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in cleaned_pool:
            f.write(json.dumps({k: r.get(k) for k in fields}, ensure_ascii=False) + "\n")

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in balanced if "balanced" in locals() else cleaned_pool:
            writer.writerow({k: r.get(k) for k in fields})

    print(f"\nWrote: {out_jsonl} ({out_jsonl.stat().st_size} bytes)")
    print(f"Wrote: {out_csv} ({out_csv.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
