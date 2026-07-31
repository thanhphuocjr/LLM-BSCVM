#!/usr/bin/env python3
"""Hợp nhất DAppSCAN(vuln+safe) + iAudit/Solodit -> dataset function-level SẠCH.
Khử confound: length-match + cân bằng safe/vuln THEO NGUỒN; dedup chéo nguồn;
chia train/val/test chống rò rỉ near-dup; báo cáo shortcut-AUC để chứng minh."""
import json, re, os, random, collections
from pathlib import Path
import numpy as np
random.seed(42); np.random.seed(42)
E2E_ROOT = Path(__file__).resolve().parents[2]
OUT = str(E2E_ROOT / "data" / "training")

INFO = {"SWC-100","SWC-102","SWC-103","SWC-108","SWC-111","SWC-118","SWC-119","SWC-129","SWC-131","SWC-135"}
def load(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

# ---------- 1) nạp + lọc ----------
dv = load(f"{OUT}/dappscan_vuln_functions.jsonl")
dv = [r for r in dv if r["match_status"] in ("line_in_span","line_near(±3)") and r["swc_id"] not in INFO]
ds = load(f"{OUT}/dappscan_safe_pool.jsonl")
ia = load(f"{OUT}/iaudit_solodit.jsonl")
for r in dv+ds: r["source"]="dappscan"
for r in ia: r["source"]="solodit"
pool = dv+ds+ia
for r in pool:
    r["n_chars"]=len(r["code"]); r.setdefault("function",None); r.setdefault("project",None)
print(f"Nạp: dappscan_vuln={len(dv)} dappscan_safe={len(ds)} solodit={len(ia)}")

# ---------- 2) dedup cấu trúc CHÉO nguồn (ưu tiên giữ Vulnerable) ----------
def sig(c): return re.sub(r"\s+","", re.sub(r"\b[A-Za-z_]\w*\b","X", c))
pool.sort(key=lambda r: 0 if r["label"]=="Vulnerable" else 1)   # vuln trước -> giữ khi trùng
seen=set(); dedup=[]
for r in pool:
    s=sig(r["code"])
    if s in seen: continue
    seen.add(s); r["_sig"]=s; dedup.append(r)
print(f"Sau dedup chéo: {len(dedup)} (bỏ {len(pool)-len(dedup)})")

# ---------- 3) length-match + cân bằng safe/vuln THEO NGUỒN ----------
def length_balance(rows, nbins=8):
    if len({r["label"] for r in rows})<2: return []
    lens=np.array([r["n_chars"] for r in rows])
    edges=np.unique(np.percentile(lens,np.linspace(0,100,nbins+1)))
    b=np.digitize(lens, edges[1:-1] if len(edges)>2 else edges)
    cell=collections.defaultdict(lambda:{"Safe":[],"Vulnerable":[]})
    for r,bi in zip(rows,b): cell[bi][r["label"]].append(r)
    out=[]
    for bi,c in cell.items():
        k=min(len(c["Safe"]),len(c["Vulnerable"]))
        random.shuffle(c["Safe"]); random.shuffle(c["Vulnerable"])
        out+=c["Safe"][:k]+c["Vulnerable"][:k]
    return out
balanced=[]
for src in ("dappscan","solodit"):
    rs=[r for r in dedup if r["source"]==src]
    bal=length_balance(rs)
    v=sum(x["label"]=="Vulnerable" for x in bal)
    print(f"  {src}: {len(rs)} -> cân bằng {len(bal)} (Vuln {v} / Safe {len(bal)-v})")
    balanced+=bal
random.shuffle(balanced)

# ---------- 4) chia train/val/test chống rò rỉ (cụm structural + near-dup tfidf>=0.85) ----------
from sklearn.feature_extraction.text import TfidfVectorizer
codes=[r["code"] for r in balanced]; n=len(balanced)
parent=list(range(n))
def find(x):
    while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
    return x
def union(a,b):
    ra,rb=find(a),find(b)
    if ra!=rb: parent[ra]=rb
# structural
sg=collections.defaultdict(list)
for i,r in enumerate(balanced): sg[r["_sig"]].append(i)
for g in sg.values():
    for j in g[1:]: union(g[0],j)
# near-dup tfidf
vec=TfidfVectorizer(analyzer="word",ngram_range=(4,6),min_df=2,max_features=120000,dtype=np.float32)
X=vec.fit_transform(codes); npair=0
for s in range(0,n,512):
    sim=X[s:s+512]@X.T; sim.data[sim.data<0.85]=0; sim.eliminate_zeros(); sim=sim.tocoo()
    for li,j in zip(sim.row,sim.col):
        gi=s+int(li)
        if gi<int(j): union(gi,int(j)); npair+=1
comp=collections.defaultdict(list)
for i in range(n): comp[find(i)].append(i)
clusters=list(comp.values()); random.Random(42).shuffle(clusters)
ntest=nval=int(0.10*n); test_i=[]; val_i=[]; train_i=[]
for cl in clusters:
    if len(test_i)+len(cl)<=ntest: test_i+=cl
    elif len(val_i)+len(cl)<=nval: val_i+=cl
    else: train_i+=cl
for idx,sp in ((train_i,"train"),(val_i,"val"),(test_i,"test")):
    for i in idx: balanced[i]["split"]=sp
print(f"\nNear-dup: {npair} cặp | {len(clusters)} cụm | split train/val/test = {len(train_i)}/{len(val_i)}/{len(test_i)}")

# rò rỉ residual val/test -> train
Xtr=X[train_i]
def resid(name,idx):
    if not idx: return
    mx=np.zeros(len(idx))
    for s in range(0,len(idx),512):
        mx[s:s+512]=(X[[idx[k] for k in range(s,min(s+512,len(idx)))]]@Xtr.T).max(axis=1).toarray().ravel()
    print(f"  {name}: {int((mx>=0.85).sum())}/{len(idx)} còn cosine>=0.85 tới train (median {np.median(mx):.2f})")
print("Rò rỉ residual:"); resid("val ",val_i); resid("test",test_i)

# ---------- 5) BÁO CÁO shortcut-AUC (chứng minh hết confound) ----------
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
y=np.array([1 if r["label"]=="Vulnerable" else 0 for r in balanced])
def auc(Xf):
    Xtr,Xte,ytr,yte=train_test_split(Xf,y,test_size=0.25,random_state=42,stratify=y)
    return roc_auc_score(yte,LogisticRegression(max_iter=2000,class_weight="balanced").fit(Xtr,ytr).predict_proba(Xte)[:,1])
Xlen=np.array([[r["n_chars"]] for r in balanced],float)
srcs=sorted({r["source"] for r in balanced}); s2i={s:i for i,s in enumerate(srcs)}
Xsrc=np.zeros((n,len(srcs)));
for i,r in enumerate(balanced): Xsrc[i,s2i[r["source"]]]=1
Xbow=TfidfVectorizer(analyzer="word",ngram_range=(1,2),min_df=3,max_features=40000,sublinear_tf=True).fit_transform(codes)
print("\n=== SHORTCUT-AUC (kỳ vọng length/source ~0.5) ===")
print(f"  length-only : {auc(Xlen):.3f}   (cũ smartbug 0.66)")
print(f"  source-only : {auc(Xsrc):.3f}   (kỳ vọng ~0.5)")
print(f"  tfidf túi-từ: {auc(Xbow):.3f}   (=trần lexical, model phải vượt)")

# per-source median length safe vs vuln
print("\n=== Median n_chars Safe vs Vuln theo nguồn (kỳ vọng hội tụ) ===")
for src in srcs:
    sf=[r['n_chars'] for r in balanced if r['source']==src and r['label']=='Safe']
    vu=[r['n_chars'] for r in balanced if r['source']==src and r['label']=='Vulnerable']
    print(f"  {src:10s} Safe={int(np.median(sf))}(n={len(sf)})  Vuln={int(np.median(vu))}(n={len(vu)})")

print("\n=== Context coverage ===")
context_fields=["contract_context","state_vars","modifiers","modifier_signatures","inheritance"]
for field in context_fields:
    have=sum(1 for r in balanced if r.get(field))
    print(f"  {field:20s}: {have:4d}/{len(balanced)} ({100*have/len(balanced):5.1f}%)")

# ---------- 6) ghi ----------
for i,r in enumerate(balanced):
    r.pop("_sig",None); r.pop("match_status",None); r.pop("swc_marker_confirmed",None)
    r["id"]=f"{r['source']}_{i}"
fields=[
    "id","source","label","swc_id","swc_category","function","project","file","n_chars","split",
    "contract_context","state_vars","modifiers","modifier_signatures","inheritance","code",
]
with open(f"{OUT}/detect_v4_functionlevel.jsonl","w",encoding="utf-8") as f:
    for r in balanced:
        f.write(json.dumps({k:r.get(k) for k in fields},ensure_ascii=False)+"\n")
nv=sum(r["label"]=="Vulnerable" for r in balanced)
print(f"\n✅ Ghi {len(balanced)} -> detect_v4_functionlevel.jsonl | Vuln {nv} / Safe {len(balanced)-nv}")
print("   split:",dict(collections.Counter(r["split"] for r in balanced)))
print("   nguồn x nhãn:", {s:{'S':sum(1 for r in balanced if r['source']==s and r['label']=='Safe'),
                              'V':sum(1 for r in balanced if r['source']==s and r['label']=='Vulnerable')} for s in srcs})
