#!/usr/bin/env python3
"""Merge the curated datasets in data/clean_dataset into two artifacts.

Inputs (data/clean_dataset):
  smartbug_wild_clean.jsonl     full Solidity contracts, comment-cleaned (5139, balanced)
  dappscan_dataset_clean.json   585 curated Vulnerable contracts + quality/swc metadata
  solodit.json                  2581 audit findings (code=mixed text, completion, ...)

Outputs (data/clean_dataset):
  detect_dataset.jsonl   {"code","label"} only       -> CodeT5/CodeBERT fine-tuning
  rag_knowledge.jsonl    unified superset schema      -> RAG knowledge base (paper stages)
  merge_report.json      stats + suggested class weights

Pipeline: normalize fields/labels -> extract Solidity from solodit fences ->
strip comments uniformly (string-aware) -> dedup globally on normalized code
(keep richest) -> drop label-conflicting codes. File 1 and File 2 share the
same unique-code rows (File 1 = {code,label} projection of File 2).
"""
import json, re, collections
from pathlib import Path

from clean_smartbug import strip_comments, tidy  # reuse the validated stripper

DATA = Path(__file__).parent / "data" / "clean_dataset"
SB   = DATA / "smartbug_wild_clean.jsonl"
DAPP = DATA / "dappscan_dataset_clean.json"
SOLO = DATA / "solodit.json"
OUT1 = DATA / "detect_dataset.jsonl"
OUT2 = DATA / "rag_knowledge.jsonl"
REPORT = DATA / "merge_report.json"

SUPERSET = ["id", "source", "code", "label", "swc_ids", "categories",
            "title", "description", "remediation", "analysis",
            "vuln_locations", "address", "tools_agreeing", "quality", "provenance"]

LABEL_MAP = {"safe": "Safe", "vulnerable": "Vulnerable"}
FENCE = re.compile(r"```[a-zA-Z]*\n?(.*?)```", re.S)
_CODE_KW = re.compile(r"\b(function|contract|modifier|require|return|mapping|constructor|library|interface)\b")
_SWC_TITLE = re.compile(r"SWC-\d+-(.*)")


def norm_label(x):
    return LABEL_MAP.get(str(x).strip().lower())


def clean_code(code: str) -> str:
    return tidy(strip_comments(code or ""))


def norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def extract_solidity(text: str) -> str:
    blocks = FENCE.findall(text or "")
    good = [b.strip() for b in blocks if _CODE_KW.search(b) and ("{" in b or ";" in b)]
    return "\n\n".join(good)


def dapp_title(swc_details):
    names = []
    for d in swc_details or []:
        cat = (d or {}).get("category", "")
        m = _SWC_TITLE.match(cat)
        name = m.group(1).strip() if m else cat
        if name and name not in names:
            names.append(name)
    return "; ".join(names) or None


def base_record():
    r = {k: None for k in SUPERSET}
    r["categories"] = []
    r["swc_ids"] = []
    return r


def load_all():
    recs = []

    # smartbug — clean full contracts
    with open(SB, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            o = json.loads(line)
            r = base_record()
            r.update(id=f"smartbug-{i}", source="smartbug_wild",
                     code=o.get("code", ""), label=norm_label(o.get("label")),
                     categories=o.get("categories", []) or [],
                     address=o.get("address"), tools_agreeing=o.get("tools_agreeing"))
            recs.append(r)

    # dappscan — curated vulnerable contracts with rich metadata
    for o in json.load(open(DAPP, encoding="utf-8")):
        md = o.get("metadata") or {}
        r = base_record()
        r.update(id=o.get("id") or f"dappscan-{len(recs)}", source="dappscan",
                 code=o.get("code", ""), label=norm_label(o.get("label")),
                 swc_ids=o.get("swc_ids", []) or [],
                 categories=o.get("categories", []) or [],
                 title=dapp_title(md.get("swc_details")),
                 vuln_locations=md.get("swc_details"),
                 quality=o.get("quality"),
                 provenance={k: md.get(k) for k in ("dapp", "file_path", "duplicate_paths")})
        recs.append(r)

    # solodit — extract Solidity from fences; completion = analysis
    for o in json.load(open(SOLO, encoding="utf-8")):
        r = base_record()
        swc = o.get("swc")
        r.update(id=f"solodit-{o.get('id')}", source="solodit",
                 code=extract_solidity(o.get("code", "")),
                 label=norm_label(o.get("label")),
                 swc_ids=[swc] if swc else [],
                 title=o.get("title"), description=o.get("description"),
                 remediation=o.get("remediation"), analysis=o.get("completion"))
        recs.append(r)

    return recs


def richness(r):
    score = sum(1 for k in ("title", "description", "remediation", "analysis",
                            "vuln_locations", "quality") if r.get(k))
    score += 1 if r.get("categories") else 0
    score += 1 if r.get("swc_ids") else 0
    return score


def main():
    raw = load_all()
    stats = {"loaded_by_source": dict(collections.Counter(r["source"] for r in raw))}

    usable, no_code, bad_label = [], 0, 0
    for r in raw:
        r["code"] = clean_code(r["code"])
        if not r["label"]:
            bad_label += 1; continue
        if len(re.sub(r"\s+", "", r["code"])) < 20:
            no_code += 1; continue
        usable.append(r)

    groups = collections.defaultdict(list)
    for r in usable:
        groups[norm_key(r["code"])].append(r)

    kept, dup_removed, conflict_codes, conflict_examples = [], 0, 0, []
    for g in groups.values():
        labels = {r["label"] for r in g}
        if len(labels) > 1:
            conflict_codes += 1
            if len(conflict_examples) < 5:
                conflict_examples.append({"labels": sorted(labels),
                                          "sources": sorted({r["source"] for r in g}),
                                          "code": g[0]["code"][:120]})
            continue
        dup_removed += len(g) - 1
        kept.append(max(g, key=richness))

    with open(OUT1, "w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps({"code": r["code"], "label": r["label"]}, ensure_ascii=False) + "\n")

    with open(OUT2, "w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps({k: r[k] for k in SUPERSET}, ensure_ascii=False) + "\n")

    labels = collections.Counter(r["label"] for r in kept)
    n = sum(labels.values())
    weights = {lab: round(n / (2 * c), 4) for lab, c in labels.items()}
    label_by_src = collections.Counter((r["source"], r["label"]) for r in kept)

    report = {
        **stats,
        "dropped_bad_label": bad_label,
        "dropped_no_code": no_code,
        "duplicate_rows_removed": dup_removed,
        "label_conflict_codes_dropped": conflict_codes,
        "conflict_examples": conflict_examples,
        "final_rows": len(kept),
        "final_label_distribution": dict(labels),
        "final_rows_by_source": dict(collections.Counter(r["source"] for r in kept)),
        "final_label_by_source": {f"{s}/{l}": c for (s, l), c in sorted(label_by_src.items())},
        "suggested_class_weights": weights,
        "outputs": {"detect": str(OUT1), "rag": str(OUT2)},
    }
    json.dump(report, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
