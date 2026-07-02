#!/usr/bin/env python3
"""Merge the 3 smart-contract datasets into two artifacts.

Inputs (in ./data):
  smartbug_wild_clean.jsonl   full Solidity contracts, already comment-cleaned
  dappscan.json               function snippets (Code/Output/Metadata)
  solodit.json                audit findings (code=mixed text, completion, ...)

Outputs (in ./data):
  detect_dataset.jsonl   {"code","label"} only  -> CodeT5/CodeBERT fine-tuning
  rag_knowledge.jsonl    unified superset schema -> RAG knowledge base
  merge_report.json      stats + suggested class weights

Pipeline: normalize fields/labels -> extract Solidity from solodit fences ->
strip comments uniformly -> dedup globally on normalized code (keep richest) ->
drop label-conflicting codes. File 1 and File 2 share the same unique-code rows;
no balancing (user chose: keep all, handle imbalance via class weights).
"""
import json, re, collections
from pathlib import Path

from clean_smartbug import strip_comments, tidy  # reuse the validated stripper

DATA = Path(__file__).parent / "data"
SB   = DATA / "smartbug_wild_clean.jsonl"
DAPP = DATA / "dappscan.json"
SOLO = DATA / "solodit.json"
OUT1 = DATA / "detect_dataset.jsonl"
OUT2 = DATA / "rag_knowledge.jsonl"
REPORT = DATA / "merge_report.json"

SUPERSET = ["id", "source", "code", "label", "swc", "categories",
            "title", "description", "remediation", "analysis",
            "address", "tools_agreeing"]

LABEL_MAP = {"safe": "Safe", "vulnerable": "Vulnerable"}
FENCE = re.compile(r"```[a-zA-Z]*\n?(.*?)```", re.S)
_CODE_KW = re.compile(r"\b(function|contract|modifier|require|return|mapping|constructor|library|interface)\b")


def norm_label(x):
    return LABEL_MAP.get(str(x).strip().lower())


def clean_code(code: str) -> str:
    return tidy(strip_comments(code or ""))


def extract_solidity(text: str) -> str:
    """Pull Solidity out of ```...``` fences in a solodit finding; drop
    non-code blocks (e.g. 'As a Caller' call lists)."""
    blocks = FENCE.findall(text or "")
    good = [b.strip() for b in blocks if _CODE_KW.search(b) and ("{" in b or ";" in b)]
    return "\n\n".join(good)


def base_record():
    return {k: None for k in SUPERSET} | {"categories": []}


def norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def load_all():
    recs = []

    # smartbug (already clean Solidity contracts)
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

    # dappscan (function snippets; Metadata only on vulnerable)
    for i, o in enumerate(json.load(open(DAPP, encoding="utf-8"))):
        md = o.get("Metadata") or {}
        r = base_record()
        r.update(id=f"dappscan-{i}", source="dappscan",
                 code=o.get("Code", ""), label=norm_label(o.get("Output")),
                 swc=md.get("SWC"), title=md.get("Title"),
                 description=md.get("Description"), remediation=md.get("Remediation"))
        recs.append(r)

    # solodit (extract Solidity from fences; completion = analysis)
    for o in json.load(open(SOLO, encoding="utf-8")):
        r = base_record()
        r.update(id=f"solodit-{o.get('id')}", source="solodit",
                 code=extract_solidity(o.get("code", "")),
                 label=norm_label(o.get("label")),
                 swc=o.get("swc"), title=o.get("title"),
                 description=o.get("description"), remediation=o.get("remediation"),
                 analysis=o.get("completion"))
        recs.append(r)

    return recs


def richness(r):
    return sum(1 for k in ("swc", "title", "description", "remediation", "analysis")
               if r.get(k)) + (1 if r.get("categories") else 0)


def main():
    raw = load_all()
    stats = {"loaded_by_source": dict(collections.Counter(r["source"] for r in raw))}

    # clean code + drop rows without usable code or label
    usable, no_code, bad_label = [], 0, 0
    for r in raw:
        r["code"] = clean_code(r["code"])
        if not r["label"]:
            bad_label += 1; continue
        if len(re.sub(r"\s+", "", r["code"])) < 20:   # empty after extraction/cleaning
            no_code += 1; continue
        usable.append(r)

    # group by normalized code
    groups = collections.defaultdict(list)
    for r in usable:
        groups[norm_key(r["code"])].append(r)

    kept, dup_removed, conflict_codes = [], 0, 0
    conflict_examples = []
    for g in groups.values():
        labels = {r["label"] for r in g}
        if len(labels) > 1:                       # same code, contradicting labels -> drop
            conflict_codes += 1
            if len(conflict_examples) < 5:
                conflict_examples.append({"labels": sorted(labels),
                                          "sources": sorted({r["source"] for r in g}),
                                          "code": g[0]["code"][:120]})
            continue
        dup_removed += len(g) - 1
        rep = max(g, key=richness)                # keep richest-metadata copy
        kept.append(rep)

    # File 1: {code, label}
    with open(OUT1, "w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps({"code": r["code"], "label": r["label"]}, ensure_ascii=False) + "\n")

    # File 2: full superset
    with open(OUT2, "w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps({k: r[k] for k in SUPERSET}, ensure_ascii=False) + "\n")

    # report + suggested class weights (user chose class-weight strategy)
    labels = collections.Counter(r["label"] for r in kept)
    n = sum(labels.values())
    weights = {lab: round(n / (2 * c), 4) for lab, c in labels.items()}
    by_src = collections.Counter(r["source"] for r in kept)
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
        "final_rows_by_source": dict(by_src),
        "final_label_by_source": {f"{s}/{l}": c for (s, l), c in sorted(label_by_src.items())},
        "suggested_class_weights": weights,
        "outputs": {"detect": str(OUT1), "rag": str(OUT2)},
    }
    json.dump(report, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
