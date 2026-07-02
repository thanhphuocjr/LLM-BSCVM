#!/usr/bin/env python3
"""Clean the SmartBugs-Wild dataset for fine-tuning.

Pipeline (per record):
  1. Drop non-Solidity records (e.g. Vyper) -> reason "non_solidity".
  2. Strip ALL comments from `code` (string-aware, never touches string
     literals) and tidy whitespace.
  3. Drop empty/stub contracts (no executable logic) -> reason "stub".
  4. Drop near-duplicates by normalized cleaned code, keep first -> "duplicate".

Outputs (next to the source file):
  smartbug_wild_clean.jsonl        kept records, `code` cleaned, all fields preserved
  smartbug_wild_dropped.jsonl      dropped records, each with `_drop_reason`
  smartbug_wild_clean_report.json  summary statistics

The original file is never modified. Re-runnable / idempotent.
"""
import json, re, hashlib, collections
from pathlib import Path

SRC = Path(__file__).parent / "data" / "smartbug_wild.jsonl"
OUT = SRC.with_name("smartbug_wild_clean.jsonl")
DROP = SRC.with_name("smartbug_wild_dropped.jsonl")
REPORT = SRC.with_name("smartbug_wild_clean_report.json")

FIELD_ORDER = ["address", "code", "label", "categories", "tools_agreeing"]


# ---------------------------------------------------------------- comments
def strip_comments(code: str) -> str:
    """Remove // , /* */ , /// , /** */ comments without touching string
    literals. Char-level state machine: CODE / LINE / BLOCK / DQ(") / SQ(').
    Block comments collapse to a single space (prevents token-joining)."""
    out = []
    i, n, state = 0, len(code), "CODE"
    while i < n:
        c = code[i]
        nxt = code[i + 1] if i + 1 < n else ""
        if state == "CODE":
            if c == "/" and nxt == "/":
                state = "LINE"; i += 2; continue
            if c == "/" and nxt == "*":
                state = "BLOCK"; i += 2; continue
            if c == '"':
                state = "DQ"; out.append(c); i += 1; continue
            if c == "'":
                state = "SQ"; out.append(c); i += 1; continue
            out.append(c); i += 1
        elif state == "LINE":
            if c == "\n":
                state = "CODE"; out.append(c)
            i += 1
        elif state == "BLOCK":
            if c == "*" and nxt == "/":
                state = "CODE"; out.append(" "); i += 2; continue
            i += 1
        elif state in ("DQ", "SQ"):
            out.append(c)
            if c == "\\":
                if i + 1 < n:
                    out.append(code[i + 1]); i += 2; continue
                i += 1; continue
            if (state == "DQ" and c == '"') or (state == "SQ" and c == "'"):
                state = "CODE"
            i += 1
    return "".join(out)


def tidy(code: str) -> str:
    """Normalize EOL, rstrip lines, collapse blank-line runs, trim ends."""
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    res, blank = [], 0
    for ln in code.split("\n"):
        ln = ln.rstrip()
        if ln == "":
            blank += 1
            if blank <= 1:
                res.append(ln)
        else:
            blank = 0
            res.append(ln)
    return "\n".join(res).strip() + "\n"


def clean_code(code: str) -> str:
    return tidy(strip_comments(code))


# ------------------------------------------------------------- classifiers
_SOL_DECL = re.compile(r"\b(contract|library|interface)\s+\w+")
_VYPER_DEF = re.compile(r"(?m)^\s*def\s+\w+\s*\(")  # Solidity has no `def`; Vyper does

def is_solidity(code: str) -> bool:
    if "pragma solidity" in code:
        return True
    if _VYPER_DEF.search(code):          # Python-style defs => Vyper / non-Solidity
        return False
    return bool(_SOL_DECL.search(code)) and "{" in code


_PRAGMA = re.compile(r"pragma[^;]*;")

def stub_reason(cleaned: str):
    """Return a reason string if cleaned Solidity code is an empty/stub
    contract (no executable logic), else None."""
    body = re.sub(r"\s+", "", cleaned)
    if len(body) < 40:
        return "tiny(<40 non-ws chars)"
    logic_semis = cleaned.count(";") - len(_PRAGMA.findall(cleaned))
    if logic_semis <= 0:
        return "no-statement(0 logic ;)"
    return None


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# -------------------------------------------------------------------- main
def main():
    kept, dropped = [], []
    reasons = collections.Counter()
    seen = {}            # hash -> first kept index
    orig_chars = clean_chars = 0

    with open(SRC, encoding="utf-8") as fh:
        records = [json.loads(l) for l in fh if l.strip()]

    for idx, o in enumerate(records):
        code = o.get("code", "")
        orig_chars += len(code)

        if not is_solidity(code):
            reasons["non_solidity"] += 1
            dropped.append({**o, "_drop_reason": "non_solidity", "_src_index": idx})
            continue

        cleaned = clean_code(code)

        sr = stub_reason(cleaned)
        if sr:
            reasons["stub"] += 1
            dropped.append({**o, "code": cleaned, "_drop_reason": f"stub:{sr}",
                            "_src_index": idx})
            continue

        h = hashlib.md5(norm(cleaned).encode("utf-8", "replace")).hexdigest()
        if h in seen:
            reasons["duplicate"] += 1
            dropped.append({**o, "code": cleaned,
                            "_drop_reason": f"duplicate_of_index:{seen[h]}",
                            "_src_index": idx})
            continue
        seen[h] = idx

        rec = {k: o.get(k) for k in FIELD_ORDER}
        rec["code"] = cleaned
        clean_chars += len(cleaned)
        kept.append(rec)

    with open(OUT, "w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(DROP, "w", encoding="utf-8") as fh:
        for r in dropped:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    label_dist = collections.Counter(r["label"] for r in kept)
    report = {
        "source_file": str(SRC),
        "total_input": len(records),
        "kept": len(kept),
        "dropped_total": len(dropped),
        "dropped_by_reason": dict(reasons),
        "kept_label_distribution": dict(label_dist),
        "code_chars_before": orig_chars,
        "code_chars_after_kept": clean_chars,
        "comment_reduction_note": "≈% removed measured on full set during validation",
        "outputs": {"clean": str(OUT), "dropped": str(DROP)},
    }
    with open(REPORT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
