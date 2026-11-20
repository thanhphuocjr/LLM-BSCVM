"""Bước offline - xây cơ sở tri thức cho Phase 1B/2: data/knowledge_base/source -> built (tất định, byte-identical).

    .venv/bin/python -m E2E.phase1b_rag.build_knowledge_base
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import joblib
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

from E2E.common.io_utils import utc_now, write_json
from E2E.common.regex_patterns import (
    RE_BULLET_PREFIX, RE_MARKDOWN_CODE_BLOCK, RE_MARKDOWN_LINK, RE_MARKDOWN_SYMBOLS, RE_SENTENCE_BREAK, RE_WHITESPACE,
)
from E2E.config import BASELINE_TAXONOMY_PATH, KNOWLEDGE_BASE_DIR, VULNERABILITY_KNOWLEDGE_PATH
from E2E.phase1b_rag.swc_agents import SEVERITY_BY_SWC, SIGNAL_PATTERNS_BY_SWC, normalize_swc_id


# Bỏ code block, link, ký hiệu Markdown -> văn bản thuần.
def clean_markdown(value: str) -> str:
    value = RE_MARKDOWN_SYMBOLS.sub(" ", RE_MARKDOWN_LINK.sub(r"\1", RE_MARKDOWN_CODE_BLOCK.sub(" ", value)))
    return RE_WHITESPACE.sub(" ", value).strip()


# Tách hướng dẫn sửa lỗi thành danh sách gạch đầu dòng (không có bullet thì tối đa 4 câu).
def checklist(remediation: str) -> list[str]:
    items = [clean_markdown(item) for line in remediation.splitlines()
             if (item := RE_BULLET_PREFIX.sub("", line).strip()) and item != line.strip()]
    if items:
        return items
    text = clean_markdown(remediation)
    sentences = [s.strip() for s in RE_SENTENCE_BREAK.split(text) if len(s.strip()) > 10]
    return [] if not text else sentences[:4] if len(sentences) > 1 else [text]


# Ghép các phần khác rỗng thành văn bản để vector hoá TF-IDF.
def _retrieval_text(*parts: str) -> str:
    return "\n".join(filter(None, parts))


# Giá trị phổ biến nhất của một trường trong các ca (bỏ rỗng / bỏ giá trị `skip`); không có -> default.
def _most_common(items: list[dict], key: str, default: str = "", skip: str | None = None) -> str:
    counter = Counter(item[key] for item in items if item[key] and item[key] != skip)
    return counter.most_common(1)[0][0] if counter else default


# Một "ca" lỗ hổng từ một hàm lỗi của dataset: SWC, root cause, lập luận audit, cách sửa, mã lỗi / mã đã sửa.
def _knowledge_item(index: int, item: dict) -> dict:
    swc = normalize_swc_id(item.get("swc_id")) or "SWC-100"
    fields = {key: str(item.get(key) or "").strip()
              for key in ("root_cause", "audit_reasoning", "mitigation_guidance", "vulnerable_code", "fixed_code")}
    title = str(item.get("vulnerability_type") or swc).strip()
    return {
        "id": str(item.get("id") or f"know_{index + 1:05d}"), "swc": swc, "title": title,
        "severity": str(item.get("severity") or SEVERITY_BY_SWC.get(swc, "Medium")).strip(),
        "root_cause": fields["root_cause"], "description": fields["audit_reasoning"],
        "remediation": fields["mitigation_guidance"], "checklist": checklist(fields["mitigation_guidance"]),
        "vulnerable_code": fields["vulnerable_code"], "fixed_code": fields["fixed_code"],
        "has_code_patch": bool(item.get("has_code_patch", bool(fields["fixed_code"]))),
        "cwe_ids": item.get("cwe_ids") or [], "contract_name": str(item.get("contract_name") or ""),
        "function_name": str(item.get("function_name") or ""),
        "source": str(item.get("source") or "final_vulnerability_knowledge"),
        "retrieval_text": _retrieval_text(swc, title, fields["root_cause"], fields["audit_reasoning"],
                                          fields["mitigation_guidance"], fields["vulnerable_code"]),
    }


# Mục registry của một SWC CÓ dữ liệu: tiêu đề / root cause / mô tả / cách sửa phổ biến nhất + tối đa 3 mẫu có bản vá.
def _registry_record(swc: str, items: list[dict], info: dict, patterns: list[str]) -> dict:
    title = _most_common(items, "title", info.get("title") or swc, skip=swc)
    remediation = _most_common(items, "remediation", info.get("remediation") or "")
    description = _most_common(items, "description", info.get("description") or "")
    root_cause = _most_common(items, "root_cause")
    samples = [{"vulnerable": x["vulnerable_code"], "fixed": x["fixed_code"], "function_name": x["function_name"],
                "root_cause": x["root_cause"]} for x in items if x["vulnerable_code"] and x["fixed_code"]][:3]
    return {
        "id": swc, "title": title, "description": description, "root_cause": root_cause, "remediation": remediation,
        "checklist": checklist(remediation), "severity": SEVERITY_BY_SWC.get(swc, "Medium"),
        "signal_patterns": patterns,
        "sample_count": len(items), "samples": samples, "source": "final_vulnerability_knowledge",
        "retrieval_text": _retrieval_text(swc, title, root_cause, description, remediation, " ".join(patterns),
                                          "\n".join(x["vulnerable"] for x in samples)),
    }


# SWC KHÔNG có dữ liệu: mục registry + một ca "nền" lấy từ baseline_taxonomy.json (để vẫn truy hồi được).
def _baseline_records(swc: str, info: dict, patterns: list[str]) -> tuple[dict, dict]:
    title = info.get("title") or swc
    description, remediation = info.get("description") or "", info.get("remediation") or ""
    samples = [s["vulnerable"] for s in info.get("samples", []) if s.get("vulnerable")]
    items = info.get("checklist") or (checklist(remediation) if remediation else [])
    text = _retrieval_text(swc, title, description, remediation, " ".join(patterns), "\n".join(samples[:4]))
    severity = SEVERITY_BY_SWC.get(swc, "Medium")
    registry = {"id": swc, "title": title, "description": description, "root_cause": "", "remediation": remediation,
                "checklist": items, "severity": severity, "signal_patterns": patterns, "sample_count": len(samples),
                "samples": [{"vulnerable": s, "fixed": ""} for s in samples[:2]], "source": "swc_baseline_taxonomy",
                "retrieval_text": text}
    knowledge = {"id": f"baseline_{swc.lower()}", "swc": swc, "title": title, "severity": severity, "root_cause": "",
                 "description": description, "remediation": remediation, "checklist": items,
                 "vulnerable_code": samples[0] if samples else "", "fixed_code": "", "has_code_patch": False,
                 "cwe_ids": [], "contract_name": "", "function_name": "", "source": "swc_baseline_taxonomy",
                 "retrieval_text": text}
    return registry, knowledge


# Dựng 2 dạng tri thức đồng bộ: knowledge_items (mỗi hàm lỗi một ca) + registry_records (37 mục SWC-100..136).
def build_knowledge_items(knowledge_file: Path, baseline_taxonomy_file: Path) -> tuple[list[dict], list[dict]]:
    raw_data = json.loads(knowledge_file.read_text(encoding="utf-8")) if knowledge_file.exists() else []
    taxonomy = json.loads(baseline_taxonomy_file.read_text(encoding="utf-8")) if baseline_taxonomy_file.exists() else {}
    vulnerable = [item for item in raw_data if item.get("is_vulnerable")]
    knowledge_items = [_knowledge_item(index, item) for index, item in enumerate(vulnerable)]
    by_swc = defaultdict(list)
    for record in knowledge_items:
        by_swc[record["swc"]].append(record)
    registry_records = []
    for swc in (f"SWC-{number:03d}" for number in range(100, 137)):
        patterns = list(SIGNAL_PATTERNS_BY_SWC.get(swc, ()))
        if swc in by_swc:
            registry_records.append(_registry_record(swc, by_swc[swc], taxonomy.get(swc, {}), patterns))
        else:
            registry, knowledge = _baseline_records(swc, taxonomy.get(swc, {}), patterns)
            registry_records.append(registry)
            knowledge_items.append(knowledge)
    return knowledge_items, registry_records


# [BƯỚC OFFLINE] Dựng tri thức -> fit TF-IDF (char_wb 3-5 gram, 60k đặc trưng) -> ghi 5 artifact vào output_dir.
def build_knowledge_base(knowledge_file: Path, output_dir: Path,
                         baseline_taxonomy_file: Path = BASELINE_TAXONOMY_PATH) -> dict:
    knowledge_items, registry_records = build_knowledge_items(knowledge_file, baseline_taxonomy_file)
    if not knowledge_items:
        raise RuntimeError("No knowledge items could be constructed.")
    output_dir.mkdir(parents=True, exist_ok=True)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=60000,
                                 sublinear_tf=True, lowercase=True, norm="l2")
    matrix = vectorizer.fit_transform(record["retrieval_text"] for record in knowledge_items)
    joblib.dump(vectorizer, output_dir / "swc_vectorizer.joblib")
    save_npz(output_dir / "swc_matrix.npz", matrix)
    write_json(output_dir / "knowledge_items.json", knowledge_items)
    write_json(output_dir / "swc_registry.json", registry_records)
    manifest = {
        "built_at": utc_now(), "knowledge_items_count": len(knowledge_items),
        "registry_records_count": len(registry_records), "matrix_shape": list(matrix.shape),
        "knowledge_source": str(knowledge_file) if knowledge_file.exists() else "none",
        "baseline_taxonomy_source": str(baseline_taxonomy_file) if baseline_taxonomy_file.exists() else "none",
        "artifacts": ["knowledge_items.json", "swc_registry.json", "swc_vectorizer.joblib", "swc_matrix.npz"],
        "version": "2.0.0",
    }
    write_json(output_dir / "knowledge_manifest.json", manifest)
    return manifest


# Điểm vào CLI: --knowledge-file, --baseline-taxonomy, --output-dir -> build và in manifest.
def main() -> None:
    parser = argparse.ArgumentParser(description="Build the vulnerability knowledge store and TF-IDF artifacts.")
    parser.add_argument("--knowledge-file", type=Path, default=VULNERABILITY_KNOWLEDGE_PATH,
                        help="Path to final-vulnerability-knowledge.json")
    parser.add_argument("--baseline-taxonomy", type=Path, default=BASELINE_TAXONOMY_PATH)
    parser.add_argument("--output-dir", type=Path, default=KNOWLEDGE_BASE_DIR)
    args = parser.parse_args()
    print(json.dumps(build_knowledge_base(args.knowledge_file, args.output_dir, args.baseline_taxonomy), indent=2))


if __name__ == "__main__":
    main()
