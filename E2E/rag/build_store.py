from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any

import joblib
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

from E2E.config import (
    E2E_ROOT,
    RUNTIME_DATA_DIR,
    VULNERABILITY_KNOWLEDGE_PATH,
)
from E2E.rag.legacy.multi_agent_vuln_detector import (
    SEVERITY_BY_SWC,
    SIGNAL_PATTERNS_BY_SWC,
    normalize_swc_id,
)
from E2E.shared.io_utils import utc_now, write_json


HEADING_RE = re.compile(r"^##?\s+(.+?)\s*$", re.MULTILINE)


def section(markdown: str, name: str) -> str:
    matches = list(HEADING_RE.finditer(markdown))
    for index, match in enumerate(matches):
        if match.group(1).strip().lower() != name.lower():
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        return markdown[match.end() : end].strip()
    return ""


def clean_markdown(value: str) -> str:
    value = re.sub(r"```[\s\S]*?```", " ", value)
    value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"[`*_>#-]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def checklist(remediation: str) -> list[str]:
    items = []
    for line in remediation.splitlines():
        item = re.sub(r"^\s*[-*]\s*", "", line).strip()
        if item and item != line.strip():
            items.append(clean_markdown(item))
    if items:
        return items
    text = clean_markdown(remediation)
    if not text:
        return []
    sentences = [s.strip() for s in re.split(r"[.;]\s+", text) if len(s.strip()) > 10]
    return sentences[:4] if len(sentences) > 1 else [text]


def build_knowledge_items(
    knowledge_file: Path | None,
    source_dir: Path | None = None,
    baseline_taxonomy_file: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Builds two synchronized knowledge representations:
    1. knowledge_items: List of granular function-level vulnerability cases for TF-IDF RAG retrieval.
    2. registry_records: 37 SWC taxonomy entries enriched with dataset empirical evidence.
    """
    raw_data: list[dict[str, Any]] = []
    if knowledge_file and knowledge_file.exists():
        if knowledge_file.suffix == ".jsonl":
            with knowledge_file.open("r", encoding="utf-8") as f:
                raw_data = [json.loads(line) for line in f if line.strip()]
        else:
            with knowledge_file.open("r", encoding="utf-8") as f:
                raw_data = json.load(f)

    baseline_tax: dict[str, Any] = {}
    if baseline_taxonomy_file and baseline_taxonomy_file.exists():
        with baseline_taxonomy_file.open("r", encoding="utf-8") as f:
            baseline_tax = json.load(f)

    # Filter vulnerable items
    vuln_items = [x for x in raw_data if x.get("is_vulnerable")]
    by_swc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    knowledge_items: list[dict[str, Any]] = []

    for idx, item in enumerate(vuln_items):
        swc = normalize_swc_id(item.get("swc_id")) or "SWC-100"
        title = str(item.get("vulnerability_type") or swc).strip()
        severity = str(item.get("severity") or SEVERITY_BY_SWC.get(swc, "Medium")).strip()
        root_cause = str(item.get("root_cause") or "").strip()
        audit_reasoning = str(item.get("audit_reasoning") or "").strip()
        mitigation = str(item.get("mitigation_guidance") or "").strip()
        vulnerable_code = str(item.get("vulnerable_code") or "").strip()
        fixed_code = str(item.get("fixed_code") or "").strip()
        item_id = str(item.get("id") or f"know_{idx + 1:05d}")

        retrieval_text = "\n".join(
            filter(
                None,
                [
                    swc,
                    title,
                    root_cause,
                    audit_reasoning,
                    mitigation,
                    vulnerable_code,
                ],
            )
        )

        record = {
            "id": item_id,
            "swc": swc,
            "title": title,
            "severity": severity,
            "root_cause": root_cause,
            "description": audit_reasoning,
            "remediation": mitigation,
            "checklist": checklist(mitigation),
            "vulnerable_code": vulnerable_code,
            "fixed_code": fixed_code,
            "has_code_patch": bool(item.get("has_code_patch", bool(fixed_code))),
            "cwe_ids": item.get("cwe_ids") or [],
            "contract_name": str(item.get("contract_name") or ""),
            "function_name": str(item.get("function_name") or ""),
            "source": str(item.get("source") or "final_vulnerability_knowledge"),
            "retrieval_text": retrieval_text,
        }
        knowledge_items.append(record)
        by_swc[swc].append(record)

    # Process 37 SWCs to build swc_registry.json
    all_swc_ids = [f"SWC-{i:03d}" for i in range(100, 137)]
    registry_records: list[dict[str, Any]] = []

    for swc in all_swc_ids:
        patterns = list(SIGNAL_PATTERNS_BY_SWC.get(swc, ()))

        # Try loading baseline if present
        base_title = swc
        base_desc = ""
        base_remediation = ""
        base_code_samples: list[str] = []
        base_checklist_items: list[str] = []

        if swc in baseline_tax:
            info = baseline_tax[swc]
            base_title = info.get("title") or swc
            base_desc = info.get("description") or ""
            base_remediation = info.get("remediation") or ""
            base_checklist_items = info.get("checklist") or []
            base_code_samples = [s["vulnerable"] for s in info.get("samples", []) if s.get("vulnerable")]
        elif source_dir and (source_dir / f"{swc}.md").exists():
            swc_md_path = source_dir / f"{swc}.md"
            base_md = swc_md_path.read_text(encoding="utf-8")
            base_title = clean_markdown(section(base_md, "Title")) or swc
            base_desc = clean_markdown(section(base_md, "Description"))
            base_remediation_raw = section(base_md, "Remediation")
            base_remediation = clean_markdown(base_remediation_raw)
            base_samples_raw = section(base_md, "Samples")
            base_code_samples = (
                re.findall(r"```(?:solidity)?\s*(.*?)```", base_samples_raw, re.DOTALL | re.IGNORECASE)
                if base_samples_raw
                else []
            )
            base_checklist_items = checklist(base_remediation_raw)

        if swc in by_swc:
            items = by_swc[swc]
            title_counter = Counter(x["title"] for x in items if x["title"] and x["title"] != swc)
            title = title_counter.most_common(1)[0][0] if title_counter else base_title or swc

            rc_counter = Counter(x["root_cause"] for x in items if x["root_cause"])
            top_rc = rc_counter.most_common(1)[0][0] if rc_counter else ""

            rem_counter = Counter(x["remediation"] for x in items if x["remediation"])
            top_remediation = rem_counter.most_common(1)[0][0] if rem_counter else base_remediation

            desc_counter = Counter(x["description"] for x in items if x["description"])
            top_desc = desc_counter.most_common(1)[0][0] if desc_counter else base_desc

            samples_with_patch = [
                {
                    "vulnerable": x["vulnerable_code"],
                    "fixed": x["fixed_code"],
                    "function_name": x["function_name"],
                    "root_cause": x["root_cause"],
                }
                for x in items
                if x["vulnerable_code"] and x["fixed_code"]
            ][:3]

            retrieval_text = "\n".join(
                filter(
                    None,
                    [
                        swc,
                        title,
                        top_rc,
                        top_desc,
                        top_remediation,
                        " ".join(patterns),
                        "\n".join(x["vulnerable"] for x in samples_with_patch),
                    ],
                )
            )

            registry_records.append(
                {
                    "id": swc,
                    "title": title,
                    "description": top_desc,
                    "root_cause": top_rc,
                    "remediation": top_remediation,
                    "checklist": checklist(top_remediation),
                    "severity": SEVERITY_BY_SWC.get(swc, "Medium"),
                    "signal_patterns": patterns,
                    "sample_count": len(items),
                    "samples": samples_with_patch,
                    "source": "final_vulnerability_knowledge",
                    "retrieval_text": retrieval_text,
                }
            )
        else:
            retrieval_text = "\n".join(
                filter(
                    None,
                    [
                        swc,
                        base_title,
                        base_desc,
                        base_remediation,
                        " ".join(patterns),
                        "\n".join(base_code_samples[:4]),
                    ],
                )
            )
            reg_item = {
                "id": swc,
                "title": base_title,
                "description": base_desc,
                "root_cause": "",
                "remediation": base_remediation,
                "checklist": base_checklist_items or (checklist(base_remediation) if base_remediation else []),
                "severity": SEVERITY_BY_SWC.get(swc, "Medium"),
                "signal_patterns": patterns,
                "sample_count": len(base_code_samples),
                "samples": [{"vulnerable": s, "fixed": ""} for s in base_code_samples[:2]],
                "source": "swc_baseline_taxonomy",
                "retrieval_text": retrieval_text,
            }
            registry_records.append(reg_item)

            knowledge_items.append(
                {
                    "id": f"baseline_{swc.lower()}",
                    "swc": swc,
                    "title": base_title,
                    "severity": SEVERITY_BY_SWC.get(swc, "Medium"),
                    "root_cause": "",
                    "description": base_desc,
                    "remediation": base_remediation,
                    "checklist": base_checklist_items or (checklist(base_remediation) if base_remediation else []),
                    "vulnerable_code": base_code_samples[0] if base_code_samples else "",
                    "fixed_code": "",
                    "has_code_patch": False,
                    "cwe_ids": [],
                    "contract_name": "",
                    "function_name": "",
                    "source": "swc_baseline_taxonomy",
                    "retrieval_text": retrieval_text,
                }
            )

    return knowledge_items, registry_records


def build_store(
    knowledge_file: Path | None,
    source_dir: Path | None,
    output_dir: Path,
) -> dict[str, Any]:
    baseline_taxonomy_file = output_dir / "baseline_taxonomy.json"
    knowledge_items, registry_records = build_knowledge_items(
        knowledge_file,
        source_dir=source_dir,
        baseline_taxonomy_file=baseline_taxonomy_file,
    )
    if not knowledge_items:
        raise RuntimeError("No knowledge items could be constructed.")

    output_dir.mkdir(parents=True, exist_ok=True)
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=60000,
        sublinear_tf=True,
        lowercase=True,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(record["retrieval_text"] for record in knowledge_items)

    joblib.dump(vectorizer, output_dir / "swc_vectorizer.joblib")
    save_npz(output_dir / "swc_matrix.npz", matrix)
    write_json(output_dir / "knowledge_items.json", knowledge_items)
    write_json(output_dir / "swc_registry.json", registry_records)

    manifest = {
        "built_at": utc_now(),
        "knowledge_items_count": len(knowledge_items),
        "registry_records_count": len(registry_records),
        "matrix_shape": list(matrix.shape),
        "knowledge_source": str(knowledge_file) if knowledge_file and knowledge_file.exists() else "none",
        "artifacts": [
            "knowledge_items.json",
            "swc_registry.json",
            "swc_vectorizer.joblib",
            "swc_matrix.npz",
            "baseline_taxonomy.json",
        ],
        "version": "2.0.0",
    }
    write_json(output_dir / "knowledge_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the vulnerability knowledge store and TF-IDF artifacts.")
    parser.add_argument(
        "--knowledge-file",
        type=Path,
        default=VULNERABILITY_KNOWLEDGE_PATH,
        help="Path to final-vulnerability-knowledge.json",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Optional fallback directory containing SWC-*.md files.",
    )
    parser.add_argument("--output-dir", type=Path, default=RUNTIME_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(build_store(args.knowledge_file, args.source_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
