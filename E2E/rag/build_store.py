from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

from E2E.config import E2E_ROOT, RUNTIME_DATA_DIR, SOURCE_DATA_DIR
from E2E.rag.legacy.multi_agent_vuln_detector import (
    SEVERITY_BY_SWC,
    SIGNAL_PATTERNS_BY_SWC,
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
    return [text] if text else []


def build_records(source_dir: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(source_dir.glob("SWC-*.md")):
        swc = path.stem.upper()
        markdown = path.read_text(encoding="utf-8")
        title = clean_markdown(section(markdown, "Title")) or swc
        description = clean_markdown(section(markdown, "Description"))
        remediation_raw = section(markdown, "Remediation")
        remediation = clean_markdown(remediation_raw)
        samples = section(markdown, "Samples")
        code_samples = re.findall(r"```(?:solidity)?\s*(.*?)```", samples, re.DOTALL | re.IGNORECASE)
        patterns = list(SIGNAL_PATTERNS_BY_SWC.get(swc, ()))
        retrieval_text = "\n".join(
            [
                swc,
                title,
                description,
                remediation,
                " ".join(patterns),
                "\n".join(code_samples[:4]),
            ]
        )
        records.append(
            {
                "id": swc,
                "title": title,
                "description": description,
                "remediation": remediation,
                "checklist": checklist(remediation_raw),
                "severity": SEVERITY_BY_SWC.get(swc, "Medium"),
                "signal_patterns": patterns,
                "sample_count": len(code_samples),
                "source": str(path.relative_to(source_dir.parent.parent)),
                "retrieval_text": retrieval_text,
            }
        )
    return records


def build_store(source_dir: Path, output_dir: Path) -> dict:
    records = build_records(source_dir)
    if not records:
        raise RuntimeError(f"No SWC markdown files found in {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_features=60000,
        sublinear_tf=True,
        lowercase=True,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(record["retrieval_text"] for record in records)
    joblib.dump(vectorizer, output_dir / "swc_vectorizer.joblib")
    save_npz(output_dir / "swc_matrix.npz", matrix)
    write_json(output_dir / "swc_registry.json", records)
    manifest = {
        "built_at": utc_now(),
        "record_count": len(records),
        "matrix_shape": list(matrix.shape),
        "source_dir": str(source_dir.relative_to(E2E_ROOT)),
        "artifacts": [
            "swc_registry.json",
            "swc_vectorizer.joblib",
            "swc_matrix.npz",
        ],
        "warning": (
            "SWC Registry is no longer actively maintained. Findings must be "
            "cross-checked with current compiler behavior and manual review."
        ),
    }
    write_json(output_dir / "knowledge_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the local SWC TF-IDF knowledge store.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=SOURCE_DATA_DIR / "swc-registry",
    )
    parser.add_argument("--output-dir", type=Path, default=RUNTIME_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(build_store(args.source_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
