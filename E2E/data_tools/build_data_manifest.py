"""
Integrity manifest of the local datasets and the detector adapter (data/data_manifest.json).

    .venv/bin/python -m E2E.data_tools.build_data_manifest

One SHA-256 per file; the SmartBugs-Wild corpus (47,398 .sol) is hashed as one tree.
Run results under data/results/ change with every run and are not hashed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from E2E.common.io_utils import utc_now, write_json
from E2E.config import DATA_DIR, DETECTOR_ADAPTER_DIR, PROJECT_ROOT, RESULTS_DIR, SMARTBUGS_WILD_DIR


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_entry(path: Path) -> dict:
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": hash_file(path),
    }


def tree_entry(path: Path) -> dict:
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = str(file_path.relative_to(path))
        file_hash = hash_file(file_path)
        size = file_path.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\0")
        file_count += 1
        total_bytes += size
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "file_count": file_count,
        "bytes": total_bytes,
        "tree_sha256": digest.hexdigest(),
    }


def build_manifest(output: Path) -> dict:
    corpora = [SMARTBUGS_WILD_DIR] if SMARTBUGS_WILD_DIR.exists() else []
    dataset_dirs = [
        item for item in DATA_DIR.iterdir()
        if item.is_dir() and item != RESULTS_DIR and item not in corpora
    ] if DATA_DIR.exists() else []
    paths = sorted(
        {
            *[
                path
                for directory in dataset_dirs
                for path in directory.rglob("*")
                if path.is_file() and not path.name.startswith(".")
            ],
            *[path for path in DETECTOR_ADAPTER_DIR.rglob("*") if path.is_file()],
        }
    )
    entries = [manifest_entry(path) for path in paths if path != output]
    trees = [tree_entry(path) for path in corpora]
    payload = {
        "created_at": utc_now(),
        "root": ".",
        "file_count": len(entries) + sum(item["file_count"] for item in trees),
        "total_bytes": sum(item["bytes"] for item in entries) + sum(item["bytes"] for item in trees),
        "files": entries,
        "trees": trees,
    }
    write_json(output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hash the local datasets and the detector adapter.")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "data_manifest.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_manifest(args.output)
    print(json.dumps({key: result[key] for key in ("file_count", "total_bytes")}, indent=2))


if __name__ == "__main__":
    main()
