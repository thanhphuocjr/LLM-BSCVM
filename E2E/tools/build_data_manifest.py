from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from E2E.config import DATA_DIR, DETECTOR_ADAPTER_DIR, E2E_ROOT
from E2E.shared.io_utils import utc_now, write_json


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_entry(path: Path) -> dict:
    return {
        "path": str(path.relative_to(E2E_ROOT)),
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
        "path": str(path.relative_to(E2E_ROOT)),
        "file_count": file_count,
        "bytes": total_bytes,
        "tree_sha256": digest.hexdigest(),
    }


def build_manifest(output: Path) -> dict:
    raw_dir = DATA_DIR / "source" / "raw"
    paths = sorted(
        {
            *[
                path
                for path in DATA_DIR.rglob("*")
                if path.is_file() and raw_dir not in path.parents
            ],
            *[path for path in DETECTOR_ADAPTER_DIR.rglob("*") if path.is_file()],
        }
    )
    entries = [manifest_entry(path) for path in paths if path != output]
    trees = [
        tree_entry(path)
        for path in sorted(item for item in raw_dir.iterdir() if item.is_dir())
    ] if raw_dir.exists() else []
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
    parser = argparse.ArgumentParser(description="Hash the E2E datasets and detector artifacts.")
    parser.add_argument("--output", type=Path, default=E2E_ROOT / "data_manifest.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_manifest(args.output)
    print(json.dumps({key: result[key] for key in ("file_count", "total_bytes")}, indent=2))


if __name__ == "__main__":
    main()
