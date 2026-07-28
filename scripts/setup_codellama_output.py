#!/usr/bin/env python3
"""Download and prepare the Kaggle CodeLlama detector output for the pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SLUG = "ntpuet/codellama"
DEFAULT_DEST = PROJECT_ROOT / "models" / "codellama-vuln-detector"


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def find_adapter_dir(root: Path) -> Path | None:
    candidates = sorted(root.rglob("adapter_config.json"), key=lambda p: len(p.parts))
    return candidates[0].parent if candidates else None


def update_env(adapter_dir: Path) -> None:
    env_path = PROJECT_ROOT / ".env"
    existing: dict[str, str] = {}
    lines: list[str] = []
    if env_path.exists():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            if raw.strip() and not raw.lstrip().startswith("#") and "=" in raw:
                key, _, value = raw.partition("=")
                existing[key.strip()] = value.strip()
            lines.append(raw)

    desired = {
        "DETECTOR_ADAPTER_PATH": str(adapter_dir.relative_to(PROJECT_ROOT)),
        "DETECTOR_BASE_MODEL": "auto",
        "DETECTOR_INPUT_MODE": "auto",
        "DETECTOR_LOAD_IN_4BIT": "auto",
    }
    changed = False
    for key, value in desired.items():
        if existing.get(key) == value:
            continue
        if key in existing:
            lines = [
                f"{key}={value}" if line.partition("=")[0].strip() == key else line
                for line in lines
            ]
        else:
            lines.append(f"{key}={value}")
        changed = True

    if changed:
        env_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        print(f"Updated {env_path.relative_to(PROJECT_ROOT)} for CodeLlama detector.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Kaggle CodeLlama output and update detector .env.")
    parser.add_argument("--slug", default=DEFAULT_SLUG, help="Kaggle kernel slug, e.g. owner/kernel-slug.")
    parser.add_argument("--dest", default=str(DEFAULT_DEST), help="Destination directory for Kaggle output.")
    parser.add_argument("--skip-download", action="store_true", help="Only inspect dest and update .env.")
    parser.add_argument("--no-env", action="store_true", help="Do not update .env.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dest = Path(args.dest)
    if not dest.is_absolute():
        dest = PROJECT_ROOT / dest
    dest.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        run(["kaggle", "kernels", "output", args.slug, "-p", str(dest)])

    adapter_dir = find_adapter_dir(dest)
    if not adapter_dir:
        print(
            f"No adapter_config.json found under {dest}.\n"
            "Check that the Kaggle output contains the final PEFT adapter directory.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    print(f"Adapter: {adapter_dir.relative_to(PROJECT_ROOT)}")
    if not args.no_env:
        update_env(adapter_dir)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
