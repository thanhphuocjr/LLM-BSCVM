#!/usr/bin/env python3
"""
Solidity Comment Stripper for SmartBugs Wild Dataset
=====================================================
Removes all comment styles (single-line //, NatSpec ///, block /* */, Doxygen /** */)
while strictly preserving:
- String literals (both single-quoted and double-quoted) containing comment-like markers, URLs, IPFS links.
- Escaped characters inside strings (\", \', \\).
- Solidity keywords, identifiers, and syntax structure.
- Clean formatting: trims trailing spaces and reduces consecutive empty lines to at most 1 blank line.

Designed for high-throughput batch processing of 47,000+ Solidity smart contracts.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Tuple


def strip_solidity_comments(source: str) -> str:
    """
    Remove all comments from a Solidity source code string while preserving
    string literals and syntax integrity.
    """
    result = []
    i = 0
    n = len(source)

    while i < n:
        char = source[i]

        # Double-quoted string literal: "..."
        if char == '"':
            start = i
            i += 1
            while i < n:
                if source[i] == '\\':
                    i += 2
                    continue
                if source[i] == '"':
                    i += 1
                    break
                i += 1
            result.append(source[start:i])

        # Single-quoted string literal: '...'
        elif char == "'":
            start = i
            i += 1
            while i < n:
                if source[i] == '\\':
                    i += 2
                    continue
                if source[i] == "'":
                    i += 1
                    break
                i += 1
            result.append(source[start:i])

        # Single-line comment: // ...
        elif char == '/' and i + 1 < n and source[i + 1] == '/':
            i += 2
            while i < n and source[i] != '\n':
                i += 1
            # Note: The newline is left in place to maintain line boundary

        # Multi-line comment: /* ... */
        elif char == '/' and i + 1 < n and source[i + 1] == '*':
            i += 2
            has_newline = False
            while i + 1 < n and not (source[i] == '*' and source[i + 1] == '/'):
                if source[i] == '\n':
                    has_newline = True
                i += 1
            if i + 1 < n:
                i += 2  # consume '*/'
            else:
                i = n   # unclosed comment reaches EOF

            if has_newline:
                result.append('\n')
            else:
                # Insert space only if needed to avoid merging adjacent tokens
                prev_is_space = bool(result and result[-1].isspace())
                next_is_space = bool(i < n and source[i].isspace())
                if not prev_is_space and not next_is_space:
                    result.append(' ')

        else:
            result.append(char)
            i += 1

    raw = ''.join(result)

    # Post-processing: clean whitespace and blank lines
    lines = [line.rstrip() for line in raw.split('\n')]
    text = '\n'.join(lines)

    # Collapse 3 or more consecutive newlines into 2 (at most one empty line between blocks)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip leading/trailing whitespace, ensure exactly one trailing newline
    text = text.strip()
    return (text + '\n') if text else ''


def process_single_file(args_tuple: Tuple[Path, Path | None, bool]) -> Tuple[bool, int, int, int, int, str | None]:
    """
    Worker function to process one contract file.
    Returns: (modified, orig_bytes, new_bytes, orig_lines, new_lines, error_msg)
    """
    file_path, output_dir, dry_run = args_tuple
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as fp:
            orig_text = fp.read()

        orig_bytes = len(orig_text.encode('utf-8'))
        orig_lines = orig_text.count('\n') + (1 if orig_text and not orig_text.endswith('\n') else 0)

        cleaned_text = strip_solidity_comments(orig_text)

        new_bytes = len(cleaned_text.encode('utf-8'))
        new_lines = cleaned_text.count('\n') + (1 if cleaned_text and not cleaned_text.endswith('\n') else 0)

        is_modified = (orig_text != cleaned_text)

        if not dry_run and is_modified:
            dest_path = file_path if output_dir is None else (output_dir / file_path.name)
            # Write safely
            temp_path = dest_path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as fp:
                fp.write(cleaned_text)
            temp_path.replace(dest_path)

        return True, orig_bytes, new_bytes, orig_lines, new_lines, None
    except Exception as e:
        return False, 0, 0, 0, 0, f"{file_path.name}: {str(e)}"


def run_pipeline(
    target_dir: Path,
    output_dir: Path | None = None,
    dry_run: bool = False,
    workers: int = 8,
    limit: int | None = None,
) -> None:
    if not target_dir.is_dir():
        print(f"[ERROR] Target directory not found: {target_dir}", file=sys.stderr)
        sys.exit(1)

    all_files = sorted(list(target_dir.glob("*.sol")))
    if limit is not None and limit > 0:
        all_files = all_files[:limit]

    total_files = len(all_files)
    print("=" * 70)
    print(f"SOLIDITY COMMENT STRIPPER PIPELINE")
    print("=" * 70)
    print(f" Target Directory : {target_dir}")
    print(f" Total Contracts  : {total_files:,}")
    print(f" Execution Mode   : {'DRY RUN (no file writes)' if dry_run else ('IN-PLACE OVERWRITE' if output_dir is None else f'OUTPUT TO {output_dir}')}")
    print(f" Parallel Workers : {workers}")
    print("=" * 70)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    tasks = [(p, output_dir, dry_run) for p in all_files]

    total_orig_bytes = 0
    total_new_bytes = 0
    total_orig_lines = 0
    total_new_lines = 0
    modified_count = 0
    unmodified_count = 0
    errors = []

    report_interval = max(1000, total_files // 20)
    processed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for success, o_b, n_b, o_l, n_l, err in executor.map(process_single_file, tasks, chunksize=100):
            processed += 1
            if success:
                total_orig_bytes += o_b
                total_new_bytes += n_b
                total_orig_lines += o_l
                total_new_lines += n_l
                if o_b != n_b:
                    modified_count += 1
                else:
                    unmodified_count += 1
            else:
                errors.append(err)

            if processed % report_interval == 0 or processed == total_files:
                pct = (processed / total_files) * 100
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                print(f"[{pct:5.1f}%] Processed {processed:,}/{total_files:,} contracts ({rate:.0f} files/s)...")

    elapsed_total = time.time() - start_time
    mb_orig = total_orig_bytes / (1024 * 1024)
    mb_new = total_new_bytes / (1024 * 1024)
    mb_saved = mb_orig - mb_new
    lines_saved = total_orig_lines - total_new_lines

    print("=" * 70)
    print("EXECUTION SUMMARY")
    print("=" * 70)
    print(f" Total Processed Contracts: {processed:,}")
    print(f" Contracts with Comments : {modified_count:,} ({modified_count/max(1, processed)*100:.1f}%)")
    print(f" Contracts Clean Already : {unmodified_count:,} ({unmodified_count/max(1, processed)*100:.1f}%)")
    print(f" Errors Encountered      : {len(errors)}")
    print(f" Original Total Lines    : {total_orig_lines:,}")
    print(f" Stripped Total Lines    : {total_new_lines:,}")
    print(f" Lines of Comments Removed: {lines_saved:,} (-{lines_saved/max(1, total_orig_lines)*100:.1f}%)")
    print(f" Original Data Size      : {mb_orig:.2f} MB")
    print(f" Cleaned Data Size       : {mb_new:.2f} MB")
    print(f" Storage Saved           : {mb_saved:.2f} MB (-{mb_saved/max(0.001, mb_orig)*100:.1f}%)")
    print(f" Total Elapsed Time      : {elapsed_total:.2f} seconds ({processed/max(0.001, elapsed_total):.0f} contracts/sec)")
    print("=" * 70)

    if errors:
        print(f"[WARNING] {len(errors)} files failed to process:")
        for err in errors[:10]:
            print(f"  - {err}")
    else:
        print("[SUCCESS] All contracts processed cleanly without errors!")


def main():
    parser = argparse.ArgumentParser(description="Strip all comments from Solidity contracts in a dataset.")
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("smartbugs_wild/contracts"),
        help="Path to folder containing .sol files (default: smartbugs_wild/contracts)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional destination directory. If not specified, performs in-place modification.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        default=True,
        help="Overwrite files in-place (default: True).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate run and print stats without modifying any files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, (os.cpu_count() or 4) * 2),
        help="Number of worker threads (default: CPU cores * 2).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of contracts to process (for testing).",
    )

    args = parser.parse_args()
    run_pipeline(
        target_dir=args.target_dir,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        workers=args.workers,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
