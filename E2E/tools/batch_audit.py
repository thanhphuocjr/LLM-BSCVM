#!/usr/bin/env python3
"""
E2E/tools/batch_audit.py
──────────────────────────────────────────────────────────────────────────────
Resumable batch audit of a folder of contracts through the server_gpu pipeline,
exported as one CSV row per repaired (target) function.

The GPU server is shared and not always on, so the batch is built to stop and
resume at any time:

  * state lives in SQLite (`E2E/batch_runs/<batch>/state.sqlite`); each contract is
    pending -> in_progress -> done | failed, and `in_progress` is reset on restart;
  * contracts run strictly one at a time (one request in flight to the GPU);
  * the server health is checked before and after every contract; a contract whose
    run overlapped a server outage is discarded and re-queued, never saved;
  * Ctrl+C, or creating the file `E2E/batch_runs/<batch>/STOP`, stops cleanly.

By default each vulnerable function gets exactly one fixer call and that patch is exported as
the result, with no verification (an external judge scores the CSV). `--verify` restores the
verify-and-retry loop; a batch is locked to the mode it was started in.

Usage (from the repository root):
  .venv/bin/python -m E2E.tools.batch_audit run    --batch pilot --input-dir E2E/data/contracts/smartbugs_wild --limit 5
  .venv/bin/python -m E2E.tools.batch_audit status --batch pilot
  .venv/bin/python -m E2E.tools.batch_audit export --batch pilot
  .venv/bin/python -m E2E.tools.batch_audit retry-failed --batch pilot
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import random
import re
import signal
import sqlite3
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from E2E.config import BATCH_RUNS_DIR
from E2E.local.orchestrator import run_pipeline
from E2E.local.server_gpu_runner import check_server_health, run_server_gpu_pipeline
from E2E.shared.function_parser import parse_code_units
from E2E.shared.io_utils import read_json, sha256_text, utc_now, write_json

BATCH_ROOT = BATCH_RUNS_DIR

# The requested columns come first; the rest let the evaluator explain each verdict.
CSV_COLUMNS = [
    "id",
    "contract_name",
    "raw_contract",
    "raw_function_code",
    "fixed_function_code",
    "related_origin_vulnerability",
    "contract_file",
    "function_id",
    "swc_ids",
    "evidence_sources",
    "root_cause",
    "patch_status",
    "patch_engine",
    "fix_rationale",
    "patch_note",
    "pipeline_verification",
    "detector_prob_before",
    "detector_prob_after",
    "compile_state",
    "repair_attempts",
    "contract_verdict",
    "run_id",
]
SUMMARY_COLUMNS = [
    "contract_file",
    "status",
    "units",
    "confirmed_findings",
    "target_functions",
    "patched_functions",
    "unchanged_functions",
    "pipeline_pass_functions",
    "dismissed_false_positives",
    "skipped_targets",
    "detector_only_not_repaired",
    "contract_verdict",
    "elapsed_s",
    "error",
]

# Error text that means the GPU server (not our code) failed during a contract.
OUTAGE_MARKERS = (
    "connection refused",
    "connectionerror",
    "urlerror",
    "remote end closed",
    "timed out",
    "timeout",
    "503",
    "502",
    "chưa sẵn sàng",
    "không thể kết nối",
)

_stop_requested = False


def _request_stop(signum, frame) -> None:  # noqa: ARG001
    global _stop_requested
    if _stop_requested:
        raise KeyboardInterrupt
    _stop_requested = True
    print("\n[batch] Stop requested: finishing the current contract, then exiting "
          "(press Ctrl+C again to abort it immediately; it will be re-queued).", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────────────
def connect(batch_dir: Path) -> sqlite3.Connection:
    batch_dir.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(batch_dir / "state.sqlite")
    db.row_factory = sqlite3.Row
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS contracts (
            seq INTEGER PRIMARY KEY,
            file TEXT UNIQUE NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            run_id TEXT,
            started_at TEXT,
            finished_at TEXT,
            elapsed_s REAL,
            last_error TEXT,
            summary_json TEXT
        );
        CREATE TABLE IF NOT EXISTS functions (
            id TEXT PRIMARY KEY,
            contract_file TEXT NOT NULL,
            row_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
        """
    )
    return db


def register_contracts(db: sqlite3.Connection, input_dir: Path, shuffle: bool, seed: int) -> int:
    """Queue every .sol file once. The order is fixed at registration so resumes are stable."""
    existing = {row["file"] for row in db.execute("SELECT file FROM contracts")}
    files = sorted(str(path.resolve()) for path in input_dir.glob("*.sol"))
    if shuffle:
        random.Random(seed).shuffle(files)
    new_files = [path for path in files if path not in existing]
    start = (db.execute("SELECT COALESCE(MAX(seq), 0) FROM contracts").fetchone()[0]) + 1
    db.executemany(
        "INSERT INTO contracts (seq, file) VALUES (?, ?)",
        [(start + index, path) for index, path in enumerate(new_files)],
    )
    db.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('input_dir', ?)", (str(input_dir.resolve()),)
    )
    db.commit()
    return len(new_files)


def counts(db: sqlite3.Connection) -> dict[str, int]:
    return {row["status"]: row["n"] for row in db.execute(
        "SELECT status, COUNT(*) AS n FROM contracts GROUP BY status"
    )}


# ──────────────────────────────────────────────────────────────────────────────
# One contract
# ──────────────────────────────────────────────────────────────────────────────
def _server_ready() -> bool:
    return check_server_health().get("status") == "ready"


def parse_active_hours(spec: str | None) -> tuple[int, int] | None:
    """'19:00-07:30' -> (start, end) in minutes since midnight; the window may cross midnight."""
    if not spec:
        return None
    match = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*", spec)
    if not match:
        raise SystemExit(f"--active-hours must look like 19:00-07:30, got {spec!r}")
    h1, m1, h2, m2 = (int(value) for value in match.groups())
    return h1 * 60 + m1, h2 * 60 + m2


def within_active_hours(window: tuple[int, int] | None, now: time.struct_time | None = None) -> bool:
    if window is None:
        return True
    now = now or time.localtime()
    minute = now.tm_hour * 60 + now.tm_min
    start, end = window
    return start <= minute < end if start < end else (minute >= start or minute < end)


def wait_until_runnable(args: argparse.Namespace, stop_file: Path, log) -> bool:
    """Block until inside the active hours and the GPU server is ready.

    Returns False when the run should end (stop requested, or server down without --wait).
    """
    window = parse_active_hours(args.active_hours)
    waiting_for = None
    while not _stop_requested and not stop_file.exists():
        if not within_active_hours(window):
            reason = f"outside active hours {args.active_hours}"
        elif not _server_ready():
            if not args.wait:
                log("GPU server not ready: stopping. Run the same command again once it is up.")
                return False
            reason = "GPU server not ready"
        else:
            if waiting_for:
                log(f"Resuming ({waiting_for} -> ok).")
            return True
        if reason != waiting_for:
            log(f"Waiting: {reason} (re-checking every {args.poll_seconds}s).")
            waiting_for = reason
        time.sleep(args.poll_seconds)
    return False


def _looks_like_outage(server_result: dict[str, Any]) -> bool:
    phases = server_result.get("phases", {})
    texts = [
        str(unit.get("error") or "")
        for unit in phases.get("detector", {}).get("original", {}).get("unit_results", [])
    ]
    texts += [str(item.get("error") or "") for item in phases.get("fixer", {}).get("patch_outcomes", [])]
    return any(marker in text.lower() for text in texts for marker in OUTAGE_MARKERS)


def _unit_code(code: str, unit_id: str) -> str:
    return next((unit.code for unit in parse_code_units(code) if unit.unit_id == unit_id), "")


def build_function_rows(
    contract_file: Path,
    source: str,
    run_id: str,
    server_result: dict[str, Any],
    verification: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One row per function that reached repair (plus Advisor-dismissed / budget-skipped ones)."""
    phases = server_result.get("phases", {})
    detector = phases.get("detector", {})
    findings = detector.get("confirmed_findings", [])
    dismissed = detector.get("dismissed_findings", [])
    deferred = detector.get("detector_only_deferred", [])
    skipped = set(detector.get("skipped_targets", []))
    verifier = phases.get("verifier", {})
    fixed_code = server_result.get("fixed_code") or source
    units = {unit.unit_id: unit for unit in parse_code_units(source)}

    prob_before = {
        item["unit_id"]: item.get("vulnerability_probability")
        for item in detector.get("original", {}).get("unit_results", [])
    }
    prob_after = {
        item["unit_id"]: item.get("vulnerability_probability")
        for item in (detector.get("fixed") or {}).get("unit_results", [])
    }
    outcome_by_unit: dict[str, dict[str, Any]] = {}
    for item in server_result.get("patch_outcomes", []):
        if item.get("unit_id"):
            current = outcome_by_unit.get(item["unit_id"])
            if current is None or item.get("status") == "applied":
                outcome_by_unit[item["unit_id"]] = item
    verify_by_unit = {item.get("unit_id"): item.get("status") for item in verifier.get("verifications", [])}
    # Only the Advisor LLM's own diagnosis: knowledge-base fallbacks describe other contracts.
    root_cause = {
        item.get("unit_id"): item.get("root_cause")
        for item in phases.get("advisor", {}).get("suggestions", [])
        if item.get("is_true_positive") and item.get("root_cause")
    }
    compile_state = (verifier.get("compilation") or {}).get("state", "")
    contract_verdict = verification.get("overall_verdict", "")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in findings:
        if item.get("unit_id"):
            grouped.setdefault(item["unit_id"], []).append(item)
    dismissed_units = {item.get("unit_id") for item in dismissed if item.get("unit_id")}
    deferred_units = {item.get("unit_id") for item in deferred if item.get("unit_id")}
    for item in [*dismissed, *deferred]:
        if item.get("unit_id"):
            grouped.setdefault(item["unit_id"], []).append(item)
    confirmed_units = {f.get("unit_id") for f in findings}

    rows = []
    for unit_id, unit_findings in grouped.items():
        unit = units.get(unit_id)
        if unit is None:
            continue
        outcome = outcome_by_unit.get(unit_id, {})
        applied = outcome.get("status") == "applied"
        unchanged = outcome.get("status") == "unchanged" or (
            not applied and "identical to original" in str(outcome.get("reason") or "")
        )
        if unit_id in dismissed_units and unit_id not in confirmed_units:
            status = "dismissed_false_positive"
        elif unit_id in deferred_units and unit_id not in confirmed_units:
            status = "detector_only_not_repaired"
        elif unit_id in skipped:
            status = "skipped_budget"
        elif unit.is_view_or_pure:
            status = "not_targeted_view_pure"
        elif applied:
            status = "applied"
        elif unchanged:
            status = "unchanged_by_model"
        elif outcome:
            status = "rejected"
        else:
            status = "not_patched"
        vulnerability_text = "; ".join(
            (
                f"{item.get('swc')} {item.get('title') or ''}".strip()
                + f" [sources={','.join(item.get('sources') or [])}"
                + (f", p_final={item.get('p_final')}" if item.get("p_final") is not None else "")
                + (f", line={item.get('line')}" if item.get("line") else "")
                + "]"
            )
            for item in unit_findings
        )
        rows.append({
            "id": f"{contract_file.stem}::{unit_id}",
            "contract_name": unit.contract_name,
            "raw_contract": source,
            "raw_function_code": unit.code,
            "fixed_function_code": (
                _unit_code(fixed_code, unit_id) if applied
                else (outcome.get("fixed_code") or unit.code) if unchanged
                else ""
            ),
            "related_origin_vulnerability": vulnerability_text,
            "contract_file": contract_file.name,
            "function_id": unit_id,
            "swc_ids": ",".join(sorted({str(item.get("swc")) for item in unit_findings})),
            "evidence_sources": ",".join(sorted({s for item in unit_findings for s in (item.get("sources") or [])})),
            "root_cause": root_cause.get(unit_id, ""),
            "patch_status": status,
            "patch_engine": outcome.get("engine", ""),
            "fix_rationale": outcome.get("rationale", ""),
            "patch_note": outcome.get("guard_warning", "") if applied else str(outcome.get("reason") or ""),
            "pipeline_verification": verify_by_unit.get(unit_id, ""),
            "detector_prob_before": prob_before.get(unit_id),
            "detector_prob_after": prob_after.get(unit_id) if applied else None,
            "compile_state": compile_state,
            "repair_attempts": verifier.get("attempts_count", 0),
            "contract_verdict": contract_verdict,
            "run_id": run_id,
        })

    summary = {
        "contract_file": contract_file.name,
        "status": "done",
        "units": len(units),
        "confirmed_findings": len(findings),
        "target_functions": sum(
            1 for row in rows if row["patch_status"] in {"applied", "unchanged_by_model", "rejected", "not_patched"}
        ),
        "patched_functions": sum(1 for row in rows if row["patch_status"] == "applied"),
        "unchanged_functions": sum(1 for row in rows if row["patch_status"] == "unchanged_by_model"),
        "pipeline_pass_functions": sum(1 for row in rows if row["pipeline_verification"] == "PASS"),
        "dismissed_false_positives": sum(1 for row in rows if row["patch_status"] == "dismissed_false_positive"),
        "skipped_targets": len(skipped),
        "detector_only_not_repaired": sum(1 for row in rows if row["patch_status"] == "detector_only_not_repaired"),
        "contract_verdict": contract_verdict,
    }
    return rows, summary


class ServerOutage(RuntimeError):
    pass


def audit_contract(path: Path, runs_dir: Path, args: argparse.Namespace) -> tuple[str, list[dict], dict]:
    source = path.read_text(encoding="utf-8", errors="replace")
    run_id = re.sub(r"[^A-Za-z0-9_.-]", "-", path.stem)[:100] + "-" + sha256_text(source)[:8]
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    common = dict(
        file_name=path.name,
        run_id=run_id,
        runs_dir=runs_dir,
        source_root=None,
        tools=["slither"],
        use_smartbugs=False,
        detector_threshold=args.detector_threshold,
        max_fix_attempts=args.max_fix_attempts,
    )
    with (run_dir / "execution.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        manifest = run_pipeline(source, **common)
        request = read_json(manifest["request_path"])
        if args.max_target_units:
            request["execution"]["repair"]["max_target_units"] = args.max_target_units
        request["execution"]["repair"]["detector_only_min_prob"] = args.detector_only_min_prob
        request["execution"]["repair"]["single_pass"] = not args.verify
        try:
            server_result = run_server_gpu_pipeline(request)
        except ConnectionError as error:
            raise ServerOutage(str(error)) from error
        if _looks_like_outage(server_result) or not _server_ready():
            raise ServerOutage("GPU server failed or went offline during this contract")
        if args.verify:
            run_pipeline(source, remote_result=server_result, local_phase1=request.get("local_phase1"), **common)
    if args.verify:
        verification = read_json(run_dir / "phase5_verification.json")
    else:
        write_json(run_dir / "server_result.json", server_result)
        verification = {"overall_verdict": "UNVERIFIED (single-pass)"}
    rows, summary = build_function_rows(path, source, run_id, server_result, verification)
    return run_id, rows, summary


# ──────────────────────────────────────────────────────────────────────────────
# Commands
# ──────────────────────────────────────────────────────────────────────────────
TARGETED_STATUSES = ("applied", "unchanged_by_model", "rejected", "not_patched")


def progress_logger(batch_dir: Path):
    """Print a message and append it, timestamped, to <batch>/progress.log."""
    path = batch_dir / "progress.log"

    def log(message: str) -> None:
        print(f"[batch] {message}", flush=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")

    return log


def function_counts(db: sqlite3.Connection) -> dict[str, int]:
    return {
        status: n for status, n in db.execute(
            "SELECT json_extract(row_json, '$.patch_status'), COUNT(*) FROM functions GROUP BY 1"
        )
    }


def totals_text(db: sqlite3.Connection, target: int | None, snapshot_every: int | None = None) -> str:
    """Cumulative totals; progress is shown toward the target, else toward the next snapshot."""
    state = counts(db)
    funcs = function_counts(db)
    done = state.get("done", 0)
    avg = db.execute("SELECT AVG(elapsed_s) FROM contracts WHERE status='done'").fetchone()[0]
    if target:
        goal = target
    elif snapshot_every:
        goal = (done // snapshot_every + 1) * snapshot_every
    else:
        goal = done + state.get("pending", 0) + state.get("in_progress", 0)
    text = (
        f"TOTAL {done}/{goal} contracts (failed {state.get('failed', 0)}) | "
        f"{sum(funcs.values())} function rows, {sum(funcs.get(s, 0) for s in TARGETED_STATUSES)} targeted: "
        f"applied {funcs.get('applied', 0)}, unchanged {funcs.get('unchanged_by_model', 0)}, "
        f"rejected {funcs.get('rejected', 0)}"
    )
    if avg and goal > done:
        left = (goal - done) * avg
        left_text = f"{left / 3600:.1f} GPU-h" if left >= 3600 else f"{left / 60:.0f} min"
        text += f" | avg {avg:.0f}s/contract, ~{left_text} left"
    return text


def command_run(args: argparse.Namespace) -> None:
    batch_dir = BATCH_ROOT / args.batch
    db = connect(batch_dir)
    log = progress_logger(batch_dir)
    parse_active_hours(args.active_hours)  # fail fast on a malformed window
    mode = "verify" if args.verify else "single_pass"
    stored = db.execute("SELECT value FROM meta WHERE key='mode'").fetchone()
    if stored is None and counts(db).get("done"):
        stored = {"value": "verify"}  # batches created before single-pass existed
    if stored is not None and stored["value"] != mode:
        sys.exit(f"[batch] Batch '{args.batch}' was run in '{stored['value']}' mode; this run is '{mode}'. "
                 "Use a new --batch name so the two kinds of results are not mixed.")
    db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('mode', ?)", (mode,))
    db.commit()
    if args.input_dir:
        added = register_contracts(db, Path(args.input_dir), args.shuffle, args.seed)
        if added:
            log(f"Registered {added} new contract(s).")
    reset = db.execute("UPDATE contracts SET status='pending' WHERE status='in_progress'").rowcount
    db.commit()
    if reset:
        log(f"Re-queued {reset} contract(s) interrupted in a previous session.")

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    stop_file = batch_dir / "STOP"
    stop_file.unlink(missing_ok=True)
    runs_dir = batch_dir / "runs"
    processed = 0
    finished_this_session = 0
    consecutive_outages = 0
    log(
        f"Session started: mode={mode}, target={args.target_done or 'all'}, "
        f"active_hours={args.active_hours or 'any'}, wait={args.wait}, export_every={args.export_every}, "
        f"snapshot_every={args.snapshot_every or 'off'}."
    )
    log(totals_text(db, args.target_done, args.snapshot_every))
    ensure_snapshots(db, args, log)

    while not _stop_requested and (args.limit is None or processed < args.limit):
        if stop_file.exists():
            log("STOP file found: exiting.")
            break
        if args.target_done and counts(db).get("done", 0) >= args.target_done:
            log(f"Target reached: {args.target_done} contracts done.")
            break
        row = db.execute(
            "SELECT seq, file, attempts FROM contracts WHERE status='pending' ORDER BY seq LIMIT 1"
        ).fetchone()
        if row is None:
            log("No pending contracts left.")
            break
        try:
            if not wait_until_runnable(args, stop_file, log):
                break
        except KeyboardInterrupt:
            break

        path = Path(row["file"])
        started = time.perf_counter()
        db.execute(
            "UPDATE contracts SET status='in_progress', started_at=?, attempts=attempts+1 WHERE seq=?",
            (utc_now(), row["seq"]),
        )
        db.commit()
        print(f"[batch] #{row['seq']} {path.name} ...", flush=True)
        try:
            run_id, rows, summary = audit_contract(path, runs_dir, args)
        except KeyboardInterrupt:
            db.execute("UPDATE contracts SET status='pending', attempts=attempts-1, last_error=? WHERE seq=?",
                       ("re-queued: interrupted", row["seq"]))
            db.commit()
            log(f"Aborted #{row['seq']} {path.name}: re-queued.")
            break
        except ServerOutage as error:
            if _server_ready():
                # The server is healthy, so this contract itself keeps timing out (e.g. huge
                # prompts): count the attempt so it ends as `failed` instead of looping forever.
                exhausted = row["attempts"] + 1 >= args.max_contract_attempts
                db.execute("UPDATE contracts SET status=?, last_error=? WHERE seq=?",
                           ("failed" if exhausted else "pending", f"server errors on this contract: {error}", row["seq"]))
            else:
                db.execute("UPDATE contracts SET status='pending', attempts=attempts-1, last_error=? WHERE seq=?",
                           (f"re-queued (server offline): {error}", row["seq"]))
            db.commit()
            consecutive_outages += 1
            log(f"SERVER PROBLEM #{row['seq']} {path.name}: {str(error)[:200]}")
            if consecutive_outages >= 3 and not args.wait:
                log("3 consecutive server failures: stopping.")
                break
            continue
        except Exception as error:  # a bug or an unparsable contract: record and move on
            status = "failed" if row["attempts"] + 1 >= args.max_contract_attempts else "pending"
            db.execute(
                "UPDATE contracts SET status=?, finished_at=?, last_error=? WHERE seq=?",
                (status, utc_now(), f"{type(error).__name__}: {error}\n{traceback.format_exc()[-2000:]}", row["seq"]),
            )
            db.commit()
            log(f"ERROR ({status}) #{row['seq']} {path.name}: {type(error).__name__}: {str(error)[:200]}")
            processed += 1
            continue

        consecutive_outages = 0
        elapsed = round(time.perf_counter() - started, 1)
        summary["elapsed_s"] = elapsed
        db.execute("DELETE FROM functions WHERE contract_file=?", (path.name,))
        db.executemany(
            "INSERT OR REPLACE INTO functions (id, contract_file, row_json) VALUES (?, ?, ?)",
            [(item["id"], path.name, json.dumps(item, ensure_ascii=False)) for item in rows],
        )
        db.execute(
            "UPDATE contracts SET status='done', run_id=?, finished_at=?, elapsed_s=?, last_error=NULL, "
            "summary_json=? WHERE seq=?",
            (run_id, utc_now(), elapsed, json.dumps(summary, ensure_ascii=False), row["seq"]),
        )
        db.commit()
        processed += 1
        finished_this_session += 1
        log(
            f"DONE #{row['seq']} {path.name} {elapsed:.0f}s: {summary['units']} units, "
            f"{summary['target_functions']} targets -> applied {summary['patched_functions']}, "
            f"unchanged {summary['unchanged_functions']}"
            + (f", pipeline-PASS {summary['pipeline_pass_functions']}" if args.verify else "")
        )
        log(totals_text(db, args.target_done, args.snapshot_every))
        ensure_snapshots(db, args, log)
        if args.export_every and finished_this_session % args.export_every == 0:
            export_batch(args.batch, quiet=True)
            log(f"CSV refreshed ({finished_this_session} new contract(s) this session).")
        if args.sleep_between:
            time.sleep(args.sleep_between)

    if finished_this_session:
        export_batch(args.batch, quiet=True)
    log(f"Session ended: {finished_this_session} contract(s) finished this session.")
    log(totals_text(db, args.target_done, args.snapshot_every))


def command_status(args: argparse.Namespace) -> None:
    db = connect(BATCH_ROOT / args.batch)
    stored = db.execute("SELECT value FROM meta WHERE key='mode'").fetchone()
    target = getattr(args, "target_done", None)
    step = getattr(args, "snapshot_every", None)
    print(f"[batch:{args.batch}] mode={stored['value'] if stored else '?'}")
    print(f"[batch:{args.batch}] contracts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts(db).items())))
    print(f"[batch:{args.batch}] functions: " + ", ".join(f"{k}={v}" for k, v in sorted(function_counts(db).items())))
    print(f"[batch:{args.batch}] {totals_text(db, target, step)}")
    for row in db.execute("SELECT file, last_error FROM contracts WHERE status='failed' LIMIT 5"):
        print(f"  failed: {Path(row['file']).name}: {str(row['last_error']).splitlines()[0]}")
    snapshots = sorted(BATCH_ROOT.joinpath(args.batch).glob(f"{args.batch}_first*_functions.csv"))
    for path in snapshots:
        print(f"  snapshot: {path}")


def snapshot_path(batch: str, first: int) -> Path:
    return BATCH_ROOT / batch / f"{batch}_first{first}_functions.csv"


def export_batch(batch: str, out: str | None = None, quiet: bool = False, first: int | None = None) -> Path:
    """Write the CSV/JSONL files atomically, so a file opened mid-run is never half-written.

    first=N exports only the first N contracts that finished (a fixed snapshot of the batch).
    Snapshots are CSV only: the JSONL twin holds the same rows, so only the live export keeps it.
    """
    batch_dir = BATCH_ROOT / batch
    db = connect(batch_dir)
    out_path = Path(out) if out else (snapshot_path(batch, first) if first else batch_dir / f"{batch}_functions.csv")
    summary_path = out_path.with_name(out_path.stem.replace("_functions", "") + "_contracts.csv")
    jsonl_path = out_path.with_suffix(".jsonl")
    rows = [json.loads(item["row_json"]) for item in db.execute("SELECT row_json FROM functions ORDER BY rowid")]
    contract_query: tuple[str, list] = (
        "SELECT * FROM contracts WHERE status != 'pending' OR last_error IS NOT NULL ORDER BY seq", []
    )
    if first:
        chosen = [item["file"] for item in db.execute(
            "SELECT file FROM contracts WHERE status='done' ORDER BY finished_at, seq LIMIT ?", (first,)
        )]
        names = {Path(file).name for file in chosen}
        rows = [row for row in rows if row.get("contract_file") in names]
        placeholders = ",".join("?" * len(chosen)) or "NULL"
        contract_query = (f"SELECT * FROM contracts WHERE file IN ({placeholders}) ORDER BY finished_at, seq", chosen)

    def write_atomic(path: Path, writer_fn) -> None:
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer_fn(handle)
        tmp.replace(path)

    def write_functions(handle) -> None:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    def write_summary(handle) -> None:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for item in db.execute(*contract_query):
            summary = json.loads(item["summary_json"]) if item["summary_json"] else {}
            writer.writerow({
                **summary,
                "contract_file": Path(item["file"]).name,
                "status": item["status"],
                "elapsed_s": item["elapsed_s"],
                "error": (item["last_error"] or "").splitlines()[0] if item["last_error"] else "",
            })

    def write_jsonl(handle) -> None:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    write_atomic(out_path, write_functions)
    write_atomic(summary_path, write_summary)
    if not first:
        write_atomic(jsonl_path, write_jsonl)
    if not quiet:
        print(f"[batch] {len(rows)} function row(s) -> {out_path}")
        print(f"[batch] contract summary -> {summary_path}")
        if not first:
            print(f"[batch] same rows as JSONL -> {jsonl_path}")
    return out_path


def ensure_snapshots(db: sqlite3.Connection, args: argparse.Namespace, log) -> None:
    """Write `<batch>_first{N}_*` once for every multiple N of --snapshot-every reached.

    The last milestone written is kept in `meta`, so a snapshot deleted to free disk space is
    not rebuilt on the next start (`export --first N` re-creates one on demand).
    """
    step = args.snapshot_every
    if not step:
        return
    done = counts(db).get("done", 0)
    row = db.execute("SELECT value FROM meta WHERE key = 'last_snapshot'").fetchone()
    last = int(row["value"]) if row else 0
    for milestone in range((last // step + 1) * step, done + 1, step):
        if not snapshot_path(args.batch, milestone).exists():
            path = export_batch(args.batch, quiet=True, first=milestone)
            log(f"SNAPSHOT: first {milestone} contracts -> {path.name} (the run continues).")
        db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('last_snapshot', ?)", (str(milestone),))
        db.commit()


def command_export(args: argparse.Namespace) -> None:
    export_batch(args.batch, args.out, first=args.first)


def command_retry_failed(args: argparse.Namespace) -> None:
    db = connect(BATCH_ROOT / args.batch)
    changed = db.execute("UPDATE contracts SET status='pending', attempts=0 WHERE status='failed'").rowcount
    db.commit()
    print(f"[batch] {changed} failed contract(s) re-queued.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resumable batch audit over the GPU server pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Process pending contracts (resumable).")
    run.add_argument("--batch", required=True, help="Batch name (folder under E2E/batch_runs/).")
    run.add_argument("--input-dir", help="Folder of .sol files to register (only needed the first time).")
    run.add_argument("--limit", type=int, help="Process at most N contracts in this session.")
    run.add_argument("--shuffle", action="store_true", help="Register contracts in a seeded random order.")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--wait", action="store_true", help="When the server is down, wait for it instead of exiting.")
    run.add_argument("--poll-seconds", type=int, default=60)
    run.add_argument("--target-done", type=int, default=None,
                     help="Stop once this many contracts are done in the batch (across all sessions).")
    run.add_argument("--active-hours", default=None,
                     help="Only start contracts inside this local time window, e.g. 19:00-07:30 (may cross midnight).")
    run.add_argument("--export-every", type=int, default=5,
                     help="Refresh the CSV/JSONL every N finished contracts (0 = only at session end).")
    run.add_argument("--snapshot-every", type=int, default=0,
                     help="Also write a fixed <batch>_first<N>_* export at every N done contracts, without stopping.")
    run.add_argument("--sleep-between", type=float, default=2.0, help="Pause between contracts (GPU breathing room).")
    run.add_argument("--max-target-units", type=int, default=None,
                     help="Repair at most N functions per contract (strongest evidence first).")
    run.add_argument("--max-fix-attempts", type=int, default=3, help="Only used with --verify.")
    run.add_argument(
        "--verify",
        action="store_true",
        help="Run the full verify-and-retry loop (compiler, detector, static/RAG, tool matrix). "
        "Default: single pass, one fix per vulnerable function, accepted as the final patch.",
    )
    run.add_argument(
        "--detector-only-min-prob",
        type=float,
        default=1.01,
        help="Repair functions flagged ONLY by the neural detector (no SWC) when its probability is at "
        "least this value. Default 1.01 = never (they are exported as detector_only_not_repaired); "
        "e.g. 0.9 repairs the strongest ones, 0 repairs all.",
    )
    run.add_argument("--max-contract-attempts", type=int, default=2,
                     help="Mark a contract failed after N crashes of our own code.")
    run.add_argument("--detector-threshold", type=float, default=None)

    for name, helptext in (("status", "Show progress."), ("retry-failed", "Re-queue failed contracts.")):
        command = sub.add_parser(name, help=helptext)
        command.add_argument("--batch", required=True)
        if name == "status":
            command.add_argument("--target-done", type=int, default=None)
            command.add_argument("--snapshot-every", type=int, default=None)
    export = sub.add_parser("export", help="Write the function CSV (+ contract summary CSV and JSONL).")
    export.add_argument("--batch", required=True)
    export.add_argument("--out")
    export.add_argument("--first", type=int, default=None,
                        help="Only the first N finished contracts, written as <batch>_first<N>_*.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    {
        "run": command_run,
        "status": command_status,
        "export": command_export,
        "retry-failed": command_retry_failed,
    }[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
