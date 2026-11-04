#!/usr/bin/env python3
"""
E2E/experiments/repair_round2.py  -  ROUND 2 (scripts/run_round2.sh)
──────────────────────────────────────────────────────────────────────────────
Second repair round for the functions the round-1 evaluation sent back
(`data/results/round1_judge/ham_can_sua_lan_2.csv`), resumable like batch_audit.

What the system sees is only its own output: no column written by the external
judge (claude_*) reaches a prompt. The 18 functions whose round-1 patch introduced a new
problem are not part of the main group (exported there as `excluded`); they run as their own
group (`--group regress`, batch round2_regress) through the same flow, reported separately. Per function:

  1. Re-scan (local, no GPU): static + RAG re-run on the round-1 patched contract,
     scoped to this function -> findings still reported / newly introduced.
  2. Advisor (LLM, every finding, no call cap): triage + root cause + fix strategy for
     each finding of the function (round-1 confirmed + round-1 dismissed), shown the
     round-1 attempt and the re-scan so it can say what the attempt missed.
  3. Fixer (LLM, one call): repairs the ORIGINAL function from the Advisor diagnosis,
     the round-1 attempt, the re-scan and the behaviour-guard note; a second call (without the
     round-1 code) when the reply copies the round-1 patch or leaves the function unchanged.
  4. Self-check (--self-check, local, no LLM): the reply is compiled with the original contract's solc,
     re-scanned (static + RAG) and run through the behaviour guard; a problem the ORIGINAL contract did
     not have (compile error, new finding, lost ETH transfer / event, removed sha256) gets one more fixer
     call with exactly those problems, and the reply with fewer problems is kept.
  No verifier: the fixer output is the round-2 result (an external judge scores the CSV).

Round-1 artifacts are reused from data/results/round1/runs/<run_id>/ (Phase 0/1 are not re-run).

Usage (from the repository root, or via scripts/run_round2.sh):
  .venv/bin/python -m E2E.experiments.repair_round2 prepare            # re-scan only, works with the server off
  .venv/bin/python -m E2E.experiments.repair_round2 preview --stt 1    # print the prompts of one function
  .venv/bin/python -m E2E.experiments.repair_round2 run --wait
  .venv/bin/python -m E2E.experiments.repair_round2 status
  .venv/bin/python -m E2E.experiments.repair_round2 export [--blind]
  .venv/bin/python -m E2E.experiments.repair_round2 retry-failed
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import re
import signal
import sqlite3
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from E2E.common.gpu_client import check_server_health
from E2E.common.io_utils import read_json, utc_now, write_json
from E2E.common.solc import compile_solidity_code, get_installed_solc_versions, select_best_solc
from E2E.config import BATCH_RUNS_DIR, DEFAULT_DETECTOR_THRESHOLD, ROUND1_JUDGE_DIR
from E2E.experiments.batch_audit import OUTAGE_MARKERS, parse_active_hours, progress_logger, within_active_hours
from E2E.phase0_parsing.code_units import CodeUnit, parse_code_units
from E2E.phase1c_detector.detector import score_units_via_server
from E2E.phase2_advisor.advisor import AdvisorAgent, RepairKnowledgeBase
from E2E.phase4_fixer.behavior_guard import validate_observable_behavior
from E2E.phase4_fixer.fixer import _build_fixer_prompt, _build_rag_knowledge_section, _splice_unit, _version_rules, patch_unit
from E2E.phase5_verifier.candidate_check import _finding_pairs
from E2E.phase5_verifier.final_verification import rerun_local_channels

DEFAULT_INPUT = ROUND1_JUDGE_DIR / "ham_can_sua_lan_2.csv"
DEFAULT_ROUND1_RUNS = BATCH_RUNS_DIR / "round1" / "runs"
# Round-1 patches the judge found to introduce a new problem: counted as failed, not re-run.
EXCLUDED_LABEL = "Đã sửa lỗi được báo nhưng bản sửa phát sinh lỗi mới"

# Round-1 judge verdicts: kept for the before/after table, never shown to the system.
JUDGE_COLUMNS = ["lan1_khac_phuc", "lan1_an_toan_hon", "lan1_ly_do_chon"]
CSV_COLUMNS = [
    "stt_goc",
    "id",
    "contract_name",
    "contract_file",
    "function_id",
    "run_id",
    "raw_contract",
    "raw_function_code",
    "related_origin_vulnerability",
    "swc_ids",
    "evidence_sources",
    "detector_prob_before",
    # round 1
    "fixed_function_code_lan1",
    "fix_rationale_lan1",
    "patch_note_lan1",
    *JUDGE_COLUMNS,
    # context the system built for round 2
    "rescan_lan1_residual",
    "rescan_lan1_introduced",
    "advisor_lan2_true_positive",
    "advisor_lan2_false_positive",
    "advisor_lan2_diagnosis",
    "advisor_all_dismissed",
    # round 2
    "fixed_function_code_lan2",
    "patch_status_lan2",
    "fix_rationale_lan2",
    "patch_note_lan2",
    "changed_vs_lan1",
    "fixer_calls_lan2",
    "prompt_trimmed_lan2",
    # --self-check (system-only regression check of the round-2 reply against the original contract)
    "selfcheck_compile_lan2",
    "selfcheck_issues_first",
    "selfcheck_retry",
    "selfcheck_issues_final",
    "rescan_lan2_residual",
    "detector_prob_lan1",
    "detector_prob_lan2",
    "round2_status",
    "round2_error",
    "round2_updated_at",
]
BLIND_DROP = set(JUDGE_COLUMNS)

_stop_requested = False


class ServerOutage(RuntimeError):
    """The GPU server failed mid-function: the function is re-queued, never saved."""


def _request_stop(signum, frame) -> None:  # noqa: ARG001
    global _stop_requested
    if _stop_requested:
        raise KeyboardInterrupt
    _stop_requested = True
    print("\n[round2] Stop requested: finishing the current function, then exiting "
          "(press Ctrl+C again to abort it now; it will be re-queued).", flush=True)


# The GPU server rejects prompts over 4096 tokens with HTTP 413, which the client wraps as "ConnectionError":
# retrying cannot help, so it must not count as an outage (that re-queued #239 forever).
_TOO_LARGE_MARKERS = ("http error 413", "content too large", "context length exceeded")
# A function that hits "outages" this many times in a row while /health answers ready is failing on its own.
OUTAGE_REPEAT_LIMIT = 5


def _too_large(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in _TOO_LARGE_MARKERS)


def _is_outage(text: str) -> bool:
    lowered = (text or "").lower()
    return not _too_large(lowered) and any(marker in lowered for marker in OUTAGE_MARKERS)


# ──────────────────────────────────────────────────────────────────────────────
# Input + state
# ──────────────────────────────────────────────────────────────────────────────
def read_input(path: Path) -> dict[int, dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["stt_goc"]): row for row in csv.DictReader(handle)}


def connect(batch_dir: Path) -> sqlite3.Connection:
    batch_dir.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(batch_dir / "state.sqlite")
    db.row_factory = sqlite3.Row
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS functions (
            stt INTEGER PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            advisor_json TEXT,
            result_json TEXT,
            error TEXT,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
        """
    )
    db.execute("UPDATE functions SET status = 'pending' WHERE status = 'in_progress'")
    db.commit()
    return db


def lock_options(db: sqlite3.Connection, args: argparse.Namespace) -> None:
    """A batch keeps the options it was started with, so every row is produced the same way."""
    wanted = {"score_detector": str(bool(args.score_detector)), "on_all_dismissed": args.on_all_dismissed,
              "group": args.group, "self_check": str(bool(args.self_check))}
    stored = {row["key"]: row["value"] for row in db.execute("SELECT key, value FROM meta")}
    if not stored:
        db.executemany("INSERT INTO meta VALUES (?, ?)", wanted.items())
        db.commit()
        return
    # Batches started before these options existed ran with their defaults.
    for key, default in (("group", "main"), ("self_check", "False")):
        if key not in stored:
            stored[key] = default
            db.execute("INSERT INTO meta VALUES (?, ?)", (key, default))
    db.commit()
    changed = {key: (stored.get(key), value) for key, value in wanted.items() if stored.get(key) != value}
    if changed:
        raise SystemExit(f"Batch was started with different options {changed}; use another --batch.")


def counts(db: sqlite3.Connection) -> dict[str, int]:
    return {row["status"]: row["n"] for row in db.execute(
        "SELECT status, COUNT(*) AS n FROM functions GROUP BY status")}


# ──────────────────────────────────────────────────────────────────────────────
# Round-1 artifacts and the local re-scan
# ──────────────────────────────────────────────────────────────────────────────
_round1_cache: dict[str, dict[str, Any]] = {}


def load_round1(runs_dir: Path, run_id: str) -> dict[str, Any]:
    if run_id not in _round1_cache:
        run_dir = runs_dir / run_id
        _round1_cache.clear()  # rows are grouped by contract: one contract in memory is enough
        _round1_cache[run_id] = {
            "server": read_json(run_dir / "server_result.json"),
            "static": read_json(run_dir / "phase1a_static.json"),
            "rag": read_json(run_dir / "phase1b_rag.json"),
        }
    return _round1_cache[run_id]


def _compact(findings: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    return [
        {"swc": item.get("swc"), "unit_id": item.get("unit_id"), "title": item.get("title") or item.get("name"),
         "line": item.get("line"), "source": source}
        for item in findings
        if item.get("swc") and item.get("unit_id")
    ]


def get_rescan(batch_dir: Path, runs_dir: Path, run_id: str) -> dict[str, Any]:
    """Static + RAG on the round-1 patched contract (cached per contract; local only)."""
    path = batch_dir / "rescan" / f"{run_id}.json"
    if path.exists():
        return read_json(path)
    round1 = load_round1(runs_dir, run_id)
    fixed_code = round1["server"].get("fixed_code") or ""
    static, rag = rerun_local_channels(fixed_code)
    original = round1["static"].get("findings", []) + round1["static"].get("candidates", []) \
        + round1["rag"].get("findings", [])
    result = {
        "run_id": run_id,
        "static_status": static.get("status"),
        "rag_status": rag.get("status"),
        "findings": _compact(static.get("findings", []), "static") + _compact(rag.get("findings", []), "rag"),
        "original_pairs": sorted(_finding_pairs(original)),
        "created_at": utc_now(),
    }
    write_json(path, result)
    return result


def rescan_for_unit(rescan: dict[str, Any], unit_id: str, target_swcs: set[str]) -> tuple[list, list]:
    known = {tuple(pair) for pair in rescan.get("original_pairs", [])}
    seen: set[str] = set()
    residual, introduced = [], []
    for item in rescan.get("findings", []):
        if item["unit_id"] != unit_id or item["swc"] in seen:
            continue
        seen.add(item["swc"])
        if item["swc"] in target_swcs:
            residual.append(item)
        elif (item["swc"], unit_id) not in known:
            introduced.append(item)
    return residual, introduced


def _fmt_rescan(items: list[dict[str, Any]]) -> str:
    return "; ".join(f"{item['swc']} {item.get('title') or ''} [{item['source']}]".strip() for item in items)


# ──────────────────────────────────────────────────────────────────────────────
# Advisor with the previous attempt in view
# ──────────────────────────────────────────────────────────────────────────────
# Round 1 showed the 14B Advisor applying this rule to every finding class and citing modifiers
# that are not in the function, so round 2 narrows it.
_ROUND1_RULE_2 = (
    "2. If the function is protected by an authorization modifier (e.g. onlyOwner, oO) or by "
    "`if (msg.sender != ...) revert/throw`, an access-control finding is a FALSE POSITIVE (`is_true_positive: false`)."
)
_ROUND2_RULE_2 = (
    "2. An authorization modifier (e.g. onlyOwner) or `if (msg.sender != ...) revert/throw` makes ONLY an "
    "access-control finding a FALSE POSITIVE. It does NOT make reentrancy, DoS / gas-limit, front-running, "
    "unchecked return value, arithmetic, timestamp, deprecated-syntax or zero-address findings false positives. "
    "Cite a guard only if it appears verbatim in <TargetFunction>."
)
_GUARD_RE = re.compile(r"\b(only[A-Z]\w*|nonReentrant|whenNotPaused|whenPaused|ownerOnly|isOwner)\b")
_ACCESS_CONTROL_SWCS = {"SWC-105", "SWC-106", "SWC-115"}


def _is_access_control(finding: dict[str, Any]) -> bool:
    title = str(finding.get("title") or "").lower()
    return finding.get("swc") in _ACCESS_CONTROL_SWCS or any(
        word in title for word in ("access control", "authorization", "unprotected"))


def ungrounded_fp_reason(finding: dict[str, Any], root_cause: str) -> str:
    """Why a false-positive verdict that rests on an authorization guard cannot stand ('' if it can)."""
    cited = set(_GUARD_RE.findall(root_cause or ""))
    if not cited:
        return ""
    code = str(finding.get("unit_code") or "")
    if not any(name in code for name in cited):
        return f"cites {', '.join(sorted(cited))}, which is not in the function"
    if not _is_access_control(finding):
        return f"an authorization guard does not rule out {finding.get('swc')}"
    return ""


class Round2Advisor(AdvisorAgent):
    previous_attempt: str = ""
    previous_attempt_short: str = ""  # without the code, used when the full prompt is too large

    def _build_prompt(self, code: str, vulnerability: dict[str, Any], knowledge_block: str) -> str:
        prompt = super()._build_prompt(code, vulnerability, knowledge_block)
        if _ROUND1_RULE_2 not in prompt:
            raise RuntimeError("Advisor prompt changed: update _ROUND1_RULE_2 in repair_round2.py")
        prompt = prompt.replace(_ROUND1_RULE_2, _ROUND2_RULE_2)
        if not self.previous_attempt:
            return prompt
        return prompt + (
            "\n\n<PreviousRepairAttempt>\n"
            "An earlier automated patch of this function was rejected in review. It is shown only so you can "
            "tell what it missed or broke; your verdict must be about <TargetFunction> (the original code). "
            "In `fix_strategy`, also state what the previous attempt missed or broke, if anything.\n"
            f"{self.previous_attempt}\n"
            "</PreviousRepairAttempt>"
        )


def previous_attempt_block(job: dict[str, Any], residual: list, introduced: list, show_code: bool = True) -> str:
    code, raw = job["fixed_lan1"].strip(), job["raw_function_code"].strip()
    if not show_code:
        lines = []
    elif len(code) > 3 * len(raw) + 1000:
        # e.g. #239: round 1 spliced 54 functions (26k chars) in place of `core`; it overflows the prompt limit.
        lines = [f"(Not shown: that patch replaced the function with {len(code)} characters of code, "
                 f"{len(code) / max(len(raw), 1):.0f}x the original, i.e. it pasted unrelated code instead of a "
                 "clean replacement of this one function.)"]
    else:
        lines = [f"```solidity\n{code}\n```"]
    if residual:
        lines.append(f"Automated re-scan of that patch still reports: {_fmt_rescan(residual)}")
    if introduced:
        lines.append(f"Automated re-scan of that patch reports NEW findings: {_fmt_rescan(introduced)}")
    if job.get("patch_note_lan1"):
        lines.append(f"Behaviour guard on that patch: {job['patch_note_lan1']}")
    return "\n".join(lines)


def run_advisor(agent: Round2Advisor, job: dict[str, Any], findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    full = agent.previous_attempt
    for finding in findings:
        suggestion = agent.advise_one(job["raw_contract"], finding)
        trimmed = ""
        for fallback in (agent.previous_attempt_short, ""):
            if not (suggestion.error and _too_large(suggestion.error)) or not agent.previous_attempt:
                break
            agent.previous_attempt = fallback
            trimmed = "previous code dropped" if fallback else "previous attempt dropped"
            suggestion = agent.advise_one(job["raw_contract"], finding)
        agent.previous_attempt = full
        if suggestion.error and _is_outage(suggestion.error):
            raise ServerOutage(suggestion.error)
        override = "" if suggestion.is_true_positive else ungrounded_fp_reason(finding, suggestion.root_cause)
        results.append({
            "prompt_trimmed": trimmed,
            "swc": finding.get("swc"),
            "title": finding.get("title"),
            "is_true_positive": bool(suggestion.is_true_positive) or bool(override),
            "fp_overridden": override,
            "root_cause": suggestion.root_cause,
            "fix_strategy": suggestion.fix_strategy,
            "attack_scenario": suggestion.attack_scenario,
            "evidence_lines": suggestion.evidence_lines,
            "error": suggestion.error,
        })
    return results


# ──────────────────────────────────────────────────────────────────────────────
# One function
# ──────────────────────────────────────────────────────────────────────────────
def is_excluded(row: dict[str, str]) -> bool:
    return row.get("claude_ly_do_chon") == EXCLUDED_LABEL


def build_job(row: dict[str, str]) -> dict[str, Any]:
    """The only round-1 fields the system may use. claude_* columns stay out of here."""
    job = {
        "stt": int(row["stt_goc"]),
        "run_id": row["run_id"],
        "function_id": row["function_id"],
        "raw_contract": row["raw_contract"],
        "raw_function_code": row["raw_function_code"],
        "fixed_lan1": row["fixed_function_code"],
        "fix_rationale_lan1": row.get("fix_rationale") or "",
        "patch_note_lan1": row.get("patch_note") or "",
        "swc_ids": [item.strip() for item in row["swc_ids"].split(",") if item.strip()],
    }
    return job


def target_findings(round1: dict[str, Any], job: dict[str, Any], unit: CodeUnit) -> list[dict[str, Any]]:
    """Every finding reported for the function in round 1 (confirmed + Advisor-dismissed), one per SWC."""
    detector = round1["server"].get("phases", {}).get("detector", {})
    found: dict[str, dict[str, Any]] = {}
    for item in [*detector.get("confirmed_findings", []), *detector.get("dismissed_findings", [])]:
        if item.get("unit_id") == job["function_id"] and item.get("swc") and item["swc"] not in found:
            found[item["swc"]] = {
                **item,
                "unit_code": item.get("unit_code") or unit.code,
                "start_line": item.get("start_line") or unit.start_line,
                "end_line": item.get("end_line") or unit.end_line,
            }
    return list(found.values())


def kb_guidance(swcs: list[str], unit_code: str) -> str:
    """Knowledge-base remediation per SWC (the same retrieval the Advisor sees); round 1 gave the fixer titles only."""
    kb = _knowledge_base()
    blocks = []
    for swc in swcs:
        retrieved = kb.retrieve(swc, code=unit_code)
        lines = [f"- [{swc}] " + " | ".join(item[:300].replace("\n", " ") for item in retrieved.get("remediations", [])[:3])]
        pair = retrieved.get("best_pair") or {}
        vulnerable, fixed = str(pair.get("vulnerable_code") or "").strip(), str(pair.get("fixed_code") or "").strip()
        if vulnerable and fixed and len(vulnerable) <= 700 and len(fixed) <= 700:
            lines.append(f"  Reference vulnerable pattern:\n```solidity\n{vulnerable}\n```\n"
                         f"  Reference fixed pattern:\n```solidity\n{fixed}\n```")
        blocks.append("\n".join(lines))
    if not blocks:
        return ""
    return ("Knowledge-base remediation guidance (from similar audited cases; adapt it, do not copy names):\n"
            + "\n".join(blocks))


def general_rules(raw_code: str) -> str:
    if re.search(r"\b(sha256|ripemd160)\s*\(", raw_code):
        return "Only `sha3` is deprecated (-> keccak256); keep `sha256` / `ripemd160` calls unchanged."
    return ""


def build_feedback_note(tp: list[dict], fp: list[dict], job: dict[str, Any], residual: list, introduced: list,
                        show_previous_code: bool = True, guidance: str = "") -> str:
    parts = []
    diagnosis = []
    for item in tp:
        if item.get("fp_overridden"):
            # The Advisor's text argues the finding is harmless; passing it on tells the fixer to do nothing.
            diagnosis.append(f"- [{item['swc']}] Treat as a real vulnerability (the false-positive verdict was "
                             f"rejected: {item['fp_overridden']}).")
            continue
        text = f"- [{item['swc']}] Root cause: {item.get('root_cause') or 'n/a'}"
        if item.get("evidence_lines"):
            text += f"\n  Evidence lines: {item['evidence_lines']}"
        if item.get("attack_scenario") and item["attack_scenario"].lower() != "none":
            text += f"\n  Attack scenario: {item['attack_scenario']}"
        diagnosis.append(text)
    if diagnosis:
        parts.append("Advisor diagnosis of the original function:\n" + "\n".join(diagnosis))
    if fp:
        parts.append(
            "Findings the Advisor judged FALSE POSITIVE (do not change code for these):\n"
            + "\n".join(f"- [{item['swc']}] {item.get('root_cause') or ''}".rstrip() for item in fp)
        )
    if guidance:
        parts.append(guidance)
    if general_rules(job["raw_function_code"]):
        parts.append(general_rules(job["raw_function_code"]))
    if show_previous_code:
        parts.append(
            "Previous repair attempt (rejected in review). Repair the ORIGINAL declaration above; keep from this "
            "attempt only what is correct and do not repeat what it missed or broke. Your replacement must NOT be "
            "identical to this attempt:\n"
            + previous_attempt_block(job, residual, introduced)
        )
    else:
        # Retry after the fixer copied the rejected attempt: describe it instead of showing it.
        summary = job.get("fix_rationale_lan1") or "a minimal change"
        parts.append(
            f"A previous repair attempt was rejected in review; it only did this: {summary}. "
            "That is NOT enough: address the root cause of every finding listed above."
            + (f"\nAutomated re-scan of that attempt still reported: {_fmt_rescan(residual)}" if residual else "")
            + (f"\nAutomated re-scan of that attempt reported NEW findings: {_fmt_rescan(introduced)}"
               if introduced else "")
            + (f"\nBehaviour guard on that attempt: {job['patch_note_lan1']}" if job.get("patch_note_lan1") else "")
        )
    return "\n\n".join(parts)


def unit_rag_section(round1: dict[str, Any], swcs: set[str]) -> str:
    """Round 1 fed the contract-level RAG cases to every function; keep only cases of this function's SWCs."""
    context = [item for item in round1["rag"].get("retrieved_context", []) if item.get("swc") in swcs]
    return _build_rag_knowledge_section(context, target_swc=next(iter(sorted(swcs)), None))


def removed_hash_calls(raw_code: str, new_code: str) -> str:
    """sha256/ripemd160 are not deprecated; replacing them changes every hash the contract stores."""
    lost = [name for name in ("sha256", "ripemd160")
            if new_code and len(re.findall(rf"\b{name}\s*\(", new_code)) < len(re.findall(rf"\b{name}\s*\(", raw_code))]
    return f"removed {'/'.join(lost)} call(s)" if lost else ""


def unescape_reply(code: str) -> str:
    """Undo a JSON-escaped reply (literal \\n, \\" instead of newlines/quotes); seen once in round 2 (#239)."""
    if not code or code.count("\\n") <= code.count("\n"):
        return code
    try:
        return json.loads(f'"{code}"')
    except ValueError:
        return code.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')


def _same_code(left: str | None, right: str | None) -> bool:
    return bool(left) and "".join(left.split()) == "".join((right or "").split())


# ──────────────────────────────────────────────────────────────────────────────
# Self-check: system-only regression check of a reply against the ORIGINAL contract
# ──────────────────────────────────────────────────────────────────────────────
_compiler_cache: dict[str, tuple[Path | None, bool, str]] = {}


def _original_compiler(run_id: str, raw_contract: str) -> tuple[Path | None, bool, str]:
    """(solc, original compiles?, label). Without a pragma select_best_solc picks the newest solc, which rejects
    0.4 syntax; then the oldest installed version the original contract compiles with is used."""
    if run_id not in _compiler_cache:
        path, reason, _ = select_best_solc(raw_contract)
        passed = compile_solidity_code(raw_contract, solc_path=path).get("passed") is True
        if not passed and reason.startswith("default"):
            for _, candidate in get_installed_solc_versions():
                if compile_solidity_code(raw_contract, solc_path=candidate).get("passed") is True:
                    path, passed = candidate, True
                    break
        _compiler_cache.clear()
        _compiler_cache[run_id] = (path, passed, path.name if path else "solc not installed")
    return _compiler_cache[run_id]


def compiler_note(job: dict[str, Any]) -> str:
    """The fixer prompt takes its version rules from the pragma; without one, name the solc the original
    compiles with (#1130: no pragma, so the fixer wrote `require(x, "msg")`, which 0.4.11 rejects)."""
    if re.search(r"pragma\s+solidity", job["raw_contract"]):
        return ""
    _, original_ok, label = _original_compiler(job["run_id"], job["raw_contract"])
    version = re.search(r"\d+\.\d+\.\d+", label)
    rules = _version_rules(f"pragma solidity {version.group(0)};") if original_ok and version else ""
    return (f"The contract has no pragma; it compiles with solc {version.group(0)}, so your code must too:\n{rules}"
            if rules else "")


def _solc_errors(text: str) -> str:
    text = re.sub(r"(/\S+?\.sol|<stdin>):(\d+):(\d+):", r"line \2:", text or "")
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return " | ".join(lines[:4])[:400]


def self_check(job: dict[str, Any], unit: CodeUnit, reply: str, original_pairs: list) -> dict[str, Any]:
    """Problems the reply has that the original contract does not (local tools only, no LLM)."""
    issues: list[str] = []
    check: dict[str, Any] = {"issues": issues, "compile": "", "residual": [], "introduced": []}
    try:
        patched = _splice_unit(job["raw_contract"], unit, reply, "round-2 self-check")
    except ValueError as error:
        issues.append(f"the reply cannot replace the function in the contract: {error}")
        return check
    solc, original_ok, label = _original_compiler(job["run_id"], job["raw_contract"])
    if not original_ok:
        check["compile"] = f"not checked: the original contract does not compile ({label})"
    else:
        compiled = compile_solidity_code(patched, solc_path=solc)
        check["compile"] = f"{'passed' if compiled.get('passed') else 'FAILED'} ({label})"
        if not compiled.get("passed"):
            issues.append(f"does not compile with {label} (the original does): {_solc_errors(compiled.get('error'))}")
    try:
        validate_observable_behavior(unit.code, reply)
    except ValueError as error:
        issues.append(f"behaviour guard: {error}")
    if removed_hash_calls(job["raw_function_code"], reply):
        issues.append(f"{removed_hash_calls(job['raw_function_code'], reply)} (sha256/ripemd160 are not deprecated)")
    static, rag = rerun_local_channels(patched)
    findings = _compact(static.get("findings", []), "static") + _compact(rag.get("findings", []), "rag")
    residual, introduced = rescan_for_unit({"findings": findings, "original_pairs": original_pairs},
                                           job["function_id"], set(job["swc_ids"]))
    check.update(residual=residual, introduced=introduced,
                 rescan_status=f"static={static.get('status')}, rag={rag.get('status')}")
    if introduced:
        issues.append(f"re-scan reports findings the original function did not have: {_fmt_rescan(introduced)}")
    return check


def self_check_feedback(reply: str, check: dict[str, Any]) -> str:
    return (
        "Automated self-check of your previous reply (shown below) found problems that the ORIGINAL function "
        "does not have:\n" + "\n".join(f"- {issue}" for issue in check["issues"])
        + "\nRepair the ORIGINAL declaration again: fix the reported vulnerabilities AND avoid these problems.\n"
        f"Your previous reply:\n```solidity\n{reply.strip()}\n```"
    )


def _detector_prob(unit: CodeUnit | None, server_url: str | None) -> float | None:
    if unit is None:
        return None
    result = score_units_via_server([unit], DEFAULT_DETECTOR_THRESHOLD, server_url)
    if result and result[0].get("status") == "error":
        if _is_outage(result[0].get("error", "")):
            raise ServerOutage(result[0]["error"])
        return None
    return result[0].get("vulnerability_probability") if result else None


def _unit_in(code: str, unit_id: str) -> CodeUnit | None:
    return next((unit for unit in parse_code_units(code) if unit.unit_id == unit_id), None)


def process_function(job: dict[str, Any], cached_advisor: list | None, args: argparse.Namespace,
                     db: sqlite3.Connection, batch_dir: Path) -> dict[str, Any]:
    round1 = load_round1(args.round1_runs, job["run_id"])
    unit = _unit_in(job["raw_contract"], job["function_id"])
    if unit is None:
        raise ValueError(f"function {job['function_id']} not found in raw_contract")
    findings = target_findings(round1, job, unit)
    if not findings:
        raise ValueError("no round-1 finding recorded for this function")

    rescan = get_rescan(batch_dir, args.round1_runs, job["run_id"])
    residual, introduced = rescan_for_unit(rescan, job["function_id"], set(job["swc_ids"]))

    if cached_advisor is None:
        agent = Round2Advisor(knowledge_base=_knowledge_base())
        agent.previous_attempt = previous_attempt_block(job, residual, introduced)
        agent.previous_attempt_short = previous_attempt_block(job, residual, introduced, show_code=False)
        cached_advisor = run_advisor(agent, job, findings)
        db.execute("UPDATE functions SET advisor_json = ? WHERE stt = ?",
                   (json.dumps(cached_advisor, ensure_ascii=False), job["stt"]))
        db.commit()
    tp = [item for item in cached_advisor if item["is_true_positive"]]
    fp = [item for item in cached_advisor if not item["is_true_positive"]]
    all_dismissed = not tp
    if all_dismissed and args.on_all_dismissed == "fix_anyway":
        # The function was sent back as genuinely vulnerable, so a blanket FP verdict is overridden:
        # repair every finding, without the Advisor's (false-positive) reasoning in the prompt.
        repair_set, diagnosed, fp_note = cached_advisor, [], []
    else:
        repair_set, diagnosed, fp_note = tp, tp, fp

    result: dict[str, Any] = {
        "rescan_residual": residual,
        "rescan_introduced": introduced,
        "advisor": cached_advisor,
        "advisor_all_dismissed": all_dismissed,
    }
    if all_dismissed and args.on_all_dismissed == "keep_raw":
        result.update(patch_status="not_repaired_advisor_fp", fixed_code=job["raw_function_code"],
                      rationale="Advisor judged every finding a false positive.", guard_warning="")
    else:
        by_swc = {item["swc"]: item for item in findings}
        unit_findings = [{"swc": item["swc"], "title": by_swc[item["swc"]].get("title", "")} for item in repair_set]
        suggestions = [
            {"unit_id": unit.unit_id, "is_true_positive": True, "fix_strategy": item.get("fix_strategy"),
             "root_cause": item.get("root_cause")}
            for item in diagnosed
            if not item.get("fp_overridden") and (item.get("fix_strategy") or item.get("root_cause"))
        ]
        guidance = kb_guidance([item["swc"] for item in repair_set], unit.code)
        compiler = compiler_note(job) if args.self_check else ""
        rag_section = unit_rag_section(round1, {item["swc"] for item in repair_set})

        # Context dropped, in order, while the server answers 413 (prompt > 4096 tokens); kept for later calls.
        trim_steps = ["previous code dropped", "RAG cases dropped", "KB guidance dropped"]
        trim = 0

        def fix(show_previous_code: bool, extra: str = "") -> dict[str, Any]:
            nonlocal trim
            while True:
                _, outcome = patch_unit(
                    job["raw_contract"],
                    unit,
                    unit_findings=unit_findings,
                    unit_suggestions=suggestions,
                    source_code=job["raw_contract"],
                    feedback_note=build_feedback_note(diagnosed, fp_note, job, residual, introduced,
                                                      show_previous_code and trim < 1, guidance if trim < 3 else "")
                    + (f"\n\n{compiler}" if compiler else "") + (f"\n\n{extra}" if extra else ""),
                    rag_section=rag_section if trim < 2 else "",
                    attempt=2,
                    server_url=None,
                    single_pass=True,
                )
                error = outcome.get("error", "") if outcome.get("status") == "rejected" else ""
                if _too_large(error) and trim < len(trim_steps):
                    trim += 1
                    continue
                if _is_outage(error):
                    raise ServerOutage(error)
                restored = unescape_reply(outcome.get("fixed_code") or "")
                if restored != (outcome.get("fixed_code") or ""):
                    outcome = {**outcome, "fixed_code": restored, "unescaped": True}
                return outcome

        outcome = fix(show_previous_code=True)
        result["fixer_calls"] = 1
        hash_warning = removed_hash_calls(job["raw_function_code"], outcome.get("fixed_code") or "")
        if outcome.get("status") == "unchanged" or _same_code(outcome.get("fixed_code"), job["fixed_lan1"]):
            result["fixer_retry_reason"] = (
                "unchanged from the original" if outcome.get("status") == "unchanged" else "copied the round-1 patch")
            outcome = fix(show_previous_code=False)
            result["fixer_calls"] = 2
        elif hash_warning:
            result["fixer_retry_reason"] = hash_warning
            outcome = fix(show_previous_code=True, extra=(
                f"Your previous reply {hash_warning}; that changes the stored hashes and breaks the contract. "
                "Keep every sha256 / ripemd160 call exactly as in the original."))
            result["fixer_calls"] = 2
        if args.self_check and outcome.get("status") == "applied" and outcome.get("fixed_code"):
            first = self_check(job, unit, outcome["fixed_code"], rescan.get("original_pairs", []))
            final, result["selfcheck_retry"] = first, ""
            if first["issues"]:
                retry = fix(show_previous_code=False, extra=self_check_feedback(outcome["fixed_code"], first))
                result["fixer_calls"] += 1
                if retry.get("status") == "applied" and retry.get("fixed_code"):
                    second = self_check(job, unit, retry["fixed_code"], rescan.get("original_pairs", []))
                    kept = len(second["issues"]) <= len(first["issues"])
                    if kept:
                        outcome, final = retry, second
                    result["selfcheck_retry"] = (f"retried: kept the {'retry' if kept else 'first reply'} "
                                                 f"({len(first['issues'])} -> {len(second['issues'])} problem(s))")
                else:
                    result["selfcheck_retry"] = f"retried: retry {retry.get('status')}, kept the first reply"
            result["selfcheck_first"], result["selfcheck_final"] = first, final
        warnings = [outcome.get("guard_warning") or outcome.get("error") or "",
                    "reply was JSON-escaped; newlines/quotes restored" if outcome.get("unescaped") else "",
                    removed_hash_calls(job["raw_function_code"], outcome.get("fixed_code") or "")]
        result["fixer_prompt_trimmed"] = ", ".join(trim_steps[:trim])
        result.update(
            patch_status=outcome.get("status"),
            fixed_code=outcome.get("fixed_code") or "",
            rationale=outcome.get("rationale") or "",
            guard_warning="; ".join(item for item in warnings if item),
        )

    if args.score_detector:
        lan1_unit = _unit_in(round1["server"].get("fixed_code") or "", job["function_id"])
        result["detector_prob_lan1"] = _detector_prob(lan1_unit, None)
        lan2_unit = None
        if result["fixed_code"]:
            with contextlib.suppress(ValueError):
                lan2_unit = _unit_in(_splice_unit(job["raw_contract"], unit, result["fixed_code"], "round 2"),
                                     job["function_id"])
        result["detector_prob_lan2"] = _detector_prob(lan2_unit, None)
    return result


_kb: RepairKnowledgeBase | None = None


def _knowledge_base() -> RepairKnowledgeBase:
    global _kb
    if _kb is None:
        _kb = RepairKnowledgeBase()
    return _kb


# ──────────────────────────────────────────────────────────────────────────────
# Commands
# ──────────────────────────────────────────────────────────────────────────────
def wait_until_runnable(args: argparse.Namespace, stop_file: Path, log) -> bool:
    window = parse_active_hours(args.active_hours)
    waiting_for = None
    while not _stop_requested and not stop_file.exists():
        if not within_active_hours(window):
            reason = f"outside active hours {args.active_hours}"
        elif check_server_health().get("status") != "ready":
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


def in_group(row: dict[str, str], group: str) -> bool:
    """main = the functions round 2 repairs; regress = the 18 whose round-1 patch introduced a new problem."""
    return is_excluded(row) == (group == "regress")


def register(db: sqlite3.Connection, rows: dict[int, dict[str, str]], group: str) -> int:
    runnable = [stt for stt in sorted(rows) if in_group(rows[stt], group)]
    db.executemany("INSERT OR IGNORE INTO functions (stt) VALUES (?)", [(stt,) for stt in runnable])
    db.commit()
    return len(runnable)


def command_run(args: argparse.Namespace) -> None:
    batch_dir = BATCH_RUNS_DIR / args.batch
    db = connect(batch_dir)
    lock_options(db, args)
    rows = read_input(args.input)
    total = register(db, rows, args.group)
    stop_file = batch_dir / "STOP"
    stop_file.unlink(missing_ok=True)
    log = progress_logger(batch_dir)
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    log(f"Round 2 [{args.group}] started: {counts(db)} ({len(rows) - total} rows of the other group; "
        f"self_check={args.self_check}, score_detector={args.score_detector}).")

    processed = 0
    outage_stt, outage_streak = None, 0  # consecutive outages of one function
    while not _stop_requested and not stop_file.exists():
        if args.limit and processed >= args.limit:
            log(f"--limit {args.limit} reached.")
            break
        only = f" AND stt IN ({','.join(str(int(stt)) for stt in args.only)})" if args.only else ""
        nxt = db.execute("SELECT stt, attempts, advisor_json FROM functions WHERE status = 'pending'"
                         f"{only} ORDER BY stt LIMIT 1").fetchone()
        if nxt is None:
            log(f"{'Selected' if args.only else 'All'} functions finished: {counts(db)}. "
                "Export with: scripts/run_round2.sh export")
            break
        if not wait_until_runnable(args, stop_file, log):
            break
        stt = nxt["stt"]
        job = build_job(rows[stt])
        db.execute("UPDATE functions SET status = 'in_progress', updated_at = ? WHERE stt = ?", (utc_now(), stt))
        db.commit()
        started = time.time()
        cached = json.loads(nxt["advisor_json"]) if nxt["advisor_json"] else None
        try:
            result = process_function(job, cached, args, db, batch_dir)
        except KeyboardInterrupt:
            db.execute("UPDATE functions SET status = 'pending' WHERE stt = ?", (stt,))
            db.commit()
            log(f"Aborted #{stt}; re-queued.")
            break
        except ServerOutage as error:
            outage_streak = outage_streak + 1 if stt == outage_stt else 1
            outage_stt = stt
            if outage_streak < OUTAGE_REPEAT_LIMIT:
                db.execute("UPDATE functions SET status = 'pending' WHERE stt = ?", (stt,))
                db.commit()
                log(f"#{stt} {job['function_id']}: server outage ({str(error)[:120]}); re-queued.")
                time.sleep(min(args.poll_seconds, 5 * outage_streak))
                continue
            outage_stt, outage_streak = None, 0
            failure = error
        except Exception as error:  # noqa: BLE001 - one bad function never stops the batch
            failure = error
        else:
            failure = None
        if failure is not None:
            attempts = nxt["attempts"] + 1
            status = "failed" if attempts >= args.max_attempts else "pending"
            trace = "".join(traceback.format_exception(failure))[-1500:]
            db.execute("UPDATE functions SET status = ?, attempts = ?, error = ?, updated_at = ? WHERE stt = ?",
                       (status, attempts, f"{type(failure).__name__}: {failure}\n{trace}", utc_now(), stt))
            db.commit()
            log(f"#{stt} {job['function_id']}: {type(failure).__name__}: {str(failure)[:200]} -> {status}.")
            continue
        outage_stt, outage_streak = None, 0
        db.execute("UPDATE functions SET status = 'done', result_json = ?, error = NULL, updated_at = ? WHERE stt = ?",
                   (json.dumps(result, ensure_ascii=False), utc_now(), stt))
        db.commit()
        processed += 1
        done = counts(db).get("done", 0)
        log(f"#{stt} {job['function_id']}: {result['patch_status']} in {time.time() - started:.0f}s "
            f"({done}/{total} done).")
        if args.export_every and processed % args.export_every == 0:
            export(args.batch, args.input, args.group, quiet=True)
        time.sleep(args.sleep_between)
    export(args.batch, args.input, args.group, quiet=True)
    log(f"Stopped: {counts(db)}.")


def command_prepare(args: argparse.Namespace) -> None:
    """Re-scan every round-1 patched contract (local static + RAG, no GPU needed)."""
    batch_dir = BATCH_RUNS_DIR / args.batch
    run_ids = sorted({row["run_id"] for row in read_input(args.input).values() if in_group(row, args.group)})
    for index, run_id in enumerate(run_ids, 1):
        if (batch_dir / "rescan" / f"{run_id}.json").exists():
            continue
        started = time.time()
        try:
            get_rescan(batch_dir, args.round1_runs, run_id)
            print(f"[{index}/{len(run_ids)}] {run_id} re-scanned in {time.time() - started:.0f}s", flush=True)
        except Exception as error:  # noqa: BLE001
            print(f"[{index}/{len(run_ids)}] {run_id} FAILED: {type(error).__name__}: {error}", flush=True)


def command_preview(args: argparse.Namespace) -> None:
    """Print what the Advisor and Fixer would receive for one function (no LLM call)."""
    batch_dir = BATCH_RUNS_DIR / args.batch
    row = read_input(args.input)[args.stt]
    if not in_group(row, args.group):
        other = "main" if args.group == "regress" else "regress"
        raise SystemExit(f"stt_goc={args.stt} belongs to the '{other}' group: add --group {other}.")
    job = build_job(row)
    round1 = load_round1(args.round1_runs, job["run_id"])
    unit = _unit_in(job["raw_contract"], job["function_id"])
    findings = target_findings(round1, job, unit)
    rescan = get_rescan(batch_dir, args.round1_runs, job["run_id"])
    residual, introduced = rescan_for_unit(rescan, job["function_id"], set(job["swc_ids"]))
    agent = Round2Advisor(knowledge_base=_knowledge_base())  # never called here
    agent.previous_attempt = previous_attempt_block(job, residual, introduced)
    first = findings[0]
    retrieved = agent.knowledge_base.retrieve(first["swc"], code=first["unit_code"])
    print(f"===== ADVISOR PROMPT ({len(findings)} finding(s); showing {first['swc']}) =====")
    print(agent._build_prompt(job["raw_contract"], first, agent._build_knowledge_block(first, retrieved)))
    placeholder = [{"swc": item["swc"], "is_true_positive": True, "root_cause": "<advisor root cause>",
                    "fix_strategy": "<advisor fix strategy>"} for item in findings]
    print("\n===== FIXER PROMPT (advisor output replaced by placeholders) =====")
    print(_build_fixer_prompt(
        unit,
        [{"swc": item["swc"], "title": item.get("title", "")} for item in findings],
        [{"is_true_positive": True, "fix_strategy": "<advisor fix strategy>"} for _ in findings],
        job["raw_contract"],
        build_feedback_note(placeholder, [], job, residual, introduced,
                            guidance=kb_guidance([item["swc"] for item in findings], unit.code)),
        unit_rag_section(round1, {item["swc"] for item in findings}),
    ))


def command_status(args: argparse.Namespace) -> None:
    batch_dir = BATCH_RUNS_DIR / args.batch
    if not (batch_dir / "state.sqlite").exists():
        print(f"Batch '{args.batch}' has not been started yet.")
        return
    db = connect(batch_dir)
    print(f"Functions: {counts(db)}")
    statuses: dict[str, int] = {}
    for row in db.execute("SELECT result_json FROM functions WHERE status = 'done'"):
        key = json.loads(row["result_json"]).get("patch_status")
        statuses[key] = statuses.get(key, 0) + 1
    print(f"Round-2 patch status: {statuses}")
    rescanned = len(list((batch_dir / "rescan").glob("*.json"))) if (batch_dir / "rescan").exists() else 0
    print(f"Contracts re-scanned: {rescanned}")
    for row in db.execute("SELECT stt, error FROM functions WHERE status = 'failed' ORDER BY stt LIMIT 10"):
        print(f"  failed #{row['stt']}: {(row['error'] or '').splitlines()[0][:150]}")


def _row_out(row: dict[str, str], state: sqlite3.Row | None, group: str = "main") -> dict[str, Any]:
    excluded = group == "main" and is_excluded(row)
    out: dict[str, Any] = {
        "stt_goc": row["stt_goc"], "id": row["id"], "contract_name": row["contract_name"],
        "contract_file": row["contract_file"], "function_id": row["function_id"], "run_id": row["run_id"],
        "raw_contract": row["raw_contract"], "raw_function_code": row["raw_function_code"],
        "related_origin_vulnerability": row["related_origin_vulnerability"], "swc_ids": row["swc_ids"],
        "evidence_sources": row["evidence_sources"], "detector_prob_before": row["detector_prob_before"],
        "fixed_function_code_lan1": row["fixed_function_code"], "fix_rationale_lan1": row.get("fix_rationale", ""),
        "patch_note_lan1": row.get("patch_note", ""),
        "lan1_khac_phuc": row.get("claude_khac_phuc", ""), "lan1_an_toan_hon": row.get("claude_an_toan_hon", ""),
        "lan1_ly_do_chon": row.get("claude_ly_do_chon", ""),
        "round2_status": "excluded" if excluded else state["status"] if state else "pending",
        "round2_error": (state["error"] or "").splitlines()[0] if state and state["error"] else "",
        "round2_updated_at": state["updated_at"] if state else "",
    }
    if excluded:
        out["patch_status_lan2"] = "not_repaired_lan1_regression"
        return out
    if not state or not state["result_json"]:
        return out
    result = json.loads(state["result_json"])
    advisor = result.get("advisor", [])
    lan2 = result.get("fixed_code") or ""
    out.update({
        "rescan_lan1_residual": _fmt_rescan(result.get("rescan_residual", [])),
        "rescan_lan1_introduced": _fmt_rescan(result.get("rescan_introduced", [])),
        "advisor_lan2_true_positive": ",".join(item["swc"] for item in advisor if item["is_true_positive"]),
        "advisor_lan2_false_positive": ",".join(item["swc"] for item in advisor if not item["is_true_positive"]),
        "advisor_lan2_diagnosis": "\n".join(
            f"[{item['swc']}] {'TP' if item['is_true_positive'] else 'FP'}"
            + (f" (Advisor said FP, overridden: {item['fp_overridden']})" if item.get("fp_overridden") else "")
            + f": {item.get('root_cause') or ''}"
            + (f" | Fix: {item['fix_strategy']}" if item.get("fix_strategy") else "")
            + (f" | error: {item['error']}" if item.get("error") else "")
            for item in advisor
        ),
        "advisor_all_dismissed": result.get("advisor_all_dismissed"),
        "fixed_function_code_lan2": lan2,
        "patch_status_lan2": result.get("patch_status"),
        "fix_rationale_lan2": result.get("rationale"),
        "patch_note_lan2": result.get("guard_warning"),
        "changed_vs_lan1": bool(lan2) and not _same_code(lan2, row["fixed_function_code"]),
        "fixer_calls_lan2": result.get("fixer_calls", 0)
        if not result.get("fixer_retry_reason") else f"{result['fixer_calls']} ({result['fixer_retry_reason']})",
        "prompt_trimmed_lan2": "; ".join(
            [f"advisor [{item['swc']}]: {item['prompt_trimmed']}" for item in advisor if item.get("prompt_trimmed")]
            + ([f"fixer: {result['fixer_prompt_trimmed']}"] if result.get("fixer_prompt_trimmed") else [])),
        "detector_prob_lan1": result.get("detector_prob_lan1"),
        "detector_prob_lan2": result.get("detector_prob_lan2"),
    })
    first, final = result.get("selfcheck_first"), result.get("selfcheck_final")
    if final:
        out.update({
            "selfcheck_compile_lan2": final["compile"],
            "selfcheck_issues_first": "\n".join(first["issues"]) or "none",
            "selfcheck_retry": result.get("selfcheck_retry") or "not needed",
            "selfcheck_issues_final": "\n".join(final["issues"]) or "none",
            "rescan_lan2_residual": _fmt_rescan(final["residual"]),
        })
    return out


def export(batch: str, input_path: Path, group: str = "main", out: Path | None = None, blind: bool = False,
           quiet: bool = False) -> Path:
    """main: all 632 rows (the regress group as `excluded`); regress: only its 18 rows."""
    batch_dir = BATCH_RUNS_DIR / batch
    db = connect(batch_dir)
    states = {row["stt"]: row for row in db.execute("SELECT * FROM functions")}
    rows = read_input(input_path)
    if group == "regress":
        rows = {stt: row for stt, row in rows.items() if in_group(row, group)}
    columns = [column for column in CSV_COLUMNS if not (blind and column in BLIND_DROP)]
    path = out or batch_dir / f"ham_sua_lan_2_ket_qua{'_blind' if blind else ''}.csv"
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for stt in sorted(rows):
            writer.writerow(_row_out(rows[stt], states.get(stt), group))
    tmp.replace(path)
    if not quiet:
        print(f"Wrote {len(rows)} rows to {path}")
    return path


def _batch_group(batch: str, fallback: str) -> str:
    """The group a batch was started with (so `export` needs no --group)."""
    path = BATCH_RUNS_DIR / batch / "state.sqlite"
    if not path.exists():
        return fallback
    with contextlib.closing(sqlite3.connect(path)) as db:
        row = db.execute("SELECT value FROM meta WHERE key = 'group'").fetchone()
    return row[0] if row else fallback


def command_retry_failed(args: argparse.Namespace) -> None:
    db = connect(BATCH_RUNS_DIR / args.batch)
    changed = db.execute("UPDATE functions SET status = 'pending', attempts = 0 WHERE status = 'failed'").rowcount
    db.commit()
    print(f"{changed} failed function(s) re-queued.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Round-2 repair of the functions sent back by the evaluation.")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--batch", default="round2", help="Folder under data/results/.")
        command.add_argument("--input", type=Path, default=DEFAULT_INPUT)
        command.add_argument("--round1-runs", type=Path, default=DEFAULT_ROUND1_RUNS)
        command.add_argument("--group", choices=("main", "regress"), default="main",
                             help="main: the functions round 1 did not repair; regress: the 18 whose round-1 "
                                  "patch introduced a new problem (run in their own batch).")

    def options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--score-detector", action="store_true",
                             help="Record the v13 detector probability of the round-1 and round-2 function "
                                  "(record only, no retry).")
        command.add_argument(
            "--on-all-dismissed", choices=("fix_anyway", "keep_raw"), default="fix_anyway",
            help="When the Advisor calls every finding of a function a false positive: repair anyway "
                 "(default; the function was sent back as vulnerable) or leave the original code.")
        command.add_argument("--self-check", action="store_true",
                             help="Compile + re-scan + behaviour-guard the reply against the original contract; "
                                  "one more fixer call when it finds a problem the original did not have.")

    run = sub.add_parser("run")
    common(run)
    options(run)
    run.add_argument("--wait", action="store_true", help="When the server is down, wait for it instead of exiting.")
    run.add_argument("--poll-seconds", type=int, default=60)
    run.add_argument("--active-hours", default=None, help="e.g. 19:00-07:30")
    run.add_argument("--limit", type=int, default=None, help="Process at most N functions in this session.")
    run.add_argument("--only", type=int, nargs="+", help="Process only these stt_goc values (for testing).")
    run.add_argument("--max-attempts", type=int, default=2, help="Mark a function failed after N errors.")
    run.add_argument("--export-every", type=int, default=10)
    run.add_argument("--sleep-between", type=float, default=1.0)

    preview = sub.add_parser("preview")
    common(preview)
    options(preview)
    preview.add_argument("--stt", type=int, required=True, help="stt_goc of the function.")

    for name in ("prepare", "status", "retry-failed"):
        common(sub.add_parser(name))
    exp = sub.add_parser("export")
    common(exp)
    exp.add_argument("--out", type=Path)
    exp.add_argument("--blind", action="store_true", help="Drop the round-1 judge verdict columns.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "run":
        command_run(args)
    elif args.command == "prepare":
        command_prepare(args)
    elif args.command == "preview":
        command_preview(args)
    elif args.command == "status":
        command_status(args)
    elif args.command == "export":
        export(args.batch, args.input, _batch_group(args.batch, args.group), args.out, blind=args.blind)
    elif args.command == "retry-failed":
        command_retry_failed(args)


if __name__ == "__main__":
    main()
