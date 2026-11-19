"""
BƯỚC 2 - chạy từng mẫu của tập test qua TOÀN BỘ pipeline (Phase 0 -> 6) rồi gom kết quả vào một file CSV.

    .venv/bin/python -m fix_eval.run_fix_eval                  chạy tiếp các mẫu chưa xong (resume được)
    .venv/bin/python -m fix_eval.run_fix_eval --limit 3        chạy thử 3 mẫu đầu
    .venv/bin/python -m fix_eval.run_fix_eval --export-only    chỉ xuất lại CSV + summary từ results.jsonl

Điều kiện chạy (để đánh giá bản vá công bằng):
  - đầu vào CHỈ có mã nguồn: data/fix_eval/test_input.json = {id, source_code}; không nhãn, không bản vá, không gợi ý;
  - RAG (Phase 1B), Advisor (Phase 2) và Fixer (Phase 4) dùng KB KHÔNG chứa test: biến môi trường
    RAG_KNOWLEDGE_BASE_DIR = data/knowledge_base/built_no_test, được kiểm tra lại trước khi chạy;
  - TẮT kiểm chứng Phase 5 (verify=False, giống CLI --no-verify): Fixer vá đúng 1 lần, bản vá giữ nguyên.
Mỗi mẫu: run_id = id của mẫu, artifact ở data/results/fix_eval/runs/<id>/ (log pipeline ở pipeline.log).
Kết quả: results.jsonl (một dòng / mẫu, ghi ngay khi xong), fix_eval_results.csv, summary.json, progress.log.
Dừng êm: tạo file data/results/fix_eval/STOP. Lời gọi GPU server bị lỗi -> mẫu được chạy lại, không ghi kết quả hỏng.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import inspect
import json
import os
import threading
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

from fix_eval import (
    NO_TEST_KB_DIR,
    PROGRESS_LOG_PATH,
    PROJECT_ROOT,
    RESULTS_CSV_PATH,
    RESULTS_DIR,
    RESULTS_JSONL_PATH,
    RUNS_DIR,
    STOP_FILE_PATH,
    SUMMARY_PATH,
    TEST_INPUT_PATH,
)

# KB không chứa test phải được chọn TRƯỚC khi E2E.config được nạp (config đọc biến môi trường lúc import).
os.environ["RAG_KNOWLEDGE_BASE_DIR"] = str(NO_TEST_KB_DIR)

from E2E import config  # noqa: E402
from E2E.common import gpu_client  # noqa: E402
from E2E.common.io_utils import read_json, utc_now, write_json  # noqa: E402
from E2E.phase1b_rag.retriever import default_retriever  # noqa: E402
from E2E.phase1b_rag.swc_agents import DEFAULT_SWC_REGISTRY  # noqa: E402
from E2E.phase2_advisor.advisor import RepairKnowledgeBase  # noqa: E402
from E2E.pipeline.gpu_pipeline import run_server_gpu_pipeline  # noqa: E402
from E2E.pipeline.orchestrator import run_pipeline  # noqa: E402

CSV_COLUMNS = [
    "id",
    "original_code",
    "vulnerability_detected",   # còn >= 1 finding sau Phase 1 (fusion) + Phase 2 (Advisor loại báo nhầm)
    "fixed_code",               # mã sau khi Fixer vá (cùng phạm vi với original_code); rỗng nếu không có bản vá
    "patch_status",             # patched / partially_patched / fix_failed / no_repair_target / dismissed_by_advisor /
                                # not_detected / error (pipeline ném lỗi)
    "detected_swc",
    "detected_vulnerabilities",
    "risk_level",
    "fused_verdict",            # kết luận fusion Phase 1 (static + RAG + detector), trước Advisor
    "fused_score",
    "detector_probability",     # P(vulnerable) lớn nhất của detector Qwen v13 trên các hàm
    "confirmed_findings",
    "dismissed_findings",
    "target_units",
    "patched_units",
    "fix_rationale",            # lý do Fixer đưa ra ("why")
    "fixer_errors",             # hàm vá thất bại: unit, bước, lỗi
    "overall_verdict",
    "run_dir",
    "elapsed_s",
    "gpu_calls",
    "gpu_errors",               # > 0: vẫn còn lời gọi GPU lỗi sau GPU_RETRIES lần chạy lại -> kết quả có thể bị suy giảm
    "error",
]
GPU_RETRIES = 2                 # số lần chạy lại một mẫu có lời gọi GPU lỗi (sau lượt chính)
SERVER_WAIT_SECONDS = 30
SERVER_MAX_WAIT_SECONDS = 3600  # GPU server không sẵn sàng quá lâu -> dừng (chạy lại lệnh để resume)


# Ghi một dòng (kèm giờ) vào progress.log và in ra màn hình.
def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    PROGRESS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


# Nhiệm vụ: đếm lời gọi HTTP tới GPU server trong một mẫu và gom lỗi của chúng. Bọc gpu_client._post_json
# (hàm HTTP duy nhất của detector, Advisor, Assessor, Fixer); lỗi vẫn được ném lại nên hành vi pipeline không đổi.
# Mục đích: Advisor/Assessor nuốt lỗi LLM (dùng tri thức KB thay thế) -> nếu không đếm ở đây, một lần mất kết nối
# sẽ lặng lẽ làm xấu kết quả. Mẫu có lỗi GPU được chạy lại.
class GpuCallMonitor:
    # Thay gpu_client._post_json bằng bản có đếm (các hàm gọi tra tên này lúc chạy nên đều đi qua đây).
    def __init__(self) -> None:
        self._original = gpu_client._post_json
        self._lock = threading.Lock()
        self.calls = 0
        self.errors: list[str] = []
        gpu_client._post_json = self._post_json

    # Bắt đầu đếm cho một mẫu mới.
    def reset(self) -> None:
        with self._lock:
            self.calls = 0
            self.errors = []

    # Gọi hàm gốc, ghi lại lỗi (endpoint + loại lỗi) rồi ném lại.
    def _post_json(self, url: str, payload: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
        with self._lock:
            self.calls += 1
        try:
            return self._original(url, payload, timeout=timeout)
        except Exception as error:
            with self._lock:
                self.errors.append(f"{url.rsplit('/', 1)[-1]}: {type(error).__name__}: {error}")
            raise


# Nhiệm vụ: trước khi chạy, xác nhận mọi nơi đọc KB trong tiến trình này (config, retriever Phase 1B, Advisor/Assessor,
# SWC agents) đều trỏ tới built_no_test/ và KB đó không chứa id nào của tập test. Sai -> dừng hẳn.
# Trả về số ca trong KB đang dùng.
def assert_no_test_knowledge_base(test_ids: set[str]) -> int:
    expected = NO_TEST_KB_DIR.resolve()
    retriever = default_retriever()
    locations = {
        "E2E.config.KNOWLEDGE_BASE_DIR": Path(config.KNOWLEDGE_BASE_DIR),
        "Phase 1B retriever": Path(retriever.store_dir),
        "Phase 2/3 RepairKnowledgeBase": Path(
            inspect.signature(RepairKnowledgeBase.__init__).parameters["knowledge_dir"].default
        ),
        "SWC agents registry": DEFAULT_SWC_REGISTRY.parent,
    }
    wrong = {name: str(path) for name, path in locations.items() if path.resolve() != expected}
    if wrong:
        raise SystemExit(f"Knowledge base is not the no-test build {expected}: {wrong}")
    leaked = [item.get("id") for item in retriever.knowledge_items if item.get("id") in test_ids]
    if leaked:
        raise SystemExit(f"The no-test knowledge base contains {len(leaked)} test id(s), e.g. {leaked[:5]}")
    return len(retriever.knowledge_items)


# Đọc results.jsonl -> {id: bản ghi MỚI NHẤT} (lần chạy lại của một mẫu ghi đè lần trước).
def load_results() -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if RESULTS_JSONL_PATH.exists():
        for line in RESULTS_JSONL_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                results[record["id"]] = record
    return results


# Thêm một bản ghi vào results.jsonl (ghi + flush ngay: dừng giữa chừng không mất mẫu đã xong).
def append_result(record: dict[str, Any]) -> None:
    RESULTS_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_JSONL_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# Chờ GPU server "ready" (kiểm tra mỗi 30 s); quá SERVER_MAX_WAIT_SECONDS -> False.
def wait_for_server() -> bool:
    waited = 0
    while True:
        health = gpu_client.check_server_health(timeout=10)
        if health.get("status") == "ready":
            return True
        if waited >= SERVER_MAX_WAIT_SECONDS:
            return False
        if waited == 0:
            log(f"GPU server not ready ({health.get('status')}: {health.get('error', '')}); waiting...")
        time.sleep(SERVER_WAIT_SECONDS)
        waited += SERVER_WAIT_SECONDS


# Nhiệm vụ: rút gọn kết quả một mẫu thành một dòng: phát hiện được gì, bản vá (nếu có), trạng thái vá, lý do.
# Đọc server_result (Phase 1C-5) + phase1_fused / phase5_verification / manifest trong thư mục run.
def summarize_run(sample: dict[str, Any], server_result: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    phases = server_result.get("phases", {})
    detector = phases.get("detector", {})
    confirmed = detector.get("confirmed_findings", [])
    dismissed = detector.get("dismissed_findings", [])
    outcomes = phases.get("fixer", {}).get("patch_outcomes", [])
    applied = [item for item in outcomes if item.get("status") == "applied"]
    failed = [item for item in outcomes if item.get("status") != "applied"]
    if applied:
        patch_status = "patched" if not failed else "partially_patched"
    elif outcomes:
        patch_status = "fix_failed"
    elif confirmed:
        patch_status = "no_repair_target"
    elif dismissed:
        patch_status = "dismissed_by_advisor"
    else:
        patch_status = "not_detected"
    fused = read_json(run_dir / "phase1_fused.json")
    verification = read_json(run_dir / "phase5_verification.json")
    manifest = read_json(run_dir / "manifest.json")
    return {
        "id": sample["id"],
        "original_code": sample["source_code"],
        "vulnerability_detected": bool(confirmed),
        "fixed_code": phases.get("fixer", {}).get("fixed_code", "") if applied else "",
        "patch_status": patch_status,
        "detected_swc": ";".join(sorted({str(item.get("swc")) for item in confirmed if item.get("swc")})),
        "detected_vulnerabilities": ";".join(dict.fromkeys(str(item.get("title") or "") for item in confirmed)),
        "risk_level": manifest.get("detection", {}).get("risk_level"),
        "fused_verdict": fused.get("verdict"),
        "fused_score": fused.get("score"),
        "detector_probability": detector.get("original", {}).get("vulnerability_probability"),
        "confirmed_findings": len(confirmed),
        "dismissed_findings": len(dismissed),
        "target_units": len(outcomes),
        "patched_units": len(applied),
        "fix_rationale": " | ".join(
            f"{item.get('unit_id')}: {item.get('rationale')}" for item in applied if item.get("rationale")
        ),
        "fixer_errors": " | ".join(
            f"{item.get('unit_id')}: {item.get('stage')}: {item.get('error')}" for item in failed
        ),
        "overall_verdict": verification.get("overall_verdict"),
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
    }


# Nhiệm vụ: chạy MỘT mẫu đúng như CLI `python -m E2E.run <file> --no-verify --max-fix-attempts 1` nhưng trong cùng
# tiến trình: run_pipeline lượt 1 (Phase 0, 1A, 1B) -> run_server_gpu_pipeline (Phase 1C -> 4, GPU server) ->
# run_pipeline lượt 2 (fusion, Phase 5 bị tắt, báo cáo Phase 6). Log của pipeline ghi vào runs/<id>/pipeline.log.
# Trả về dòng kết quả (gpu_errors = số lời gọi GPU lỗi; patch_status "error" nếu pipeline ném lỗi).
def run_sample(sample: dict[str, Any], monitor: GpuCallMonitor) -> dict[str, Any]:
    sample_id, code = sample["id"], sample["source_code"]
    run_dir = RUNS_DIR / sample_id
    run_dir.mkdir(parents=True, exist_ok=True)
    common = dict(
        file_name=f"{sample_id}.sol",
        runs_dir=RUNS_DIR,
        max_fix_attempts=1,
        verify=False,
    )
    monitor.reset()
    started = time.perf_counter()
    record: dict[str, Any]
    try:
        with (run_dir / "pipeline.log").open("w", encoding="utf-8") as pipeline_log, \
                contextlib.redirect_stdout(pipeline_log):
            manifest = run_pipeline(code, run_id=sample_id, **common)
            request = read_json(manifest["request_path"])
            server_result = run_server_gpu_pipeline(request)
            run_pipeline(
                code,
                run_id=sample_id,
                remote_result=server_result,
                local_phase1=request.get("local_phase1"),
                **common,
            )
        record = summarize_run(sample, server_result, run_dir)
        record["error"] = ""
    except Exception as error:  # noqa: BLE001 - one broken sample must not stop the batch
        record = {"id": sample_id, "original_code": code, "patch_status": "error",
                  "error": f"{type(error).__name__}: {error}"}
        (run_dir / "runner_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    if monitor.errors:
        record["error"] = "; ".join(filter(None, [record.get("error"), *monitor.errors[:3]]))
    record["elapsed_s"] = round(time.perf_counter() - started, 1)
    record["gpu_calls"] = monitor.calls
    record["gpu_errors"] = len(monitor.errors)
    record["finished_at"] = utc_now()
    return record


# Nhiệm vụ: ghi fix_eval_results.csv (một dòng / mẫu, sắp theo id như file ground truth; UTF-8 có BOM để Excel
# đọc đúng) và summary.json (đếm theo trạng thái vá, thời gian, lỗi). Chỉ dùng kết quả pipeline, không dùng nhãn.
def export(results: dict[str, dict[str, Any]], total_samples: int) -> None:
    rows = [results[key] for key in sorted(results)]
    RESULTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})
    times = [float(row["elapsed_s"]) for row in rows if row.get("elapsed_s") is not None]
    summary = {
        "updated_at": utc_now(),
        "samples_total": total_samples,
        "samples_done": len(rows),
        "patch_status": dict(Counter(row.get("patch_status") for row in rows)),
        "vulnerability_detected": sum(1 for row in rows if row.get("vulnerability_detected")),
        "with_fixed_code": sum(1 for row in rows if row.get("fixed_code")),
        "fused_verdict": dict(Counter(row.get("fused_verdict") for row in rows)),
        "samples_with_gpu_errors": sum(1 for row in rows if row.get("gpu_errors")),
        "seconds_total": round(sum(times), 1),
        "seconds_mean": round(sum(times) / len(times), 1) if times else None,
        "settings": {
            "knowledge_base": str(NO_TEST_KB_DIR.relative_to(PROJECT_ROOT)),
            "verification": "disabled (--no-verify): one Fixer attempt, patch not verified",
            "input": str(TEST_INPUT_PATH.relative_to(PROJECT_ROOT)),
            "server": gpu_client.get_server_url(),
        },
    }
    write_json(SUMMARY_PATH, summary)


# Tham số CLI: --limit (chạy thử N mẫu), --export-only, --rerun-errors (chạy lại cả mẫu status=error),
# --ids (chạy lại đúng các id này).
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the test split through the pipeline (no-test KB, no verification).")
    parser.add_argument("--limit", type=int, default=None, help="Run at most N pending samples (smoke test).")
    parser.add_argument("--export-only", action="store_true", help="Only rebuild the CSV and summary from results.jsonl.")
    parser.add_argument("--rerun-errors", action="store_true", help="Also re-run samples whose last status is 'error'.")
    parser.add_argument("--ids", default="", help="Comma-separated sample ids to (re-)run, even if already done.")
    return parser.parse_args()


# [HÀM CHÍNH BƯỚC 2] Nạp tập đầu vào, kiểm tra KB không chứa test, chạy lần lượt các mẫu chưa xong (chờ GPU server khi
# cần, dừng khi có file STOP), chạy lại mẫu lỗi GPU tối đa GPU_RETRIES lần, rồi xuất CSV + summary.
def main() -> None:
    args = parse_args()
    samples = read_json(TEST_INPUT_PATH)
    if args.export_only:
        export(load_results(), len(samples))
        print(f"CSV: {RESULTS_CSV_PATH}")
        return

    kb_items = assert_no_test_knowledge_base({sample["id"] for sample in samples})
    STOP_FILE_PATH.unlink(missing_ok=True)
    results = load_results()

    # Mẫu cần chạy: chưa có kết quả, hoặc lần trước còn lỗi GPU, hoặc (--rerun-errors) pipeline ném lỗi.
    def needs_run(sample_id: str) -> bool:
        record = results.get(sample_id)
        return record is None or bool(record.get("gpu_errors")) or (
            args.rerun_errors and record.get("patch_status") == "error"
        )

    pending = [sample for sample in samples if needs_run(sample["id"])]
    if args.ids:
        wanted = {value.strip() for value in args.ids.split(",") if value.strip()}
        pending = [sample for sample in samples if sample["id"] in wanted]
    if args.limit is not None:
        pending = pending[: args.limit]
    log(f"START {len(pending)} pending / {len(samples)} samples | KB {NO_TEST_KB_DIR.name} ({kb_items} cases, "
        f"no test ids) | verification OFF | server {gpu_client.get_server_url()}")

    monitor = GpuCallMonitor()
    queue = [(sample, 0) for sample in pending]
    done = 0
    session_started = time.perf_counter()
    while queue:
        if STOP_FILE_PATH.exists():
            log("STOP file found: exiting (run the command again to resume).")
            break
        if not wait_for_server():
            log("GPU server unavailable for too long: exiting (run the command again to resume).")
            break
        sample, retries = queue.pop(0)
        record = run_sample(sample, monitor)
        if record["gpu_errors"] and retries < GPU_RETRIES:
            log(f"GPU error on {sample['id']} ({record['error'][:160]}); will retry later.")
            queue.append((sample, retries + 1))
            continue
        append_result(record)
        results[sample["id"]] = record
        done += 1
        rate = (time.perf_counter() - session_started) / done
        remaining = sum(1 for s in samples if needs_run(s["id"]))
        log(
            f"[{sum(1 for s in samples if not needs_run(s['id']))}/{len(samples)}] {sample['id']} "
            f"{record['patch_status']} (detected={record.get('vulnerability_detected')}, patched "
            f"{record.get('patched_units', 0)}/{record.get('target_units', 0)}) {record['elapsed_s']}s, "
            f"{record['gpu_calls']} GPU call(s){', GPU ERRORS ' + str(record['gpu_errors']) if record['gpu_errors'] else ''}"
            f" | avg {rate:.1f}s, ~{remaining * rate / 3600:.1f} h left"
        )
        if done % 25 == 0:
            export(results, len(samples))
    export(results, len(samples))
    log(f"END session: {done} sample(s) finished | CSV {RESULTS_CSV_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
