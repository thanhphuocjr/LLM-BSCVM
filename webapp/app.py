"""
Web app (FastAPI) of the audit pipeline: ./scripts/run_webapp.sh -> http://127.0.0.1:8080.

The app has no audit logic of its own: POST /api/audit/submit runs the CLI
(`python -m E2E.run <file>`, Phase 0 -> 6) in a subprocess and the other endpoints read the
artifacts of data/results/audit_runs/<run_id>/ for the page in webapp/static/.
"""

from __future__ import annotations

import difflib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from E2E.common.gpu_client import check_server_health
from E2E.common.regex_patterns import RE_RUN_ID, RE_UNSAFE_NAME_CHARS
from E2E.config import DEMO_CONTRACTS_DIR, PROJECT_ROOT, RUNS_DIR

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Smart Contract Vulnerability Audit API (LLM-BSCVM v13 Qwen2.5-Coder-14B)",
    description="Web app API of the LLM-BSCVM audit pipeline",
    version="2.0.0",
)


# Middleware: thêm header no-cache cho mọi response để trình duyệt luôn lấy trạng thái/log mới nhất.
@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# In-memory execution tracker: run_id -> {"process": Popen | None, "status": str, "started_at": float}
ACTIVE_TASKS: Dict[str, Dict[str, Any]] = {}
TASK_LOCK = threading.Lock()


# Trả thư mục data/results/audit_runs/<run_id>; từ chối run_id có ".." hoặc ký tự lạ (chống path traversal) -> HTTP 400.
def run_dir_for(run_id: str) -> Path:
    """Resolve a run directory, rejecting ids that could escape RUNS_DIR (e.g. '..')."""
    if not RE_RUN_ID.fullmatch(run_id) or ".." in run_id:
        raise HTTPException(status_code=400, detail="Invalid run_id.")
    return RUNS_DIR / run_id


# Làm sạch tên file người dùng gửi lên thành "<tên>.sol" đơn giản (không thư mục, không traversal, tối đa 64 ký tự).
def safe_contract_file_name(file_name: str) -> str:
    """Keep only a plain `<stem>.sol` name: no directories, no traversal."""
    stem = RE_UNSAFE_NAME_CHARS.sub("-", Path(file_name or "").stem).strip("-") or "Contract"
    return f"{stem[:64]}.sol"


# Đọc file JSON; file không có hoặc hỏng -> None (giao diện không bị lỗi khi run đang chạy dở).
def read_json_safe(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# Nhiệm vụ: trạng thái của một run, dùng chung cho danh sách, chi tiết và log.
#   - Run do web app này khởi chạy (ACTIVE_TASKS): running / completed / failed / cancelled theo tiến trình CLI.
#   - Run khác (chạy từ CLI, hoặc trước khi web app khởi động lại): completed nếu manifest.json đã qua Phase 6
#     (status complete/partial); còn lại (dừng ở pending_remote hay chưa có manifest) là failed.
def run_status(run_id: str, run_dir: Path) -> str:
    with TASK_LOCK:
        task = ACTIVE_TASKS.get(run_id)
        if task:
            process = task.get("process")
            if task["status"] == "running" and process and process.poll() is not None:
                task["status"] = "completed" if process.returncode == 0 else "failed"
            return task["status"]
    manifest = read_json_safe(run_dir / "manifest.json") or {}
    return "completed" if manifest.get("status") in {"complete", "partial"} else "failed"


# Body của POST /api/audit/submit: mã Solidity và tên file (ngưỡng detector lấy từ threshold_config.json).
class AuditRequest(BaseModel):
    code: str = Field(..., description="Solidity smart contract code")
    file_name: str = Field(default="Contract.sol", description="Contract filename")


# GET /api/system/status: GPU server có sẵn sàng không + tên GPU (KHÔNG lộ IP/port ra trình duyệt).
@app.get("/api/system/status")
def get_system_status() -> Dict[str, Any]:
    """Returns whether the GPU server is ready."""
    health = check_server_health(timeout=1)
    return {
        "server_gpu_online": health.get("status") == "ready",
        "server_gpu_info": {"status": health.get("status"), "gpu_device": health.get("gpu_device")},
    }


# GET /api/contracts/examples: trả 3 contract mẫu trong demo_contracts/ (nút "Mẫu" trên giao diện).
@app.get("/api/contracts/examples")
def get_example_contracts() -> List[Dict[str, Any]]:
    """Demo contracts shown under "Mẫu" (demo_contracts/)."""
    examples = []
    for file in sorted(DEMO_CONTRACTS_DIR.glob("*.sol")):
        code = file.read_text(encoding="utf-8")
        vuln_type = "Unknown"
        if "VulnerableBank" in file.name:
            vuln_type = "Reentrancy (SWC-107)"
        elif "AccessControl" in file.name:
            vuln_type = "Access Control / tx.origin (SWC-105/115)"
        elif "Safe" in file.name:
            vuln_type = "Secure Benchmark (Safe)"
        examples.append({
            "file_name": file.name,
            "title": file.stem,
            "vuln_type": vuln_type,
            "code": code,
            "line_count": len(code.splitlines()),
        })
    return examples


# [ENDPOINT CHÍNH] POST /api/audit/submit: kiểm tra đầu vào, tạo run_id + thư mục run, lưu file .sol, rồi chạy
# lệnh CLI `python -m E2E.run <file> --run-id ... --max-fix-attempts 3` trong một subprocess ở luồng nền;
# stdout được ghi vào execution.log. Trả về run_id ngay để giao diện theo dõi.
# Mục đích: web app KHÔNG có logic audit riêng - nó tái sử dụng đúng CLI orchestrator.
@app.post("/api/audit/submit")
def submit_audit(req: AuditRequest) -> Dict[str, Any]:
    """Submits a new contract audit job and starts processing in the background."""
    if not req.code.strip():
        raise HTTPException(status_code=400, detail="Contract code cannot be empty.")

    from E2E.pipeline.orchestrator import make_run_id  # lazy: the pipeline modules load sklearn/scipy

    file_name = safe_contract_file_name(req.file_name)
    run_id = make_run_id(file_name, req.code)
    run_dir = run_dir_for(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    input_file = run_dir / file_name
    input_file.write_text(req.code, encoding="utf-8")

    cmd = [sys.executable, "-u", "-m", "E2E.run", str(input_file), "--run-id", run_id, "--max-fix-attempts", "3"]
    log_file = run_dir / "execution.log"

    with TASK_LOCK:
        ACTIVE_TASKS[run_id] = {"process": None, "status": "running", "started_at": time.time()}

    # Luồng nền: chạy subprocess CLI, ghi từng dòng log vào execution.log, cập nhật trạng thái completed/failed khi xong.
    def run_worker():
        with open(log_file, "w", encoding="utf-8") as out:
            out.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting E2E audit run: {run_id}\n")
            out.write(f"Command: {' '.join(cmd)}\n")
            out.write("Backend: server_gpu\n\n")
            out.flush()

            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            with TASK_LOCK:
                ACTIVE_TASKS[run_id]["process"] = proc

            if proc.stdout:
                for line in iter(proc.stdout.readline, ""):
                    out.write(line)
                    out.flush()
                proc.stdout.close()

            proc.wait()
            out.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Process exited with code {proc.returncode}\n")
            out.flush()

            with TASK_LOCK:
                if ACTIVE_TASKS[run_id]["status"] == "running":
                    ACTIVE_TASKS[run_id]["status"] = "completed" if proc.returncode == 0 else "failed"

    threading.Thread(target=run_worker, daemon=True).start()
    return {"status": "started", "run_id": run_id}


# GET /api/runs: liệt kê các lần audit (mới nhất trước) với tên file, verdict, điểm, số lỗi, trạng thái - cho trang lịch sử.
@app.get("/api/runs")
def list_runs() -> List[Dict[str, Any]]:
    """Returns list of all audit runs from newest to oldest."""
    runs = []
    if not RUNS_DIR.exists():
        return runs
    for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue
        manifest = read_json_safe(run_dir / "manifest.json") or {}
        phase0 = read_json_safe(run_dir / "phase0_input.json") or {}
        detection = manifest.get("detection", {})
        run_info: Dict[str, Any] = {
            "run_id": run_dir.name,
            "status": run_status(run_dir.name, run_dir),
            "created_at": manifest.get("completed_at") or phase0.get("created_at"),
            "contract_name": phase0.get("source", {}).get("file_name") or run_dir.name,
            "verdict": detection.get("verdict", "Unknown"),
            "score": detection.get("score", 0.0),
            "finding_count": detection.get("finding_count", 0),
        }
        if run_info["status"] == "running":
            with TASK_LOCK:
                started_at = ACTIVE_TASKS.get(run_dir.name, {}).get("started_at", time.time())
            run_info["elapsed_seconds"] = round(time.time() - started_at, 1)
        runs.append(run_info)
    return runs


# GET /api/runs/{run_id}: những gì trang kết quả cần: mã gốc / mã vá + diff, kết luận (manifest, fusion),
# findings giống hệt báo cáo, CVSS của Phase 3, kết luận Phase 5 và báo cáo nào đã có.
@app.get("/api/runs/{run_id}")
def get_run_detail(run_id: str) -> Dict[str, Any]:
    """Returns the artifacts the result page shows for one run."""
    run_dir = run_dir_for(run_id)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found.")

    manifest = read_json_safe(run_dir / "manifest.json") or {}
    phase0 = read_json_safe(run_dir / "phase0_input.json") or {}
    fused = read_json_safe(run_dir / "phase1_fused.json") or {}
    assessor = read_json_safe(run_dir / "phase3_assessor.json") or {}
    fixer = read_json_safe(run_dir / "phase4_fixer.json") or {}
    verification = read_json_safe(run_dir / "phase5_verification.json") or {}
    report_data = read_json_safe(run_dir / "phase6_report_data.json") or {}

    original_code = phase0.get("source", {}).get("code", "")
    fixed_code = fixer.get("fixed_code", "")
    diff = ""
    if original_code and fixed_code and original_code != fixed_code:
        diff = "".join(difflib.unified_diff(
            original_code.splitlines(keepends=True),
            fixed_code.splitlines(keepends=True),
            fromfile="Original.sol",
            tofile="Fixed.sol",
        ))

    return {
        "run_id": run_id,
        "manifest": manifest,
        "contract_name": phase0.get("source", {}).get("file_name") or "Contract.sol",
        "original_code": original_code,
        "fixed_code": fixed_code,
        "diff": diff,
        "phases": {"phase1_fused": fused, "phase3_assessor": assessor},
        # Same list as the audit report: after the GPU phases, findings the Advisor dismissed as
        # false positives are excluded (they stay in phase1_fused.json for transparency).
        "findings": report_data["findings"] if "findings" in report_data else fused.get("findings", []),
        "verification_verdict": verification.get("overall_verdict", "Pending"),
        "has_pdf": (run_dir / "phase6_audit_report.pdf").exists(),
        "has_html": (run_dir / "phase6_audit_report.html").exists(),
    }


# GET /api/runs/{run_id}/logs: log realtime (execution.log) + trạng thái run - giao diện đọc tag "[PHASE x]"
# trong log để tô tiến độ 5 bước.
@app.get("/api/runs/{run_id}/logs")
def get_run_logs(run_id: str) -> Dict[str, Any]:
    """Returns realtime logs of a run."""
    run_dir = run_dir_for(run_id)
    log_file = run_dir / "execution.log"
    logs = ""
    if log_file.exists():
        try:
            logs = log_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            logs = "(Unable to read log file)"
    return {"run_id": run_id, "status": run_status(run_id, run_dir), "logs": logs}


# GET /api/runs/{run_id}/report/html: trả báo cáo HTML của Phase 6.
@app.get("/api/runs/{run_id}/report/html")
def get_report_html(run_id: str):
    """Serves the generated HTML audit report."""
    path = run_dir_for(run_id) / "phase6_audit_report.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="HTML report not yet generated.")
    return HTMLResponse(content=path.read_text(encoding="utf-8"))


# GET /api/runs/{run_id}/report/pdf: tải báo cáo PDF.
@app.get("/api/runs/{run_id}/report/pdf")
def get_report_pdf(run_id: str):
    """Downloads the generated PDF audit report."""
    path = run_dir_for(run_id) / "phase6_audit_report.pdf"
    if not path.exists():
        raise HTTPException(status_code=404, detail="PDF report not yet generated.")
    return FileResponse(path, media_type="application/pdf", filename=f"audit_report_{run_id}.pdf")


# POST /api/runs/{run_id}/cancel: dừng subprocess audit đang chạy (terminate, sau 0.5s thì kill).
@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str) -> Dict[str, Any]:
    """Cancels a currently active run process."""
    with TASK_LOCK:
        task = ACTIVE_TASKS.get(run_id)
        proc = task.get("process") if task else None
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                time.sleep(0.5)
                if proc.poll() is None:
                    proc.kill()
                task["status"] = "cancelled"
                return {"status": "cancelled", "run_id": run_id}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to cancel process: {e}")
    raise HTTPException(status_code=404, detail=f"Run {run_id} is not running.")


STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# GET /: trả trang giao diện webapp/static/index.html.
@app.api_route("/", methods=["GET", "HEAD"])
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>Web App Static Files Initializing...</h1>", status_code=200)
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
