from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
E2E_ROOT = PROJECT_ROOT / "E2E"
RUNS_DIR = E2E_ROOT / "runs"
EXAMPLES_DIR = E2E_ROOT / "examples"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Smart Contract Vulnerability Audit API (LLM-BSCVM v13 Qwen2.5-Coder-14B)",
    description="Full-featured Web App API for Smart Contract Security Auditing",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# In-memory execution tracker: run_id -> { "process": Popen, "log": str, "status": str, "started_at": float }
ACTIVE_TASKS: Dict[str, Dict[str, Any]] = {}
TASK_LOCK = threading.Lock()


class AuditRequest(BaseModel):
    code: str = Field(..., description="Solidity smart contract code")
    file_name: str = Field(default="Contract.sol", description="Contract filename")
    backend: str = Field(default="server_gpu", description="Backend: server_gpu, local_fast, local, kaggle, ollama")
    detector_threshold: float = Field(default=0.700, description="Detector threshold (default 0.700 for v13)")
    tools: str = Field(default="slither", description="Security tools: slither or all")


def read_json_safe(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_env_file(path: Path = PROJECT_ROOT / ".env") -> None:
    """Populate os.environ from .env file so settings take effect immediately."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value

load_env_file()


@app.get("/api/system/status")
def get_system_status() -> Dict[str, Any]:
    """Returns system status, active models, and environment readiness."""
    load_env_file()
    adapter_path = PROJECT_ROOT / "models" / "qwen25-v13-14b-detector-final"
    if not adapter_path.exists():
        adapter_path = E2E_ROOT / "models" / "codellama-vuln-detector"
    threshold_cfg = read_json_safe(adapter_path / "threshold_config.json") or {}
    
    # Check kaggle CLI
    kaggle_available = shutil.which("kaggle") is not None

    # Check Dedicated GPU Server (RTX 4070 Ti SUPER v13)
    server_gpu_online = False
    server_gpu_info = {}
    server_url = os.environ.get("SERVER_GPU_URL", "http://192.168.91.176:6868").rstrip("/")
    try:
        import urllib.request
        req = urllib.request.Request(f"{server_url}/api/v1/health", headers={"User-Agent": "LLM-BSCVM-Webapp"})
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            if resp.status == 200:
                raw_info = json.loads(resp.read().decode("utf-8"))
                server_gpu_online = raw_info.get("status") == "ready"
                # Bảo mật: Không để lộ IP/Port ra bên ngoài client
                server_gpu_info = {
                    "status": raw_info.get("status"),
                    "gpu_device": raw_info.get("gpu_device", "NVIDIA GeForce RTX 4070 Ti SUPER"),
                    "model": raw_info.get("model", "qwen25-v13-14b-detector-final"),
                    "vram_allocated_gb": raw_info.get("vram_allocated_gb"),
                }
    except Exception:
        server_gpu_online = False

    return {
        "model_version": threshold_cfg.get("model_name", "qwen25-v13-14b-detector-final"),
        "train_mode": threshold_cfg.get("train_mode", "Qwen2.5-Coder-14B-Instruct v13 Quality-First (Focal Loss + Decoupled Line Head + Multi-GPU)"),
        "base_model": threshold_cfg.get("base_model", "Qwen/Qwen2.5-Coder-14B-Instruct"),
        "best_threshold": threshold_cfg.get("best_threshold", 0.700),
        "max_context_length": threshold_cfg.get("max_len", 1024),
        "lora_rank": threshold_cfg.get("lora_r", 64),
        "lora_alpha": threshold_cfg.get("lora_alpha", 128),
        "adapter_present": (adapter_path / "adapter_model.safetensors").exists(),
        "adapter_size_bytes": (adapter_path / "adapter_model.safetensors").stat().st_size if (adapter_path / "adapter_model.safetensors").exists() else 0,
        "knowledge_store_cases": 11869,
        "server_gpu_online": server_gpu_online,
        "server_gpu_info": server_gpu_info,
        "kaggle_cli_installed": kaggle_available,
        "python_env": sys.executable,
    }


@app.get("/api/contracts/examples")
def get_example_contracts() -> List[Dict[str, Any]]:
    """Returns list of curated example contracts."""
    examples = []
    if EXAMPLES_DIR.exists():
        for file in sorted(EXAMPLES_DIR.glob("*.sol")):
            code = file.read_text(encoding="utf-8")
            vuln_type = "Unknown"
            if "VulnerableBank" in file.name:
                vuln_type = "Reentrancy (SWC-107)"
            elif "AccessControl" in file.name:
                vuln_type = "Access Control / tx.origin (SWC-105/115)"
            elif "Signature" in file.name:
                vuln_type = "Signature Replay (SWC-114/115)"
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


@app.post("/api/audit/submit")
def submit_audit(req: AuditRequest) -> Dict[str, Any]:
    """Submits a new contract audit job and starts processing in the background."""
    if not req.code.strip():
        raise HTTPException(status_code=400, detail="Contract code cannot be empty.")

    stem = re.sub(r"[^a-zA-Z0-9_-]+", "-", Path(req.file_name).stem).strip("-") or "contract"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    import hashlib
    sha_prefix = hashlib.sha256(req.code.encode("utf-8")).hexdigest()[:8]
    run_id = f"{timestamp}-{stem}-{sha_prefix}"

    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    input_file = run_dir / req.file_name
    input_file.write_text(req.code, encoding="utf-8")

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "E2E.run",
        str(input_file),
        "--run-id",
        run_id,
        "--backend",
        req.backend,
        "--detector-threshold",
        str(req.detector_threshold),
        "--max-fix-attempts",
        "1",
    ]
    if req.tools:
        cmd.extend(["--tools", req.tools])

    log_file = run_dir / "execution.log"

    with TASK_LOCK:
        ACTIVE_TASKS[run_id] = {
            "process": None,
            "status": "running",
            "log_file": str(log_file),
            "started_at": time.time(),
        }

    def run_worker():
        with open(log_file, "w", encoding="utf-8") as out:
            out.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting E2E audit run: {run_id}\n")
            out.write(f"Command: {' '.join(cmd)}\n")
            out.write(f"Backend: {req.backend} | Threshold: {req.detector_threshold}\n\n")
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
                if run_id in ACTIVE_TASKS:
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
                if run_id in ACTIVE_TASKS:
                    ACTIVE_TASKS[run_id]["status"] = "completed" if proc.returncode == 0 else "failed"

    threading.Thread(target=run_worker, daemon=True).start()

    return {
        "status": "started",
        "run_id": run_id,
        "backend": req.backend,
        "detector_threshold": req.detector_threshold,
        "run_dir": str(run_dir),
    }


@app.get("/api/runs")
def list_runs() -> List[Dict[str, Any]]:
    """Returns list of all audit runs from newest to oldest."""
    runs = []
    if RUNS_DIR.exists():
        for d in sorted(RUNS_DIR.iterdir(), reverse=True):
            if not d.is_dir() or d.name.startswith("."):
                continue

            manifest = read_json_safe(d / "manifest.json")
            phase0 = read_json_safe(d / "phase0_input.json")

            has_pdf = (d / "phase6_audit_report.pdf").exists()
            has_html = (d / "phase6_audit_report.html").exists()
            has_md = (d / "phase6_audit_report.md").exists()
            has_report = has_pdf or has_html or has_md

            run_info: Dict[str, Any] = {
                "run_id": d.name,
                "status": "completed",
                "created_at": None,
                "contract_name": "Contract.sol",
                "verdict": "Unknown",
                "score": 0.0,
                "risk_level": "None",
                "finding_count": 0,
                "has_pdf": has_pdf,
                "has_html": has_html,
                "has_md": has_md,
            }

            if manifest:
                run_info["created_at"] = manifest.get("completed_at")
                det = manifest.get("detection", {})
                run_info["verdict"] = det.get("verdict", "Unknown")
                run_info["score"] = det.get("score", 0.0)
                run_info["risk_level"] = det.get("risk_level", "None")
                run_info["finding_count"] = det.get("finding_count", 0)

            if phase0:
                run_info["contract_name"] = phase0.get("source", {}).get("file_name", d.name)
                if not run_info["created_at"]:
                    run_info["created_at"] = phase0.get("created_at")

            # Xác định trạng thái thực thi chính xác
            is_actively_running = False
            with TASK_LOCK:
                if d.name in ACTIVE_TASKS:
                    proc_info = ACTIVE_TASKS[d.name]
                    if proc_info["status"] == "running":
                        proc = proc_info.get("process")
                        if proc and proc.poll() is not None:
                            proc_info["status"] = "completed" if proc.returncode == 0 else "failed"
                        else:
                            is_actively_running = True
                            run_info["status"] = "running"
                            run_info["elapsed_seconds"] = round(time.time() - proc_info["started_at"], 1)

            if not is_actively_running:
                if has_report or (manifest and manifest.get("completed_at")):
                    run_info["status"] = "completed"
                elif manifest and manifest.get("status") == "awaiting_kaggle":
                    run_info["status"] = "awaiting_kaggle"
                elif manifest and manifest.get("status") == "failed":
                    run_info["status"] = "failed"
                else:
                    run_info["status"] = "completed"

            runs.append(run_info)

    return runs


@app.get("/api/runs/{run_id}")
def get_run_detail(run_id: str) -> Dict[str, Any]:
    """Returns all detail artifacts and phase states for a run."""
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found.")

    manifest = read_json_safe(run_dir / "manifest.json") or {}
    phase0 = read_json_safe(run_dir / "phase0_input.json") or {}
    static = read_json_safe(run_dir / "phase1a_static.json") or {}
    rag = read_json_safe(run_dir / "phase1b_rag.json") or {}
    fused = read_json_safe(run_dir / "phase1_fused.json") or {}
    advisor = read_json_safe(run_dir / "phase2_advisor.json") or {}
    assessor = read_json_safe(run_dir / "phase3_assessor.json") or {}
    fixer = read_json_safe(run_dir / "phase4_fixer.json") or {}
    verifier = read_json_safe(run_dir / "phase5_verification.json") or {}
    report_data = read_json_safe(run_dir / "phase6_report_data.json") or {}

    original_code = phase0.get("source", {}).get("code", "")
    contract_name = phase0.get("source", {}).get("file_name", "")

    # Fallback 1: Direct .sol files in run directory
    if not original_code:
        sol_files = sorted(run_dir.glob("*.sol"))
        if sol_files:
            try:
                original_code = sol_files[0].read_text(encoding="utf-8")
                if not contract_name:
                    contract_name = sol_files[0].name
            except Exception:
                pass

    # Fallback 2: request.json
    if not original_code:
        req_json = read_json_safe(run_dir / "request.json") or {}
        original_code = req_json.get("source", {}).get("code", "") or req_json.get("code", "")
        if not contract_name:
            contract_name = req_json.get("source", {}).get("file_name", "") or req_json.get("file_name", "")

    # Fallback 3: phase6_report_data.json
    if not original_code:
        rep_data = read_json_safe(run_dir / "phase6_report_data.json") or {}
        original_code = rep_data.get("original_code", "")
        if not contract_name:
            contract_name = rep_data.get("contract_name", "")

    if not contract_name:
        parts = run_id.split("-")
        if len(parts) >= 3:
            contract_name = f"{parts[2]}.sol"
        else:
            contract_name = "Contract.sol"

    fixed_code = fixer.get("fixed_code", "")

    has_pdf = (run_dir / "phase6_audit_report.pdf").exists()
    has_html = (run_dir / "phase6_audit_report.html").exists()
    has_md = (run_dir / "phase6_audit_report.md").exists()
    has_report = has_pdf or has_html or has_md

    # Xác định trạng thái
    status = "completed" if has_report or manifest.get("completed_at") else "unknown"
    is_actively_running = False
    with TASK_LOCK:
        if run_id in ACTIVE_TASKS:
            proc_info = ACTIVE_TASKS[run_id]
            if proc_info["status"] == "running":
                proc = proc_info.get("process")
                if proc and proc.poll() is not None:
                    proc_info["status"] = "completed" if proc.returncode == 0 else "failed"
                    status = proc_info["status"]
                else:
                    is_actively_running = True
                    status = "running"
            else:
                status = proc_info["status"]

    if not is_actively_running:
        if has_report or manifest.get("completed_at"):
            status = "completed"
            is_awaiting_kaggle = False
        elif manifest.get("status") == "awaiting_kaggle":
            status = "awaiting_kaggle"
            is_awaiting_kaggle = True
        elif manifest.get("status") == "failed":
            status = "failed"
            is_awaiting_kaggle = False
        else:
            status = "completed"
            is_awaiting_kaggle = False
    else:
        is_awaiting_kaggle = False

    diff_text = fixer.get("diff", "")
    if not diff_text and original_code and fixed_code and original_code != fixed_code:
        import difflib
        diff_lines = list(difflib.unified_diff(
            original_code.splitlines(keepends=True),
            fixed_code.splitlines(keepends=True),
            fromfile="Original.sol",
            tofile="Fixed.sol",
        ))
        diff_text = "".join(diff_lines)

    return {
        "run_id": run_id,
        "status": status,
        "is_awaiting_kaggle": is_awaiting_kaggle,
        "manifest": manifest,
        "contract_name": contract_name,
        "original_code": original_code,
        "fixed_code": fixed_code,
        "diff": diff_text,
        "phases": {
            "phase0_input": bool(phase0),
            "phase1a_static": static,
            "phase1b_rag": rag,
            "phase1c_llm": {"status": "pending" if is_awaiting_kaggle else "ok"},
            "phase1_fused": fused,
            "phase2_advisor": advisor,
            "phase3_assessor": assessor,
            "phase4_fixer": fixer,
            "phase5_verification": verifier,
            "phase6_report": (bool(report_data) or has_report) and not is_awaiting_kaggle,
        },
        "findings": fused.get("findings", []),
        "patch_outcomes": fixer.get("patch_outcomes", []),
        "verification_verdict": verifier.get("policy_verdict") or verifier.get("overall_verdict", "Pending"),
        "has_pdf": has_pdf,
        "has_html": has_html,
        "has_md": has_md,
    }


@app.get("/api/runs/{run_id}/logs")
def get_run_logs(run_id: str) -> Dict[str, Any]:
    """Returns realtime logs of a run."""
    run_dir = RUNS_DIR / run_id
    log_file = run_dir / "execution.log"
    logs = ""
    if log_file.exists():
        try:
            logs = log_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            logs = "(Unable to read log file)"

    status = "unknown"
    with TASK_LOCK:
        if run_id in ACTIVE_TASKS:
            proc_info = ACTIVE_TASKS[run_id]
            proc = proc_info.get("process")
            if proc_info.get("status") == "running":
                if proc and proc.poll() is not None:
                    proc_info["status"] = "completed" if proc.returncode == 0 else "failed"
                    status = proc_info["status"]
                else:
                    status = "running"
            else:
                status = proc_info.get("status", "unknown")
        else:
            # Task not active in memory (historical run or after server reload)
            if "Run completed successfully." in logs or "Process exited with code 0" in logs:
                status = "completed"
            elif "Process exited with code" in logs or "Traceback (most recent call last)" in logs:
                status = "failed"
            elif (run_dir / "manifest.json").exists():
                manifest = read_json_safe(run_dir / "manifest.json") or {}
                if manifest.get("status") == "complete":
                    status = "completed"

    def is_phase_real(filename: str) -> bool:
        p = run_dir / filename
        if not p.exists():
            return False
        data = read_json_safe(p)
        if isinstance(data, dict):
            if data.get("status") in {"pending", "awaiting_kaggle"}:
                return False
            if "overall_verdict" in data and "pending" in str(data.get("overall_verdict", "")).lower():
                return False
        return True

    is_p6_done = (
        (status == "completed" or "[PHASE 6] Completed" in logs or "Run completed successfully" in logs)
        and (run_dir / "phase6_audit_report.html").exists()
    )

    phase_files = {
        "phase0": is_phase_real("phase0_input.json"),
        "phase1a": is_phase_real("phase1a_static.json"),
        "phase1b": is_phase_real("phase1b_rag.json"),
        "fusion": is_phase_real("phase1_fused.json"),
        "phase2": is_phase_real("phase2_advisor.json"),
        "phase3": is_phase_real("phase3_assessor.json"),
        "phase4": is_phase_real("phase4_fixer.json"),
        "phase5": is_phase_real("phase5_verification.json"),
        "phase6": is_p6_done,
    }

    return {
        "run_id": run_id,
        "status": status,
        "logs": logs,
        "phase_files": phase_files,
    }


@app.get("/api/runs/{run_id}/report/html")
def get_report_html(run_id: str):
    """Serves the generated HTML audit report."""
    path = RUNS_DIR / run_id / "phase6_audit_report.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="HTML report not yet generated.")
    return HTMLResponse(content=path.read_text(encoding="utf-8"))


@app.get("/api/runs/{run_id}/report/pdf")
def get_report_pdf(run_id: str):
    """Downloads the generated PDF audit report."""
    path = RUNS_DIR / run_id / "phase6_audit_report.pdf"
    if not path.exists():
        raise HTTPException(status_code=404, detail="PDF report not yet generated.")
    return FileResponse(path, media_type="application/pdf", filename=f"audit_report_{run_id}.pdf")


@app.get("/api/runs/{run_id}/report/md")
def get_report_md(run_id: str):
    """Returns markdown report text."""
    path = RUNS_DIR / run_id / "phase6_audit_report.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Markdown report not yet generated.")
    return PlainTextResponse(content=path.read_text(encoding="utf-8"))


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str) -> Dict[str, Any]:
    """Cancels a currently active run process."""
    with TASK_LOCK:
        if run_id in ACTIVE_TASKS:
            task = ACTIVE_TASKS[run_id]
            proc = task.get("process")
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
    return {"status": "not_running", "run_id": run_id}


STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.api_route("/", methods=["GET", "HEAD"])
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>Web App Static Files Initializing...</h1>", status_code=200)
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    print(f"🚀 Launching Smart Contract Audit Web App on http://localhost:{port}")
    uvicorn.run("webapp.app:app", host="0.0.0.0", port=port, reload=True)

