#!/usr/bin/env python3
"""
kaggle_kernel.py
══════════════════════════════════════════════════════════════════════════════
Chạy v13 API server (serve_v13_api.py) trên GPU T4 của Kaggle thay cho server 176.
KHÔNG chạy trực tiếp: scripts/kaggle_server.py điền cấu hình + mã server vào 2 hằng CONFIG_B64,
SERVER_SOURCE_B64 rồi đẩy file này lên Kaggle (kernel riêng tư, có GPU + Internet).

  mode "serve"   : cài thư viện -> tải Qwen gốc -> mở tunnel (ngrok domain cố định, hoặc Cloudflare quick tunnel
                   khi không có ngrok token) -> chạy server trên 2 GPU T4 -> phục vụ tối đa N giờ rồi tự tắt.
  mode "selftest": không mở ngrok; với từng dtype chấm lại các hàm tham chiếu của server 176 và đo tốc độ
                   /generate, ghi /kaggle/working/selftest.json rồi thoát.
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import base64
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

# scripts/kaggle_server.py thay 2 giá trị này lúc đẩy lên Kaggle (file trong repo không chứa bí mật).
CONFIG_B64 = "@@CONFIG_B64@@"
SERVER_SOURCE_B64 = "@@SERVER_SOURCE_B64@@"

WORK_DIR = Path("/kaggle/working")
INPUT_DIR = Path("/kaggle/input")
PORT = 6868
CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
LONG_PROMPT_TOKENS = 3900  # gần giới hạn 4.096 token của /generate = prompt dài nhất pipeline có thể gửi
VERSION_PROBE = (
    "import torch, transformers, peft, bitsandbytes; "
    "print(f'torch {torch.__version__} | transformers {transformers.__version__} | peft {peft.__version__} | '"
    "f'bitsandbytes {bitsandbytes.__version__} | GPU {torch.cuda.get_device_name(0)} x{torch.cuda.device_count()} | '"
    "f'compute capability {torch.cuda.get_device_capability(0)} (bf16 gốc cần >= 8.0)')"
)


# In một dòng log có giờ UTC (Kaggle lưu stdout làm log của kernel; xem bằng: kaggle_server.py logs).
def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# Cài thư viện đúng phiên bản lúc train (giữ torch có sẵn của Kaggle) và trả về dòng phiên bản thực tế.
def install_packages(packages: list[str]) -> str:
    log(f"pip install {' '.join(packages)}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=True)
    probe = subprocess.run([sys.executable, "-c", VERSION_PROBE], capture_output=True, text=True)
    versions = (probe.stdout or probe.stderr).strip()
    log(versions)
    return versions


# Tìm thư mục adapter LoRA v13 trong dataset đã gắn vào kernel (/kaggle/input/...).
def find_adapter_dir() -> Path:
    for config in sorted(INPUT_DIR.rglob("adapter_config.json")):
        if (config.parent / "adapter_model.safetensors").exists():
            log(f"Adapter: {config.parent}")
            return config.parent
    raise SystemExit("Không thấy adapter v13 trong /kaggle/input - chạy trước: scripts/kaggle_server.py upload")


# Tải backbone Qwen gốc (~30 GB) vào ổ còn trống nhiều nhất (/kaggle/working giới hạn 20 GB nên không dùng).
def download_base_model(repo_id: str) -> Path:
    root = max((Path(p) for p in ("/kaggle/tmp", "/tmp", "/root") if Path(p).exists()),
               key=lambda path: shutil.disk_usage(path).free)
    log(f"Tải {repo_id} vào {root} (còn trống {shutil.disk_usage(root).free / 1e9:.0f} GB)...")
    started = time.time()
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id, local_dir=str(root / repo_id.split("/")[-1]), max_workers=8,
                             allow_patterns=["*.json", "*.safetensors", "*.txt"])
    log(f"Tải xong sau {time.time() - started:.0f}s")
    return Path(path)


# Mở tunnel ngrok tới server local bằng domain cố định của tài khoản (SERVER_GPU_URL trên máy local không đổi).
def open_ngrok_tunnel(token: str, domain: str) -> str:
    from pyngrok import ngrok
    ngrok.set_auth_token(token)
    tunnel = ngrok.connect(addr=f"127.0.0.1:{PORT}", domain=domain) if domain else ngrok.connect(addr=f"127.0.0.1:{PORT}")
    return tunnel.public_url


# Mở Cloudflare quick tunnel (không cần tài khoản; URL *.trycloudflare.com đổi mỗi lần, mỗi request tối đa ~100 giây).
def open_cloudflare_tunnel() -> str:
    binary = Path("/tmp/cloudflared")
    urllib.request.urlretrieve(CLOUDFLARED_URL, binary)
    binary.chmod(0o755)
    process = subprocess.Popen([str(binary), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{PORT}"],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: [lines.put(line) for line in process.stdout], daemon=True).start()
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            line = lines.get(timeout=5)
        except queue.Empty:
            continue
        if match := re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line):
            return match.group(0)
    raise SystemExit("cloudflared không trả về URL sau 3 phút.")


# Mở tunnel theo cấu hình (ngrok nếu có token, ngược lại Cloudflare) và in dòng PUBLIC_URL= cho máy local đọc.
def open_tunnel(config: dict) -> str:
    url = open_ngrok_tunnel(config["ngrok_token"], config["ngrok_domain"]) if config.get("ngrok_token") \
        else open_cloudflare_tunnel()
    log(f"PUBLIC_URL={url}")
    return url


# Chạy serve_v13_api.py ở tiến trình con (API key, cho phép shutdown từ xa, chia model ra 2 T4); log in thẳng ra log kernel.
def start_server(server_path: Path, adapter_dir: Path, base_dir: Path, dtype: str, api_key: str) -> subprocess.Popen:
    env = {**os.environ, "SERVER_GPU_API_KEY": api_key, "SERVER_GPU_ALLOW_SHUTDOWN": "1",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "TOKENIZERS_PARALLELISM": "false"}
    command = [sys.executable, "-u", str(server_path), "--host", "127.0.0.1", "--port", str(PORT),
               "--adapter-dir", str(adapter_dir), "--base-model", str(base_dir), "--dtype", dtype, "--multi-gpu"]
    log(f"Khởi động server (dtype {dtype}, multi-GPU)...")
    return subprocess.Popen(command, env=env)


# Gọi API của server local với header X-API-Key (giống client E2E/common/gpu_client.py).
def call_api(path: str, payload: dict | None = None, api_key: str = "", timeout: int = 900) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=data,
                                     headers={"Content-Type": "application/json", "X-API-Key": api_key})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


# Chờ server nạp xong model (health = ready); server thoát giữa chừng -> SystemExit kèm mã lỗi.
def wait_ready(process: subprocess.Popen, limit_s: int = 5400) -> dict:
    started = time.time()
    while time.time() - started < limit_s:
        if process.poll() is not None:
            raise SystemExit(f"Server thoát khi đang nạp model (mã {process.returncode}) - xem log phía trên.")
        try:
            health = call_api("/api/v1/health", timeout=5)
            if health.get("status") == "ready":
                log(f"Server sẵn sàng sau {time.time() - started:.0f}s: {health}")
                return health
        except Exception:
            pass
        time.sleep(10)
    raise SystemExit("Quá thời gian chờ server nạp model.")


# Dừng server con (SIGTERM, quá 30 giây thì kill) để trả VRAM.
def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


# Chấm lại các hàm tham chiếu và so với điểm server 176 đã lưu (cùng adapter, bf16 trên RTX 4070 Ti SUPER).
def check_reference(reference: dict, api_key: str) -> list[dict]:
    threshold = float(reference["threshold"])
    rows = []
    for case in reference["cases"]:
        started = time.time()
        row = {"unit_id": case["unit_id"], "score_176": case["score_176"]}
        try:
            score = float(call_api("/api/v1/detect", {"code": case["code"], "threshold": threshold},
                                   api_key)["vulnerable_probability"])
            row.update(score=round(score, 4), diff=round(score - case["score_176"], 4),
                       same_label=(score >= threshold) == (case["score_176"] >= threshold))
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        row["elapsed_s"] = round(time.time() - started, 2)
        log(f"  detect {row}")
        rows.append(row)
    return rows


# Đo /generate: câu trả lời ngắn (tốc độ sinh token) và prompt dài ~3.900 token (prompt dài nhất pipeline có thể gửi).
def check_generate(api_key: str, tokenizer, reference: dict) -> dict:
    short_prompt = ("Explain in 4 short sentences what a reentrancy vulnerability in Solidity is "
                    "and how to prevent it.")
    filler = "\n\n".join(case["code"] for case in reference["cases"])
    long_prompt = "Review the Solidity code below and answer with one word: OK or BUG.\n\n" + filler
    while len(tokenizer(long_prompt)["input_ids"]) < LONG_PROMPT_TOKENS:
        long_prompt += "\n\n" + filler
    long_prompt = tokenizer.decode(tokenizer(long_prompt)["input_ids"][:LONG_PROMPT_TOKENS])
    result = {"long_prompt_tokens": len(tokenizer(long_prompt)["input_ids"])}
    for name, prompt, max_tokens in (("short", short_prompt, 256), ("long_prompt", long_prompt, 32)):
        started = time.time()
        try:
            text = call_api("/api/v1/generate", {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0},
                            api_key)["response"]
            elapsed = time.time() - started
            tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            result[name] = {"elapsed_s": round(elapsed, 1), "new_tokens": tokens,
                            "tokens_per_s": round(tokens / elapsed, 2) if elapsed else None, "text": text[:800]}
        except Exception as error:
            result[name] = {"error": f"{type(error).__name__}: {error}", "elapsed_s": round(time.time() - started, 1)}
        log(f"  generate {name}: {result[name]}")
    return result


# Chế độ selftest: lần lượt từng dtype -> nạp server -> chấm tham chiếu + đo generate -> ghi selftest.json.
def run_selftest(config: dict, server_path: Path, adapter_dir: Path, base_dir: Path, versions: str) -> None:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    report = {"versions": versions, "runs": []}
    for dtype in config["dtypes"]:
        started = time.time()
        process = start_server(server_path, adapter_dir, base_dir, dtype, config["api_key"])
        run = {"dtype": dtype}
        try:
            run["health"] = wait_ready(process)
            run["load_s"] = round(time.time() - started, 1)
            run["detect"] = check_reference(config["reference"], config["api_key"])
            run["generate"] = check_generate(config["api_key"], tokenizer, config["reference"])
        except SystemExit as error:
            run["error"] = str(error)
        finally:
            stop_server(process)
        report["runs"].append(run)
        (WORK_DIR / "selftest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log("Selftest xong: /kaggle/working/selftest.json")


# Chế độ serve: mở tunnel -> nạp server -> tự kiểm tra -> phục vụ tới hết N giờ hoặc tới khi client gọi /shutdown.
def run_serve(config: dict, server_path: Path, adapter_dir: Path, base_dir: Path) -> None:
    public_url = open_tunnel(config)
    process = start_server(server_path, adapter_dir, base_dir, config["dtypes"][0], config["api_key"])
    try:
        wait_ready(process)
        check_reference(config["reference"], config["api_key"])
        deadline = time.time() + float(config["hours"]) * 3600
        log(f"ĐANG PHỤC VỤ tại {public_url} tới {time.strftime('%H:%M UTC', time.gmtime(deadline))}")
        last_report = time.time()
        while process.poll() is None and time.time() < deadline:
            time.sleep(20)
            if time.time() - last_report >= 600:
                last_report = time.time()
                try:
                    health = call_api("/api/v1/health", timeout=30)
                except Exception as error:
                    health = f"health lỗi: {error}"
                log(f"Còn {(deadline - time.time()) / 60:.0f} phút | {health}")
    finally:
        stop_server(process)
    log("Server đã dừng, kernel kết thúc (quota GPU ngừng tính).")


# Điểm vào: giải mã cấu hình + mã server -> cài thư viện -> adapter + backbone -> selftest hoặc serve.
def main() -> None:
    if CONFIG_B64.startswith("@@"):
        raise SystemExit("File mẫu chưa có cấu hình - đẩy lên Kaggle bằng: .venv/bin/python scripts/kaggle_server.py start")
    config = json.loads(base64.b64decode(CONFIG_B64))
    server_path = WORK_DIR / "serve_v13_api.py"
    server_path.write_bytes(base64.b64decode(SERVER_SOURCE_B64))
    log(f"Mode {config['mode']} | dtype {config['dtypes']} | base {config['base_model']}")
    versions = install_packages(config["packages"])
    adapter_dir = find_adapter_dir()
    base_dir = download_base_model(config["base_model"])
    if config["mode"] == "selftest":
        run_selftest(config, server_path, adapter_dir, base_dir, versions)
    else:
        run_serve(config, server_path, adapter_dir, base_dir)


if __name__ == "__main__":
    main()
