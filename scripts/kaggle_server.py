#!/usr/bin/env python3
"""Điều khiển GPU server v13 chạy trên Kaggle T4 (thay server 176) từ máy local. Chạy từ thư mục gốc repo:

  .venv/bin/python scripts/kaggle_server.py setup      nhập ngrok authtoken + domain -> ghi SERVER_GPU_URL, API key vào .env
  .venv/bin/python scripts/kaggle_server.py upload     đẩy adapter models/qwen25-v13-14b-detector-final lên Kaggle (1 lần)
  .venv/bin/python scripts/kaggle_server.py selftest   chạy thử trên Kaggle (không cần ngrok): so điểm với server 176
  .venv/bin/python scripts/kaggle_server.py result     tải + in kết quả selftest
  .venv/bin/python scripts/kaggle_server.py start      bật server qua ngrok (--hours 3, --dtype fp16|bf16), tự tắt sau N giờ
  .venv/bin/python scripts/kaggle_server.py start --cloudflare   không cần ngrok: chờ URL trycloudflare rồi tự ghi vào .env
  .venv/bin/python scripts/kaggle_server.py url        đọc URL tunnel từ log kernel đang chạy và ghi vào .env (khi start --cloudflare
                                                       đã thoát trước khi có URL, vd. Kaggle xếp hàng GPU quá lâu)
  .venv/bin/python scripts/kaggle_server.py status     trạng thái kernel Kaggle + server qua tunnel
  .venv/bin/python scripts/kaggle_server.py check      chấm lại 6 hàm tham chiếu qua ngrok, so với server 176
  .venv/bin/python scripts/kaggle_server.py logs       log kernel (--follow để xem trực tiếp, --selftest cho kernel thử)
  .venv/bin/python scripts/kaggle_server.py stop       tắt server ngay (ngừng tính quota GPU)
  .venv/bin/python scripts/kaggle_server.py quota      số giờ GPU Kaggle còn lại trong tuần
"""

from __future__ import annotations

import argparse
import base64
import getpass
import json
import os
import queue
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from E2E.common.gpu_client import detect_via_server, generate_via_server, get_server_url, load_env_file  # noqa: E402

ENV_FILE = ROOT / ".env"
ADAPTER_DIR = ROOT / "models" / "qwen25-v13-14b-detector-final"
KERNEL_TEMPLATE = ROOT / "gpu_server" / "kaggle_kernel.py"
SERVER_SOURCE = ROOT / "gpu_server" / "serve_v13_api.py"
REFERENCE_FILE = ROOT / "gpu_server" / "reference_scores.json"
DATASET_SLUG = "llm-bscvm-v13-adapter"
KERNEL_SLUGS = {"serve": "llm-bscvm-v13-server", "selftest": "llm-bscvm-v13-selftest"}
BASE_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"
# Cùng phiên bản lúc train v13 (data/results/detector_v13_training/train_v13_*.log, adapter_config.json); torch giữ bản Kaggle.
PACKAGES = ["transformers==5.17.0", "peft==0.20.0", "bitsandbytes", "accelerate", "fastapi", "uvicorn", "pyngrok"]
ACCELERATOR = "NvidiaTeslaT4"
SETUP_HOURS = 1.0       # thời gian dự phòng cho cài thư viện + tải ~30 GB + nạp model
MAX_SESSION_HOURS = 12  # giới hạn một phiên GPU của Kaggle


# Đường dẫn Kaggle CLI (cài trong .venv cạnh python đang chạy, nếu không thì tìm trong PATH).
def kaggle_cli() -> str:
    local = Path(sys.executable).parent / "kaggle"
    if local.exists():
        return str(local)
    if found := shutil.which("kaggle"):
        return found
    sys.exit("Không thấy Kaggle CLI: .venv/bin/pip install kaggle")


# Chạy một lệnh Kaggle CLI, in lệnh ra màn hình; check=True -> lỗi thì dừng; capture -> bỏ dòng nhắc nâng cấp CLI.
def run_kaggle(*args: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    command = [kaggle_cli(), *args]
    if not capture:
        print("$ kaggle " + " ".join(args), flush=True)
    result = subprocess.run(command, text=True, capture_output=capture)
    if capture:
        result.stdout = "\n".join(line for line in (result.stdout or "").splitlines() if "outdated" not in line)
    if check and result.returncode != 0:
        sys.exit(f"Lệnh Kaggle thất bại (mã {result.returncode}): {(result.stderr or '').strip()}")
    return result


# Tên tài khoản Kaggle (KAGGLE_USERNAME hoặc trường username trong kaggle.json; không đọc key).
def kaggle_user() -> str:
    if user := os.environ.get("KAGGLE_USERNAME"):
        return user
    for path in (Path.home() / ".kaggle" / "kaggle.json", Path.home() / ".config" / "kaggle" / "kaggle.json"):
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["username"]
    sys.exit("Chưa cấu hình Kaggle CLI (~/.kaggle/kaggle.json).")


# Đọc .env vào os.environ rồi trả giá trị khoá bắt buộc; thiếu -> nhắc chạy setup.
def require_env(key: str) -> str:
    load_env_file()
    value = os.environ.get(key, "").strip()
    if not value:
        sys.exit(f"Thiếu {key} trong .env - chạy trước: .venv/bin/python scripts/kaggle_server.py setup")
    return value


# Ghi/đổi các khoá trong .env, giữ các dòng khác; SERVER_GPU_URL cũ được giữ lại thành comment để quay về 176.
def update_env(values: dict[str, str]) -> None:
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
    output, written = [], set()
    for line in lines:
        key, sep, old = line.partition("=")
        key = key.strip()
        if not sep or line.lstrip().startswith("#") or key not in values:
            output.append(line)
            continue
        if key in written:
            continue
        if key == "SERVER_GPU_URL" and old.strip() != values[key]:
            output.append(f"# SERVER_GPU_URL={old.strip()}")
        output.append(f"{key}={values[key]}")
        written.add(key)
    missing = [key for key in values if key not in written]
    header = "# GPU server trên Kaggle (scripts/kaggle_server.py)"
    if missing:
        output += ([] if header in output else ["", header]) + [f"{key}={values[key]}" for key in missing]
    ENV_FILE.write_text("\n".join(output) + "\n", encoding="utf-8")


# setup: ngrok authtoken + domain cố định -> SERVER_GPU_URL, SERVER_GPU_API_KEY (tạo ngẫu nhiên), NGROK_AUTHTOKEN,
# SERVER_GPU_TIMEOUT=300 trong .env.
def cmd_setup(args: argparse.Namespace) -> None:
    load_env_file()
    token = args.ngrok_token or getpass.getpass("ngrok authtoken (dashboard.ngrok.com > Your Authtoken): ").strip()
    domain = args.ngrok_domain or input("ngrok domain (dashboard.ngrok.com > Domains, vd. abc-xyz.ngrok-free.app): ")
    domain = domain.strip().removeprefix("https://").removeprefix("http://").rstrip("/")
    if not token or not domain:
        sys.exit("Cần cả authtoken và domain.")
    api_key = os.environ.get("SERVER_GPU_API_KEY", "").strip() or secrets.token_urlsafe(32)
    timeout = os.environ.get("SERVER_GPU_TIMEOUT", "").strip() or "300"
    update_env({"SERVER_GPU_URL": f"https://{domain}", "SERVER_GPU_API_KEY": api_key, "NGROK_AUTHTOKEN": token,
                "SERVER_GPU_TIMEOUT": timeout})
    print(f"Đã ghi .env: SERVER_GPU_URL=https://{domain}, SERVER_GPU_API_KEY, NGROK_AUTHTOKEN, SERVER_GPU_TIMEOUT={timeout}")
    print("Bước tiếp: .venv/bin/python scripts/kaggle_server.py start")


# upload: đẩy adapter v13 (trừ training_args.bin) thành dataset RIÊNG TƯ <user>/llm-bscvm-v13-adapter, chờ Kaggle xử lý xong.
def cmd_upload(args: argparse.Namespace) -> None:
    slug = f"{kaggle_user()}/{DATASET_SLUG}"
    files = [path for path in sorted(ADAPTER_DIR.iterdir()) if path.is_file() and path.name != "training_args.bin"]
    if not any(path.name == "adapter_model.safetensors" for path in files):
        sys.exit(f"Không thấy adapter_model.safetensors trong {ADAPTER_DIR}")
    exists = run_kaggle("datasets", "status", slug, check=False, capture=True).returncode == 0
    with tempfile.TemporaryDirectory(prefix="kaggle-adapter-") as tmp:
        stage = Path(tmp)
        for path in files:
            # cp -c = bản sao APFS (không tốn thêm 1 GB); ổ khác thì copy thường.
            if subprocess.run(["cp", "-c", str(path), str(stage / path.name)], capture_output=True).returncode != 0:
                shutil.copy2(path, stage / path.name)
        metadata = {"title": "LLM-BSCVM v13 adapter", "id": slug, "licenses": [{"name": "other"}]}
        (stage / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if exists:
            run_kaggle("datasets", "version", "-p", str(stage), "-m", args.message)
        else:
            run_kaggle("datasets", "create", "-p", str(stage))
    print("Đang chờ Kaggle xử lý dataset...", flush=True)
    for _ in range(80):
        status = run_kaggle("datasets", "status", slug, check=False, capture=True).stdout.strip()
        if status.lower() == "ready":
            print(f"Dataset sẵn sàng (riêng tư): https://www.kaggle.com/datasets/{slug}")
            return
        time.sleep(15)
    print(f"Dataset chưa xong (trạng thái: {status}); kiểm tra lại: kaggle datasets status {slug}")


# Base64 của chuỗi/bytes (để nhúng cấu hình và mã server vào kernel).
def b64(data: str | bytes) -> str:
    return base64.b64encode(data.encode("utf-8") if isinstance(data, str) else data).decode("ascii")


# Dựng kernel từ gpu_server/kaggle_kernel.py (nhúng cấu hình + serve_v13_api.py hiện tại) và đẩy lên Kaggle với GPU T4,
# Internet, riêng tư, gắn dataset adapter; Kaggle chạy ngay sau khi đẩy, tối đa timeout_hours.
def push_kernel(mode: str, config: dict, timeout_hours: float) -> str:
    user = kaggle_user()
    kernel_id = f"{user}/{KERNEL_SLUGS[mode]}"
    source = KERNEL_TEMPLATE.read_text(encoding="utf-8")
    source = source.replace("@@CONFIG_B64@@", b64(json.dumps(config)), 1)
    source = source.replace("@@SERVER_SOURCE_B64@@", b64(SERVER_SOURCE.read_bytes()), 1)
    metadata = {"id": kernel_id, "title": KERNEL_SLUGS[mode], "code_file": "kaggle_kernel.py", "language": "python",
                "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
                "dataset_sources": [f"{user}/{DATASET_SLUG}"], "competition_sources": [], "kernel_sources": [],
                "model_sources": []}
    seconds = int(min(timeout_hours, MAX_SESSION_HOURS) * 3600)
    with tempfile.TemporaryDirectory(prefix="kaggle-kernel-") as tmp:
        (Path(tmp) / "kaggle_kernel.py").write_text(source, encoding="utf-8")
        (Path(tmp) / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        run_kaggle("kernels", "push", "-p", tmp, "--accelerator", ACCELERATOR, "-t", str(seconds))
    return kernel_id


# Nội dung gpu_server/reference_scores.json: 6 hàm demo + điểm detector đã chấm trên server 176.
def load_reference() -> dict:
    return json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))


# selftest: chạy thử trên Kaggle không cần ngrok (bf16 rồi fp16), mất ~30-45 phút quota GPU.
def cmd_selftest(args: argparse.Namespace) -> None:
    config = {"mode": "selftest", "dtypes": args.dtypes.split(","), "packages": PACKAGES, "base_model": BASE_MODEL,
              "api_key": secrets.token_urlsafe(16), "reference": load_reference()}
    kernel_id = push_kernel("selftest", config, timeout_hours=2.5)
    print(f"\nĐã đẩy {kernel_id}. Theo dõi: .venv/bin/python scripts/kaggle_server.py logs --selftest --follow")
    print("Khi xong (~30-45 phút): .venv/bin/python scripts/kaggle_server.py result")


# result: tải selftest.json của lần chạy thử gần nhất và in bảng so sánh với server 176.
def cmd_result(args: argparse.Namespace) -> None:
    kernel_id = f"{kaggle_user()}/{KERNEL_SLUGS['selftest']}"
    print(run_kaggle("kernels", "status", kernel_id, check=False, capture=True).stdout.strip())
    with tempfile.TemporaryDirectory(prefix="kaggle-out-") as tmp:
        run_kaggle("kernels", "output", kernel_id, "-p", tmp, "--file-pattern", r"selftest\.json", "-o", "-q",
                   check=False, capture=True)
        report_file = Path(tmp) / "selftest.json"
        if not report_file.exists():
            sys.exit("Chưa có selftest.json (kernel còn chạy hoặc lỗi) - xem: scripts/kaggle_server.py logs --selftest")
        report = json.loads(report_file.read_text(encoding="utf-8"))
    print(f"\n{report.get('versions')}")
    for run in report["runs"]:
        print(f"\n── dtype {run['dtype']} " + (f"| nạp model {run['load_s']}s" if "load_s" in run else ""))
        if "error" in run:
            print(f"   LỖI: {run['error']}")
        print_detect_rows(run.get("detect", []))
        for name, item in (run.get("generate") or {}).items():
            if isinstance(item, dict):
                print(f"   generate {name}: " + ", ".join(f"{key}={value}" for key, value in item.items() if key != "text"))
                if item.get("text"):
                    print(f"      » {item['text'][:300]!r}")


# In bảng điểm detector: hàm, điểm 176, điểm mới, chênh lệch, cùng nhãn hay không.
def print_detect_rows(rows: list[dict]) -> None:
    if rows:
        print(f"   {'hàm':30} {'176':>7} {'mới':>7} {'lệch':>8}  nhãn   thời gian")
    for row in rows:
        if "error" in row:
            print(f"   {row['unit_id']:30} {row['score_176']:>7} LỖI: {row['error']}")
            continue
        same = "giống" if row["same_label"] else "KHÁC"
        print(f"   {row['unit_id']:30} {row['score_176']:>7.4f} {row['score']:>7.4f} {row['diff']:>+8.4f}  {same}  "
              f"{row['elapsed_s']}s")
    if rows and all("diff" in row for row in rows):
        print(f"   lệch tuyệt đối lớn nhất: {max(abs(row['diff']) for row in rows):.4f}")


# start: đẩy kernel serve (tunnel ngrok từ .env, hoặc --cloudflare), server tự tắt sau --hours giờ.
def cmd_start(args: argparse.Namespace) -> None:
    load_env_file()
    api_key = os.environ.get("SERVER_GPU_API_KEY", "").strip()
    if not api_key:
        api_key = secrets.token_urlsafe(32)
        update_env({"SERVER_GPU_API_KEY": api_key})
    tunnel = {"ngrok_token": "", "ngrok_domain": ""}
    if not args.cloudflare:
        url = require_env("SERVER_GPU_URL")
        if ".ngrok" not in url:
            sys.exit(f"SERVER_GPU_URL={url} chưa trỏ tới ngrok - chạy setup, hoặc dùng: start --cloudflare")
        tunnel = {"ngrok_token": require_env("NGROK_AUTHTOKEN"),
                  "ngrok_domain": url.removeprefix("https://").removeprefix("http://")}
    config = {"mode": "serve", "dtypes": [args.dtype], "hours": args.hours, "packages": PACKAGES,
              "base_model": BASE_MODEL, "api_key": api_key, **tunnel, "reference": load_reference()}
    kernel_id = push_kernel("serve", config, timeout_hours=args.hours + SETUP_HOURS)
    print(f"\nĐã đẩy {kernel_id} (dtype {args.dtype}, phục vụ {args.hours} giờ sau khi nạp xong).")
    if args.cloudflare:
        save_public_url(wait_for_public_url(kernel_id))
    print("Cài thư viện + tải Qwen + nạp model mất ~10-15 phút. Theo dõi:")
    print("  .venv/bin/python scripts/kaggle_server.py status          (ready = dùng được)")
    print("  .venv/bin/python scripts/kaggle_server.py logs --follow")


# Sửa chữ tiếng Việt bị Kaggle CLI giải mã sai (UTF-8 đọc thành Latin-1), dòng nào không sửa được thì giữ nguyên.
def fix_text(line: str) -> str:
    try:
        return line.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return line


# Theo dõi log kernel (kaggle kernels logs --follow), in ra màn hình; until -> dừng khi gặp dòng khớp regex và trả match.
# Luồng log của Kaggle đôi khi treo (lỗi 500 rồi im lặng): quá stall_s giây không có dòng mới thì kết nối lại.
def follow_logs(kernel_id: str, until: str | None = None, limit_s: float = 2400, stall_s: float = 180) -> re.Match | None:
    printed, deadline, queued_notice = 0, time.time() + limit_s, False
    while time.time() < deadline:
        process = subprocess.Popen([kaggle_cli(), "kernels", "logs", kernel_id, "--follow"], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, errors="replace")
        lines: queue.Queue = queue.Queue()
        threading.Thread(target=lambda: [lines.put(raw) for raw in process.stdout] + [lines.put(None)],
                         daemon=True).start()
        seen = 0
        while time.time() < deadline:
            try:
                raw = lines.get(timeout=stall_s)
            except queue.Empty:
                process.terminate()  # treo: kết nối lại
                break
            if raw is None:  # luồng kết thúc (kernel xong hoặc chưa chạy)
                break
            line = fix_text(raw.rstrip("\n"))
            # "outdated": nhắc nâng cấp CLI; "logs/stream" 500: kernel chưa có phiên chạy (đang xếp hàng) -> bỏ qua.
            if "outdated" in line or ("Server Error" in line and "logs/stream" in line):
                continue
            seen += 1
            if seen > printed:  # lần kết nối lại phát lại từ đầu: bỏ các dòng đã in
                print(line, flush=True)
                printed = seen
            if until and (match := re.search(until, line)):
                process.terminate()
                return match
        process.wait()
        status = run_kaggle("kernels", "status", kernel_id, check=False, capture=True).stdout.upper()
        if any(state in status for state in ("COMPLETE", "ERROR", "CANCEL")):
            return None  # kernel đã kết thúc: không còn log mới
        if "QUEUED" in status and not queued_notice:
            print("Kaggle đang xếp hàng cấp GPU (QUEUED) - thường vài phút, script tự đợi...", flush=True)
            queued_notice = True
        time.sleep(15)  # kernel còn xếp hàng/chưa có log, hoặc vừa treo: thử lại
    return None


# Chờ dòng PUBLIC_URL=... trong log kernel serve (~3 phút sau khi có GPU; Kaggle có lúc xếp hàng GPU > 30 phút).
def wait_for_public_url(kernel_id: str) -> str:
    print("Đang chờ kernel mở tunnel (log trực tiếp bên dưới)...\n", flush=True)
    match = follow_logs(kernel_id, until=r"PUBLIC_URL=(https://\S+)", limit_s=5400)
    if not match:
        sys.exit("Không thấy PUBLIC_URL trong log - kernel còn chạy thì thử lại: .venv/bin/python scripts/kaggle_server.py url")
    return match.group(1)


# Ghi URL tunnel vào .env (SERVER_GPU_URL; SERVER_GPU_TIMEOUT mặc định 300 giây).
def save_public_url(url: str) -> None:
    load_env_file()
    update_env({"SERVER_GPU_URL": url, "SERVER_GPU_TIMEOUT": os.environ.get("SERVER_GPU_TIMEOUT") or "300"})
    print(f"\nĐã ghi SERVER_GPU_URL={url} vào .env (URL Cloudflare đổi sau mỗi lần start).")


# url: đọc lại PUBLIC_URL từ log kernel serve đang chạy và ghi vào .env.
def cmd_url(args: argparse.Namespace) -> None:
    save_public_url(wait_for_public_url(f"{kaggle_user()}/{KERNEL_SLUGS['serve']}"))
    print("Kiểm tra: .venv/bin/python scripts/kaggle_server.py status")


# Gọi thẳng /api/v1/health qua SERVER_GPU_URL và diễn giải lỗi của ngrok (tunnel chưa mở / server đang nạp).
def probe_server() -> str:
    url = f"{get_server_url()}/api/v1/health"
    request = urllib.request.Request(url, headers={"ngrok-skip-browser-warning": "1"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            health = json.loads(response.read().decode("utf-8"))
        return (f"{health.get('status')} | {health.get('gpu_device')} | dtype {health.get('dtype')} | "
                f"VRAM {health.get('vram_used_mb')}/{health.get('vram_total_mb')} MB | uptime {health.get('uptime_seconds')}s")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "replace")
        if "ERR_NGROK_8012" in body or error.code == 502:
            return "tunnel đã mở, server đang nạp model (chưa dùng được)"
        if "ERR_NGROK_3200" in body or error.code in (404, 530):
            return "offline - tunnel chưa mở (kernel đang cài/tải hoặc đã tắt)"
        return f"HTTP {error.code}: {body[:200]}"
    except Exception as error:
        return f"offline ({type(error).__name__}: {error})"


# status: trạng thái kernel serve trên Kaggle + health của server qua ngrok.
def cmd_status(args: argparse.Namespace) -> None:
    load_env_file()
    kernel_id = f"{kaggle_user()}/{KERNEL_SLUGS['serve']}"
    print(run_kaggle("kernels", "status", kernel_id, check=False, capture=True).stdout.strip())
    print(f"Server {get_server_url()}: {probe_server()}")


# check: chấm lại 6 hàm tham chiếu qua ngrok (đúng đường client của pipeline) + đo một lần /generate.
def cmd_check(args: argparse.Namespace) -> None:
    load_env_file()
    reference = load_reference()
    threshold = float(reference["threshold"])
    print(f"Server {get_server_url()}: {probe_server()}\n")
    rows = []
    for case in reference["cases"]:
        started = time.time()
        row = {"unit_id": case["unit_id"], "score_176": case["score_176"]}
        try:
            score = float(detect_via_server(case["code"], threshold=threshold)["vulnerable_probability"])
            row.update(score=score, diff=score - case["score_176"],
                       same_label=(score >= threshold) == (case["score_176"] >= threshold))
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        row["elapsed_s"] = round(time.time() - started, 2)
        rows.append(row)
    print_detect_rows(rows)
    started = time.time()
    try:
        text = generate_via_server("In one sentence, what is SWC-107?", temperature=0.0, max_tokens=80)
        print(f"\n   generate: {time.time() - started:.1f}s » {text[:200]!r}")
    except Exception as error:
        print(f"\n   generate LỖI sau {time.time() - started:.1f}s: {type(error).__name__}: {error}")


# logs: log kernel serve (hoặc selftest), đã sửa lỗi hiển thị tiếng Việt; --follow xem trực tiếp khi kernel đang chạy.
def cmd_logs(args: argparse.Namespace) -> None:
    kernel_id = f"{kaggle_user()}/{KERNEL_SLUGS['selftest' if args.selftest else 'serve']}"
    if args.follow:
        follow_logs(kernel_id)
        return
    for line in run_kaggle("kernels", "logs", kernel_id, check=False, capture=True).stdout.splitlines():
        print(fix_text(line))


# stop: gọi /api/v1/shutdown (cần API key) -> server dừng -> kernel kết thúc, quota GPU ngừng tính.
def cmd_stop(args: argparse.Namespace) -> None:
    url = f"{get_server_url()}/api/v1/shutdown"
    headers = {"X-API-Key": require_env("SERVER_GPU_API_KEY"), "ngrok-skip-browser-warning": "1"}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=b"", headers=headers), timeout=15) as response:
            print(f"Server trả lời: {response.read().decode('utf-8')}")
    except Exception as error:
        print(f"Không gọi được shutdown ({error}). Nếu kernel vẫn chạy: Kaggle > Your Work > "
              f"{KERNEL_SLUGS['serve']} > Cancel run.")


# quota: số giờ GPU Kaggle đã dùng / còn lại trong tuần.
def cmd_quota(args: argparse.Namespace) -> None:
    run_kaggle("quota", check=False)


# Điểm vào: phân tích lệnh con và gọi hàm tương ứng.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("setup", help="ghi cấu hình ngrok + API key vào .env")
    setup.add_argument("--ngrok-token", default="")
    setup.add_argument("--ngrok-domain", default="")
    upload = commands.add_parser("upload", help="đẩy adapter v13 lên Kaggle (dataset riêng tư)")
    upload.add_argument("-m", "--message", default="Update v13 adapter")
    selftest = commands.add_parser("selftest", help="chạy thử trên Kaggle, so điểm với server 176")
    selftest.add_argument("--dtypes", default="bf16,fp16", help="danh sách dtype cần thử, cách nhau dấu phẩy")
    commands.add_parser("result", help="in kết quả selftest")
    start = commands.add_parser("start", help="bật server trên Kaggle")
    start.add_argument("--hours", type=float, default=3.0, help="số giờ phục vụ sau khi nạp xong (mặc định 3)")
    start.add_argument("--dtype", choices=["bf16", "fp16"], default=None,
                       help="mặc định: SERVER_GPU_DTYPE trong .env, không có thì fp16 (T4: nhanh gấp 2-3 lần bf16)")
    start.add_argument("--cloudflare", action="store_true",
                       help="dùng Cloudflare quick tunnel thay ngrok (không cần tài khoản, request tối đa ~100 giây)")
    commands.add_parser("url", help="đọc URL tunnel từ log kernel đang chạy, ghi vào .env")
    commands.add_parser("status", help="trạng thái kernel + server")
    commands.add_parser("check", help="so điểm detector qua ngrok với server 176")
    logs = commands.add_parser("logs", help="log kernel")
    logs.add_argument("--follow", action="store_true")
    logs.add_argument("--selftest", action="store_true")
    commands.add_parser("stop", help="tắt server")
    commands.add_parser("quota", help="quota GPU Kaggle")
    args = parser.parse_args()
    if args.command == "start" and args.dtype is None:
        load_env_file()
        # Kaggle T4 không có bf16 gốc: selftest 2026-11-25 cho fp16 cùng nhãn với server 176 trên 6/6 hàm tham chiếu
        # (lệch điểm <= 0.04) và đọc prompt 3.900 token trong 15 giây thay vì 49 giây.
        args.dtype = os.environ.get("SERVER_GPU_DTYPE", "").strip() or "fp16"
    handlers = {"setup": cmd_setup, "upload": cmd_upload, "selftest": cmd_selftest, "result": cmd_result,
                "start": cmd_start, "url": cmd_url, "status": cmd_status, "check": cmd_check, "logs": cmd_logs, "stop": cmd_stop,
                "quota": cmd_quota}
    handlers[args.command](args)


if __name__ == "__main__":
    main()
