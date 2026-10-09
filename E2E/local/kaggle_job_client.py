from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from E2E.config import KAGGLE_DIR
from E2E.shared.io_utils import read_json, write_json


TERMINAL_SUCCESS = ("complete",)
TERMINAL_FAILURE = ("error", "cancel", "failed")


def _resolve_kaggle_cmd(command: list[str]) -> list[str]:
    if command and command[0] == "kaggle":
        venv_kaggle = Path(sys.executable).parent / "kaggle"
        if venv_kaggle.exists():
            return [str(venv_kaggle)] + command[1:]
        return [sys.executable, "-m", "kaggle"] + command[1:]
    return command


def _run(command: list[str], *, timeout: int = 300) -> str:
    cmd = _resolve_kaggle_cmd(command)
    completed = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(cmd)}\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    return f"{completed.stdout}\n{completed.stderr}".strip()


def get_kernel_status_safe(
    kernel_id: str,
    max_consecutive_errors: int = 10,
    retry_delay: float = 12.0,
) -> str:
    """Queries kernel status, gracefully tolerating transient network or 5xx API errors."""
    consecutive_errors = 0
    while True:
        try:
            output = _run(["kaggle", "kernels", "status", kernel_id], timeout=60)
            return output.strip()
        except Exception as exc:
            consecutive_errors += 1
            err_msg = str(exc)
            is_transient = any(
                code in err_msg
                for code in ("503", "502", "504", "429", "timeout", "timed out", "ConnectionError", "RemoteDisconnected")
            )
            if consecutive_errors < max_consecutive_errors and (is_transient or "Server Error" in err_msg):
                print(
                    f"[KaggleJobClient] Transient status check warning ({consecutive_errors}/{max_consecutive_errors}): "
                    f"{err_msg.splitlines()[0] if err_msg else 'Error'}. Retrying in {retry_delay:.0f}s..."
                )
                time.sleep(retry_delay)
            else:
                raise


def inject_request(template: Path, request: dict[str, Any], output: Path) -> Path:
    notebook = json.loads(template.read_text(encoding="utf-8"))
    payload = base64.b64encode(
        json.dumps(request, ensure_ascii=False).encode("utf-8")
    ).decode("ascii")
    parameter_cells = [
        cell
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
        and "parameters" in cell.get("metadata", {}).get("tags", [])
    ]
    if len(parameter_cells) != 1:
        raise ValueError("Notebook must contain exactly one tagged parameters cell")
    parameter_cells[0]["source"] = [
        "# Generated for a private Kaggle execution.\n",
        f"REQUEST_B64 = {json.dumps(payload)}\n",
        'DETECTOR_ADAPTER_PATH = ""\n',
        'DETECTOR_BASE_PATH = ""\n',
        'GENERATOR_MODEL_PATH = ""\n',
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return output


class KaggleJobClient:
    def __init__(
        self,
        *,
        kernel_id: str = "thanhphuocjr/codellama-e2e-smart-contract-audit",
        template_notebook: str | Path = KAGGLE_DIR / "codellama_e2e.ipynb",
        metadata_template: str | Path = KAGGLE_DIR / "kernel-metadata.json",
        poll_seconds: int = 15,
        timeout_seconds: int = 3 * 60 * 60,
    ) -> None:
        self.kernel_id = kernel_id
        self.template_notebook = Path(template_notebook)
        self.metadata_template = Path(metadata_template)
        self.poll_seconds = max(10, poll_seconds)
        self.timeout_seconds = timeout_seconds

    def prepare(self, request_path: str | Path, build_dir: str | Path) -> Path:
        build = Path(build_dir)
        build.mkdir(parents=True, exist_ok=True)
        request = read_json(request_path)
        notebook_name = "codellama_e2e.ipynb"
        inject_request(self.template_notebook, request, build / notebook_name)
        metadata = read_json(self.metadata_template)
        metadata["id"] = self.kernel_id
        metadata["code_file"] = notebook_name
        metadata["is_private"] = True
        metadata["enable_gpu"] = True
        metadata["enable_internet"] = True
        metadata["machine_shape"] = "NvidiaTeslaT4"
        write_json(build / "kernel-metadata.json", metadata)
        return build

    def push(self, build_dir: str | Path) -> str:
        output = _run(
            [
                "kaggle",
                "kernels",
                "push",
                "-p",
                str(build_dir),
                "--accelerator",
                "NvidiaTeslaT4",
            ],
            timeout=900,
        )
        match = re.search(r"kaggle\.com/code/([A-Za-z0-9_-]+/[A-Za-z0-9_-]+)", output)
        if match:
            self.kernel_id = match.group(1)
        return output

    def wait(self, expected_run_id: str | None = None) -> str:
        deadline = time.monotonic() + self.timeout_seconds
        last_status = ""
        saw_active_state = False
        time.sleep(10)
        while time.monotonic() < deadline:
            output = get_kernel_status_safe(self.kernel_id)
            normalized = output.lower()
            last_status = output.strip()
            if any(state in normalized for state in ("queued", "running", "starting", "initializing", "preparing")):
                saw_active_state = True
            if any(status in normalized for status in TERMINAL_SUCCESS):
                if saw_active_state or expected_run_id is None:
                    return last_status
                if (time.monotonic() + self.timeout_seconds - deadline) > 30:
                    return last_status
            if any(status in normalized for status in TERMINAL_FAILURE):
                raise RuntimeError(f"Kaggle kernel did not complete successfully: {last_status}")
            time.sleep(self.poll_seconds)
        raise TimeoutError(f"Timed out waiting for {self.kernel_id}. Last status: {last_status}")

    def download_result(self, output_dir: str | Path, max_retries: int = 3) -> tuple[dict[str, Any], Path]:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, max_retries + 1):
            try:
                _run(["kaggle", "kernels", "output", self.kernel_id, "-p", str(destination)], timeout=900)
                break
            except Exception as e:
                if attempt < max_retries:
                    print(f"[KaggleJobClient] Download attempt {attempt} failed ({e}). Retrying in 10s...")
                    time.sleep(10)
                else:
                    raise
        candidates = sorted(destination.glob("**/result.json"))
        if not candidates:
            raise FileNotFoundError(f"Kaggle output did not contain result.json in {destination}")
        return read_json(candidates[0]), candidates[0]

    def wait_and_download(
        self,
        request_path: str | Path,
        run_dir: str | Path,
        push_output: str = "",
    ) -> dict[str, Any]:
        run_path = Path(run_dir)
        request = read_json(request_path)
        expected_run_id = str(request.get("run_id") or "")
        print(f"[KaggleJobClient] Awaiting remote run {self.kernel_id} with expected run_id: {expected_run_id}...")

        deadline = time.monotonic() + self.timeout_seconds
        last_status = ""
        time.sleep(5)

        try:
            while time.monotonic() < deadline:
                output = get_kernel_status_safe(self.kernel_id)
                normalized = output.lower()
                last_status = output.strip()
                print(f"[KaggleJobClient] Status: {last_status}")

                if any(status in normalized for status in TERMINAL_FAILURE):
                    raise RuntimeError(f"Kaggle kernel did not complete successfully: {last_status}")

                if any(status in normalized for status in TERMINAL_SUCCESS):
                    output_dir = run_path / "kaggle-output"
                    try:
                        result, result_path = self.download_result(output_dir)
                        result_run_id = str(result.get("run_id") or "")
                        if not expected_run_id or result_run_id == expected_run_id:
                            shutil.copy2(result_path, run_path / "kaggle_result.json")
                            print(f"[KaggleJobClient] Verified result for run_id: {result_run_id}")
                            return {
                                "result": result,
                                "push_output": push_output,
                                "status_output": last_status,
                                "result_path": str(run_path / "kaggle_result.json"),
                            }
                        else:
                            print(
                                f"[KaggleJobClient] Output contains stale run_id '{result_run_id}' "
                                f"(expected '{expected_run_id}'). Waiting for new kernel execution to finish..."
                            )
                    except Exception as exc:
                        print(f"[KaggleJobClient] Downloading result pending ({exc}). Continuing to wait...")

                time.sleep(self.poll_seconds)
        except KeyboardInterrupt:
            print("\n[KaggleJobClient] Interrupted by user (Ctrl+C).")
            print(f"[KaggleJobClient] ⚠️ LƯU Ý: Máy ảo Kaggle ({self.kernel_id}) vẫn có thể đang chạy trên cloud.")
            print(f"[KaggleJobClient] 👉 Để dừng ngay trên Kaggle và tiết kiệm quota GPU: Mở https://www.kaggle.com/code/{self.kernel_id} và bấm 'Cancel run'.")
            raise

        raise TimeoutError(
            f"Timed out waiting for Kaggle run_id '{expected_run_id}'. Last status: {last_status}"
        )

    def submit_and_wait(
        self,
        request_path: str | Path,
        run_dir: str | Path,
    ) -> dict[str, Any]:
        run_path = Path(run_dir)
        build = self.prepare(request_path, run_path / "kaggle-kernel-build")
        push_output = self.push(build)
        print(f"[KaggleJobClient] Successfully pushed kernel {self.kernel_id}.")
        return self.wait_and_download(request_path, run_path, push_output=push_output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit or await one private CodeLlama audit job on Kaggle.")
    parser.add_argument("request", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--kernel-id",
        default="thanhphuocjr/codellama-e2e-smart-contract-audit",
    )
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--timeout-seconds", type=int, default=3 * 60 * 60)
    parser.add_argument("--wait-only", action="store_true", help="Do not push new kernel; only wait and download results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or args.request.parent
    client = KaggleJobClient(
        kernel_id=args.kernel_id,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    if args.wait_only:
        result = client.wait_and_download(args.request, run_dir)
    else:
        result = client.submit_and_wait(args.request, run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
