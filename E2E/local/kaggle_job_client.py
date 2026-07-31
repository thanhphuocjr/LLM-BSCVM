from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from E2E.config import KAGGLE_DIR
from E2E.shared.io_utils import read_json, write_json


TERMINAL_SUCCESS = ("complete",)
TERMINAL_FAILURE = ("error", "cancel", "failed")


def _run(command: list[str], *, timeout: int = 300) -> str:
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    return f"{completed.stdout}\n{completed.stderr}".strip()


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
        poll_seconds: int = 20,
        timeout_seconds: int = 3 * 60 * 60,
    ) -> None:
        self.kernel_id = kernel_id
        self.template_notebook = Path(template_notebook)
        self.metadata_template = Path(metadata_template)
        self.poll_seconds = poll_seconds
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

    def wait(self) -> str:
        deadline = time.monotonic() + self.timeout_seconds
        last_status = ""
        while time.monotonic() < deadline:
            output = _run(["kaggle", "kernels", "status", self.kernel_id])
            normalized = output.lower()
            last_status = output.strip()
            if any(status in normalized for status in TERMINAL_SUCCESS):
                return last_status
            if any(status in normalized for status in TERMINAL_FAILURE):
                raise RuntimeError(f"Kaggle kernel did not complete successfully: {last_status}")
            time.sleep(self.poll_seconds)
        raise TimeoutError(f"Timed out waiting for {self.kernel_id}. Last status: {last_status}")

    def download_result(self, output_dir: str | Path) -> tuple[dict[str, Any], Path]:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        _run(["kaggle", "kernels", "output", self.kernel_id, "-p", str(destination)], timeout=900)
        candidates = sorted(destination.glob("**/result.json"))
        if not candidates:
            raise FileNotFoundError(f"Kaggle output did not contain result.json in {destination}")
        return read_json(candidates[0]), candidates[0]

    def submit_and_wait(
        self,
        request_path: str | Path,
        run_dir: str | Path,
    ) -> dict[str, Any]:
        run_path = Path(run_dir)
        build = self.prepare(request_path, run_path / "kaggle-kernel-build")
        push_output = self.push(build)
        status_output = self.wait()
        result, result_path = self.download_result(run_path / "kaggle-output")
        shutil.copy2(result_path, run_path / "kaggle_result.json")
        return {
            "result": result,
            "push_output": push_output,
            "status_output": status_output,
            "result_path": str(run_path / "kaggle_result.json"),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit one private CodeLlama audit job to Kaggle.")
    parser.add_argument("request", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--kernel-id",
        default="thanhphuocjr/codellama-e2e-smart-contract-audit",
    )
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=int, default=3 * 60 * 60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or args.request.parent
    client = KaggleJobClient(
        kernel_id=args.kernel_id,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    result = client.submit_and_wait(args.request, run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
