from __future__ import annotations

import argparse
import json
from pathlib import Path

from E2E.config import KAGGLE_DIR


def build_notebook(runtime_path: Path, output_path: Path) -> Path:
    runtime = runtime_path.read_text(encoding="utf-8")
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "overview",
                "metadata": {},
                "source": [
                    "# CodeLlama Smart Contract Audit E2E\n",
                    "\n",
                    "Private Kaggle GPU worker for the fine-tuned detector and CodeLlama Instruct agents. "
                    "Attach the detector adapter, CodeLlama-7b-hf, and CodeLlama-7b-Instruct-hf before running."
                ],
            },
            {
                "cell_type": "code",
                "id": "parameters",
                "execution_count": None,
                "metadata": {"tags": ["parameters"]},
                "outputs": [],
                "source": [
                    "# Populated automatically by E2E.local.kaggle_job_client.\n",
                    "REQUEST_B64 = \"\"\n",
                    "DETECTOR_ADAPTER_PATH = \"\"\n",
                    "DETECTOR_BASE_PATH = \"\"\n",
                    "GENERATOR_MODEL_PATH = \"\"\n",
                ],
            },
            {
                "cell_type": "code",
                "id": "runtime",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": runtime.splitlines(keepends=True),
            },
        ],
        "metadata": {
            "kaggle": {
                "accelerator": "gpu",
                "dataSources": [],
                "dockerImageVersionId": None,
                "isGpuEnabled": True,
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook",
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the self-contained Kaggle notebook.")
    parser.add_argument("--runtime", type=Path, default=KAGGLE_DIR / "kaggle_runtime.py")
    parser.add_argument("--output", type=Path, default=KAGGLE_DIR / "codellama_e2e.ipynb")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(build_notebook(args.runtime, args.output))


if __name__ == "__main__":
    main()
