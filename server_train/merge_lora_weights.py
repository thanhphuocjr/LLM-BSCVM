#!/usr/bin/env python3
"""
Merge LoRA Weights into Base Model (Version 13 / Qwen2.5-Coder-14B-Instruct)
══════════════════════════════════════════════════════════════════════════════
Script này thực hiện hợp nhất (merge & unload) trọng số LoRA adapter của v13
vào trực tiếp mô hình nền (Base Model) để tạo ra checkpoint độc lập hoàn chỉnh,
phục vụ cho:
  - Inference tốc độ cao không cần PEFT runtime (vLLM, TGI, SGLang)
  - Chuyển đổi định dạng GGUF / AWQ / GPTQ cho Ollama hoặc llama.cpp
  - Triển khai độc lập trong hệ thống kiểm tra lỗ hổng bảo mật.
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("merge_weights")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_ADAPTER_DIR = PROJECT_ROOT / "models" / "qwen25-v13-14b-detector-final"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "qwen25-v13-14b-merged"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ghép trọng số LoRA adapter vào mô hình nền LLM (v13)."
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=DEFAULT_ADAPTER_DIR,
        help="Đường dẫn thư mục LoRA adapter (mặc định: models/qwen25-v13-14b-detector-final)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="Tên hoặc đường dẫn mô hình nền Hugging Face (mặc định: Qwen/Qwen2.5-Coder-14B-Instruct)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Thư mục lưu mô hình đã ghép trọng số hoàn chỉnh (mặc định: models/qwen25-v13-14b-merged)",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Kiểu dữ liệu lưu tensor (bfloat16 hoặc float16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Thiết bị thực thi (auto, cuda, cpu, mps)",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype_name: str, torch_module):
    if dtype_name == "bfloat16":
        return torch_module.bfloat16
    if dtype_name == "float16":
        return torch_module.float16
    return torch_module.float32


def merge_lora_into_base(
    adapter_dir: Path,
    output_dir: Path,
    base_model_name: str = DEFAULT_BASE_MODEL,
    dtype_str: str = "bfloat16",
    device_name: str = "auto",
) -> None:
    start_time = time.time()
    logger.info("════════════════════════════════════════════════════════════════════")
    logger.info("  BẮT ĐẦU GHÉP TRỌNG SỐ LORA VÀO BASE MODEL (VERSION 13)")
    logger.info(f"  LoRA Adapter : {adapter_dir}")
    logger.info(f"  Base Model   : {base_model_name}")
    logger.info(f"  Output Dir   : {output_dir}")
    logger.info(f"  Dtype        : {dtype_str}")
    logger.info("════════════════════════════════════════════════════════════════════")

    if not adapter_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục adapter: {adapter_dir}")

    # Đọc threshold config nếu có để cập nhật thông số base_model
    cfg_file = adapter_dir / "threshold_config.json"
    cfg = {}
    if cfg_file.exists():
        try:
            cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
            if not base_model_name or base_model_name == DEFAULT_BASE_MODEL:
                base_model_name = cfg.get("base_model") or cfg.get("hf_model") or base_model_name
        except Exception as e:
            logger.warning(f"Không thể đọc threshold_config.json: {e}")

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        logger.error(
            "Cần cài đặt transformers, peft, torch. Hãy chạy trong môi trường .venv hoặc server: "
            "pip install transformers peft torch"
        )
        raise exc

    device = resolve_device(device_name)
    dtype = resolve_dtype(dtype_str, torch)
    logger.info(f"Target device: {device} | Dtype: {dtype}")

    # 1. Nạp tokenizer
    logger.info(f"==> [1/4] Đang nạp tokenizer từ {adapter_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Nạp Base Model
    logger.info(f"==> [2/4] Đang nạp mô hình nền từ '{base_model_name}'...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2,
        id2label={0: "Safe", 1: "Vulnerable"},
        label2id={"Safe": 0, "Vulnerable": 1},
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        base_model.to(device)

    # 3. Nạp LoRA Adapter và thực hiện ghép trọng số (merge_and_unload)
    logger.info(f"==> [3/4] Đang nạp adapter và thực hiện merge_and_unload()...")
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    merged_model = peft_model.merge_and_unload()
    merged_model.eval()

    # 4. Lưu mô hình hoàn chỉnh
    logger.info(f"==> [4/4] Đang lưu mô hình đã ghép vào: {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(
        str(output_dir),
        safe_serialization=True,
    )
    tokenizer.save_pretrained(str(output_dir))

    # Sao chép các tệp bổ trợ nếu có (line_head.pt, threshold_config.json, etc.)
    for aux_name in ("line_head.pt", "threshold_config.json", "chat_template.jinja"):
        aux_src = adapter_dir / aux_name
        if aux_src.exists():
            shutil.copy2(aux_src, output_dir / aux_name)
            logger.info(f"  • Đã sao chép tệp bổ trợ: {aux_name}")

    elapsed = time.time() - start_time
    logger.info("════════════════════════════════════════════════════════════════════")
    logger.info(f"✅ HOÀN TẤT GHÉP TRỌNG SỐ V13!")
    logger.info(f"  Thời gian thực thi : {elapsed:.2f}s")
    logger.info(f"  Thư mục mô hình    : {output_dir}")
    logger.info("════════════════════════════════════════════════════════════════════")


def main() -> None:
    args = parse_args()
    merge_lora_into_base(
        adapter_dir=args.adapter_dir,
        output_dir=args.output_dir,
        base_model_name=args.base_model,
        dtype_str=args.torch_dtype,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
