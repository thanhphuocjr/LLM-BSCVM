#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Qwen2.5-Coder-14B-Instruct v13 — Quality-First Smart Contract Detector      ║
║  QLoRA 4-bit Fine-Tuning | GPU 0 Dedicated (16GB VRAM)                       ║
║                                                                              ║
║  Triết lý v13: CHẤT LƯỢNG LÀ TIÊN QUYẾT (QUALITY-FIRST)                      ║
║  • Nâng cấp Backbone: Qwen/Qwen2.5-Coder-14B-Instruct                        ║
║    (14.7B params, 48 layers, hidden_size 5120, vocab 152k tokens)            ║
║  • Chạy chuyên biệt trên GPU 0 (14.1GB VRAM free, không ảnh hưởng GPU 1)      ║
║  • Ngữ cảnh tối ưu: MAX_LEN = 1024 (Bao phủ 99.92% code DAppSCAN thực tế)      ║
║  • LoRA mở rộng: r = 64, alpha = 128 trên 7 projection layers               ║
║  • Kiến trúc Decoupled Dual-Head:                                           ║
║    - score_head: nn.Linear(5120, 2) cho Sequence Classification (Safe/Vuln)  ║
║    - line_head:  nn.Linear(5120, 1) chuyên trách định vị dòng lỗi           ║
║  • Hàm mất mát kết hợp Focal Loss (gamma=2.0) + Source Weighting             ║
║    (DAppSCAN x2.5, iAudit x1.5, Kaggle x0.8) - Không bỏ dữ liệu              ║
║  • Huấn luyện chuyên sâu: 10 Epochs, Patience = 5, Cosine Annealing          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Chạy trên server:
    bash run_train.sh
Hoặc:
    python3 train_qwen25_v13.py 2>&1 | tee train_v13.log
"""

import os
import sys
import json
import glob
import random
import collections
import csv
import inspect
import math
import time
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import transformers
from torch.utils.data import WeightedRandomSampler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# ══════════════════════════════════════════════════════════════════════════════
# ENV CONFIG — Tối ưu cho Dual GPU Server (2x RTX 4070 Ti SUPER)
# ══════════════════════════════════════════════════════════════════════════════
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Chống phân mảnh bộ nhớ CUDA (tương thích cả PyTorch mới và cũ)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Sử dụng cả 2 GPU để chia tải (GPU 0 gánh ~70% layers, GPU 1 gánh ~30% layers vào 5.5GB trống)
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# SEED
# ══════════════════════════════════════════════════════════════════════════════
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS — V13 QUALITY FIRST CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# ─── Model Backbone: Qwen2.5-Coder-14B-Instruct ───
HF_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"

# ─── Context & Batch Size (Chất lượng tối đa, MAX_LEN=1024) ───
MAX_LEN = 1024       # Bao phủ 99.92% DAppSCAN (2,643/2,645 mẫu) và 99.87% toàn dataset
TRAIN_BS = 1         # Batch size 1 mỗi forward step
EVAL_BS = 1          # Batch size 1 an toàn khi đánh giá
GRAD_ACCUM = 16      # Effective batch size = 1 × 16 = 16 (Tăng tần suất cập nhật gradient)

# ─── Training Schedule & Quality ───
EPOCHS = 10          # 10 epochs đầy đủ
LR = 2e-5            # Learning rate tối ưu cho Instruct-tuned Qwen2.5-Coder
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.08  # 8% warmup cho sự khởi đầu ổn định
PATIENCE = 5         # Kiên nhẫn chờ hội tụ sâu cho các mẫu DAppSCAN khó
LABEL_SMOOTHING = 0.01
USE_GRADIENT_CHECKPOINTING = True

# ─── LoRA Capacity (Giữ nguyên Rank 64 cho năng lực mô hình hóa tối đa) ───
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ─── Loss Functions: Focal Loss + Source Weighting + Line Guidance ───
FOCAL_GAMMA = 2.0    # Triệt tiêu gradient từ mẫu dễ (như Kaggle pt ~ 1.0)
AUX_LINE_WEIGHT = 0.20  # Trọng số định vị dòng lỗi với line_head độc lập
SOURCE_WEIGHT_MAP = {
    "dappscan": 2.5,         # DAppSCAN được ưu tiên cao nhất
    "iaudit_trustllm": 1.5,  # Hợp đồng thực tế iAudit
    "kaggle_ds2": 0.8,       # Giữ 100% dữ liệu Kaggle nhưng giảm trọng số
}

# ─── Class Balance ───
USE_CLASS_BALANCED_SAMPLER = True
CLASS_BALANCE_POWER = 0.7

# ─── Threshold Tuning ───
THRESHOLD_OBJECTIVE = "f1"
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.005), 3)

# ─── Labels ───
LABEL2ID = {"Safe": 0, "Vulnerable": 1}
ID2LABEL = {0: "Safe", 1: "Vulnerable"}

# ─── Output paths ───
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = str(SCRIPT_DIR / "qwen25-v13-14b-detector")
SAVE_DIR = OUTPUT_DIR + "-final"

# ─── Dataset path ───
DATA_FILENAME = "final-dataset-v12-train.json"

# ─── Instruction Prompt Template cho Qwen2.5-Coder-Instruct ───
INSTRUCTION_PROMPT = (
    "Analyze the following Solidity smart contract code for security vulnerabilities. "
    "Determine whether this function is Safe or Vulnerable:\n\n"
)


def print_banner():
    """In banner thông tin đầu phiên train v13 Quality-First."""
    sep = "═" * 74
    logger.info(sep)
    logger.info("  Qwen2.5-Coder-14B-Instruct v13 — Quality-First Vulnerability Detector")
    logger.info("  QLoRA 4-bit bfloat16 | Multi-GPU 32GB VRAM | Decoupled Dual-Head")
    logger.info(sep)
    logger.info(f"  Thời gian bắt đầu : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Base model         : {HF_MODEL}")
    logger.info(f"  Dataset            : {DATA_FILENAME} (Toàn bộ 100% dữ liệu)")
    logger.info(f"  MAX_LEN            : {MAX_LEN} tokens (Ngữ cảnh đầy đủ)")
    logger.info(f"  Effective BS       : {TRAIN_BS} × {GRAD_ACCUM} = {TRAIN_BS * GRAD_ACCUM}")
    logger.info(f"  Epochs             : {EPOCHS} (patience={PATIENCE})")
    logger.info(f"  Learning Rate      : {LR}")
    logger.info(f"  LoRA rank/alpha    : {LORA_R}/{LORA_ALPHA} (All 7 projections)")
    logger.info(f"  Line Guidance (λ)  : {AUX_LINE_WEIGHT} (Đầu dò độc lập line_head)")
    logger.info(f"  Focal Loss Gamma   : {FOCAL_GAMMA} (Tự động downweight mẫu dễ)")
    logger.info(f"  Source Weighting   : {SOURCE_WEIGHT_MAP}")

    n_gpus = torch.cuda.device_count()
    logger.info(f"  Số lượng GPU       : {n_gpus}")
    for i in range(n_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        vram = torch.cuda.get_device_properties(i).total_memory / 1024**3
        logger.info(f"    • GPU {i}: {gpu_name} ({vram:.1f} GB VRAM)")
    logger.info(f"  transformers       : {transformers.__version__}")
    logger.info(f"  torch              : {torch.__version__}")
    logger.info(sep)


# ══════════════════════════════════════════════════════════════════════════════
# 1. TÌM VÀ NẠP DATASET
# ══════════════════════════════════════════════════════════════════════════════

def find_data():
    """Tìm dataset: ưu tiên cùng thư mục script -> data/ -> các path khác."""
    candidates = [
        str(SCRIPT_DIR / DATA_FILENAME),
        str(SCRIPT_DIR / "data" / DATA_FILENAME),
        str(SCRIPT_DIR.parent / DATA_FILENAME),
        str(SCRIPT_DIR.parent / "final_dataset_train" / DATA_FILENAME),
    ]
    for c in candidates:
        if os.path.isfile(c):
            logger.info(f"Tìm thấy dataset tại: {c}")
            return c

    raise FileNotFoundError(
        f"❌ Không tìm thấy dataset '{DATA_FILENAME}'!\n"
        f"   Đã tìm kiếm tại:\n"
        + "\n".join(f"     - {c}" for c in candidates)
        + f"\n   Vui lòng kiểm tra file trong thư mục {SCRIPT_DIR}/"
    )


def load_data(data_path):
    """Nạp và chuẩn hóa 100% dữ liệu (không cắt giảm)."""
    logger.info(f"Đang nạp dataset: {data_path}")

    with open(data_path, encoding="utf-8") as f:
        raw_rows = json.load(f)

    rows = []
    for r in raw_rows:
        if not r.get("code"):
            continue
        label = r.get("label")
        label_text = r.get("label_text", "")

        if isinstance(label, int) and label in (0, 1):
            r["label_text"] = ID2LABEL[label]
        elif isinstance(label, str) and label in LABEL2ID:
            r["label_text"] = label
            r["label"] = LABEL2ID[label]
        elif isinstance(label, str) and label.isdigit() and int(label) in (0, 1):
            r["label"] = int(label)
            r["label_text"] = ID2LABEL[int(label)]
        elif label_text in LABEL2ID:
            r["label"] = LABEL2ID[label_text]
        else:
            continue
        rows.append(r)

    logger.info(f"Tổng số mẫu nạp đầy đủ 100%: {len(rows)} / {len(raw_rows)}")
    return rows


def split_data(rows):
    """Phân chia theo trường split có sẵn trong dataset."""
    splits = {"train": [], "val": [], "test": []}
    for r in rows:
        split_name = r.get("split", "train")
        splits.get(split_name, splits["train"]).append(r)

    sep = "═" * 74
    logger.info(f"\n{sep}")
    logger.info("PHÂN BỔ CÁC TẬP SPLIT (ĐẦY ĐỦ 100% MẪU DỮ LIỆU)")
    logger.info(sep)
    role_map = {
        "train": "Huấn luyện (Focal Loss + Source Weighting)",
        "val": "Validation + Calibration & Threshold Tuning",
        "test": "Benchmark Kiểm Thử Độc Lập",
    }
    for split_name, items in splits.items():
        n_vuln = sum(1 for x in items if x["label"] == 1)
        n_safe = len(items) - n_vuln
        by_source = dict(
            collections.Counter(x.get("source", "unknown") for x in items)
        )
        n_with_lines = sum(1 for x in items if x.get("vulnerable_lines"))
        role_desc = role_map.get(split_name, "")
        logger.info(
            f"  {split_name:5s}: {len(items):6d} mẫu "
            f"(Safe {n_safe:5d} / Vuln {n_vuln:5d}) | Có câu lỗi: {n_with_lines:4d} — {role_desc}"
        )
        logger.info(f"         Nguồn: {by_source}")
    logger.info(sep)
    return splits


# ══════════════════════════════════════════════════════════════════════════════
# 2. INSTRUCTION-AWARE TOKENIZATION & DATASET WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

def build_model_input(row):
    """
    Xây dựng input text cho Qwen2.5-Coder-14B-Instruct:
    Thêm instruction prefix chuyên biệt để kích hoạt khả năng audit của mô hình Instruct.
    TUYỆT ĐỐI KHÔNG chứa label hay thông tin metadata tránh Data Leakage.
    """
    code = row["code"].strip()
    return f"{INSTRUCTION_PROMPT}{code}"


def build_class_label_weights(items):
    """Tính sample weights cho Class-Balanced Sampler (kết hợp Source-Aware Weighting)."""
    labels = [ID2LABEL[r["label"]] for r in items]
    counts = collections.Counter(labels)
    n_classes = max(1, len(counts))
    base = []
    for i, r in enumerate(items):
        l = labels[i]
        w = (len(items) / (n_classes * counts[l])) ** CLASS_BALANCE_POWER
        # Nhân trọng số nguồn để tăng cường tần suất lấy mẫu DAppSCAN
        src = r.get("source", "unknown")
        w *= SOURCE_WEIGHT_MAP.get(src, 1.0)
        base.append(w)

    weights = torch.as_tensor(base, dtype=torch.double)
    weights = weights / weights.mean()

    logger.info("Train class sampler distribution (kết hợp Source Weighting):")
    for l in sorted(counts):
        class_weights = [
            weights[i].item() for i, ll in enumerate(labels) if ll == l
        ]
        logger.info(
            f"  {l:12s}: n={counts[l]:5d}  mean_weight={np.mean(class_weights):.4f}"
        )
    return weights


class VulnDatasetV13(torch.utils.data.Dataset):
    """
    Dataset wrapper v13 cho Qwen2.5-Coder:
    - Instruction-Aware Input format
    - Token-level Line Guidance Labels với MAX_LEN=2048
    - Per-sample source weight
    """

    def __init__(self, items, tokenizer, sample_weights=None, max_length=MAX_LEN):
        self.items = items
        self.texts = [build_model_input(r) for r in items]
        self.sample_weights = sample_weights
        self.labels = [r["label"] for r in items]
        self.sources = [r.get("source", "unknown") for r in items]
        self.source_weights = [SOURCE_WEIGHT_MAP.get(s, 1.0) for s in self.sources]
        self.max_length = max_length

        logger.info(f"Tokenizing {len(items)} mẫu với max_length={max_length} kèm offset mapping...")
        self.enc = tokenizer(
            self.texts,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )

        # Xây dựng nhãn token-level cho câu bị lỗi
        self.line_labels = []
        self.has_line_supervision = []

        sup_count = 0
        for idx, r in enumerate(items):
            text = self.texts[idx]
            offsets = self.enc["offset_mapping"][idx]
            bug_lines = r.get("vulnerable_lines", []) or []

            # Giám sát câu lỗi cho mẫu Vulnerable có vulnerable_lines
            if bug_lines and r.get("label") == 1:
                spans = []
                for bl in bug_lines:
                    bl = bl.strip()
                    if not bl:
                        continue
                    p = 0
                    while True:
                        pos = text.find(bl, p)
                        if pos == -1:
                            break
                        spans.append((pos, pos + len(bl)))
                        p = pos + len(bl)

                token_mask = []
                for (s, e) in offsets:
                    if s == e:  # special token hoặc padding
                        token_mask.append(0.0)
                    elif any(s >= sp[0] and e <= sp[1] for sp in spans):
                        token_mask.append(1.0)
                    else:
                        token_mask.append(0.0)

                if sum(token_mask) > 0:
                    self.line_labels.append(token_mask)
                    self.has_line_supervision.append(1.0)
                    sup_count += 1
                else:
                    self.line_labels.append([0.0] * len(offsets))
                    self.has_line_supervision.append(0.0)
            else:
                self.line_labels.append([0.0] * len(offsets))
                self.has_line_supervision.append(0.0)

        logger.info(f"  -> Số mẫu có giám sát dòng lỗi (Mask M=1): {sup_count} / {len(items)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.enc["input_ids"][idx],
            "attention_mask": self.enc["attention_mask"][idx],
            "labels": self.labels[idx],
            "line_labels": self.line_labels[idx],
            "has_line_supervision": self.has_line_supervision[idx],
            "source_weight": self.source_weights[idx],
        }


class VulnDataCollatorV13:
    """Collator động v13: Pad input_ids, attention_mask và line_labels đồng bộ."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        labels = [f["labels"] for f in features]
        has_sup = [f["has_line_supervision"] for f in features]
        src_weights = [f["source_weight"] for f in features]

        # Pad sequence tokens
        batch = self.tokenizer.pad(
            [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features],
            padding=True,
            return_tensors="pt",
        )

        # Pad line_labels theo max length của batch
        max_len = batch["input_ids"].shape[1]
        padded_lines = []
        for f in features:
            ll = f["line_labels"]
            padded = ll + [0.0] * (max_len - len(ll))
            padded_lines.append(padded[:max_len])

        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        batch["line_labels"] = torch.tensor(padded_lines, dtype=torch.float32)
        batch["has_line_supervision"] = torch.tensor(has_sup, dtype=torch.float32)
        batch["source_weight"] = torch.tensor(src_weights, dtype=torch.float32)
        return batch


# ══════════════════════════════════════════════════════════════════════════════
# 3. FOCAL LOSS & DECOUPLED DUAL-HEAD ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) cho Binary/Multi-class Classification:
      FL(p_t) = - (1 - p_t)^gamma * log(p_t)
    Mẫu dễ có xác suất cao p_t -> 1 sẽ bị triệt tiêu gradient.
    Mẫu khó (như DAppSCAN) p_t -> 0.5 sẽ giữ nguyên lực cập nhật gradient.
    """

    def __init__(self, gamma=2.0, label_smoothing=0.01, reduction="none"):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits,
            targets.to(logits.device),
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        focal_weight = (1.0 - pt) ** self.gamma
        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def setup_line_head(model):
    """Khởi tạo và gán line_head độc lập (nn.Linear(hidden_dim, 1)) vào model."""
    score_head = None
    for attr_path in ["score", "base_model.model.score", "base_model.score"]:
        obj = model
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "modules_to_save"):
                score_head = obj.modules_to_save.get("default", None)
                if score_head is not None:
                    break
            elif isinstance(obj, nn.Linear):
                score_head = obj
                break
        except AttributeError:
            continue

    if score_head is not None:
        target_device = score_head.weight.device
        target_dtype = score_head.weight.dtype
    else:
        target_device = next(model.parameters()).device
        target_dtype = torch.bfloat16

    hidden_dim = model.config.hidden_size
    line_head = nn.Linear(hidden_dim, 1, bias=True, dtype=target_dtype, device=target_device)
    nn.init.xavier_uniform_(line_head.weight)
    nn.init.zeros_(line_head.bias)

    # Đảm bảo requires_grad = True
    model.line_head = line_head

    # Đăng ký Forward Hook trên score_head để lấy trực tiếp last_hidden_state
    # TUYỆT ĐỐI KHÔNG CẦN output_hidden_states=True -> Tiết kiệm ngay ~490 MB VRAM!
    class LastHiddenHolder:
        def __init__(self):
            self.val = None
        def __call__(self, module, inp, out):
            if isinstance(inp, tuple) and len(inp) > 0:
                self.val = inp[0]

    holder = LastHiddenHolder()
    if score_head is not None:
        score_head.register_forward_hook(holder)
    model.last_hidden_holder = holder

    logger.info(
        f"✅ Đã khởi tạo Decoupled line_head: Linear({hidden_dim}, 1) tại {target_device} "
        f"với dtype={target_dtype} (Hook bắt last_hidden tiết kiệm 490MB VRAM)"
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 4. MODEL LOADING & QLORA (MULTI-GPU 32GB VRAM)
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer():
    """
    Nạp Qwen2.5-Coder-14B-Instruct:
    - 4-bit NF4 Quantization với bfloat16 compute
    - FlashAttention-2 kích hoạt để tiết kiệm VRAM activation
    - Tự động phân bổ qua 2 GPU (device_map="auto")
    """
    logger.info(f"Đang nạp tokenizer từ: {HF_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    logger.info("Cấu hình BitsAndBytes 4-bit NF4 (bfloat16 compute)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        # Cấu hình rõ ràng (explicit device map) cho Qwen2.5-Coder-14B (48 layers):
        # GPU 0 (có ~14 GB free): gánh 32 layers đầu + embeddings (~5.8 GB 4-bit) -> Dư tới ~8.2 GB cho activation/AdamW!
        # GPU 1 (có ~5.5 GB free): gánh 16 layers cuối + norm + score (~2.9 GB 4-bit) -> Dư ~2.6 GB an toàn, không đụng tới tác vụ nền!
        # Bằng cách gán device_map dictionary cụ thể, Accelerate sẽ KHÔNG ước tính unquantized (28GB) và KHÔNG đẩy sang CPU.
        device_map = {
            "model.embed_tokens": 0,
            "model.rotary_emb": 0,
        }
        for i in range(32):
            device_map[f"model.layers.{i}"] = 0
        for i in range(32, 48):
            device_map[f"model.layers.{i}"] = 1
        device_map["model.norm"] = 1
        device_map["score"] = 1
        logger.info(
            f"Phát hiện {n_gpus} GPU CUDA -> Kích hoạt Dual-GPU Custom Device Map: "
            f"GPU 0 (32 layers + embed), GPU 1 (16 layers + norm + score)."
        )
    else:
        device_map = {"": 0}
        logger.info("Phát hiện 1 GPU CUDA -> Nạp model vào GPU 0.")

    logger.info(f"Đang nạp base model {HF_MODEL}...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            HF_MODEL,
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        logger.info("✅ FlashAttention-2 đã được kích hoạt thành công!")
    except Exception as e:
        logger.warning(f"⚠️  Không thể dùng flash_attention_2 ({e}), fallback sang sdpa...")
        model = AutoModelForSequenceClassification.from_pretrained(
            HF_MODEL,
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    # Chuẩn bị cho kbit QLoRA
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING
    )

    # Cấu hình LoRA rank cao (r=64, alpha=128)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=LORA_TARGETS,
        modules_to_save=["score"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Thiết lập Decoupled line_head riêng biệt
    model = setup_line_head(model)

    model.is_parallelizable = True
    model.model_parallel = True

    if torch.cuda.is_available():
        for i in range(n_gpus):
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            resv = torch.cuda.memory_reserved(i) / 1024**3
            logger.info(f"GPU {i} VRAM sau nạp model: allocated={alloc:.2f}GB, reserved={resv:.2f}GB")

    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# 5. CUSTOM TRAINER V13 VỚI FOCAL LOSS & DUAL-HEAD LINE GUIDANCE
# ══════════════════════════════════════════════════════════════════════════════

def logits_from_predictions(predictions):
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    return np.asarray(predictions)


def probs_from_logits(logits):
    logits = np.asarray(logits)
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp[:, 1] / exp.sum(axis=-1)


def paper_metrics(labels, pred):
    """Metrics chuẩn: macro-averaged precision, recall, F1."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, pred, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_metrics(eval_pred):
    logits = logits_from_predictions(eval_pred.predictions)
    labels = eval_pred.label_ids
    pred = np.argmax(logits, axis=-1)
    m = paper_metrics(labels, pred)
    return {
        "accuracy": m["accuracy"],
        "precision": m["precision"],
        "recall": m["recall"],
        "f1_macro": m["f1"],
        "f1_vuln": f1_score(labels, pred, pos_label=1, zero_division=0),
    }


class ClearCacheCallback(transformers.TrainerCallback):
    """Giải phóng bộ nhớ đệm CUDA trước và sau khi đánh giá để chống phân mảnh VRAM."""

    def on_evaluate(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_epoch_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class MaskedLineGuidanceTrainerV13(Trainer):
    """
    Custom Trainer v13:
      Total Loss = Loss_cls (Focal Loss x Source Weight) + AUX_LINE_WEIGHT * Loss_line_head
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss_fn = FocalLoss(gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)

    def _default_train_sampler(self, train_dataset=None):
        method = super()._get_train_sampler
        if len(inspect.signature(method).parameters) == 0:
            return method()
        return method(train_dataset)

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if not USE_CLASS_BALANCED_SAMPLER:
            return self._default_train_sampler(train_dataset)
        if dataset is None or getattr(dataset, "sample_weights", None) is None:
            return self._default_train_sampler(train_dataset)
        generator = torch.Generator()
        generator.manual_seed(SEED)
        return WeightedRandomSampler(
            weights=dataset.sample_weights,
            num_samples=len(dataset.sample_weights),
            replacement=True,
            generator=generator,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        line_labels = inputs.pop("line_labels", None)
        has_line_supervision = inputs.pop("has_line_supervision", None)
        source_weight = inputs.pop("source_weight", None)

        is_train = model.training
        need_line_loss = (
            is_train
            and line_labels is not None
            and has_line_supervision is not None
            and (has_line_supervision.sum() > 0)
            and AUX_LINE_WEIGHT > 0
        )

        # Forward pass: TUYỆT ĐỐI KHÔNG lưu 48 tầng hidden states trung gian (tiết kiệm ~490 MB VRAM)
        outputs = model(**inputs, output_hidden_states=False)

        # 1. Phân loại toàn hàm với Focal Loss
        raw_logits = outputs.logits  # [batch_size, 2]
        cls_loss_per_sample = self.focal_loss_fn(raw_logits, inputs["labels"])  # [batch_size]

        if source_weight is not None:
            weights = source_weight.to(cls_loss_per_sample.device)
            cls_loss = (cls_loss_per_sample * weights).mean()
        else:
            cls_loss = cls_loss_per_sample.mean()

        total_loss = cls_loss

        # 2. Định vị dòng lỗi với Decoupled line_head (Lấy last_hidden từ Hook)
        if need_line_loss:
            holder = getattr(model, "last_hidden_holder", None)
            if holder is None and hasattr(model, "base_model"):
                holder = getattr(model.base_model, "last_hidden_holder", None)
            last_hidden = holder.val if holder is not None else None

            # Fallback an toàn nếu hook chưa có giá trị
            if last_hidden is None and hasattr(outputs, "hidden_states") and outputs.hidden_states:
                last_hidden = outputs.hidden_states[-1]

            # Tìm line_head
            line_head = getattr(model, "line_head", None)
            if line_head is None and hasattr(model, "base_model"):
                line_head = getattr(model.base_model, "line_head", None)

            if line_head is not None and last_hidden is not None:
                token_logits = line_head(last_hidden.to(line_head.weight.dtype)).squeeze(-1).float()  # [batch, seq]
                att_mask = inputs.get("attention_mask", torch.ones_like(line_labels)).float().to(token_logits.device)
                target_labels = line_labels.to(token_logits.device)

                # Binary Cross Entropy cho từng token
                bce_fn = nn.BCEWithLogitsLoss(reduction="none")
                per_token_loss = bce_fn(token_logits, target_labels)  # [batch, seq]

                token_counts = att_mask.sum(dim=-1).clamp(min=1.0)
                sample_line_loss = (per_token_loss * att_mask).sum(dim=-1) / token_counts  # [batch]

                mask_m = has_line_supervision.to(sample_line_loss.device)
                num_supervised = mask_m.sum().clamp(min=1.0)
                aux_line_loss = (sample_line_loss * mask_m).sum() / num_supervised

                total_loss = cls_loss + AUX_LINE_WEIGHT * aux_line_loss

            if holder is not None:
                holder.val = None

        if return_outputs:
            clean_output = transformers.modeling_outputs.SequenceClassifierOutput(
                loss=total_loss,
                logits=outputs.logits,
            )
            return (total_loss, clean_output)
        return total_loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict,
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ):
        eval_inputs = {
            k: v
            for k, v in inputs.items()
            if k not in ("line_labels", "has_line_supervision", "source_weight")
        }
        loss, logits, labels = super().prediction_step(
            model,
            eval_inputs,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
        )
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        return loss, logits, labels


def build_training_args():
    """Xây dựng TrainingArguments tối ưu v13."""
    kwargs = {
        "output_dir": OUTPUT_DIR,
        "num_train_epochs": EPOCHS,
        "per_device_train_batch_size": TRAIN_BS,
        "per_device_eval_batch_size": EVAL_BS,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "eval_accumulation_steps": 10,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "lr_scheduler_type": "cosine_with_restarts",
        "bf16": True,
        "optim": "paged_adamw_8bit",
        "gradient_checkpointing": USE_GRADIENT_CHECKPOINTING,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "max_grad_norm": 0.3,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_macro",
        "greater_is_better": True,
        "logging_steps": 25,
        "save_total_limit": 2,
        "report_to": "none",
        "seed": SEED,
        "remove_unused_columns": False,
        "dataloader_pin_memory": False,
    }

    signature = inspect.signature(TrainingArguments.__init__)
    parameters = signature.parameters
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )
    supported = set(parameters) - {"self"}

    if not accepts_kwargs:
        if (
            "eval_strategy" in kwargs
            and "eval_strategy" not in supported
            and "evaluation_strategy" in supported
        ):
            kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
        elif (
            "evaluation_strategy" in kwargs
            and "evaluation_strategy" not in supported
            and "eval_strategy" in supported
        ):
            kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")

        dropped = sorted(k for k in kwargs if k not in supported)
        for key in dropped:
            kwargs.pop(key)
        if dropped:
            logger.warning(f"Dropped unsupported TrainingArguments: {', '.join(dropped)}")

    return TrainingArguments(**kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# 6. THRESHOLD TUNING & SMOKE TESTS
# ══════════════════════════════════════════════════════════════════════════════

def metrics_at_threshold(labels, probs, threshold):
    pred = (probs >= threshold).astype(int)
    return paper_metrics(labels, pred)


def tune_threshold(labels, probs, thresholds=THRESHOLD_GRID, objective=THRESHOLD_OBJECTIVE):
    results = []
    for th in thresholds:
        m = metrics_at_threshold(labels, probs, th)
        results.append({"threshold": float(th), **m})
    best = max(
        results,
        key=lambda r: (r[objective], r["f1"], r["recall"], r["precision"]),
    )
    return best, results


SMOKE_TESTS = [
    {
        "name": "SWC-107 Reentrancy",
        "expected": "Vulnerable",
        "code": """function withdraw(uint amount) public {
    require(balances[msg.sender] >= amount);
    (bool ok,) = msg.sender.call{value: amount}("");
    require(ok);
    balances[msg.sender] -= amount;
}""",
    },
    {
        "name": "SWC-104 Unchecked Call Return",
        "expected": "Vulnerable",
        "code": """function pay(address payable to, uint256 amount) public {
    to.call{value: amount}("");
}""",
    },
    {
        "name": "Access Control Missing",
        "expected": "Vulnerable",
        "code": """function setOwner(address newOwner) public {
    owner = newOwner;
}""",
    },
    {
        "name": "Safe Guarded Withdraw",
        "expected": "Safe",
        "code": """function withdraw(uint amount) public nonReentrant {
    require(balances[msg.sender] >= amount);
    balances[msg.sender] -= amount;
    (bool ok,) = msg.sender.call{value: amount}("");
    require(ok, "Transfer failed");
}""",
    },
    {
        "name": "Safe AccessControl onlyOwner",
        "expected": "Safe",
        "code": """function setFee(uint256 newFee) public onlyOwner {
    fee = newFee;
    emit FeeUpdated(newFee);
}""",
    },
    {
        "name": "tx.origin Auth (Vulnerable)",
        "expected": "Vulnerable",
        "code": """function transferTo(address payable dest) public {
    require(tx.origin == owner);
    dest.transfer(address(this).balance);
}""",
    },
    {
        "name": "SWC-135 Code With No Effects",
        "expected": "Vulnerable",
        "code": """function calculateReward(uint256 balance, uint256 rate) public pure returns (uint256) {
    uint256 reward;
    balance * rate;
    return reward;
}""",
    },
]


def run_smoke_tests(model, tokenizer, threshold):
    model.eval()

    @torch.no_grad()
    def predict(code_text):
        input_text = f"{INSTRUCTION_PROMPT}{code_text.strip()}"
        enc = tokenizer(
            input_text,
            truncation=True,
            max_length=MAX_LEN,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(next(model.parameters()).device)
        logits = model(**enc).logits
        prob_vuln = torch.softmax(logits, dim=-1)[0, 1].item()
        return ("Vulnerable" if prob_vuln >= threshold else "Safe"), prob_vuln

    logger.info(f"\nSmoke test v13 (Qwen2.5-Coder) với threshold = {threshold:.3f}")
    header = f"{'Case':35s} {'Expect':12s} {'Pred':12s} {'P(vuln)':>8s}"
    logger.info(header)
    all_pass = True
    for t in SMOKE_TESTS:
        pred_label, p = predict(t["code"])
        status = "✅" if pred_label == t["expected"] else "❌"
        if pred_label != t["expected"]:
            all_pass = False
        logger.info(
            f"{t['name']:35s} {t['expected']:12s} {pred_label:12s} {p:8.3f} {status}"
        )

    sep = "═" * 60
    if all_pass:
        logger.info(f"\n{sep}\n✅ TẤT CẢ SMOKE TESTS V13 ĐỀU PASS!\n{sep}")
    else:
        logger.info(
            f"\n{sep}\n⚠️  Một số smoke test chưa pass. "
            f"Kiểm tra lại threshold hoặc xem xét train thêm.\n{sep}"
        )
    return all_pass


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE V13 QUALITY FIRST
# ══════════════════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()
    print_banner()

    # ─── 1. Nạp 100% data ───
    data_path = find_data()
    rows = load_data(data_path)
    splits = split_data(rows)

    # ─── 2. Load model & tokenizer ───
    model, tokenizer = load_model_and_tokenizer()

    # ─── 3. Tokenize & build datasets v13 (MAX_LEN = 2048) ───
    logger.info("Đang tokenize 3 tập v13 (train/val/test) kèm nhãn dòng lỗi và source weight...")
    train_weights = (
        build_class_label_weights(splits["train"])
        if USE_CLASS_BALANCED_SAMPLER
        else None
    )
    train_ds = VulnDatasetV13(splits["train"], tokenizer, sample_weights=train_weights, max_length=MAX_LEN)
    val_ds = VulnDatasetV13(splits["val"], tokenizer, max_length=MAX_LEN)
    test_ds = VulnDatasetV13(splits["test"], tokenizer, max_length=MAX_LEN)
    collator = VulnDataCollatorV13(tokenizer)

    # ─── 4. Khởi tạo Trainer v13 ───
    args = build_training_args()
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": collator,
        "compute_metrics": compute_metrics,
        "callbacks": [
            EarlyStoppingCallback(early_stopping_patience=PATIENCE),
            ClearCacheCallback(),
        ],
    }
    init_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in init_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = MaskedLineGuidanceTrainerV13(**trainer_kwargs)

    # Ngăn Trainer tự ý wrap DataParallel khi đã dùng device_map="auto"
    trainer.args._n_gpu = 1

    steps_per_epoch = max(
        1, math.ceil(len(train_ds) / max(1, TRAIN_BS) / max(1, GRAD_ACCUM))
    )
    total_steps = steps_per_epoch * EPOCHS

    sep = "═" * 74
    logger.info(f"\n{sep}")
    logger.info("BẮT ĐẦU HUẤN LUYỆN V13 QUALITY FIRST")
    logger.info(f"  Train mẫu      : {len(train_ds):,}")
    logger.info(f"  Val mẫu        : {len(val_ds):,}")
    logger.info(f"  Epochs         : {EPOCHS} (patience={PATIENCE})")
    logger.info(f"  Steps/epoch    : ~{steps_per_epoch}")
    logger.info(f"  Total steps    : ~{total_steps}")
    logger.info(f"  Effective BS   : {TRAIN_BS * GRAD_ACCUM}")
    logger.info(f"  MAX_LEN        : {MAX_LEN} tokens")
    logger.info(f"  LoRA rank      : {LORA_R}")
    logger.info(f"  Aux Line λ     : {AUX_LINE_WEIGHT}")
    logger.info(f"  Focal Gamma    : {FOCAL_GAMMA}")
    logger.info(sep)

    trainer.train()

    # Log epoch results
    logger.info("\nEpoch | val_acc | val_prec | val_recall | val_f1 | val_f1_vuln")
    for h in trainer.state.log_history:
        if "eval_f1_macro" in h:
            ep = round(h["epoch"])
            logger.info(
                f"  {ep:3d} | {h['eval_accuracy']:.4f} | "
                f"{h['eval_precision']:.4f} | "
                f"{h['eval_recall']:.4f} | {h['eval_f1_macro']:.4f} | "
                f"{h['eval_f1_vuln']:.4f}"
            )

    # ─── 5. Threshold Tuning trên Validation ───
    logger.info(f"\n{'═' * 74}")
    logger.info("THRESHOLD TUNING trên Validation set (v13)")
    logger.info("═" * 74)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    val_out = trainer.predict(val_ds)
    val_logits = logits_from_predictions(val_out.predictions)
    val_probs = probs_from_logits(val_logits)
    val_labels = val_out.label_ids
    best_threshold_info, threshold_rows = tune_threshold(val_labels, val_probs)
    best_threshold = best_threshold_info["threshold"]

    logger.info(f"Best threshold ({THRESHOLD_OBJECTIVE}): {best_threshold:.3f}")
    logger.info(
        f"  Val Acc={best_threshold_info['accuracy']:.4f}  "
        f"Prec={best_threshold_info['precision']:.4f}  "
        f"Recall={best_threshold_info['recall']:.4f}  "
        f"F1={best_threshold_info['f1']:.4f}"
    )

    # ─── 6. Benchmark trên Test set ───
    logger.info(f"\n{'═' * 74}")
    logger.info("BENCHMARK TRÊN TEST SET (v13 - ĐÁNH GIÁ CHÍNH THỨC)")
    logger.info("═" * 74)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    test_out = trainer.predict(test_ds)
    test_logits = logits_from_predictions(test_out.predictions)
    test_probs = probs_from_logits(test_logits)
    test_labels = test_out.label_ids
    test_preds = (test_probs >= best_threshold).astype(int)

    test_metrics = paper_metrics(test_labels, test_preds)
    test_f1_vuln = f1_score(test_labels, test_preds, pos_label=1, zero_division=0)
    cm = confusion_matrix(test_labels, test_preds)
    tn, fp, fn, tp = cm.ravel()

    logger.info(f"\nTest kết quả (threshold = {best_threshold:.3f}):")
    logger.info(
        f"  Macro-F1    : {test_metrics['f1']:.4f}\n"
        f"  Accuracy    : {test_metrics['accuracy']:.4f}\n"
        f"  Precision   : {test_metrics['precision']:.4f}\n"
        f"  Recall      : {test_metrics['recall']:.4f}\n"
        f"  Vuln-F1     : {test_f1_vuln:.4f}"
    )
    logger.info(f"\nConfusion Matrix [[TN, FP], [FN, TP]]:\n{cm}")
    logger.info(f"  TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    logger.info(
        f"\nClassification Report:\n"
        f"{classification_report(test_labels, test_preds, target_names=['Safe', 'Vulnerable'])}"
    )

    # Đánh giá chi tiết từng nguồn dữ liệu (Đặc biệt DAppSCAN)
    logger.info("\n--- Kết quả kiểm thử v13 theo từng nguồn dữ liệu ---")
    test_items = splits["test"]
    sources = sorted(set(x.get("source", "unknown") for x in test_items))
    for src in sources:
        indices = [
            i for i, x in enumerate(test_items) if x.get("source") == src
        ]
        if not indices:
            continue
        sub_labels = test_labels[indices]
        sub_preds = test_preds[indices]
        sub_acc = accuracy_score(sub_labels, sub_preds)
        sub_f1 = f1_score(sub_labels, sub_preds, average="macro", zero_division=0)
        logger.info(f"  {src:20s}: n={len(indices):4d}  Acc={sub_acc:.4f}  F1={sub_f1:.4f}")

    # Đánh giá riêng hợp đồng thực tế (Real-world only)
    real_indices = [
        i
        for i, x in enumerate(test_items)
        if x.get("source") in ("dappscan", "iaudit_trustllm")
    ]
    if real_indices:
        r_labels = test_labels[real_indices]
        r_preds = test_preds[real_indices]
        r_metrics = paper_metrics(r_labels, r_preds)
        logger.info("\n--- ĐÁNH GIÁ ĐỘC LẬP TRÊN HỢP ĐỒNG THỰC TẾ (REAL-WORLD ONLY) ---")
        logger.info(
            f"  real_world_total   : n={len(real_indices):4d}  "
            f"Acc={r_metrics['accuracy']:.4f}  F1={r_metrics['f1']:.4f}  "
            f"Prec={r_metrics['precision']:.4f}  Recall={r_metrics['recall']:.4f}"
        )

    # ─── 7. Smoke tests ───
    run_smoke_tests(model, tokenizer, best_threshold)

    # ─── 8. Lưu model và metadata ───
    logger.info(f"\n{'═' * 74}")
    logger.info("LƯU MODEL V13 & CONFIG")
    logger.info("═" * 74)

    os.makedirs(SAVE_DIR, exist_ok=True)
    trainer.save_model(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    # Lưu thêm trọng số line_head riêng biệt
    line_head = getattr(model, "line_head", None)
    if line_head is None and hasattr(model, "base_model"):
        line_head = getattr(model.base_model, "line_head", None)
    if line_head is not None:
        torch.save(line_head.state_dict(), os.path.join(SAVE_DIR, "line_head.pt"))
        logger.info(f"✅ Đã lưu trọng số line_head tại: {os.path.join(SAVE_DIR, 'line_head.pt')}")

    threshold_config = {
        "best_threshold": float(best_threshold),
        "threshold_objective": THRESHOLD_OBJECTIVE,
        "max_len": MAX_LEN,
        "id2label": ID2LABEL,
        "label2id": LABEL2ID,
        "test_metrics": {
            "accuracy": float(test_metrics["accuracy"]),
            "precision": float(test_metrics["precision"]),
            "recall": float(test_metrics["recall"]),
            "f1_macro": float(test_metrics["f1"]),
            "f1_vuln": float(test_f1_vuln),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "train_mode": "Qwen2.5-Coder-14B-Instruct v13 Quality-First (Focal Loss + Decoupled Line Head + Multi-GPU)",
        "dataset": DATA_FILENAME,
        "aux_line_weight": AUX_LINE_WEIGHT,
        "focal_gamma": FOCAL_GAMMA,
        "source_weights": SOURCE_WEIGHT_MAP,
        "model_name": "qwen25-v13-14b-detector-final",
        "instruction_prompt": INSTRUCTION_PROMPT,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(SAVE_DIR, "threshold_config.json"), "w") as f:
        json.dump(threshold_config, f, indent=2)

    logger.info(f"✅ Đã lưu model v13 & config tại: {SAVE_DIR}")

    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, _ = divmod(remainder, 60)
    logger.info(f"\n{'═' * 74}")
    logger.info(f"  HOÀN TẤT HUẤN LUYỆN V13 — Tổng thời gian: {int(hours)}h {int(minutes)}m")
    logger.info(f"  Model saved     : {SAVE_DIR}")
    logger.info(f"  Test F1 (macro) : {test_metrics['f1']:.4f}")
    logger.info(f"  Best threshold  : {best_threshold:.3f}")
    logger.info(f"{'═' * 74}")


if __name__ == "__main__":
    main()
