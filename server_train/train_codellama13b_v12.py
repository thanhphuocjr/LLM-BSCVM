#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  CodeLlama-13B v12 — Masked Line-Guided Smart Contract Vulnerability Detector  ║
║  QLoRA 4-bit Fine-Tuning trên final-dataset-v12-train.json                   ║
║                                                                              ║
║  Đột phá v12:                                                                ║
║  • Kế thừa nền tảng Sequence Classification chất lượng cao của v11.         ║
║  • Bổ sung cơ chế MASKED LINE GUIDANCE LOSS (Giám sát dòng lỗi có mặt nạ):   ║
║    - Mẫu DAppSCAN Vulnerable: Phạt hàm mất mát hướng sự chú ý (Attention     ║
║      Guidance) vào đúng các token của câu bị lỗi (vulnerable_lines).         ║
║    - Mẫu Solodit, Kaggle và mẫu Safe: TẮT HOÀN TOÀN loss phụ (Mask = 0).     ║
║    - Không gây Data Leakage: Input code hoàn toàn sạch 100%.                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

Chạy trên server:
    bash run_train.sh
Hoặc:
    python3 train_codellama13b_v12.py 2>&1 | tee train_v12.log
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
# ENV CONFIG — Tối ưu cho GPU Server
# ══════════════════════════════════════════════════════════════════════════════
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Dùng GPU 0 (GPU 1 phục vụ việc khác)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
# HYPERPARAMETERS — V12 MASKED ATTENTION & QUALITY FIRST
# ══════════════════════════════════════════════════════════════════════════════

# ─── Model: CodeLlama 13B ───
HF_MODEL = "codellama/CodeLlama-13b-hf"

# ─── Context & Batch Size (RTX 4070 Ti SUPER 16GB VRAM) ───
MAX_LEN = 1024       # Bao phủ 99.93% mẫu code
TRAIN_BS = 1         # Batch size 1 cho 13B trên 16GB
EVAL_BS = 1          # Batch size 1 an toàn
GRAD_ACCUM = 32      # Effective batch size = 1 × 32 = 32

# ─── Training Schedule ───
EPOCHS = 10          # 10 epochs
LR = 5e-5            # Learning rate cho 13B QLoRA
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
PATIENCE = 4         # Early stopping patience
LABEL_SMOOTHING = 0.02
USE_GRADIENT_CHECKPOINTING = True

# ─── LoRA Config ───
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ─── Masked Line Guidance / Attention Guidance Loss (v12) ───
# Trọng số loss phụ định vị dòng lỗi: Lambda = 0.2
# Giúp kéo điểm DAppSCAN lên bằng cách giám sát fine-grained token saliency
AUX_LINE_WEIGHT = 0.2

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
OUTPUT_DIR = str(SCRIPT_DIR / "codellama-v12-13b-detector")
SAVE_DIR = OUTPUT_DIR + "-final"

# ─── Dataset path (v12) ───
DATA_FILENAME = "final-dataset-v12-train.json"


def print_banner():
    """In banner thông tin đầu phiên train v12."""
    sep = "═" * 72
    logger.info(sep)
    logger.info("  CodeLlama-13B v12 — Masked Line-Guided Vulnerability Detector")
    logger.info("  QLoRA 4-bit | Sequence Classification + Masked Line Guidance")
    logger.info(sep)
    logger.info(f"  Thời gian bắt đầu : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Base model         : {HF_MODEL}")
    logger.info(f"  Dataset            : {DATA_FILENAME}")
    logger.info(f"  MAX_LEN            : {MAX_LEN}")
    logger.info(f"  Effective BS       : {TRAIN_BS} × {GRAD_ACCUM} = {TRAIN_BS * GRAD_ACCUM}")
    logger.info(f"  Epochs             : {EPOCHS} (patience={PATIENCE})")
    logger.info(f"  Learning Rate      : {LR}")
    logger.info(f"  LoRA rank/alpha    : {LORA_R}/{LORA_ALPHA}")
    logger.info(f"  Line Guidance (λ)  : {AUX_LINE_WEIGHT} (Masked: chỉ bật cho DAppSCAN vuln)")
    if torch.cuda.is_available():
        logger.info(f"  GPU                : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"  VRAM               : {vram:.1f} GB")
    logger.info(f"  transformers       : {transformers.__version__}")
    logger.info(f"  torch              : {torch.__version__}")
    logger.info(sep)


# ══════════════════════════════════════════════════════════════════════════════
# 1. TÌM VÀ NẠP DATASET V12
# ══════════════════════════════════════════════════════════════════════════════

def find_data():
    """Tìm dataset v12: ưu tiên cùng thư mục script -> data/ -> các path khác."""
    candidates = [
        str(SCRIPT_DIR / DATA_FILENAME),
        str(SCRIPT_DIR / "data" / DATA_FILENAME),
        str(SCRIPT_DIR.parent / DATA_FILENAME),
        str(SCRIPT_DIR.parent / "final_dataset_train" / DATA_FILENAME),
    ]
    for c in candidates:
        if os.path.isfile(c):
            logger.info(f"Tìm thấy dataset v12 tại: {c}")
            return c

    raise FileNotFoundError(
        f"❌ Không tìm thấy dataset '{DATA_FILENAME}'!\n"
        f"   Đã tìm kiếm tại:\n"
        + "\n".join(f"     - {c}" for c in candidates)
        + f"\n   Vui lòng kiểm tra file trong thư mục {SCRIPT_DIR}/"
    )


def load_data(data_path):
    """Nạp và chuẩn hóa dataset từ .json v12."""
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

    logger.info(f"Tổng mẫu hợp lệ: {len(rows)} / {len(raw_rows)} nạp")
    return rows


def split_data(rows):
    """Phân chia theo trường split có sẵn trong dataset."""
    splits = {"train": [], "val": [], "test": []}
    for r in rows:
        split_name = r.get("split", "train")
        splits.get(split_name, splits["train"]).append(r)

    sep = "═" * 72
    logger.info(f"\n{sep}")
    logger.info("PHÂN BỔ CÁC TẬP SPLIT (final-dataset-v12-train.json)")
    logger.info(sep)
    role_map = {
        "train": "Huấn luyện",
        "val": "Validation + Threshold Tuning",
        "test": "Benchmark Kiểm Thử",
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
# 2. TOKENIZE & MASKED LINE SUPERVISION DATASET
# ══════════════════════════════════════════════════════════════════════════════

def build_model_input(row):
    """Xây dựng input text cho model: Raw code sạch hoàn toàn để tránh data leakage."""
    return row["code"].strip()


def build_class_label_weights(items):
    """Tính sample weights cho Class-Balanced Sampler (cân bằng Safe/Vulnerable)."""
    labels = [ID2LABEL[r["label"]] for r in items]
    counts = collections.Counter(labels)
    n_classes = max(1, len(counts))
    base = []
    for l in labels:
        w = (len(items) / (n_classes * counts[l])) ** CLASS_BALANCE_POWER
        base.append(w)
    weights = torch.as_tensor(base, dtype=torch.double)
    weights = weights / weights.mean()
    logger.info("Train class sampler distribution:")
    for l in sorted(counts):
        class_weights = [
            weights[i].item() for i, ll in enumerate(labels) if ll == l
        ]
        logger.info(
            f"  {l:12s}: n={counts[l]:5d}  mean_weight={np.mean(class_weights):.4f}"
        )
    return weights


class VulnDatasetV12(torch.utils.data.Dataset):
    """Dataset wrapper cho Sequence Classification kèm Masked Line Guidance Labels."""

    def __init__(self, items, tokenizer, sample_weights=None, max_length=MAX_LEN):
        self.items = items
        self.texts = [build_model_input(r) for r in items]
        self.sample_weights = sample_weights
        self.labels = [r["label"] for r in items]
        self.max_length = max_length

        logger.info(f"Tokenizing {len(items)} mẫu kèm offset mapping...")
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

            # CHỈ ÁP DỤNG cho mẫu Vulnerable có vulnerable_lines (từ DAppSCAN)
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
                # Kaggle, Solodit, DAppSCAN Safe: MẶT NẠ M = 0 (Không tính loss câu lỗi)
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
        }


class VulnDataCollatorV12:
    """Collator động: Pad input_ids, attention_mask và line_labels đồng bộ."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        labels = [f["labels"] for f in features]
        has_sup = [f["has_line_supervision"] for f in features]

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
        return batch


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL LOADING & QLORA
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer():
    """Nạp CodeLlama-13B (4-bit NF4) + cấu hình QLoRA."""
    logger.info(f"Đang nạp tokenizer từ: {HF_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info(f"Đang nạp model CodeLlama-13B (4-bit QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        HF_MODEL,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    # Prepare for QLoRA training
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=LORA_TARGETS,
        modules_to_save=["score"],  # Classification head
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.is_parallelizable = True
    model.model_parallel = True

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        logger.info(f"VRAM sau khi load model: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")

    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# 4. CUSTOM TRAINER VỚI MASKED LINE GUIDANCE LOSS
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


class MaskedLineGuidanceTrainer(Trainer):
    """Custom Trainer kết hợp Class Balancing + Masked Line Guidance Loss."""

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
        """
        Tính Total Loss:
          Loss_total = Loss_cls + AUX_LINE_WEIGHT * M * Loss_line_guidance
          trong đó M = 1 nếu mẫu có câu lỗi (DAppSCAN), M = 0 cho các mẫu khác.
        """
        line_labels = inputs.pop("line_labels", None)
        has_line_supervision = inputs.pop("has_line_supervision", None)

        is_train = model.training
        need_line_loss = (
            is_train
            and line_labels is not None
            and has_line_supervision is not None
            and (has_line_supervision.sum() > 0)
            and AUX_LINE_WEIGHT > 0
        )

        # Forward pass: CHỈ tính hidden_states khi đang train VÀ có mẫu cần giám sát dòng lỗi
        outputs = model(**inputs, output_hidden_states=need_line_loss)
        cls_loss = outputs.loss
        total_loss = cls_loss

        # Tính loss phụ nếu có mẫu có giám sát dòng lỗi
        if need_line_loss:
            # Last hidden states: [batch_size, seq_len, hidden_dim]
            last_hidden = outputs.hidden_states[-1]

            # Truy xuất score head của model (nn.Linear(hidden_dim, 2))
            score_head = getattr(model, "score", None)
            if score_head is None and hasattr(model, "base_model"):
                score_head = getattr(model.base_model, "score", None)

            if score_head is not None:
                # Đảm bảo dtype khớp
                weight_dtype = next(score_head.parameters()).dtype
                token_logits = score_head(last_hidden.to(weight_dtype))  # [batch, seq, 2]

                # Relative vulnerability logit (Logit class 1 - Logit class 0)
                token_vuln_logits = (token_logits[:, :, 1] - token_logits[:, :, 0]).float()

                # Attention mask (chỉ tính loss trên token thật, bỏ qua padding)
                att_mask = inputs.get("attention_mask", torch.ones_like(line_labels)).float()
                target_labels = line_labels.to(token_vuln_logits.device)

                # Binary Cross Entropy per-token
                bce_fn = nn.BCEWithLogitsLoss(reduction="none")
                per_token_loss = bce_fn(token_vuln_logits, target_labels)  # [batch, seq]

                # Trung bình loss trên các token hợp lệ
                token_counts = att_mask.sum(dim=-1).clamp(min=1.0)
                sample_line_loss = (per_token_loss * att_mask).sum(dim=-1) / token_counts  # [batch]

                # Áp dụng mặt nạ has_line_supervision (Mask M)
                mask_m = has_line_supervision.to(sample_line_loss.device)
                num_supervised = mask_m.sum().clamp(min=1.0)
                aux_line_loss = (sample_line_loss * mask_m).sum() / num_supervised

                total_loss = cls_loss + AUX_LINE_WEIGHT * aux_line_loss
            del last_hidden

        if return_outputs:
            # BẢO VỆ VRAM: Trainer lưu trữ outputs trong suốt evaluation_loop.
            # TUYỆT ĐỐI KHÔNG để hidden_states trong outputs trả về cho Trainer,
            # vì sẽ khiến Trainer gom 41 layer x 1061 mẫu lên VRAM gây tràn CUDA OOM!
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
        eval_inputs = {k: v for k, v in inputs.items() if k not in ("line_labels", "has_line_supervision")}
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
    """Xây dựng TrainingArguments v12."""
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
        "lr_scheduler_type": "cosine",
        "fp16": True,
        "optim": "paged_adamw_8bit",
        "gradient_checkpointing": USE_GRADIENT_CHECKPOINTING,
        "label_smoothing_factor": LABEL_SMOOTHING,
        "max_grad_norm": 0.3,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_macro",
        "greater_is_better": True,
        "logging_steps": 50,
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
# 5. THRESHOLD TUNING & SMOKE TESTS
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
]


def run_smoke_tests(model, tokenizer, threshold):
    model.eval()

    @torch.no_grad()
    def predict(code_text):
        enc = tokenizer(
            code_text.strip(),
            truncation=True,
            max_length=MAX_LEN,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(next(model.parameters()).device)
        prob_vuln = torch.softmax(model(**enc).logits, dim=-1)[0, 1].item()
        return ("Vulnerable" if prob_vuln >= threshold else "Safe"), prob_vuln

    logger.info(f"\nSmoke test v12 với threshold = {threshold:.3f}")
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
        logger.info(f"\n{sep}\n✅ TẤT CẢ SMOKE TESTS V12 ĐỀU PASS!\n{sep}")
    else:
        logger.info(
            f"\n{sep}\n⚠️  Một số smoke test chưa pass. "
            f"Kiểm tra lại threshold hoặc xem xét train thêm.\n{sep}"
        )
    return all_pass


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE V12
# ══════════════════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()
    print_banner()

    # ─── 1. Nạp data v12 ───
    data_path = find_data()
    rows = load_data(data_path)
    splits = split_data(rows)

    # ─── 2. Load model & tokenizer ───
    model, tokenizer = load_model_and_tokenizer()

    # ─── 3. Tokenize & build datasets v12 ───
    logger.info("Đang tokenize 3 tập v12 (train/val/test) kèm nhãn dòng lỗi...")
    train_weights = (
        build_class_label_weights(splits["train"])
        if USE_CLASS_BALANCED_SAMPLER
        else None
    )
    train_ds = VulnDatasetV12(splits["train"], tokenizer, sample_weights=train_weights)
    val_ds = VulnDatasetV12(splits["val"], tokenizer)
    test_ds = VulnDatasetV12(splits["test"], tokenizer)
    collator = VulnDataCollatorV12(tokenizer)

    # ─── 4. Khởi tạo Masked Line Guidance Trainer ───
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
    trainer = MaskedLineGuidanceTrainer(**trainer_kwargs)

    trainer.args._n_gpu = 1

    steps_per_epoch = max(
        1, math.ceil(len(train_ds) / max(1, TRAIN_BS) / max(1, GRAD_ACCUM))
    )
    total_steps = steps_per_epoch * EPOCHS

    sep = "═" * 60
    logger.info(f"\n{sep}")
    logger.info("BẮT ĐẦU HUẤN LUYỆN V12")
    logger.info(f"  Train mẫu:    {len(train_ds):,}")
    logger.info(f"  Val mẫu:      {len(val_ds):,}")
    logger.info(f"  Epochs:       {EPOCHS}")
    logger.info(f"  Steps/epoch:  ~{steps_per_epoch}")
    logger.info(f"  Total steps:  ~{total_steps}")
    logger.info(f"  Effective BS: {TRAIN_BS * GRAD_ACCUM}")
    logger.info(f"  LoRA rank:    {LORA_R}")
    logger.info(f"  Aux Line λ:   {AUX_LINE_WEIGHT}")
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
    logger.info(f"\n{'═' * 72}")
    logger.info("THRESHOLD TUNING trên Validation set (v12)")
    logger.info("═" * 72)

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
    logger.info(f"\n{'═' * 72}")
    logger.info("BENCHMARK TRÊN TEST SET (v12 - ĐÁNH GIÁ CHÍNH THỨC)")
    logger.info("═" * 72)

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

    # Đánh giá chi tiết từng nguồn dữ liệu (Đặc biệt là DAppSCAN)
    logger.info("\n--- Kết quả kiểm thử v12 theo từng nguồn dữ liệu ---")
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
    logger.info(f"\n{'═' * 72}")
    logger.info("LƯU MODEL V12 & CONFIG")
    logger.info("═" * 72)

    os.makedirs(SAVE_DIR, exist_ok=True)
    trainer.save_model(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

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
        "train_mode": "CodeLlama-13B v12 Sequence Classification + Masked Line Guidance (10,356 samples)",
        "dataset": DATA_FILENAME,
        "aux_line_weight": AUX_LINE_WEIGHT,
        "model_name": "codellama-v12-13b-detector-final",
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(SAVE_DIR, "threshold_config.json"), "w") as f:
        json.dump(threshold_config, f, indent=2)

    logger.info(f"✅ Đã lưu model v12 & config tại: {SAVE_DIR}")

    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, _ = divmod(remainder, 60)
    logger.info(f"\n{'═' * 72}")
    logger.info(f"  HOÀN TẤT HUẤN LUYỆN V12 — Tổng thời gian: {int(hours)}h {int(minutes)}m")
    logger.info(f"  Model saved: {SAVE_DIR}")
    logger.info(f"  Test F1 (macro): {test_metrics['f1']:.4f}")
    logger.info(f"  Best threshold:  {best_threshold:.3f}")
    logger.info(f"{'═' * 72}")


if __name__ == "__main__":
    main()
