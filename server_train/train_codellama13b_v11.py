#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  CodeLlama-13B v11 — Balanced Real-World Smart Contract Vulnerability Detector                  ║
║  QLoRA 4-bit Fine-Tuning trên final-dataset-v11-train.json (10,356 mẫu: 56% Real / 44% Synthetic)        ║
║                                                                              ║
║  Upgrade từ v9 (7B → 13B) — Chạy trên RTX 4070 Ti SUPER 16GB              ║
║  Ưu tiên CHẤT LƯỢNG tối đa: LoRA r=64, 8 epochs, patience=4               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Chạy trên server:
    nohup python train_codellama13b.py > train.log 2>&1 &

Hoặc:
    python train_codellama13b.py 2>&1 | tee train.log
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
import transformers
from torch.utils.data import WeightedRandomSampler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
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
# ENV CONFIG — Tối ưu cho server GPU đơn
# ══════════════════════════════════════════════════════════════════════════════
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Dùng GPU 0 (GPU 1 đang chạy embeddings)

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
# HYPERPARAMETERS — ƯU TIÊN CHẤT LƯỢNG TỐI ĐA
# ══════════════════════════════════════════════════════════════════════════════

# ─── Model: CodeLlama 13B (upgrade từ 7B) ───
HF_MODEL = "codellama/CodeLlama-13b-hf"

# ─── Context & Batch Size (RTX 4070 Ti SUPER 16GB VRAM) ───
MAX_LEN = 1024       # Bao phủ 99.93% mẫu, giữ nguyên để không mất thông tin
TRAIN_BS = 1         # Batch size 1 — 13B lớn hơn, cần tiết kiệm VRAM
EVAL_BS = 1          # Giảm từ 2 → 1 cho 13B (an toàn VRAM)
GRAD_ACCUM = 32      # Effective batch size = 1 × 32 = 32

# ─── Training Schedule (ưu tiên chất lượng tối đa) ───
EPOCHS = 10        # Tăng từ 6 → 10 epochs để model 13B hội tụ đầy đủ
LR = 5e-5            # Giảm LR: 13B cần LR nhỏ hơn để train ổn định
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06  # ~0.5 epoch warmup trên 21K mẫu
PATIENCE = 4         # Tăng patience → cho phép model "phục hồi" qua local minima
LABEL_SMOOTHING = 0.02
USE_GRADIENT_CHECKPOINTING = True  # Bắt buộc cho 13B trên 16GB

# ─── LoRA Config (rank cao → biểu diễn phong phú hơn) ───
LORA_R = 64          # Tăng từ 32 → 64: tận dụng model 13B lớn hơn
LORA_ALPHA = 128     # Alpha = 2 × rank (chuẩn best practice)
LORA_DROPOUT = 0.05
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

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
OUTPUT_DIR = str(SCRIPT_DIR / "codellama-v11-13b-detector")
SAVE_DIR = OUTPUT_DIR + "-final"

# ─── Dataset path ───
DATA_FILENAME = "final-dataset-v11-train.json"


def print_banner():
    """In banner thông tin đầu phiên train."""
    sep = "═" * 72
    logger.info(sep)
    logger.info("  CodeLlama-13B v11 — Balanced Real-World Smart Contract Vulnerability Detector")
    logger.info("  QLoRA 4-bit Fine-Tuning | Ưu tiên CHẤT LƯỢNG TỐI ĐA")
    logger.info(sep)
    logger.info(f"  Thời gian bắt đầu : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Base model         : {HF_MODEL}")
    logger.info(f"  MAX_LEN            : {MAX_LEN}")
    logger.info(f"  Effective BS       : {TRAIN_BS} × {GRAD_ACCUM} = {TRAIN_BS * GRAD_ACCUM}")
    logger.info(f"  Epochs             : {EPOCHS} (patience={PATIENCE})")
    logger.info(f"  Learning Rate      : {LR}")
    logger.info(f"  LoRA rank/alpha    : {LORA_R}/{LORA_ALPHA}")
    logger.info(f"  LoRA targets       : {len(LORA_TARGETS)} layers")
    logger.info(f"  Label Smoothing    : {LABEL_SMOOTHING}")
    if torch.cuda.is_available():
        logger.info(f"  GPU                : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"  VRAM               : {vram:.1f} GB")
    logger.info(f"  transformers       : {transformers.__version__}")
    logger.info(f"  torch              : {torch.__version__}")
    logger.info(sep)


# ══════════════════════════════════════════════════════════════════════════════
# 1. TÌM VÀ NẠP DATASET
# ══════════════════════════════════════════════════════════════════════════════

def find_data():
    """
    Tìm dataset: ưu tiên cùng thư mục script → data/ subfolder → các path khác.
    Hỗ trợ .json, .jsonl, .csv
    """
    search_paths = [
        # Cùng thư mục script
        str(SCRIPT_DIR / DATA_FILENAME),
        str(SCRIPT_DIR / "data" / DATA_FILENAME),
        # Relative paths (CWD)
        DATA_FILENAME,
        f"data/{DATA_FILENAME}",
        # Fallback paths tuyệt đối
        str(SCRIPT_DIR.parent / "final_dataset_train" / DATA_FILENAME),
    ]
    for path in search_paths:
        if os.path.isfile(path):
            return path
    # Glob search
    for pattern in [f"**/{DATA_FILENAME}", f"../**/{DATA_FILENAME}"]:
        hits = sorted(glob.glob(pattern, recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"❌ Không tìm thấy dataset '{DATA_FILENAME}'!\n"
        f"   Vui lòng đặt file vào cùng thư mục với script hoặc subfolder data/"
    )


def load_data(data_path):
    """Nạp và chuẩn hóa dataset từ .json/.jsonl/.csv"""
    logger.info(f"Đang nạp dataset: {data_path}")

    if data_path.endswith(".csv"):
        with open(data_path, encoding="utf-8") as f:
            raw_rows = list(csv.DictReader(f))
        for r in raw_rows:
            if "label" in r and isinstance(r["label"], str) and r["label"].isdigit():
                r["label"] = int(r["label"])
    elif data_path.endswith(".jsonl"):
        raw_rows = [
            json.loads(line)
            for line in open(data_path, encoding="utf-8")
            if line.strip()
        ]
    else:
        with open(data_path, encoding="utf-8") as f:
            raw_rows = json.load(f)

    # Lọc mẫu hợp lệ: có code + label hợp lệ
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
    logger.info("PHÂN BỔ CÁC TẬP SPLIT (final-dataset-train.json)")
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
        role_desc = role_map.get(split_name, "")
        logger.info(
            f"  {split_name:5s}: {len(items):6d} mẫu "
            f"(Safe {n_safe:5d} / Vuln {n_vuln:5d}) — {role_desc}"
        )
        logger.info(f"         Nguồn: {by_source}")
    logger.info(sep)
    return splits


# ══════════════════════════════════════════════════════════════════════════════
# 2. TOKENIZE & CLASS-BALANCED SAMPLER
# ══════════════════════════════════════════════════════════════════════════════

def build_model_input(row):
    """Xây dựng input text cho model: chỉ dùng raw code, không thêm prefix."""
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


class VulnDataset(torch.utils.data.Dataset):
    """Dataset wrapper cho sequence classification."""

    def __init__(self, items, tokenizer_fn, sample_weights=None):
        self.items = items
        self.texts = [build_model_input(r) for r in items]
        self.enc = tokenizer_fn(
            self.texts,
            truncation=True,
            max_length=MAX_LEN,
            add_special_tokens=True,
        )
        self.labels = [r["label"] for r in items]
        self.sample_weights = sample_weights

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: self.enc[k][idx] for k in self.enc}
        item["labels"] = self.labels[idx]
        return item


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL LOADING
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
        modules_to_save=["score"],  # Classification head cần trainable đầy đủ
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.is_parallelizable = True
    model.model_parallel = True

    # Log VRAM usage after model load
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        logger.info(f"VRAM sau khi load model: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")

    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# 4. METRICS & TRAINER
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
    """Metrics chuẩn theo bài báo: macro-averaged precision, recall, F1."""
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


class ClassBalancedTrainer(Trainer):
    """Custom Trainer với WeightedRandomSampler để cân bằng class."""

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


def build_training_args():
    """Xây dựng TrainingArguments, tự động xử lý tương thích phiên bản."""
    kwargs = {
        "output_dir": OUTPUT_DIR,
        "num_train_epochs": EPOCHS,
        "per_device_train_batch_size": TRAIN_BS,
        "per_device_eval_batch_size": EVAL_BS,
        "gradient_accumulation_steps": GRAD_ACCUM,
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
        # Handle eval_strategy vs evaluation_strategy
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
# 5. THRESHOLD TUNING
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


# ══════════════════════════════════════════════════════════════════════════════
# 6. SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

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
    require(ok);
}""",
    },
    {
        "name": "Safe AccessControl onlyOwner",
        "expected": "Safe",
        "code": """function setLimit(uint256 _limit) external onlyOwner {
    require(_limit > 0, "invalid");
    limit = _limit;
}""",
    },
    {
        "name": "tx.origin Auth (Vulnerable)",
        "expected": "Vulnerable",
        "code": """function emergencyWithdraw() external {
    require(tx.origin == owner, "not owner");
    payable(msg.sender).transfer(address(this).balance);
}""",
    },
]


def run_smoke_tests(model, tokenizer, threshold):
    """Chạy smoke test inference."""
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

    logger.info(f"\nSmoke test với threshold = {threshold:.3f}")
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
        logger.info(f"\n{sep}\n✅ TẤT CẢ SMOKE TESTS ĐỀU PASS!\n{sep}")
    else:
        logger.info(
            f"\n{sep}\n⚠️  Một số smoke test chưa pass. "
            f"Kiểm tra lại threshold hoặc xem xét train thêm.\n{sep}"
        )
    return all_pass


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — PIPELINE CHÍNH
# ══════════════════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()
    print_banner()

    # ─── 1. Nạp data ───
    data_path = find_data()
    rows = load_data(data_path)
    splits = split_data(rows)

    # ─── 2. Load model & tokenizer ───
    model, tokenizer = load_model_and_tokenizer()

    # ─── 3. Tokenize & build datasets ───
    logger.info("Đang tokenize 3 tập (train/val/test)...")
    train_weights = (
        build_class_label_weights(splits["train"])
        if USE_CLASS_BALANCED_SAMPLER
        else None
    )
    train_ds = VulnDataset(splits["train"], tokenizer, sample_weights=train_weights)
    val_ds = VulnDataset(splits["val"], tokenizer)
    test_ds = VulnDataset(splits["test"], tokenizer)
    collator = DataCollatorWithPadding(tokenizer)

    # Thống kê token bị cắt
    ntr = sum(
        1
        for text in train_ds.texts
        if len(tokenizer(text, add_special_tokens=True)["input_ids"]) > MAX_LEN
    )
    logger.info(
        f"Train bị cắt >{MAX_LEN} token: {ntr}/{len(train_ds)} "
        f"({100 * ntr / len(train_ds):.2f}%)"
    )

    # Thống kê token length (sample 2000 đầu)
    sample_texts = train_ds.texts[:2000]
    tok_lens = [
        len(tokenizer(t, add_special_tokens=True)["input_ids"]) for t in sample_texts
    ]
    logger.info(
        f"Token length (2000 mẫu đầu): mean={np.mean(tok_lens):.0f}, "
        f"median={np.median(tok_lens):.0f}, "
        f"p95={np.percentile(tok_lens, 95):.0f}, max={max(tok_lens)}"
    )

    # ─── 4. Training ───
    args = build_training_args()
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": collator,
        "compute_metrics": compute_metrics,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    }
    try:
        trainer = ClassBalancedTrainer(
            processing_class=tokenizer, **trainer_kwargs
        )
    except TypeError:
        trainer = ClassBalancedTrainer(tokenizer=tokenizer, **trainer_kwargs)

    trainer.args._n_gpu = 1

    steps_per_epoch = max(
        1, math.ceil(len(train_ds) / max(1, TRAIN_BS) / max(1, GRAD_ACCUM))
    )
    total_steps = steps_per_epoch * EPOCHS

    sep = "═" * 60
    logger.info(f"\n{sep}")
    logger.info("BẮT ĐẦU HUẤN LUYỆN")
    logger.info(f"  Train mẫu:    {len(train_ds):,}")
    logger.info(f"  Val mẫu:      {len(val_ds):,}")
    logger.info(f"  Epochs:       {EPOCHS}")
    logger.info(f"  Steps/epoch:  ~{steps_per_epoch}")
    logger.info(f"  Total steps:  ~{total_steps}")
    logger.info(f"  Effective BS: {TRAIN_BS * GRAD_ACCUM}")
    logger.info(f"  LoRA rank:    {LORA_R}")
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
    logger.info("THRESHOLD TUNING trên Validation set")
    logger.info("═" * 72)

    val_out = trainer.predict(val_ds)
    val_logits = logits_from_predictions(val_out.predictions)
    val_probs = probs_from_logits(val_logits)
    val_labels = val_out.label_ids
    best_threshold_info, threshold_rows = tune_threshold(val_labels, val_probs)
    best_threshold = best_threshold_info["threshold"]

    logger.info(
        f"Best threshold by validation {THRESHOLD_OBJECTIVE}: {best_threshold:.3f}"
    )
    logger.info(
        f">> VAL: Accuracy={best_threshold_info['accuracy']:.4f}  "
        f"Precision={best_threshold_info['precision']:.4f}  "
        f"Recall={best_threshold_info['recall']:.4f}  "
        f"F1={best_threshold_info['f1']:.4f}"
    )

    logger.info("\nTop 10 thresholds trên Validation:")
    header = f"{'thr':>6s} {'acc':>7s} {'prec':>7s} {'recall':>7s} {'f1':>7s}"
    logger.info(header)
    for r in sorted(
        threshold_rows, key=lambda x: x[THRESHOLD_OBJECTIVE], reverse=True
    )[:10]:
        logger.info(
            f"{r['threshold']:6.3f} {r['accuracy']:7.4f} "
            f"{r['precision']:7.4f} {r['recall']:7.4f} {r['f1']:7.4f}"
        )

    # ─── 6. Đánh giá trên TEST set ───
    logger.info(f"\n{'═' * 72}")
    logger.info(
        f"TEST (n={len(test_ds)}) — CodeLlama-13B v10 QLoRA — "
        f"threshold={best_threshold:.3f}"
    )
    logger.info("═" * 72)

    pred_out = trainer.predict(test_ds)
    test_logits = logits_from_predictions(pred_out.predictions)
    test_probs = probs_from_logits(test_logits)
    yt = pred_out.label_ids
    yp = (test_probs >= best_threshold).astype(int)

    logger.info(
        "\n"
        + classification_report(
            yt, yp, target_names=["Safe", "Vulnerable"], digits=4, zero_division=0
        )
    )
    logger.info(f"Confusion [[TN FP][FN TP]]:\n{confusion_matrix(yt, yp)}")
    m = paper_metrics(yt, yp)
    logger.info(
        f"\n>> Accuracy={m['accuracy']:.4f}  Precision={m['precision']:.4f}  "
        f"Recall={m['recall']:.4f}  F1={m['f1']:.4f}"
    )

    # Phân bố nhãn trên test set
    safe_total = sum(yt == 0)
    vuln_total = sum(yt == 1)
    logger.info(
        f"\nTest ground truth: Safe={safe_total} ({safe_total / len(yt):.1%}), "
        f"Vulnerable={vuln_total} ({vuln_total / len(yt):.1%})"
    )

    # Đánh giá theo từng nguồn dữ liệu
    test_items = splits["test"]
    sources_in_test = set(r.get("source", "unknown") for r in test_items)
    real_metrics = None
    if len(sources_in_test) > 1:
        logger.info("\n--- Kết quả theo từng nguồn dữ liệu ---")
        for src in sorted(sources_in_test):
            idxs = [
                i
                for i, r in enumerate(test_items)
                if r.get("source", "unknown") == src
            ]
            if len(idxs) < 10:
                continue
            src_yt = yt[idxs]
            src_yp = yp[idxs]
            sm = paper_metrics(src_yt, src_yp)
            logger.info(
                f"  {src:20s}: n={len(idxs):5d}  "
                f"Acc={sm['accuracy']:.4f}  F1={sm['f1']:.4f}"
            )

        # Đánh giá riêng cho toàn bộ dữ liệu hợp đồng thực tế (Real-World: dappscan + iaudit_trustllm)
        real_idxs = [
            i for i, r in enumerate(test_items)
            if r.get("source", "unknown") in ("dappscan", "iaudit_trustllm")
        ]
        if real_idxs:
            real_yt = yt[real_idxs]
            real_yp = yp[real_idxs]
            real_metrics = paper_metrics(real_yt, real_yp)
            logger.info("\n--- ĐÁNH GIÁ ĐỘC LẬP TRÊN HỢP ĐỒNG THỰC TẾ (REAL-WORLD ONLY) ---")
            logger.info(
                f"  real_world_total   : n={len(real_idxs):5d}  "
                f"Acc={real_metrics['accuracy']:.4f}  F1={real_metrics['f1']:.4f}  "
                f"Prec={real_metrics['precision']:.4f}  Recall={real_metrics['recall']:.4f}"
            )

    # ─── 7. Lưu model & config ───
    logger.info(f"\n{'═' * 72}")
    logger.info("LƯU MODEL & CONFIG")
    logger.info("═" * 72)

    os.makedirs(SAVE_DIR, exist_ok=True)
    trainer.save_model(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    threshold_config = {
        "base_model": HF_MODEL,
        "hf_model": HF_MODEL,
        "best_threshold": best_threshold,
        "threshold_objective": THRESHOLD_OBJECTIVE,
        "validation_metrics": best_threshold_info,
        "max_len": MAX_LEN,
        "label2id": LABEL2ID,
        "id2label": ID2LABEL,
        "train_mode": "CodeLlama-13B v11 sequence classification QLoRA 4-bit (final-dataset-v11-train 10,356 samples: 56% Real / 44% Synthetic)",
        "dataset": DATA_FILENAME,
        "dataset_splits": {
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        },
        "dataset_sources": dict(
            collections.Counter(r.get("source", "unknown") for r in rows)
        ),
        "dataset_total_samples": len(rows),
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "effective_batch_size": TRAIN_BS * GRAD_ACCUM,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "label_smoothing": LABEL_SMOOTHING,
        "class_balanced_sampler": USE_CLASS_BALANCED_SAMPLER,
        "class_balance_power": CLASS_BALANCE_POWER,
        "test_metrics": m,
        "real_world_test_metrics": real_metrics,
    }
    config_path = Path(SAVE_DIR) / "threshold_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(threshold_config, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ Đã lưu model & config tại: {SAVE_DIR}")
    logger.info(f"Files: {os.listdir(SAVE_DIR)}")

    # ─── 8. Smoke Test ───
    run_smoke_tests(model, tokenizer, best_threshold)

    # ─── Tổng kết ───
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    logger.info(f"\n{'═' * 72}")
    logger.info(f"  HOÀN TẤT — Tổng thời gian: {hours}h {minutes}m")
    logger.info(f"  Model saved: {SAVE_DIR}")
    logger.info(f"  Test F1 (macro): {m['f1']:.4f}")
    logger.info(f"  Best threshold:  {best_threshold:.3f}")
    logger.info(f"{'═' * 72}")


if __name__ == "__main__":
    main()
