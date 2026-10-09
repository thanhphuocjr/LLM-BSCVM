# %% [markdown]
# # CodeLlama-v5 - Vuln Detector (Function-Level) - QLoRA 4-bit
# 
# Huấn luyện mô hình **CodeLlama-7B** trên dataset chuẩn hóa **`detect_v5_functionlevel.jsonl`** với bài toán phân loại nhị phân: **`Safe`** vs **`Vulnerable`**.
# 
# ### Điểm mới ở bản v5:
# - **Dataset v5 sạch 100%**: Đã loại bỏ interface rỗng, bóc tách chuẩn hàm đơn lẻ, khử nhãn sai DAppSCAN và loại bỏ boilerplate rác.
# - **Không bị Truncate**: Toàn bộ mẫu v5 đều $\le 3,800$ ký tự, khớp trọn vẹn context `MAX_LEN = 1024` trên GPU Kaggle T4.
# - **Zero-Label-Smearing**: Không bị gán nhãn sai cho các hàm helper.
# - **Cân bằng domain gap**: Source-balanced sampling giữa DAppSCAN và Solodit.
# - **Tự động tối ưu Threshold**: Tìm ngưỡng F1 tối ưu trên tập Validation và lưu vào `threshold_config.json`.
# 
# ### Hướng dẫn chạy trên Kaggle:
# 1. **Add Data** -> Upload tệp `detect_v5_functionlevel.jsonl` (hoặc `.csv`).
# 2. **Bật GPU** (T4 x1 hoặc T4 x2, Accelerator: GPU T4).
# 3. Chạy cell cài đặt, **Restart Kernel**, rồi chọn **Run All**.
# 4. Nếu CodeLlama bị gate trên Hugging Face: Add Input -> Models -> `CodeLlama-7B` hoặc khai báo `HF_TOKEN` trong Kaggle Secrets.

# %% [markdown]
# ## 0. Cài đặt package & môi trường

# %%
# Cài đặt phiên bản ổn định nhất cho Kaggle GPU T4
# !pip install -q "transformers==4.46.3" "tokenizers==0.20.3" "huggingface_hub==0.25.2" "accelerate==1.0.1" "peft==0.13.2" "bitsandbytes>=0.43.0" scikit-learn sentencepiece
# !pip uninstall -y torchao
print("✅ Cài đặt hoàn tất -> Chọn Run -> Restart Kernel -> Run All")

# %% [markdown]
# ## 1. Cấu hình Hyperparameters

# %%
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Tránh lỗi DataParallel khi dùng 2xT4

import json, glob, random, collections, re, csv
from pathlib import Path
import numpy as np, torch
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
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Model base CodeLlama 7B
HF_MODEL = "codellama/CodeLlama-7b-hf"

# Dataset v5 đã được chuẩn hóa length <= 3800 chars -> MAX_LEN 1024 vừa vặn 100% mẫu
MAX_LEN = 1024
TRAIN_BS = 1       # Batch size 1 trên GPU T4 để tối ưu VRAM khi max_len=1024
EVAL_BS = 2
GRAD_ACCUM = 16    # Effective batch size = TRAIN_BS * GRAD_ACCUM = 16

EPOCHS = 4
LR = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
PATIENCE = 2
LABEL_SMOOTHING = 0.02
USE_GRADIENT_CHECKPOINTING = True

# Cấu hình LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Giảm domain gap giữa 2 nguồn DAppSCAN và Solodit
USE_SOURCE_BALANCED_SAMPLER = True
SOURCE_BALANCE_POWER = 0.7

# Ngưỡng phân loại tối ưu
THRESHOLD_OBJECTIVE = "f1"  # accuracy, precision, recall, f1
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.005), 3)

LABEL2ID = {"Safe": 0, "Vulnerable": 1}
ID2LABEL = {0: "Safe", 1: "Vulnerable"}
OUTPUT_DIR = "./codellama-v5-detector"
SAVE_DIR = OUTPUT_DIR + "-final"

print(
    "transformers:", transformers.__version__,
    "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "| base:", HF_MODEL,
    "| mode: QLoRA 4-bit",
    "| max_len:", MAX_LEN,
)

# %% [markdown]
# ## 2. Tìm model và nạp Dataset v5

# %%
def find_model():
    candidates = []
    for cfg in glob.glob("/kaggle/input/**/config.json", recursive=True):
        model_dir = os.path.dirname(cfg)
        try:
            model_type = json.load(open(cfg, encoding="utf-8")).get("model_type", "")
        except Exception:
            continue
        weights = (
            glob.glob(model_dir + "/*.safetensors")
            or glob.glob(model_dir + "/pytorch_model*.bin")
            or glob.glob(model_dir + "/*.index.json")
        )
        if model_type == "llama" and weights:
            candidates.append(model_dir)
    if candidates:
        candidates.sort(key=len)
        print("MODEL local:", candidates[0])
        return candidates[0]
    print("MODEL Hugging Face:", HF_MODEL)
    return HF_MODEL

def find_data():
    candidates = [
        "/kaggle/input/**/detect_v5_functionlevel.jsonl",
        "detect_v5_functionlevel.jsonl",
        "/kaggle/input/**/detect_v5_functionlevel.csv",
        "detect_v5_functionlevel.csv",
        "data/training/detect_v5_functionlevel.jsonl",
        "../data/training/detect_v5_functionlevel.jsonl",
        "E2E/data/training/detect_v5_functionlevel.jsonl",
    ]
    for pattern in candidates:
        hits = sorted(glob.glob(pattern, recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError("Chưa thấy dataset v5! Vui lòng Add Data: upload detect_v5_functionlevel.jsonl")

MODEL_PATH = find_model()
DATA_PATH = find_data()
print("DATASET v5:", DATA_PATH)

if DATA_PATH.endswith(".csv"):
    with open(DATA_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
else:
    rows = [json.loads(line) for line in open(DATA_PATH, encoding="utf-8") if line.strip()]

rows = [r for r in rows if r.get("code") and r.get("label") in LABEL2ID]
print(f"Tổng số mẫu nạp: {len(rows)}")

splits = {"train": [], "val": [], "test": []}
for r in rows:
    splits.get(r.get("split", "train"), splits["train"]).append(r)

for split_name, items in splits.items():
    n_vuln = sum(x["label"] == "Vulnerable" for x in items)
    by_source = dict(collections.Counter(x.get("source", "unknown") for x in items))
    by_group = dict(collections.Counter((x.get("source", "unknown"), x["label"]) for x in items))
    print(f"  {split_name:5s}: {len(items):5d} (Vuln {n_vuln}/Safe {len(items)-n_vuln}) | {by_source}")
    print("          source,label:", by_group)

# %% [markdown]
# ## 3. Tokenize và xây dựng Weighted Sampler

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def build_model_input(row):
    return row["code"].strip()

def build_source_label_weights(items):
    groups = [(r.get("source", "unknown"), r["label"]) for r in items]
    counts = collections.Counter(groups)
    n_groups = max(1, len(counts))
    base = []
    for g in groups:
        w = (len(items) / (n_groups * counts[g])) ** SOURCE_BALANCE_POWER
        base.append(w)
    weights = torch.as_tensor(base, dtype=torch.double)
    weights = weights / weights.mean()
    print("\nTrain sampler groups:")
    for g in sorted(counts):
        group_weights = [weights[i].item() for i, gg in enumerate(groups) if gg == g]
        print(f"  {g}: n={counts[g]:4d} weight={np.mean(group_weights):.3f}")
    return weights

class VulnDataset(torch.utils.data.Dataset):
    def __init__(self, items, sample_weights=None):
        self.items = items
        self.texts = [build_model_input(r) for r in items]
        self.enc = tokenizer(
            self.texts,
            truncation=True,
            max_length=MAX_LEN,
            add_special_tokens=True,
        )
        self.labels = [LABEL2ID[r["label"]] for r in items]
        self.sample_weights = sample_weights

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: self.enc[k][idx] for k in self.enc}
        item["labels"] = self.labels[idx]
        return item

train_weights = build_source_label_weights(splits["train"]) if USE_SOURCE_BALANCED_SAMPLER else None
train_ds = VulnDataset(splits["train"], sample_weights=train_weights)
val_ds = VulnDataset(splits["val"])
test_ds = VulnDataset(splits["test"])
collator = DataCollatorWithPadding(tokenizer)

ntr = sum(1 for text in train_ds.texts if len(tokenizer(text, add_special_tokens=True)["input_ids"]) > MAX_LEN)
print(f"\nTrain bị cắt >{MAX_LEN} token: {ntr}/{len(train_ds)} ({100*ntr/len(train_ds):.1f}%)")
print("Ví dụ input mẫu v5:\n", train_ds.texts[0][:400])

# %% [markdown]
# ## 4. Nạp CodeLlama 7B (4-bit NF4) + QLoRA

# %%
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=2,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.float16,
)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING)
lora = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    target_modules=LORA_TARGETS,
    modules_to_save=["score"],
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

model.is_parallelizable = True
model.model_parallel = True

# %% [markdown]
# ## 5. Huấn luyện mô hình (Fine-Tuning)

# %%
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

class SourceBalancedTrainer(Trainer):
    def _get_train_sampler(self):
        if not USE_SOURCE_BALANCED_SAMPLER:
            return super()._get_train_sampler()
        if self.train_dataset is None or getattr(self.train_dataset, "sample_weights", None) is None:
            return super()._get_train_sampler()
        generator = torch.Generator()
        generator.manual_seed(SEED)
        return WeightedRandomSampler(
            weights=self.train_dataset.sample_weights,
            num_samples=len(self.train_dataset.sample_weights),
            replacement=True,
            generator=generator,
        )

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=TRAIN_BS,
    per_device_eval_batch_size=EVAL_BS,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    fp16=True,
    optim="paged_adamw_8bit",
    gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
    label_smoothing_factor=LABEL_SMOOTHING,
    max_grad_norm=0.3,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_steps=25,
    save_total_limit=2,
    report_to="none",
    seed=SEED,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
)

trainer = SourceBalancedTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
)
trainer.args._n_gpu = 1

trainer.train()

print("\nEpoch | val_acc | val_prec | val_recall | val_f1 | val_f1_vuln")
for h in trainer.state.log_history:
    if "eval_f1_macro" in h:
        print(
            f"  {round(h['epoch']):3d} | {h['eval_accuracy']:.4f} | {h['eval_precision']:.4f} | "
            f"{h['eval_recall']:.4f} | {h['eval_f1_macro']:.4f} | {h['eval_f1_vuln']:.4f}"
        )

# %% [markdown]
# ## 6. Tìm Threshold tối ưu trên Validation

# %%
def metrics_at_threshold(labels, probs, threshold):
    pred = (probs >= threshold).astype(int)
    return paper_metrics(labels, pred)

def tune_threshold(labels, probs, thresholds=THRESHOLD_GRID, objective=THRESHOLD_OBJECTIVE):
    rows = []
    for th in thresholds:
        m = metrics_at_threshold(labels, probs, th)
        rows.append({"threshold": float(th), **m})
    best = max(rows, key=lambda r: (r[objective], r["f1"], r["recall"], r["precision"]))
    return best, rows

val_out = trainer.predict(val_ds)
val_logits = logits_from_predictions(val_out.predictions)
val_probs = probs_from_logits(val_logits)
val_labels = val_out.label_ids
BEST_THRESHOLD_INFO, threshold_rows = tune_threshold(val_labels, val_probs)
BEST_THRESHOLD = BEST_THRESHOLD_INFO["threshold"]

print(f"Best threshold by validation {THRESHOLD_OBJECTIVE}: {BEST_THRESHOLD:.3f}")
print(
    f">> VAL: Accuracy={BEST_THRESHOLD_INFO['accuracy']:.4f}  "
    f"Precision={BEST_THRESHOLD_INFO['precision']:.4f}  "
    f"Recall={BEST_THRESHOLD_INFO['recall']:.4f}  F1={BEST_THRESHOLD_INFO['f1']:.4f}"
)

print("\nTop thresholds:")
print(f"{'thr':>6s} {'acc':>7s} {'prec':>7s} {'recall':>7s} {'f1':>7s}")
for r in sorted(threshold_rows, key=lambda x: x[THRESHOLD_OBJECTIVE], reverse=True)[:10]:
    print(f"{r['threshold']:6.3f} {r['accuracy']:7.4f} {r['precision']:7.4f} {r['recall']:7.4f} {r['f1']:7.4f}")

# %% [markdown]
# ## 7. Đánh giá kiểm thử trên tập TEST

# %%
pred_out = trainer.predict(test_ds)
test_logits = logits_from_predictions(pred_out.predictions)
test_probs = probs_from_logits(test_logits)
yt = pred_out.label_ids
yp = (test_probs >= BEST_THRESHOLD).astype(int)

print("=" * 64 + f"\nTEST (n={len(yt)}) - CodeLlama-v5 QLoRA - threshold={BEST_THRESHOLD:.3f}\n" + "=" * 64)
print(classification_report(yt, yp, target_names=["Safe", "Vulnerable"], digits=4, zero_division=0))
print("Confusion [[TN FP][FN TP]]:\n", confusion_matrix(yt, yp))
m = paper_metrics(yt, yp)
print(f"\n>> Accuracy={m['accuracy']:.4f}  Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  F1={m['f1']:.4f}")

src = np.array([r.get("source", "unknown") for r in splits["test"]])
print("\n--- Theo nguồn ---")
print(f"{'source':10s} {'n':>4s} {'acc':>7s} {'prec':>7s} {'recall':>7s} {'f1':>7s}")
for source in sorted(set(src)):
    idx = np.where(src == source)[0]
    sm = paper_metrics(yt[idx], yp[idx])
    print(f"{source:10s} {len(idx):4d} {sm['accuracy']:7.3f} {sm['precision']:7.3f} {sm['recall']:7.3f} {sm['f1']:7.3f}")

# %% [markdown]
# ## 8. Lưu Adapter QLoRA và Threshold Config

# %%
os.makedirs(SAVE_DIR, exist_ok=True)
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

threshold_config = {
    "base_model": MODEL_PATH,
    "hf_model": HF_MODEL,
    "best_threshold": BEST_THRESHOLD,
    "threshold_objective": THRESHOLD_OBJECTIVE,
    "validation_metrics": BEST_THRESHOLD_INFO,
    "max_len": MAX_LEN,
    "label2id": LABEL2ID,
    "id2label": ID2LABEL,
    "train_mode": "CodeLlama v5 sequence classification QLoRA 4-bit",
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "label_smoothing": LABEL_SMOOTHING,
    "source_balanced_sampler": USE_SOURCE_BALANCED_SAMPLER,
    "source_balance_power": SOURCE_BALANCE_POWER,
}
with open(Path(SAVE_DIR) / "threshold_config.json", "w", encoding="utf-8") as f:
    json.dump(threshold_config, f, ensure_ascii=False, indent=2)

print("Đã lưu model & config tại:", SAVE_DIR)
print(os.listdir(SAVE_DIR))

# %% [markdown]
# ## 9. Smoke Test Inference

# %%
model.eval()

@torch.no_grad()
def predict(code, threshold=None):
    if threshold is None:
        threshold = BEST_THRESHOLD
    enc = tokenizer(
        code.strip(),
        truncation=True,
        max_length=MAX_LEN,
        add_special_tokens=True,
        return_tensors="pt",
    ).to(next(model.parameters()).device)
    prob_vuln = torch.softmax(model(**enc).logits, dim=-1)[0, 1].item()
    return ("Vulnerable" if prob_vuln >= threshold else "Safe"), prob_vuln

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
        "name": "Access Control Missing (vulnerable)",
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
]

print(f"Smoke test với threshold = {BEST_THRESHOLD:.3f}")
print(f"{'Case':35s} {'Expect':12s} {'Pred':12s} {'P(vuln)':>8s}")
for t in SMOKE_TESTS:
    pred, p = predict(t["code"])
    status = "✅" if pred == t["expected"] else "❌"
    print(f"{t['name']:35s} {t['expected']:12s} {pred:12s} {p:8.3f} {status}")
