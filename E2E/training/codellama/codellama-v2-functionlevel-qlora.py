# %% [markdown]
# # CodeLlama-v2 - Vuln Detector function-level - QLoRA 4-bit
#
# Train tren `detect_v4_functionlevel.jsonl` voi bai toan binary classification:
# `Safe` vs `Vulnerable`.
#
# Ban v2 tap trung vao van de da thay trong ket qua CodeLlama v1:
# - giu CodeLlama QLoRA 4-bit lam baseline chinh
# - source-balanced sampler theo `(source, label)` de giam domain gap DAppSCAN/Solodit
# - optional context input neu dataset co `state_vars`, `modifiers`, `contract_context`, ...
# - label smoothing nhe + gradient checkpointing
# - threshold tuning tren validation bang Accuracy/Precision/Recall/F1
# - luu `threshold_config.json` de inference dung threshold da chon
#
# Kaggle:
# 1. Add Data -> upload `detect_v4_functionlevel.jsonl`
# 2. Bat GPU
# 3. Chay cell cai dat, Restart Kernel, roi Run All
# 4. Neu CodeLlama bi gate tren Hugging Face: Add Input -> Models -> CodeLlama-7B
#    hoac khai bao `HF_TOKEN` trong Kaggle Secrets.
#
# Mac dinh van dung QLoRA 4-bit vi CodeLlama-7B full fine-tune khong phu hop Kaggle T4.

# %% [markdown]
# ## 0. Cai dat package

# %%
# Cai ban on dinh cho Kaggle, roi RESTART KERNEL sau cell nay
# Notebook setup command:
# pip install -q transformers==4.46.3 tokenizers==0.20.3 huggingface_hub==0.25.2 accelerate==1.0.1 peft==0.13.2 bitsandbytes>=0.43.0 scikit-learn sentencepiece
# pip uninstall -y torchao
print("Xong -> Run -> Restart Kernel -> Run All")

# %% [markdown]
# ## 1. Cau hinh

# %%
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # tranh Trainer DataParallel tren 2xT4

import json, glob, random, collections, re
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

# Official CodeLlama co the can accept license/HF_TOKEN. Neu Kaggle da add model local,
# cell find_model() ben duoi se uu tien /kaggle/input truoc.
HF_MODEL = "codellama/CodeLlama-7b-hf"

MAX_LEN = 512      # Neu Kaggle con du VRAM/thoi gian: thu 768 hoac 1024.
TRAIN_BS = 2       # OOM thi giam 1
EVAL_BS = 4
GRAD_ACCUM = 8     # effective batch = TRAIN_BS * GRAD_ACCUM

EPOCHS = 6
LR = 2e-4
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.03
PATIENCE = 2
LABEL_SMOOTHING = 0.03
USE_GRADIENT_CHECKPOINTING = True

# QLoRA. De on dinh tren Kaggle T4, mac dinh r=16 nhu baseline CodeLlama da chay duoc.
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Giam domain gap: moi group (source, label) co xac suat duoc sample gan bang nhau.
USE_SOURCE_BALANCED_SAMPLER = True
SOURCE_BALANCE_POWER = 0.7   # 1.0 = can bang manh; 0.5 = nhe hon

# Input context: dataset hien tai co the chi co `code`, nhung neu sau nay co them field nay thi tu dong dung.
USE_CONTEXT_INPUT = True
CONTEXT_FIELDS = [
    "contract_context",
    "state_vars",
    "state_variables",
    "modifiers",
    "modifier_signatures",
    "inheritance",
]

# Chon threshold tren validation bang cac chi so paper.
# Chon "f1" de can bang chung; chon "recall" neu muon giam false negative hon.
THRESHOLD_OBJECTIVE = "f1"  # one of: accuracy, precision, recall, f1
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.005), 3)

LABEL2ID = {"Safe": 0, "Vulnerable": 1}
ID2LABEL = {0: "Safe", 1: "Vulnerable"}
OUTPUT_DIR = "./codellama-v2-qlora-balanced"
SAVE_DIR = OUTPUT_DIR + "-final"

assert THRESHOLD_OBJECTIVE in {"accuracy", "precision", "recall", "f1"}
print(
    "transformers", transformers.__version__,
    "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "| base:", HF_MODEL,
    "| mode: QLoRA 4-bit",
    "| max_len:", MAX_LEN,
)

# %% [markdown]
# ## 2. Tim model + nap du lieu + leakage guard nhe

# %%
def find_model():
    # Uu tien model da Add Input trong Kaggle, tranh loi gated HF.
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
        "/kaggle/input/**/detect_v4_functionlevel.jsonl",
        "detect_v4_functionlevel.jsonl",
        "../DatasetBuild/output/detect_v4_functionlevel.jsonl",
        "DatasetBuild/output/detect_v4_functionlevel.jsonl",
    ]
    for pattern in candidates:
        hits = sorted(glob.glob(pattern, recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError("Chua thay data -> Add Data: upload detect_v4_functionlevel.jsonl")


SOLIDITY_KEYWORDS = set("""
abstract after alias anonymous apply as assembly auto bool break calldata case catch constant
constructor continue contract default delete do else emit enum error event external false fallback
fixed for function hex if immutable import in indexed inline int interface internal is library mapping
memory modifier new null of override payable pragma private public pure receive return returns storage
string struct super supports switch this throw true try type typedef ufixed uint unchecked unicode using
var view virtual while wei ether gwei seconds minutes hours days weeks years address bytes msg tx block
abi assert require revert selfdestruct keccak256 sha256 ripemd160 ecrecover addmod mulmod now
""".split())
TOKEN_RE = re.compile(r'("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|0x[0-9a-fA-F]+|\b\d+(?:\.\d+)?\b|\b[A-Za-z_]\w*\b|\s+|.)', re.S)


def strip_comments(code):
    code = re.sub(r"//.*", " ", code)
    code = re.sub(r"/\*.*?\*/", " ", code, flags=re.S)
    return code


def structural_signature(code):
    out = []
    for tok in TOKEN_RE.findall(strip_comments(code)):
        if tok.isspace():
            continue
        if tok[0] in "'\"":
            out.append("STR")
        elif re.fullmatch(r"0x[0-9a-fA-F]+|\d+(?:\.\d+)?", tok):
            out.append("NUM")
        elif re.fullmatch(r"[A-Za-z_]\w*", tok):
            out.append(tok if tok in SOLIDITY_KEYWORDS else "ID")
        else:
            out.append(tok)
    return " ".join(out)


def report_split_leakage_sanity(splits):
    exact = collections.defaultdict(set)
    structural = collections.defaultdict(set)
    for split_name, items in splits.items():
        for r in items:
            code = r["code"]
            exact[re.sub(r"\s+", "", code)].add(split_name)
            structural[structural_signature(code)].add(split_name)
    exact_leaks = sum(1 for v in exact.values() if len(v) > 1)
    structural_leaks = sum(1 for v in structural.values() if len(v) > 1)
    print(f"Leakage guard: exact_cross_split={exact_leaks} | structural_like_cross_split={structural_leaks}")
    if exact_leaks:
        print("  CANH BAO: Co exact duplicate giua cac split. Nen build lai dataset bang DatasetBuild/scripts/merge_v4.py")
    if structural_leaks:
        print("  Note: structural_like la check rat manh tay, co the bao false positive; dung de canh bao, khong thay the Union-Find split.")


MODEL_PATH = find_model()
DATA_PATH = find_data()
print("DATA:", DATA_PATH)

rows = [json.loads(line) for line in open(DATA_PATH, encoding="utf-8") if line.strip()]
rows = [r for r in rows if r.get("code") and r.get("label") in LABEL2ID]

splits = {"train": [], "val": [], "test": []}
for r in rows:
    splits.get(r.get("split", "train"), splits["train"]).append(r)

for split_name, items in splits.items():
    n_vuln = sum(x["label"] == "Vulnerable" for x in items)
    by_source = dict(collections.Counter(x.get("source", "unknown") for x in items))
    by_group = dict(collections.Counter((x.get("source", "unknown"), x["label"]) for x in items))
    print(f"  {split_name:5s}: {len(items):5d} (Vuln {n_vuln}/Safe {len(items)-n_vuln}) | {by_source}")
    print("          source,label:", by_group)

report_split_leakage_sanity(splits)

# %% [markdown]
# ## 3. Tokenize + source-balanced weights

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def _as_text(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(str(x) for x in value if x)
    if isinstance(value, dict):
        return "\n".join(f"{k}: {v}" for k, v in value.items() if v)
    return str(value)


def build_model_input(row):
    code = row["code"]
    if not USE_CONTEXT_INPUT:
        return code
    blocks = []
    for field in CONTEXT_FIELDS:
        text = _as_text(row.get(field)).strip()
        if text:
            blocks.append(f"[{field.upper()}]\n{text}")
    blocks.append(f"[FUNCTION]\n{code}")
    return "\n\n".join(blocks)


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
print(f"\nTrain bi cat >{MAX_LEN} token: {ntr}/{len(train_ds)} ({100*ntr/len(train_ds):.1f}%)")
print("Input example:\n", train_ds.texts[0][:700])

# %% [markdown]
# ## 4. Nap CodeLlama 4-bit + gan QLoRA

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

# Kaggle co the cap 2 GPU, nhung model da nam tren 1 GPU.
model.is_parallelizable = True
model.model_parallel = True

# %% [markdown]
# ## 5. Huan luyen

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
    tokenizer=tokenizer,
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
# ## 6. Threshold tuning tren validation

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
# ## 7. Danh gia TEST bang threshold da chon

# %%
pred_out = trainer.predict(test_ds)
test_logits = logits_from_predictions(pred_out.predictions)
test_probs = probs_from_logits(test_logits)
yt = pred_out.label_ids
yp = (test_probs >= BEST_THRESHOLD).astype(int)

print("=" * 64 + f"\nTEST (n={len(yt)}) - CodeLlama-v2 QLoRA - threshold={BEST_THRESHOLD:.3f}\n" + "=" * 64)
print(classification_report(yt, yp, target_names=["Safe", "Vulnerable"], digits=4, zero_division=0))
print("Confusion [[TN FP][FN TP]]:\n", confusion_matrix(yt, yp))
m = paper_metrics(yt, yp)
print(f"\n>> Accuracy={m['accuracy']:.4f}  Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  F1={m['f1']:.4f}")

src = np.array([r.get("source", "unknown") for r in splits["test"]])
print("\n--- Theo nguon ---")
print(f"{'source':10s} {'n':>4s} {'acc':>7s} {'prec':>7s} {'recall':>7s} {'f1':>7s}")
for source in sorted(set(src)):
    idx = np.where(src == source)[0]
    sm = paper_metrics(yt[idx], yp[idx])
    print(f"{source:10s} {len(idx):4d} {sm['accuracy']:7.3f} {sm['precision']:7.3f} {sm['recall']:7.3f} {sm['f1']:7.3f}")

print("\n--- Loi theo nguon ---")
print(f"{'source':10s} {'FP':>4s} {'FN':>4s}")
for source in sorted(set(src)):
    idx = np.where(src == source)[0]
    fp = int(np.sum((yt[idx] == LABEL2ID["Safe"]) & (yp[idx] == LABEL2ID["Vulnerable"])))
    fn = int(np.sum((yt[idx] == LABEL2ID["Vulnerable"]) & (yp[idx] == LABEL2ID["Safe"])))
    print(f"{source:10s} {fp:4d} {fn:4d}")

# %% [markdown]
# ## 8. Luu adapter + threshold

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
    "train_mode": "CodeLlama sequence classification QLoRA 4-bit",
    "use_context_input": USE_CONTEXT_INPUT,
    "context_fields": CONTEXT_FIELDS,
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "label_smoothing": LABEL_SMOOTHING,
    "source_balanced_sampler": USE_SOURCE_BALANCED_SAMPLER,
    "source_balance_power": SOURCE_BALANCE_POWER,
}
with open(Path(SAVE_DIR) / "threshold_config.json", "w", encoding="utf-8") as f:
    json.dump(threshold_config, f, ensure_ascii=False, indent=2)

print("Da luu:", SAVE_DIR, os.listdir(SAVE_DIR))
print("Threshold config:", threshold_config)

# %% [markdown]
# ## 9. Inference + smoke test SWC

# %%
model.eval()


def build_inference_text(code, **context):
    row = {"code": code}
    row.update(context)
    return build_model_input(row)


@torch.no_grad()
def predict(code, threshold=None, **context):
    if threshold is None:
        threshold = BEST_THRESHOLD
    text = build_inference_text(code, **context)
    enc = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LEN,
        add_special_tokens=True,
        return_tensors="pt",
    ).to(next(model.parameters()).device)
    prob_vuln = torch.softmax(model(**enc).logits, dim=-1)[0, 1].item()
    return ("Vulnerable" if prob_vuln >= threshold else "Safe"), prob_vuln


SMOKE_TESTS = [
    {
        "name": "SWC-107 reentrancy",
        "expected": "Vulnerable",
        "context": {"state_vars": "mapping(address => uint256) balances;"},
        "code": """function withdraw(uint amount) public {
    require(balances[msg.sender] >= amount);
    (bool ok,) = msg.sender.call{value: amount}("");
    require(ok);
    balances[msg.sender] -= amount;
}""",
    },
    {
        "name": "SWC-104 unchecked call",
        "expected": "Vulnerable",
        "code": """function pay(address payable to, uint256 amount) public {
    to.call{value: amount}("");
}""",
    },
    {
        "name": "SWC-115 tx.origin auth",
        "expected": "Vulnerable",
        "context": {"state_vars": "address owner;"},
        "code": """function emergencyWithdraw(address payable to) public {
    require(tx.origin == owner);
    to.transfer(address(this).balance);
}""",
    },
    {
        "name": "Access control missing",
        "expected": "Vulnerable",
        "context": {"state_vars": "address owner;"},
        "code": """function setOwner(address newOwner) public {
    owner = newOwner;
}""",
    },
    {
        "name": "Safe guarded withdraw",
        "expected": "Safe",
        "context": {
            "state_vars": "mapping(address => uint256) balances;",
            "modifiers": "nonReentrant",
        },
        "code": """function withdraw(uint256 amount) public nonReentrant {
    require(balances[msg.sender] >= amount);
    balances[msg.sender] -= amount;
    (bool ok,) = msg.sender.call{value: amount}("");
    require(ok);
}""",
    },
]

print(f"Smoke test threshold={BEST_THRESHOLD:.3f}")
print(f"{'case':28s} {'expect':12s} {'pred':12s} {'P(vuln)':>8s}")
for t in SMOKE_TESTS:
    label, prob = predict(t["code"], **t.get("context", {}))
    print(f"{t['name'][:28]:28s} {t['expected']:12s} {label:12s} {prob:8.3f}")
