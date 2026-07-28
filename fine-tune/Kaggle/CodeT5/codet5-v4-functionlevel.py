# %% [markdown]
# # CodeT5 - Vuln Detector v4 (Kaggle-ready) - LoRA / full fine-tune
#
# Train tren `detect_v4_functionlevel.jsonl` voi CodeT5.
#
# Ban nay them:
# - source-balanced sampler de giam domain gap DAppSCAN/Solodit
# - optional context input (`state_vars`, `modifiers`, `contract_context` neu dataset co)
# - LoRA manh hon + label smoothing + gradient checkpointing
# - threshold tuning tren validation bang Accuracy/Precision/Recall/F1
# - leakage guard nhe truoc khi train
# - smoke test theo cac mau SWC thong dung
#
# Kaggle:
# 1. Add Data -> upload `detect_v4_functionlevel.jsonl`
# 2. Bat GPU
# 3. Chay cell cai dat, Restart Kernel, roi Run All

# %% [markdown]
# ## 0. Cai dat package

# %%
# Cai ban on dinh cho Kaggle, roi RESTART KERNEL sau cell nay
!pip install -q "transformers==4.46.3" "tokenizers==0.20.3" "huggingface_hub==0.25.2" "accelerate==1.0.1" "peft==0.13.2" scikit-learn
!pip uninstall -y torchao
print("Xong -> Run -> Restart Kernel -> Run All")

# %% [markdown]
# ## 1. Cau hinh

# %%
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json, glob, random, collections, re
from pathlib import Path
import numpy as np, torch
import transformers
from torch.utils.data import WeightedRandomSampler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
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

MODEL_NAME = "Salesforce/codet5-base"
MAX_LEN = 512
TRAIN_BS = 8       # OOM tren T4 thi giam 4 va tang GRAD_ACCUM
EVAL_BS = 16
GRAD_ACCUM = 2

# CodeT5 classification. Mac dinh LoRA de vua Kaggle T4.
USE_LORA = True
LORA_R, LORA_ALPHA, LORA_DROPOUT = 32, 64, 0.1
LORA_TARGETS = ["q", "k", "v", "o"]

# Neu GPU manh hon, co the thu full fine-tune: USE_LORA=False, LR=2e-5.
EPOCHS = 12 if USE_LORA else 8
LR = 2e-4 if USE_LORA else 2e-5
WEIGHT_DECAY, WARMUP_RATIO, PATIENCE = 0.01, 0.06, 3
LABEL_SMOOTHING = 0.05
USE_GRADIENT_CHECKPOINTING = True

# Giam domain gap: moi group (source, label) co xac suat duoc sample gan bang nhau.
USE_SOURCE_BALANCED_SAMPLER = True
SOURCE_BALANCE_POWER = 1.0   # 1.0 = can bang manh; 0.5 = nhe hon

# Input context: dataset hien tai chi co `code`, nhung neu sau nay co them cac field nay thi tu dong dung.
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
OUTPUT_DIR = "./codet5-v4-lora-balanced" if USE_LORA else "./codet5-v4-full-balanced"
SAVE_DIR = OUTPUT_DIR + "-final"

assert THRESHOLD_OBJECTIVE in {"accuracy", "precision", "recall", "f1"}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    "transformers", transformers.__version__,
    "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "| model:", MODEL_NAME,
    "| mode:", "LoRA" if USE_LORA else "full-FT",
    "| LR", LR,
    "| LoRA r", LORA_R if USE_LORA else "-",
)

# %% [markdown]
# ## 2. Nap du lieu + leakage guard nhe

# %%
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


PATH = find_data()
print("DATA:", PATH)
rows = [json.loads(line) for line in open(PATH, encoding="utf-8") if line.strip()]
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
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


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
            blocks.append(f"<{field.upper()}>\n{text}")
    blocks.append(f"<FUNCTION>\n{code}")
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
# ## 4. Model + huan luyen

# %%
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

if USE_LORA:
    from peft import LoraConfig, get_peft_model, TaskType

    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        modules_to_save=["classification_head"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

if USE_GRADIENT_CHECKPOINTING:
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "config"):
        model.config.use_cache = False

model = model.to(device)


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
    fp16=torch.cuda.is_available(),
    gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
    label_smoothing_factor=LABEL_SMOOTHING,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_steps=50,
    save_total_limit=2,
    report_to="none",
    seed=SEED,
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

trainer.train()

print("\nEpoch | val_acc | val_prec | val_recall | val_f1")
for h in trainer.state.log_history:
    if "eval_f1_macro" in h:
        print(
            f"  {round(h['epoch']):3d} | {h['eval_accuracy']:.4f} | {h['eval_precision']:.4f} | "
            f"{h['eval_recall']:.4f} | {h['eval_f1_macro']:.4f}"
        )

# %% [markdown]
# ## 5. Threshold tuning tren validation

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
# ## 6. Danh gia TEST bang threshold da chon

# %%
pred_out = trainer.predict(test_ds)
test_logits = logits_from_predictions(pred_out.predictions)
test_probs = probs_from_logits(test_logits)
yt = pred_out.label_ids
yp = (test_probs >= BEST_THRESHOLD).astype(int)

print("=" * 64 + f"\nTEST (n={len(yt)}) - {'LoRA' if USE_LORA else 'full-FT'} - threshold={BEST_THRESHOLD:.3f}\n" + "=" * 64)
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

# %% [markdown]
# ## 7. Luu model + threshold

# %%
os.makedirs(SAVE_DIR, exist_ok=True)
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

threshold_config = {
    "best_threshold": BEST_THRESHOLD,
    "threshold_objective": THRESHOLD_OBJECTIVE,
    "validation_metrics": BEST_THRESHOLD_INFO,
    "model_name": MODEL_NAME,
    "max_len": MAX_LEN,
    "use_context_input": USE_CONTEXT_INPUT,
    "context_fields": CONTEXT_FIELDS,
    "use_lora": USE_LORA,
    "lora_r": LORA_R if USE_LORA else None,
    "source_balanced_sampler": USE_SOURCE_BALANCED_SAMPLER,
}
with open(Path(SAVE_DIR) / "threshold_config.json", "w", encoding="utf-8") as f:
    json.dump(threshold_config, f, ensure_ascii=False, indent=2)

print("Da luu:", SAVE_DIR, os.listdir(SAVE_DIR))
print("Threshold config:", threshold_config)

# %% [markdown]
# ## 8. Inference + smoke test SWC

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
    ).to(device)
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

