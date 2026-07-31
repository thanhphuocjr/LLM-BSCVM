# %% [markdown]
# # CodeLlama - Vuln Detector v4 (function-level) - QLoRA 4-bit
#
# Train tren `detect_v4_functionlevel.jsonl` voi bai toan binary classification:
# `Safe` vs `Vulnerable`.
#
# Kaggle:
# 1. Add Data -> upload `detect_v4_functionlevel.jsonl`
# 2. Bat GPU
# 3. Chay cell cai dat, Restart Kernel, roi Run All
# 4. Neu model CodeLlama bi gate tren Hugging Face: Add Input -> Models -> CodeLlama-7B
#    hoac khai bao `HF_TOKEN` trong Kaggle Secrets.
#
# Mac dinh dung QLoRA 4-bit vi CodeLlama-7B full fine-tune khong phu hop Kaggle T4.

# %% [markdown]
# ## 0. Cai dat package

# %%
# Cai ban on dinh cho Kaggle, roi RESTART KERNEL
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

import json, glob, random, collections
import numpy as np, torch
import transformers
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

MAX_LEN = 512
TRAIN_BS = 2       # OOM thi giam 1
EVAL_BS = 4
GRAD_ACCUM = 8     # effective batch = TRAIN_BS * GRAD_ACCUM

EPOCHS = 5
LR = 2e-4
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.03
PATIENCE = 2

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

LABEL2ID = {"Safe": 0, "Vulnerable": 1}
ID2LABEL = {0: "Safe", 1: "Vulnerable"}
OUTPUT_DIR = "./codellama-v4-qlora"
SAVE_DIR = OUTPUT_DIR + "-final"

print(
    "transformers", transformers.__version__,
    "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "| base:", HF_MODEL,
)

# %% [markdown]
# ## 2. Tim model va nap du lieu

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
    by_source = dict(collections.Counter(x["source"] for x in items))
    print(f"  {split_name:5s}: {len(items):5d} (Vuln {n_vuln}/Safe {len(items)-n_vuln}) | {by_source}")

# %% [markdown]
# ## 3. Tokenize

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


class VulnDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items
        self.enc = tokenizer(
            [r["code"] for r in items],
            truncation=True,
            max_length=MAX_LEN,
            add_special_tokens=True,
        )
        self.labels = [LABEL2ID[r["label"]] for r in items]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: self.enc[k][idx] for k in self.enc}
        item["labels"] = self.labels[idx]
        return item


train_ds = VulnDataset(splits["train"])
val_ds = VulnDataset(splits["val"])
test_ds = VulnDataset(splits["test"])
collator = DataCollatorWithPadding(tokenizer)

ntr = sum(1 for r in splits["train"] if len(tokenizer(r["code"], add_special_tokens=True)["input_ids"]) > MAX_LEN)
print(f"Train bi cat >{MAX_LEN} token: {ntr}/{len(splits['train'])} ({100*ntr/len(splits['train']):.1f}%)")

# %% [markdown]
# ## 4. Nap CodeLlama 4-bit + gan LoRA

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

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
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
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    pred = np.argmax(logits, axis=-1)
    precision, recall, f1_macro, _ = precision_recall_fscore_support(
        labels, pred, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, pred),
        "precision": precision,
        "recall": recall,
        "f1_macro": f1_macro,
        "f1_vuln": f1_score(labels, pred, pos_label=1, zero_division=0),
    }


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

trainer = Trainer(
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
# ## 6. Danh gia TEST

# %%
pred_out = trainer.predict(test_ds)
yp = np.argmax(pred_out.predictions, axis=-1)
yt = pred_out.label_ids

print("=" * 58 + f"\nTEST (n={len(yt)}) - CodeLlama QLoRA\n" + "=" * 58)
print(classification_report(yt, yp, target_names=["Safe", "Vulnerable"], digits=4, zero_division=0))
print("Confusion [[TN FP][FN TP]]:\n", confusion_matrix(yt, yp))
precision, recall, f1_macro, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
print(f"\n>> Accuracy={accuracy_score(yt, yp):.4f}  Precision={precision:.4f}  Recall={recall:.4f}  F1={f1_macro:.4f}")

src = np.array([r["source"] for r in splits["test"]])
print("\n--- Theo nguon ---")
print(f"{'source':10s} {'n':>4s} {'acc':>7s} {'prec':>7s} {'recall':>7s} {'f1':>7s}")
for source in sorted(set(src)):
    idx = np.where(src == source)[0]
    p, r, f, _ = precision_recall_fscore_support(yt[idx], yp[idx], average="macro", zero_division=0)
    print(f"{source:10s} {len(idx):4d} {accuracy_score(yt[idx], yp[idx]):7.3f} {p:7.3f} {r:7.3f} {f:7.3f}")

# %% [markdown]
# ## 7. Luu adapter

# %%
os.makedirs(SAVE_DIR, exist_ok=True)
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
json.dump(
    {
        "base_model": MODEL_PATH,
        "max_len": MAX_LEN,
        "label2id": LABEL2ID,
        "id2label": ID2LABEL,
        "train_mode": "CodeLlama sequence classification QLoRA 4-bit",
    },
    open(os.path.join(SAVE_DIR, "training_config.json"), "w", encoding="utf-8"),
    indent=2,
)
print("Da luu:", SAVE_DIR, os.listdir(SAVE_DIR))

# %% [markdown]
# ## 8. Thu inference

# %%
model.eval()


@torch.no_grad()
def predict(code, threshold=0.5):
    enc = tokenizer(
        code,
        truncation=True,
        max_length=MAX_LEN,
        add_special_tokens=True,
        return_tensors="pt",
    ).to(model.device)
    prob_vuln = torch.softmax(model(**enc).logits, dim=-1)[0, 1].item()
    return ("Vulnerable" if prob_vuln >= threshold else "Safe"), prob_vuln


TEST_CODE = """function withdraw(uint amount) public {
    require(balances[msg.sender] >= amount);
    (bool ok,) = msg.sender.call{value: amount}("");
    require(ok);
    balances[msg.sender] -= amount;
}"""

label, prob = predict(TEST_CODE)
print(f"Du doan: {label}  (P(Vulnerable)={prob:.3f})")
