# %% [markdown]
# # CodeLlama - Vuln Detector v4 AST-flow (function-level) - QLoRA 4-bit
#
# Train tren `detect_v4_functionlevel.jsonl` voi bai toan binary classification:
# `Safe` vs `Vulnerable`.
#
# Ban nay khong chi dua raw Solidity vao model. Moi sample duoc them mot
# AST/control-flow summary tu parser Solidity de CodeLlama thay ro hon:
# - cau truc AST preorder va tan suat node;
# - signature, modifier, control statements;
# - thu tu guard -> external call -> state write;
# - sensitive ops nhu msg.sender, msg.value, call/delegatecall/transfer.
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
# pip install -q "transformers==4.46.3" "tokenizers==0.20.3" "huggingface_hub==0.25.2" "accelerate==1.0.1" "peft==0.13.2" "bitsandbytes>=0.43.0" scikit-learn sentencepiece "solidity_parser==0.1.1"
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

MAX_LEN = 1024     # 1536 token lam Kaggle T4 vuot quota 12h; 1024 nhanh va on dinh hon
TRAIN_BS = 1       # CodeLlama 7B QLoRA tren T4 nen batch 1
EVAL_BS = 2
GRAD_ACCUM = 16    # effective batch = TRAIN_BS * GRAD_ACCUM

EPOCHS = 3
LR = 2e-4
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.03
PATIENCE = 1

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

USE_AST_AUGMENTATION = True
AST_MAX_NODES = 48
AST_MAX_FLOW_EVENTS = 40
AST_MAX_FEATURE_CHARS = 1400
CODE_MAX_CHARS = 5200

LABEL2ID = {"Safe": 0, "Vulnerable": 1}
ID2LABEL = {0: "Safe", 1: "Vulnerable"}
OUTPUT_DIR = "./codellama-v4-ast-qlora"
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
# ## 3. AST/control-flow encoding

# %%
def _load_solidity_parser():
    try:
        from solidity_parser import parser as sol_parser
        return sol_parser, None
    except Exception as exc:
        return None, repr(exc)


SOLIDITY_PARSER, AST_IMPORT_ERROR = _load_solidity_parser()
AST_FEATURE_CACHE = {}
AST_STATS = collections.Counter()
CONTROL_NODE_TYPES = {
    "IfStatement",
    "ForStatement",
    "WhileStatement",
    "DoWhileStatement",
    "ReturnStatement",
    "EmitStatement",
    "ThrowStatement",
    "RevertStatement",
    "TryStatement",
}
LOW_LEVEL_CALLS = ("call", "delegatecall", "staticcall", "send", "transfer")
SENSITIVE_PATTERNS = {
    "msg.sender": r"\bmsg\s*\.\s*sender\b",
    "msg.value": r"\bmsg\s*\.\s*value\b",
    "tx.origin": r"\btx\s*\.\s*origin\b",
    "block.timestamp": r"\bblock\s*\.\s*timestamp\b|\bnow\b",
    "block.number": r"\bblock\s*\.\s*number\b",
    "selfdestruct": r"\bselfdestruct\s*\(",
    "delegatecall": r"\.\s*delegatecall\s*(?:\{|\()",
    "low_level_call": r"\.\s*call\s*(?:\{|\()",
    "value_transfer": r"\.\s*(?:send|transfer)\s*\(",
}


def _short(text, limit=120):
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _unique(values, limit=24):
    out, seen = [], set()
    for value in values:
        value = _short(value, 100)
        if not value or value in seen:
            continue
        out.append(value)
        seen.add(value)
        if len(out) >= limit:
            break
    return out


def _join(values, empty="none", sep=", "):
    return sep.join(values) if values else empty


def _mask_noncode(code):
    chars = list(code)
    i = 0
    while i < len(chars):
        if code.startswith("//", i):
            j = code.find("\n", i + 2)
            j = len(code) if j < 0 else j
            for k in range(i, j):
                chars[k] = " "
            i = j
            continue
        if code.startswith("/*", i):
            j = code.find("*/", i + 2)
            j = len(code) if j < 0 else j + 2
            for k in range(i, j):
                if chars[k] != "\n":
                    chars[k] = " "
            i = j
            continue
        if code[i] in ("'", '"'):
            quote = code[i]
            j = i + 1
            while j < len(code):
                if code[j] == "\\":
                    j += 2
                    continue
                if code[j] == quote:
                    j += 1
                    break
                j += 1
            for k in range(i, min(j, len(chars))):
                if chars[k] != "\n":
                    chars[k] = " "
            i = j
            continue
        i += 1
    return "".join(chars)


def _node_type(node):
    if isinstance(node, dict):
        return str(node.get("type") or node.get("nodeType") or "")
    return ""


def _iter_ast_children(node):
    skip = {"loc", "range", "src", "id", "typeDescriptions"}
    if isinstance(node, dict):
        for key, value in node.items():
            if key in skip or key.startswith("_"):
                continue
            if isinstance(value, (dict, list)):
                yield value
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, (dict, list)):
                yield item


def _walk_ast(node):
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            yield current
        children = list(_iter_ast_children(current))
        stack.extend(reversed(children))


def _expr_name(node):
    if node is None:
        return ""
    if isinstance(node, str):
        return _short(node, 100)
    if isinstance(node, (int, float, bool)):
        return str(node)
    if isinstance(node, list):
        return _join([_expr_name(x) for x in node[:4]])
    if not isinstance(node, dict):
        return ""

    ntype = _node_type(node)
    if ntype in {"Identifier", "ElementaryTypeName", "UserDefinedTypeName"}:
        return _short(node.get("name") or node.get("namePath") or node.get("baseName") or ntype)
    if ntype in {"NumberLiteral", "StringLiteral", "BooleanLiteral", "HexLiteral", "DecimalNumber", "Literal"}:
        return _short(node.get("number") or node.get("value") or node.get("hexValue") or ntype)
    if ntype == "MemberAccess":
        base = _expr_name(node.get("expression"))
        member = _short(node.get("memberName") or node.get("member_name") or "")
        return f"{base}.{member}" if base and member else base or member
    if ntype == "IndexAccess":
        base = _expr_name(node.get("base") or node.get("expression"))
        index = _expr_name(node.get("index"))
        return f"{base}[{index}]" if index else f"{base}[]"
    if ntype == "FunctionCall":
        return _expr_name(node.get("expression"))
    if ntype == "FunctionCallOptions":
        return _expr_name(node.get("expression"))
    if ntype == "Assignment":
        left = _expr_name(node.get("left") or node.get("leftHandSide"))
        op = node.get("operator") or "="
        right = _expr_name(node.get("right") or node.get("rightHandSide"))
        return _short(f"{left} {op} {right}")
    if ntype == "BinaryOperation":
        left = _expr_name(node.get("left"))
        op = node.get("operator") or "?"
        right = _expr_name(node.get("right"))
        return _short(f"{left} {op} {right}")
    if ntype == "UnaryOperation":
        op = node.get("operator") or ""
        return _short(f"{op}{_expr_name(node.get('subExpression'))}")
    if ntype == "TupleExpression":
        return "(" + _join([_expr_name(x) for x in node.get("components", [])]) + ")"

    for key in ("name", "memberName", "operator", "value"):
        if node.get(key):
            return _short(node[key])
    if node.get("expression") is not None:
        return _expr_name(node.get("expression"))
    return ntype


def _base_identifier(node):
    if isinstance(node, str):
        match = re.search(r"[A-Za-z_$][\w$]*", node)
        return match.group(0) if match else ""
    if not isinstance(node, dict):
        return ""
    ntype = _node_type(node)
    if ntype == "Identifier":
        return str(node.get("name") or "")
    if ntype in {"IndexAccess", "MemberAccess", "FunctionCallOptions"}:
        return _base_identifier(node.get("base") or node.get("expression"))
    return _base_identifier(node.get("left") or node.get("leftHandSide") or node.get("expression"))


def _decl_list(node):
    if isinstance(node, dict):
        for key in ("parameters", "params", "variables"):
            if isinstance(node.get(key), list):
                return node[key]
        return [node]
    if isinstance(node, list):
        return node
    return []


def _decl_name(node):
    if isinstance(node, dict):
        return _short(node.get("name") or node.get("identifier") or "")
    return ""


def _type_name(node):
    if isinstance(node, dict):
        return _expr_name(node.get("typeName") or node.get("typeDescriptions") or node)
    return _expr_name(node)


def _first_node(ast, wanted_type):
    for node in _walk_ast(ast):
        if _node_type(node) == wanted_type:
            return node
    return None


def _function_meta(ast):
    func = _first_node(ast, "FunctionDefinition")
    if not func:
        return {
            "name": "unknown",
            "visibility": "unknown",
            "mutability": "unknown",
            "params": [],
            "returns": [],
            "modifiers": [],
        }

    params = []
    for param in _decl_list(func.get("parameters")):
        pname = _decl_name(param)
        ptype = _type_name(param)
        params.append(_short(f"{ptype} {pname}".strip()))

    returns = []
    for param in _decl_list(func.get("returnParameters") or func.get("returns")):
        rname = _decl_name(param)
        rtype = _type_name(param)
        returns.append(_short(f"{rtype} {rname}".strip()))

    modifiers = []
    for modifier in func.get("modifiers") or []:
        modifiers.append(_expr_name(modifier.get("modifierName") or modifier.get("name") or modifier))

    return {
        "name": _short(func.get("name") or "constructor"),
        "visibility": _short(func.get("visibility") or "default"),
        "mutability": _short(func.get("stateMutability") or func.get("state_mutability") or "nonpayable"),
        "params": _unique(params, 16),
        "returns": _unique(returns, 8),
        "modifiers": _unique(modifiers, 12),
    }


def _declared_names(ast):
    names = set()
    for node in _walk_ast(ast):
        if _node_type(node) == "VariableDeclaration":
            name = _decl_name(node)
            if name:
                names.add(name)
    return names


def _ast_shape(ast):
    seq = []
    counts = collections.Counter()

    def visit(node, depth=0):
        if isinstance(node, dict):
            ntype = _node_type(node)
            if ntype:
                counts[ntype] += 1
                if len(seq) < AST_MAX_NODES:
                    seq.append(f"{depth}:{ntype}")
        for child in _iter_ast_children(node):
            visit(child, depth + 1)

    visit(ast)
    return seq, counts


def _call_kind(call_name):
    lowered = call_name.lower().replace(" ", "")
    if lowered in {"require", "assert"}:
        return "GUARD"
    if lowered.endswith(".require") or lowered.endswith(".assert"):
        return "GUARD"
    if lowered in {"revert"} or lowered.endswith(".revert"):
        return "REVERT"
    if "selfdestruct" in lowered or lowered.endswith(".suicide"):
        return "SELFDESTRUCT"
    for op in LOW_LEVEL_CALLS:
        if lowered == op or lowered.endswith("." + op):
            return "VALUE_TRANSFER" if op in {"send", "transfer"} else "EXTERNAL_CALL"
    return "CALL"


def _ast_flow_events(ast):
    events = []
    declared = _declared_names(ast)
    for node in _walk_ast(ast):
        ntype = _node_type(node)
        if ntype == "IfStatement":
            events.append("BRANCH:if")
        elif ntype in {"ForStatement", "WhileStatement", "DoWhileStatement"}:
            events.append("LOOP:" + ntype.replace("Statement", "").lower())
        elif ntype == "ReturnStatement":
            events.append("RETURN")
        elif ntype == "EmitStatement":
            events.append("EMIT:" + _expr_name(node.get("eventCall") or node.get("expression")))
        elif ntype in {"ThrowStatement", "RevertStatement"}:
            events.append("REVERT")
        elif ntype == "Assignment":
            left = node.get("left") or node.get("leftHandSide")
            left_name = _expr_name(left)
            base = _base_identifier(left)
            write_kind = "LOCAL_WRITE" if base in declared else "STATE_WRITE"
            events.append(f"{write_kind}:{left_name}")
        elif ntype == "FunctionCall":
            call_name = _expr_name(node)
            events.append(f"{_call_kind(call_name)}:{call_name}")
        if len(events) >= AST_MAX_FLOW_EVENTS * 2:
            break
    return events[:AST_MAX_FLOW_EVENTS]


def _raw_sensitive_ops(code):
    masked = _mask_noncode(code)
    return [name for name, pattern in SENSITIVE_PATTERNS.items() if re.search(pattern, masked)]


def _flow_flags(events):
    kinds = [event.split(":", 1)[0] for event in events]
    external_idx = [i for i, kind in enumerate(kinds) if kind in {"EXTERNAL_CALL", "VALUE_TRANSFER", "SELFDESTRUCT"}]
    state_write_idx = [i for i, kind in enumerate(kinds) if kind == "STATE_WRITE"]
    guard_idx = [i for i, kind in enumerate(kinds) if kind == "GUARD"]
    if not external_idx:
        return "has_external_call=false"
    first_external = external_idx[0]
    flags = {
        "has_external_call": True,
        "guard_before_external": any(i < first_external for i in guard_idx),
        "state_write_before_external": any(i < first_external for i in state_write_idx),
        "external_before_state_write": any(i > first_external for i in state_write_idx),
    }
    return " ".join(f"{key}={str(value).lower()}" for key, value in flags.items())


def _feature_lines(parser_name, code, meta, shape, counts, events, reason=None):
    count_bits = [f"{key}:{value}" for key, value in counts.most_common(18)]
    control_counts = [f"{key}:{counts[key]}" for key in sorted(CONTROL_NODE_TYPES) if counts.get(key)]
    calls = [event.split(":", 1)[1] for event in events if event.startswith(("CALL:", "GUARD:", "EXTERNAL_CALL:", "VALUE_TRANSFER:"))]
    external_calls = [event.split(":", 1)[1] for event in events if event.startswith(("EXTERNAL_CALL:", "VALUE_TRANSFER:", "SELFDESTRUCT:"))]
    writes = [event.split(":", 1)[1] for event in events if event.startswith("STATE_WRITE:")]
    lines = [
        f"parser={parser_name}" + (f" reason={_short(reason, 80)}" if reason else ""),
        (
            "function="
            f"{meta['name']} visibility={meta['visibility']} mutability={meta['mutability']} "
            f"modifiers={_join(meta['modifiers'])} params={_join(meta['params'])} returns={_join(meta['returns'])}"
        ),
        "node_counts=" + _join(count_bits, sep=" "),
        "control_counts=" + _join(control_counts, sep=" "),
        "ast_preorder=" + _join(shape[:AST_MAX_NODES], sep=" > "),
        "flow=" + _join(events[:AST_MAX_FLOW_EVENTS], sep=" -> "),
        "flow_flags=" + _flow_flags(events),
        "state_writes=" + _join(_unique(writes, 16)),
        "calls=" + _join(_unique(calls, 24)),
        "external_calls=" + _join(_unique(external_calls, 12)),
        "sensitive_ops=" + _join(_raw_sensitive_ops(code)),
    ]
    text = "\n".join(lines)
    return text[:AST_MAX_FEATURE_CHARS]


def _fallback_ast_summary(code, reason):
    masked = _mask_noncode(code)
    signature = re.search(r"\bfunction\s+([A-Za-z_$][\w$]*)\s*\((.*?)\)", masked, flags=re.S)
    name = signature.group(1) if signature else "unknown"
    params = _unique(re.findall(r"\b(?:address|uint\d*|int\d*|bool|string|bytes\d*)\s+(?:memory|storage|calldata\s+)?([A-Za-z_$][\w$]*)", signature.group(2) if signature else ""), 16)
    visibility = next((kw for kw in ("external", "public", "internal", "private") if re.search(r"\b" + kw + r"\b", masked)), "default")
    mutability = next((kw for kw in ("payable", "view", "pure") if re.search(r"\b" + kw + r"\b", masked)), "nonpayable")
    modifier_text = " ".join(re.findall(r"\)\s*(.*?)\s*(?:returns\s*\(|\{)", masked, flags=re.S)[:1])
    modifier_text = re.sub(
        r"\b(?:external|public|internal|private|payable|view|pure|virtual|override|returns)\b",
        " ",
        modifier_text,
    )
    modifiers = _unique(re.findall(r"\b[A-Za-z_$][\w$]*\b", modifier_text), 8)

    flow_patterns = [
        ("BRANCH:if", r"\bif\s*\("),
        ("LOOP:for", r"\bfor\s*\("),
        ("LOOP:while", r"\bwhile\s*\("),
        ("GUARD:require", r"\brequire\s*\("),
        ("GUARD:assert", r"\bassert\s*\("),
        ("REVERT:revert", r"\brevert\s*(?:\(|[A-Za-z_$])"),
        ("EXTERNAL_CALL:delegatecall", r"\.\s*delegatecall\s*(?:\{|\()"),
        ("EXTERNAL_CALL:staticcall", r"\.\s*staticcall\s*(?:\{|\()"),
        ("EXTERNAL_CALL:call", r"\.\s*call\s*(?:\{|\()"),
        ("VALUE_TRANSFER:transfer", r"\.\s*transfer\s*\("),
        ("VALUE_TRANSFER:send", r"\.\s*send\s*\("),
        ("SELFDESTRUCT:selfdestruct", r"\bselfdestruct\s*\("),
        ("EMIT:emit", r"\bemit\s+[A-Za-z_$][\w$]*"),
        ("RETURN", r"\breturn\b"),
    ]
    positioned = []
    for event, pattern in flow_patterns:
        for match in re.finditer(pattern, masked):
            positioned.append((match.start(), event))

    assign_re = re.compile(
        r"(?<![=!<>])\b(?P<lhs>[A-Za-z_$][\w$]*(?:\s*\[[^\]]+\]|\s*\.\s*[A-Za-z_$][\w$]*)*)\s*(?:[+\-*/%]?=|\+\+|--)(?!=)"
    )
    local_names = set(params)
    local_names.update(
        re.findall(
            r"\b(?:address|uint\d*|int\d*|bool|string|bytes\d*|bytes|var)\s+(?:memory|storage|calldata\s+)?([A-Za-z_$][\w$]*)",
            masked,
        )
    )
    for match in assign_re.finditer(masked):
        lhs = _short(match.group("lhs"))
        base = re.match(r"[A-Za-z_$][\w$]*", lhs)
        write_kind = "LOCAL_WRITE" if base and base.group(0) in local_names else "STATE_WRITE"
        positioned.append((match.start(), f"{write_kind}:{lhs}"))

    positioned.sort(key=lambda item: item[0])
    events = [event for _, event in positioned[:AST_MAX_FLOW_EVENTS]]
    calls = []
    call_re = re.compile(r"\b([A-Za-z_$][\w$]*(?:\s*\.\s*[A-Za-z_$][\w$]*)*)\s*(?:\{[^{}]*\})?\s*\(")
    for match in call_re.finditer(masked):
        call = re.sub(r"\s+", "", match.group(1))
        prefix = masked[max(0, match.start() - 18) : match.start()]
        if call in {"if", "for", "while", "return"}:
            continue
        if re.search(r"\b(?:function|modifier|event|error)\s*$", prefix):
            continue
        calls.append(call)
    for call in calls:
        kind = _call_kind(call)
        if kind != "CALL":
            continue
        if len(events) < AST_MAX_FLOW_EVENTS:
            events.append(f"CALL:{call}")

    pseudo_shape = ["0:SourceUnit", "1:ContractDefinition", "2:FunctionDefinition", "3:Block"]
    counts = collections.Counter({"SourceUnit": 1, "ContractDefinition": 1, "FunctionDefinition": 1, "Block": 1})
    for event in events:
        kind = event.split(":", 1)[0]
        node = {
            "BRANCH": "IfStatement",
            "LOOP": "LoopStatement",
            "GUARD": "FunctionCall",
            "EXTERNAL_CALL": "FunctionCall",
            "VALUE_TRANSFER": "FunctionCall",
            "SELFDESTRUCT": "FunctionCall",
            "STATE_WRITE": "Assignment",
            "LOCAL_WRITE": "Assignment",
            "EMIT": "EmitStatement",
            "RETURN": "ReturnStatement",
            "REVERT": "RevertStatement",
            "CALL": "FunctionCall",
        }.get(kind, kind)
        counts[node] += 1
        if len(pseudo_shape) < AST_MAX_NODES:
            pseudo_shape.append(f"4:{node}")

    meta = {
        "name": name,
        "visibility": visibility,
        "mutability": mutability,
        "params": params,
        "returns": [],
        "modifiers": modifiers,
    }
    return _feature_lines("fallback_lexical_ast", code, meta, pseudo_shape, counts, events, reason=reason)


def _parse_solidity(code):
    wrapped = "pragma solidity ^0.8.0;\ncontract __AstWrap {\n" + code.strip() + "\n}"
    try:
        return SOLIDITY_PARSER.parse(wrapped, loc=False)
    except TypeError:
        return SOLIDITY_PARSER.parse(wrapped)


def ast_flow_features(code):
    if not USE_AST_AUGMENTATION:
        return ""
    key = code.strip()
    if key in AST_FEATURE_CACHE:
        return AST_FEATURE_CACHE[key]
    if SOLIDITY_PARSER is None:
        AST_STATS["fallback_import"] += 1
        features = _fallback_ast_summary(code, AST_IMPORT_ERROR or "parser_import_failed")
    else:
        try:
            ast = _parse_solidity(code)
            meta = _function_meta(ast)
            shape, counts = _ast_shape(ast)
            events = _ast_flow_events(ast)
            features = _feature_lines("solidity_parser_ast", code, meta, shape, counts, events)
            AST_STATS["solidity_parser"] += 1
        except Exception as exc:
            AST_STATS["fallback_parse"] += 1
            features = _fallback_ast_summary(code, repr(exc))
    AST_FEATURE_CACHE[key] = features
    return features


def build_model_input(row):
    code = str(row["code"]).strip()
    if not USE_AST_AUGMENTATION:
        return code
    if len(code) > CODE_MAX_CHARS:
        code = code[:CODE_MAX_CHARS] + "\n/* code truncated for token budget */"
    ast_features = ast_flow_features(code)
    return (
        "<SOLIDITY_CODE>\n"
        + code
        + "\n</SOLIDITY_CODE>\n<AST_FLOW>\n"
        + ast_features
        + "\n</AST_FLOW>"
    )


print("AST parser:", "solidity_parser" if SOLIDITY_PARSER is not None else f"fallback lexical ({AST_IMPORT_ERROR})")
if USE_AST_AUGMENTATION:
    for split_name, items in splits.items():
        for row in items:
            row["model_input"] = build_model_input(row)
    print("AST feature stats:", dict(AST_STATS))
    sample = next((row for row in splits["train"] if row.get("model_input")), None)
    if sample:
        print("Sample model input preview:\n" + sample["model_input"][:1200])
else:
    for items in splits.values():
        for row in items:
            row["model_input"] = row["code"]

# %% [markdown]
# ## 4. Tokenize

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


class VulnDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items
        self.enc = tokenizer(
            [r["model_input"] for r in items],
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

raw_ntr = sum(1 for r in splits["train"] if len(tokenizer(r["code"], add_special_tokens=True)["input_ids"]) > MAX_LEN)
aug_ntr = sum(1 for r in splits["train"] if len(tokenizer(r["model_input"], add_special_tokens=True)["input_ids"]) > MAX_LEN)
print(f"Raw code bi cat >{MAX_LEN} token: {raw_ntr}/{len(splits['train'])} ({100*raw_ntr/len(splits['train']):.1f}%)")
print(f"AST+code bi cat >{MAX_LEN} token: {aug_ntr}/{len(splits['train'])} ({100*aug_ntr/len(splits['train']):.1f}%)")

# %% [markdown]
# ## 5. Nap CodeLlama 4-bit + gan LoRA

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
# ## 6. Huan luyen

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

checkpoint_candidates = sorted(
    glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")),
    key=lambda path: int(re.search(r"checkpoint-(\d+)$", path).group(1)) if re.search(r"checkpoint-(\d+)$", path) else -1,
)
resume_checkpoint = checkpoint_candidates[-1] if checkpoint_candidates else None
if resume_checkpoint:
    print("Resume training from:", resume_checkpoint)

trainer.train(resume_from_checkpoint=resume_checkpoint)

print("\nEpoch | val_acc | val_prec | val_recall | val_f1 | val_f1_vuln")
for h in trainer.state.log_history:
    if "eval_f1_macro" in h:
        print(
            f"  {round(h['epoch']):3d} | {h['eval_accuracy']:.4f} | {h['eval_precision']:.4f} | "
            f"{h['eval_recall']:.4f} | {h['eval_f1_macro']:.4f} | {h['eval_f1_vuln']:.4f}"
        )

# %% [markdown]
# ## 7. Danh gia TEST

# %%
pred_out = trainer.predict(test_ds)
yp = np.argmax(pred_out.predictions, axis=-1)
yt = pred_out.label_ids

print("=" * 58 + f"\nTEST (n={len(yt)}) - CodeLlama QLoRA + AST-flow\n" + "=" * 58)
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
# ## 8. Luu adapter

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
        "train_mode": "CodeLlama sequence classification QLoRA 4-bit + AST/control-flow augmented input",
        "input_format": "AST_FLOW + SOLIDITY_CODE",
        "ast_augmentation": {
            "enabled": USE_AST_AUGMENTATION,
            "parser": "solidity_parser" if SOLIDITY_PARSER is not None else "fallback_lexical_ast",
            "max_nodes": AST_MAX_NODES,
            "max_flow_events": AST_MAX_FLOW_EVENTS,
            "max_feature_chars": AST_MAX_FEATURE_CHARS,
            "code_max_chars": CODE_MAX_CHARS,
        },
    },
    open(os.path.join(SAVE_DIR, "training_config.json"), "w", encoding="utf-8"),
    indent=2,
)
print("Da luu:", SAVE_DIR, os.listdir(SAVE_DIR))

# %% [markdown]
# ## 9. Thu inference

# %%
model.eval()


@torch.no_grad()
def predict(code, threshold=0.5):
    model_input = build_model_input({"code": code})
    enc = tokenizer(
        model_input,
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
