from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
E2E_ROOT = Path(__file__).resolve().parent
# Every dataset lives under E2E/data (git-ignored, local only).
DATA_DIR = E2E_ROOT / "data"
RUNTIME_DATA_DIR = DATA_DIR / "runtime"          # RAG artifacts built by E2E.rag.build_store
KNOWLEDGE_DIR = DATA_DIR / "knowledge"           # vulnerability knowledge base (build_store input)
CONTRACTS_DIR = DATA_DIR / "contracts"           # contract corpora for batch audits / benchmarks
EVAL_DATA_DIR = DATA_DIR / "eval"                # evaluation sets and benchmark results
RUNS_DIR = E2E_ROOT / "runs"
BATCH_RUNS_DIR = E2E_ROOT / "batch_runs"
# Model weights live in <repo>/models (git-ignored); one folder per adapter.
MODEL_DIR = PROJECT_ROOT / "models"
DETECTOR_ADAPTER_DIR = MODEL_DIR / "qwen25-v13-14b-detector-final"
SCHEMA_DIR = E2E_ROOT / "schemas"
KAGGLE_DIR = E2E_ROOT / "kaggle"
VULNERABILITY_KNOWLEDGE_PATH = KNOWLEDGE_DIR / "final-vulnerability-knowledge.json"
SMARTBUGS_WILD_DIR = CONTRACTS_DIR / "smartbugs_wild"
# Train/val/test split of the active v13 detector; kept inside the GPU-server bundle.
DETECTOR_DATASET_PATH = PROJECT_ROOT / "server_train" / "final-dataset-v12-train.json"

SCHEMA_VERSION = "1.0.0"
PIPELINE_VERSION = "1.0.0"

DEFAULT_FINAL_THRESHOLD = 0.50
DEFAULT_DETECTOR_THRESHOLD = 0.700
DEFAULT_DETECTOR_BASE_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"
DEFAULT_DETECTOR_MAX_LEN = 1024
DEFAULT_COMPONENT_WEIGHTS = {
    "llm_detector": 0.65,
    "rag": 0.15,
    "static": 0.20,
}

RISK_ORDER = {
    "Critical": 4,
    "High": 3,
    "Medium": 2,
    "Low": 1,
    "Informational": 0,
    "None": 0,
    "Unknown": 0,
}
