from __future__ import annotations

from pathlib import Path


E2E_ROOT = Path(__file__).resolve().parent
DATA_DIR = E2E_ROOT / "data"
SOURCE_DATA_DIR = DATA_DIR / "source"
RUNTIME_DATA_DIR = DATA_DIR / "runtime"
RUNS_DIR = E2E_ROOT / "runs"
MODEL_DIR = E2E_ROOT / "models"
DETECTOR_ADAPTER_DIR = MODEL_DIR / "codellama-vuln-detector"
SCHEMA_DIR = E2E_ROOT / "schemas"
KAGGLE_DIR = E2E_ROOT / "kaggle"

SCHEMA_VERSION = "1.0.0"
PIPELINE_VERSION = "1.0.0"

DEFAULT_FINAL_THRESHOLD = 0.50
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
