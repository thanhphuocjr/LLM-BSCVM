"""Nơi DUY NHẤT khai báo đường dẫn, ngưỡng, trọng số và số lần gọi LLM của pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
E2E_ROOT = Path(__file__).resolve().parent

# Dữ liệu (chỉ có trên máy local, xem data/README.md)
DATA_DIR = PROJECT_ROOT / "data"
KNOWLEDGE_SOURCE_DIR = DATA_DIR / "knowledge_base" / "source"
VULNERABILITY_KNOWLEDGE_PATH = KNOWLEDGE_SOURCE_DIR / "final-vulnerability-knowledge.json"
BASELINE_TAXONOMY_PATH = KNOWLEDGE_SOURCE_DIR / "baseline_taxonomy.json"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base" / "built"
SMARTBUGS_WILD_DIR = DATA_DIR / "smartbugs_wild"
RESULTS_DIR = DATA_DIR / "results"
RUNS_DIR = RESULTS_DIR / "audit_runs"

# Model, contract mẫu, trọng số fusion
MODEL_DIR = PROJECT_ROOT / "models"
DETECTOR_ADAPTER_DIR = MODEL_DIR / "qwen25-v13-14b-detector-final"
DEMO_CONTRACTS_DIR = PROJECT_ROOT / "demo_contracts"
FUSION_CALIBRATION_PATH = E2E_ROOT / "phase1_fusion" / "fusion_calibration.json"

SCHEMA_VERSION = "1.0.0"
PIPELINE_VERSION = "1.0.0"

DEFAULT_FINAL_THRESHOLD = 0.50
DEFAULT_DETECTOR_THRESHOLD = 0.700
DEFAULT_COMPONENT_WEIGHTS = {"llm_detector": 0.65, "rag": 0.15, "static": 0.20}
# Số lần sửa tối đa của vòng Fixer -> Verifier.
MAX_FIX_ATTEMPTS = 2
# Số lời gọi LLM tối đa cho một contract (phần còn lại dùng mặc định từ KB / SWC).
MAX_LLM_ADVISOR_CALLS = 40
MAX_LLM_ASSESSOR_CALLS = 5

RISK_ORDER = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1, "Informational": 0, "None": 0, "Unknown": 0}
