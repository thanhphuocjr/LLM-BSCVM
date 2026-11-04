"""Single place for every path, threshold and weight used by the pipeline."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
E2E_ROOT = Path(__file__).resolve().parent

# ── Data (git-ignored, local only; layout described in data/README.md) ────────
DATA_DIR = PROJECT_ROOT / "data"
# Vulnerability knowledge base: source files -> build_knowledge_base -> built artifacts.
KNOWLEDGE_SOURCE_DIR = DATA_DIR / "knowledge_base" / "source"
VULNERABILITY_KNOWLEDGE_PATH = KNOWLEDGE_SOURCE_DIR / "final-vulnerability-knowledge.json"
BASELINE_TAXONOMY_PATH = KNOWLEDGE_SOURCE_DIR / "baseline_taxonomy.json"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base" / "built"      # read by Phase 1B (RAG) and Phase 2 (Advisor)
# Train/val/test split of the v13 detector (copied to the GPU server by `scripts/gpu_server.sh deploy`).
DETECTOR_DATASET_PATH = DATA_DIR / "detector_dataset" / "final-dataset-v12-train.json"
SMARTBUGS_WILD_DIR = DATA_DIR / "smartbugs_wild"                # input corpus of the round-1 batch
SOLODIT_EVAL_DIR = DATA_DIR / "solodit_eval"                    # evaluation set of the 9-tool benchmark

# ── Results ──────────────────────────────────────────────────────────────────
RESULTS_DIR = DATA_DIR / "results"
RUNS_DIR = RESULTS_DIR / "audit_runs"                           # one folder per audit (web app / CLI)
BATCH_RUNS_DIR = RESULTS_DIR                                    # batches: round1, round2, round2_regress
ROUND1_JUDGE_DIR = RESULTS_DIR / "round1_judge"                 # judge scores of round 1 = input of round 2
SOLODIT_BENCHMARK_RESULTS_DIR = RESULTS_DIR / "solodit_9tools"

# ── Models and demo inputs ───────────────────────────────────────────────────
MODEL_DIR = PROJECT_ROOT / "models"
DETECTOR_ADAPTER_DIR = MODEL_DIR / "qwen25-v13-14b-detector-final"
DEMO_CONTRACTS_DIR = PROJECT_ROOT / "demo_contracts"
FUSION_CALIBRATION_PATH = E2E_ROOT / "phase1_fusion" / "fusion_calibration.json"

SCHEMA_VERSION = "1.0.0"
PIPELINE_VERSION = "1.0.0"

DEFAULT_FINAL_THRESHOLD = 0.50
DEFAULT_DETECTOR_THRESHOLD = 0.700
DEFAULT_COMPONENT_WEIGHTS = {
    "llm_detector": 0.65,
    "rag": 0.15,
    "static": 0.20,
}
MAX_FIX_ATTEMPTS = 3   # Fixer -> Verifier repair loop

RISK_ORDER = {
    "Critical": 4,
    "High": 3,
    "Medium": 2,
    "Low": 1,
    "Informational": 0,
    "None": 0,
    "Unknown": 0,
}
