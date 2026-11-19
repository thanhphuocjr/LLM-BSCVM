"""
Đánh giá bản vá (Fixer) trên tập test của cơ sở tri thức - nằm NGOÀI pipeline chính (E2E/).

  split_test_set.py   BƯỚC 1: tách tập test ra file riêng + xây KB không chứa test (built_no_test/)
  run_fix_eval.py     BƯỚC 2: chạy từng mẫu test qua pipeline Phase 0 -> 6 (KB không chứa test, tắt kiểm chứng
                      Phase 5) rồi gom kết quả vào một file CSV

Mọi đường dẫn của phần đánh giá khai báo ở đây. File này KHÔNG import E2E: run_fix_eval phải đặt biến môi
trường RAG_KNOWLEDGE_BASE_DIR trước khi E2E.config được nạp lần đầu.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Đầu vào
KNOWLEDGE_SOURCE_PATH = DATA_DIR / "knowledge_base" / "source" / "final-vulnerability-knowledge.json"
DETECTOR_DATASET_PATH = DATA_DIR / "detector_dataset" / "final-dataset-v12-train.json"

# Bước 1: tập test + KB không chứa test
FIX_EVAL_DATA_DIR = DATA_DIR / "fix_eval"
GROUND_TRUTH_PATH = FIX_EVAL_DATA_DIR / "test_set_ground_truth.json"   # mọi trường của mẫu test (để so sánh)
TEST_INPUT_PATH = FIX_EVAL_DATA_DIR / "test_input.json"                # chỉ {id, source_code} -> pipeline
SPLIT_MANIFEST_PATH = FIX_EVAL_DATA_DIR / "split_manifest.json"
NO_TEST_KNOWLEDGE_PATH = DATA_DIR / "knowledge_base" / "source" / "final-vulnerability-knowledge.no_test.json"
NO_TEST_KB_DIR = DATA_DIR / "knowledge_base" / "built_no_test"

# Bước 2: kết quả chạy pipeline
RESULTS_DIR = DATA_DIR / "results" / "fix_eval"
RUNS_DIR = RESULTS_DIR / "runs"                     # một thư mục artifact / mẫu (run_id = id của mẫu)
RESULTS_JSONL_PATH = RESULTS_DIR / "results.jsonl"  # một dòng / mẫu, ghi ngay khi mẫu chạy xong (resume)
RESULTS_CSV_PATH = RESULTS_DIR / "fix_eval_results.csv"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
PROGRESS_LOG_PATH = RESULTS_DIR / "progress.log"
STOP_FILE_PATH = RESULTS_DIR / "STOP"               # tạo file này để dừng êm sau mẫu đang chạy

TEST_SPLIT = "test"
REAL_WORLD_SOURCE = "iaudit_trustllm"               # mẫu thực tế chạy trước, kaggle_ds2 (tổng hợp) chạy sau
