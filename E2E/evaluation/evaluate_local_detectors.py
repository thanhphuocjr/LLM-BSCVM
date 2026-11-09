from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from E2E.common.io_utils import utc_now, write_json
from E2E.config import DETECTOR_DATASET_PATH, PROJECT_ROOT, RESULTS_DIR
from E2E.phase1a_static.static_runner import run_static
from E2E.phase1b_rag.retriever import LocalRAGRetriever


# Tính precision, recall, F1, accuracy từ tp/fp/tn/fn.
def metrics(counts: dict[str, int]) -> dict:
    tp, fp, tn, fn = (counts[key] for key in ("tp", "fp", "tn", "fn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = tp + fp + tn + fn
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "accuracy": round((tp + tn) / total, 6) if total else 0.0,
    }


# Đọc dataset detector (mảng JSON v12 hoặc JSONL) và lọc theo split (train/val/test).
def load_rows(dataset: Path, split: str) -> list[dict]:
    """A JSON array (the v12 detector dataset) or JSONL, filtered to one split."""
    text = dataset.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [row for row in rows if row.get("split") == split]


# Đánh giá RIÊNG kênh static và kênh RAG như hai bộ phân loại nhị phân trên test split của dataset detector
# (mỗi hàm: dự đoán Vulnerable/Safe so với nhãn), tính chỉ số tổng và theo từng nguồn dữ liệu
# (dappscan, iaudit_trustllm, kaggle_ds2). Mục đích: số liệu cho luận văn về độ tin cậy của 2 kênh phụ trợ.
def evaluate(dataset: Path, split: str) -> dict:
    rows = load_rows(dataset, split)
    retriever = LocalRAGRetriever()
    overall = {name: defaultdict(int) for name in ("static", "rag")}
    by_source = {
        name: defaultdict(lambda: defaultdict(int))
        for name in ("static", "rag")
    }
    started = time.perf_counter()
    for row in rows:
        # v12 stores label "0"/"1" next to label_text; the JSONL sets use label="Vulnerable".
        expected = (row.get("label_text") or row.get("label")) == "Vulnerable"
        predictions = {
            "static": run_static(row["code"])["verdict"] == "Vulnerable",
            "rag": retriever.analyze(row["code"])["final_verdict"] == "Vulnerable",
        }
        for name, predicted in predictions.items():
            key = ("t" if predicted == expected else "f") + ("p" if predicted else "n")
            overall[name][key] += 1
            by_source[name][row.get("source", "unknown")][key] += 1
    return {
        "created_at": utc_now(),
        "dataset": str(dataset.resolve().relative_to(PROJECT_ROOT)) if PROJECT_ROOT in dataset.resolve().parents else str(dataset),
        "split": split,
        "rows": len(rows),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "overall": {name: metrics(counts) for name, counts in overall.items()},
        "by_source": {
            name: {
                source: metrics(counts)
                for source, counts in source_groups.items()
            }
            for name, source_groups in by_source.items()
        },
        "interpretation": (
            "Static and local RAG are supporting evidence channels. They are not "
            "replacements for the fine-tuned Qwen v13 detector or manual review."
        ),
    }


# Tham số: --dataset (mặc định final-dataset-v12-train.json), --split (mặc định test), --output.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local static and RAG detectors.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DETECTOR_DATASET_PATH,
        help="JSON array or JSONL with code/label/split (default: the v13 train/val/test set)",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "local_detector_eval.json",
    )
    return parser.parse_args()


# Điểm vào: chạy evaluate và ghi data/results/local_detector_eval.json.
def main() -> None:
    args = parse_args()
    result = evaluate(args.dataset, args.split)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
