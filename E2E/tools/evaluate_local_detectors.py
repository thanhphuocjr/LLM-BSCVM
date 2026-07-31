from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from E2E.config import DATA_DIR, E2E_ROOT
from E2E.local.static_runner import run_static
from E2E.rag.retriever import LocalRAGRetriever
from E2E.shared.io_utils import utc_now, write_json


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


def evaluate(dataset: Path, split: str) -> dict:
    rows = []
    with dataset.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") == split:
                rows.append(row)
    retriever = LocalRAGRetriever()
    overall = {name: defaultdict(int) for name in ("static", "rag")}
    by_source = {
        name: defaultdict(lambda: defaultdict(int))
        for name in ("static", "rag")
    }
    started = time.perf_counter()
    for row in rows:
        expected = row["label"] == "Vulnerable"
        predictions = {
            "static": run_static(row["code"])["verdict"] == "Vulnerable",
            "rag": retriever.analyze(row["code"])["final_verdict"] == "Vulnerable",
        }
        for name, predicted in predictions.items():
            key = ("t" if expected else "f") + ("p" if predicted else "n")
            overall[name][key] += 1
            by_source[name][row.get("source", "unknown")][key] += 1
    return {
        "created_at": utc_now(),
        "dataset": str(dataset.relative_to(E2E_ROOT)),
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
            "replacements for the fine-tuned CodeLlama classifier or manual review."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local static and RAG detectors.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATA_DIR / "training" / "detect_v4_functionlevel.jsonl",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "training" / "reports" / "local_detector_eval.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(args.dataset, args.split)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
