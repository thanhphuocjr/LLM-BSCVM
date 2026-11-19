"""
BƯỚC 1 - tách tập test của cơ sở tri thức ra file riêng và xây một bản KB KHÔNG chứa test.

    .venv/bin/python -m fix_eval.split_test_set

Tập test = các mẫu split == "test" của data/knowledge_base/source/final-vulnerability-knowledge.json.
Ghi ra:
  data/fix_eval/test_set_ground_truth.json  mọi trường của mẫu test (is_vulnerable, swc_id, root_cause,
                                            mitigation_guidance, fixed_code, ...) + detector_v12_split
  data/fix_eval/test_input.json             CHỈ {id, source_code} = đầu vào của pipeline (không nhãn, không
                                            bản vá, không gợi ý); mẫu thực tế (iAudit) xếp trước kaggle_ds2
  data/knowledge_base/source/final-vulnerability-knowledge.no_test.json   các mẫu còn lại (train + val)
  data/knowledge_base/built_no_test/        KB build từ file trên bằng đúng build_knowledge_base của Phase 1B
  data/fix_eval/split_manifest.json         số lượng, SHA-256, kết quả kiểm tra rò rỉ
KB chính (data/knowledge_base/built/) giữ nguyên.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from E2E.common.io_utils import read_json, utc_now, write_json
from E2E.common.regex_patterns import RE_WHITESPACE
from E2E.config import BASELINE_TAXONOMY_PATH
from E2E.phase1b_rag.build_knowledge_base import build_knowledge_base
from fix_eval import (
    DETECTOR_DATASET_PATH,
    GROUND_TRUTH_PATH,
    KNOWLEDGE_SOURCE_PATH,
    NO_TEST_KB_DIR,
    NO_TEST_KNOWLEDGE_PATH,
    PROJECT_ROOT,
    REAL_WORLD_SOURCE,
    SPLIT_MANIFEST_PATH,
    TEST_INPUT_PATH,
    TEST_SPLIT,
)

# Hậu tố ngẫu nhiên 6 chữ thường mà kaggle_ds2 gắn vào tên (vd "withdraw_tolabw", "balances_qgmxvc").
# Bỏ hậu tố đi thì các hàm sinh từ cùng một template trở thành giống hệt nhau. Chỉ dùng để thống kê.
RE_KAGGLE_RANDOM_SUFFIX = re.compile(r"_[a-z]{6}\b")


# Bỏ mọi khoảng trắng -> so khớp mã không phụ thuộc xuống dòng / thụt lề (mã iAudit trong KB bị gộp thành 1 dòng).
def squash(code: str | None) -> str:
    return RE_WHITESPACE.sub("", code or "")


# Khung template của một hàm kaggle_ds2: thay hậu tố ngẫu nhiên bằng "_X" rồi bỏ khoảng trắng.
def kaggle_template(code: str | None) -> str:
    return squash(RE_KAGGLE_RANDOM_SUFFIX.sub("_X", code or ""))


# SHA-256 của một file (ghi vào manifest để biết chính xác file nào đã được dùng).
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Đường dẫn tương đối so với thư mục gốc repo (ghi vào manifest).
def relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


# Nhiệm vụ: trả về hàm cho biết một mẫu test nằm ở split nào của dataset detector v12 ("train"/"val"/"test",
# None nếu không có trong v12). kaggle_ds2 khớp theo original_id + "_vuln" (mã v12 có thể kèm khai báo biến
# state nên so theo "chứa"), iAudit khớp theo mã (đã bỏ khoảng trắng).
# Mục đích: detector Qwen v13 học trên split train của v12 -> người đánh giá biết mẫu nào detector đã thấy.
def detector_split_lookup(detector_rows: list[dict[str, Any]]) -> Callable[[dict[str, Any]], str | None]:
    by_original_id = {row["original_id"]: row for row in detector_rows}
    splits_by_code: dict[str, set[str]] = {}
    for row in detector_rows:
        splits_by_code.setdefault(squash(row["code"]), set()).add(row["split"])

    # Split detector v12 của MỘT mẫu KB (nhiều split thì nối bằng "/").
    def lookup(item: dict[str, Any]) -> str | None:
        code = squash(item.get("vulnerable_code"))
        twin = by_original_id.get(f"{item.get('original_id')}_vuln")
        if twin is not None and code and code in squash(twin["code"]):
            return twin["split"]
        splits = splits_by_code.get(code)
        return "/".join(sorted(splits)) if splits else None

    return lookup


# Nhiệm vụ: kiểm tra KB vừa build không còn dấu vết của tập test: id, mã lỗi, mã đã sửa (ground truth) trong
# knowledge_items.json và trong các mẫu (samples) của swc_registry.json. Trả về số lần trùng của từng loại.
# Mục đích: chứng minh RAG (Phase 1B), Advisor và Fixer không thể truy hồi lời giải của chính mẫu test.
def leak_checks(kb_dir: Path, test_rows: list[dict[str, Any]]) -> dict[str, int]:
    items = read_json(kb_dir / "knowledge_items.json")
    registry = read_json(kb_dir / "swc_registry.json")
    test_ids = {item["id"] for item in test_rows}
    test_code = {squash(item.get("vulnerable_code")) for item in test_rows if item.get("vulnerable_code")}
    test_fixed = {squash(item.get("fixed_code")) for item in test_rows if item.get("fixed_code")}
    test_any = test_code | test_fixed
    samples = [sample for record in registry for sample in record.get("samples", []) if isinstance(sample, dict)]
    return {
        "test_ids_in_knowledge_items": sum(item.get("id") in test_ids for item in items),
        "test_code_in_knowledge_items": sum(
            squash(item.get(field)) in test_any
            for item in items
            for field in ("vulnerable_code", "fixed_code")
            if item.get(field)
        ),
        "test_code_in_registry_samples": sum(
            squash(sample.get(field)) in test_any
            for sample in samples
            for field in ("vulnerable", "fixed")
            if sample.get(field)
        ),
    }


# [HÀM CHÍNH BƯỚC 1] Tách tập test, ghi file ground truth + file đầu vào (id, source_code), ghi nguồn KB không
# chứa test (loại cả mẫu train/val có mã trùng mã test - trên dữ liệu hiện tại là 0), build KB vào built_no_test/,
# kiểm tra rò rỉ (lỗi -> dừng) và ghi split_manifest.json.
def main() -> None:
    knowledge = read_json(KNOWLEDGE_SOURCE_PATH)
    test_rows = [item for item in knowledge if item.get("split") == TEST_SPLIT]
    test_ids = {item["id"] for item in test_rows}
    test_any = {squash(item.get(field)) for item in test_rows for field in ("vulnerable_code", "fixed_code") if item.get(field)}
    duplicates = [
        item for item in knowledge
        if item["id"] not in test_ids
        and any(squash(item.get(field)) in test_any for field in ("vulnerable_code", "fixed_code") if item.get(field))
    ]
    removed_ids = test_ids | {item["id"] for item in duplicates}
    no_test_rows = [item for item in knowledge if item["id"] not in removed_ids]

    lookup = detector_split_lookup(read_json(DETECTOR_DATASET_PATH))
    ground_truth = [{**item, "detector_v12_split": lookup(item)} for item in test_rows]
    # Mẫu thực tế (iAudit) chạy trước, kaggle_ds2 (tổng hợp) sau; trong mỗi nhóm giữ thứ tự id (sort ổn định).
    run_order = sorted(test_rows, key=lambda item: item.get("source") != REAL_WORLD_SOURCE)
    test_input = [{"id": item["id"], "source_code": item["vulnerable_code"]} for item in run_order]

    write_json(GROUND_TRUTH_PATH, ground_truth)
    write_json(TEST_INPUT_PATH, test_input)
    write_json(NO_TEST_KNOWLEDGE_PATH, no_test_rows)
    kb_manifest = build_knowledge_base(NO_TEST_KNOWLEDGE_PATH, NO_TEST_KB_DIR, BASELINE_TAXONOMY_PATH)

    leaks = leak_checks(NO_TEST_KB_DIR, test_rows)
    if any(leaks.values()):
        raise SystemExit(f"Leak check failed - the no-test knowledge base still contains test data: {leaks}")

    kaggle_test = [item for item in test_rows if item.get("source") == "kaggle_ds2"]
    kb_templates = {
        kaggle_template(item.get("vulnerable_code"))
        for item in no_test_rows
        if item.get("source") == "kaggle_ds2" and item.get("is_vulnerable")
    }
    manifest = {
        "created_at": utc_now(),
        "knowledge_source": {
            "path": relative(KNOWLEDGE_SOURCE_PATH),
            "sha256": sha256_file(KNOWLEDGE_SOURCE_PATH),
            "rows": len(knowledge),
            "by_split": dict(Counter(item.get("split") for item in knowledge)),
        },
        "test_set": {
            "rows": len(test_rows),
            "by_source_label_fix": dict(Counter(
                f"{item.get('source')} | {'vulnerable' if item.get('is_vulnerable') else 'safe'} | "
                f"{'has fixed_code' if item.get('fixed_code') else 'no fixed_code'}"
                for item in test_rows
            )),
            "detector_v12_split_by_source": dict(Counter(
                f"{item.get('source')} | {item.get('detector_v12_split')}" for item in ground_truth
            )),
            "run_order": "iaudit_trustllm first, then kaggle_ds2 (id order inside each group)",
        },
        "no_test_knowledge": {
            "rows": len(no_test_rows),
            "removed_test_rows": len(test_rows),
            "removed_train_val_rows_with_test_code": len(duplicates),
            "by_split": dict(Counter(item.get("split") for item in no_test_rows)),
        },
        "built_no_test": {"dir": relative(NO_TEST_KB_DIR), **kb_manifest},
        "leak_checks": leaks,
        "kaggle_template_note": {
            "kaggle_test_rows": len(kaggle_test),
            "distinct_templates_in_test": len({kaggle_template(item.get("vulnerable_code")) for item in kaggle_test}),
            "test_rows_whose_template_is_still_in_kb": sum(
                kaggle_template(item.get("vulnerable_code")) in kb_templates for item in kaggle_test
            ),
            "meaning": "kaggle_ds2 is synthetic: the same few templates with random name suffixes, so the KB "
                       "keeps same-template cases (different names) for every kaggle test row.",
        },
        "files": {
            relative(path): sha256_file(path)
            for path in (GROUND_TRUTH_PATH, TEST_INPUT_PATH, NO_TEST_KNOWLEDGE_PATH)
        },
    }
    write_json(SPLIT_MANIFEST_PATH, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
