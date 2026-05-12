"""
merge_swc.py
------------
Merge extra_swc_samples.json vào dappscan.json.

Mapping cấu trúc:
  extra_swc_samples  →  dappscan
  ─────────────────────────────────────────────────
  Code               →  Code
  Label              →  Output
  SWC  (lookup)      →  Metadata.SWC
  Title (from extra) →  Metadata.Title
  description  (registry) → Metadata.Description
  remediation  (registry) → Metadata.Remediation

Các trường Description / Remediation còn thiếu
được bổ sung từ swc_registry.json theo khóa SWC id.
"""

import json
import os

# ── Đường dẫn file (chỉnh nếu cần) ──────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DAPPSCAN_PATH   = os.path.join(BASE_DIR, "dappscan.json")
EXTRA_PATH      = os.path.join(BASE_DIR, "extra_swc_samples.json")
REGISTRY_PATH   = os.path.join(BASE_DIR, "swc_registry.json")
OUTPUT_PATH     = os.path.join(BASE_DIR, "merged_dataset.json")


def load_json(path: str) -> list | dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_registry_lookup(registry: list) -> dict:
    """Tạo dict  {SWC-xxx: {title, description, remediation}}."""
    return {
        item["id"]: {
            "title":       item.get("title", ""),
            "description": item.get("description", ""),
            "remediation": item.get("remediation", ""),
        }
        for item in registry
    }


def convert_extra_item(item: dict, registry_lookup: dict) -> dict:
    """
    Chuyển 1 record từ định dạng extra_swc_samples
    sang định dạng dappscan.
    """
    swc_id = item.get("SWC", "")
    reg    = registry_lookup.get(swc_id, {})

    # Title: ưu tiên lấy từ registry (chuẩn hơn), fallback về extra
    title       = reg.get("title") or item.get("Title", "")
    description = reg.get("description", "")
    remediation = reg.get("remediation", "")

    # Label → Output
    label = item.get("Label", "Vulnerable")

    return {
        "Code":   item.get("Code", ""),
        "Output": label,
        "Metadata": {
            "SWC":         swc_id,
            "Title":       title,
            "Description": description,
            "Remediation": remediation,
        },
    }


def main():
    print("── Đọc file dappscan.json …")
    dappscan = load_json(DAPPSCAN_PATH)
    print(f"   {len(dappscan):,} records gốc")

    print("── Đọc file extra_swc_samples.json …")
    extras = load_json(EXTRA_PATH)
    print(f"   {len(extras):,} records cần merge")

    print("── Đọc file swc_registry.json …")
    registry = load_json(REGISTRY_PATH)
    lookup   = build_registry_lookup(registry)
    print(f"   {len(lookup)} SWC entries trong registry")

    # ── Chuyển đổi và gộp ───────────────────────────────────────────────────
    converted = []
    skipped   = 0
    for item in extras:
        swc_id = item.get("SWC", "")
        if swc_id not in lookup:
            print(f"   [WARN] Không tìm thấy {swc_id!r} trong registry → giữ trống Description/Remediation")
            skipped += 1
        converted.append(convert_extra_item(item, lookup))

    merged = dappscan + converted

    # ── Lưu kết quả ─────────────────────────────────────────────────────────
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    # ── Thống kê ─────────────────────────────────────────────────────────────
    print("\n── Hoàn thành! ─────────────────────────────────────────────────")
    print(f"   Records gốc (dappscan)  : {len(dappscan):,}")
    print(f"   Records thêm (extra)    : {len(converted):,}")
    print(f"   Tổng sau merge          : {len(merged):,}")
    if skipped:
        print(f"   [WARN] {skipped} record không khớp SWC id trong registry")
    print(f"   Output → {OUTPUT_PATH}")

    # ── Kiểm tra nhanh ───────────────────────────────────────────────────────
    print("\n── Mẫu record đầu tiên từ extra sau khi convert:")
    print(json.dumps(converted[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()