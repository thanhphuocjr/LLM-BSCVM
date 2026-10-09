#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  DAppSCAN Line-Level Vulnerability Extractor & Dataset v12 Builder           ║
║  Mục đích: Trích xuất chính xác câu bị lỗi (vulnerable lines) từ DAppSCAN    ║
║            RAW để làm giàu dataset v11 thành v12 phục vụ huấn luyện.         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import glob
import os
import re
from pathlib import Path
from collections import Counter

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SWC_ROOT = REPO_ROOT / "E2E" / "data" / "source" / "raw" / "DAppSCAN" / "DAppSCAN-source" / "SWCsource"
CONTRACTS_ROOT = REPO_ROOT / "E2E" / "data" / "source" / "raw" / "DAppSCAN"

V11_PATH = SCRIPT_DIR / "final-dataset-v11-train.json"
V12_PATH = SCRIPT_DIR / "final-dataset-v12-train.json"


def clean_norm(c: str) -> str:
    """Loại bỏ comment và whitespace để đối chiếu cấu trúc code."""
    c = re.sub(r"/\*[\s\S]*?\*/", "", c)
    c = re.sub(r"//.*", "", c)
    return re.sub(r"\s+", "", c)


def parse_line_numbers(ln_str: str, total_lines: int) -> list[int]:
    """Parse các định dạng lineNumber: L158, L584-L588, L720, L721..."""
    if not ln_str:
        return []
    nums = [int(n) for n in re.findall(r"\d+", str(ln_str))]
    if not nums:
        return []
    if "-" in str(ln_str) and len(nums) == 2:
        start, end = nums[0], nums[1]
        # Nếu range hợp lý (<= 40 dòng), lấy toàn bộ range
        if end >= start and (end - start) <= 40:
            return [x for x in range(start, end + 1) if 1 <= x <= total_lines]
        return [start] if 1 <= start <= total_lines else []
    return [x for x in nums if 1 <= x <= total_lines]


def build_dataset_v12():
    print("=" * 72)
    print("  BẮT ĐẦU TRÍCH XUẤT CÂU LỖI TỪ DAPPSCAN RAW & TẠO DATASET V12")
    print("=" * 72)

    # 1. Nạp dataset v11
    print(f"\n[1/5] Đang nạp dataset v11 từ: {V11_PATH}")
    with open(V11_PATH, "r", encoding="utf-8") as f:
        v11_data = json.load(f)
    print(f"      -> Tổng số mẫu nạp: {len(v11_data)}")

    # 2. Nạp toàn bộ metadata SWCsource
    print(f"\n[2/5] Đang quét và lập chỉ mục nhãn SWC từ: {SWC_ROOT}")
    swc_files = glob.glob(f"{SWC_ROOT}/**/*.json", recursive=True)
    contract_swcs = {}
    for sf in swc_files:
        try:
            with open(sf, "r", encoding="utf-8", errors="replace") as f:
                meta = json.load(f)
                rel = meta.get("filePath")
                if rel:
                    contract_swcs[rel] = meta.get("SWCs", [])
        except Exception as e:
            print(f"      Cảnh báo lỗi đọc file {sf}: {e}")
    print(f"      -> Đã nạp nhãn SWC cho {len(contract_swcs)} hợp đồng.")

    # 3. Nạp và lập chỉ mục nội dung các file .sol
    print(f"\n[3/5] Đang nạp và tiền xử lý mã nguồn hợp đồng .sol...")
    sol_lines = {}
    sol_norms = {}
    for rel in contract_swcs:
        full_p = CONTRACTS_ROOT / rel
        if full_p.exists():
            try:
                with open(full_p, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                    sol_lines[rel] = lines
                    sol_norms[rel] = clean_norm("".join(lines))
            except Exception as e:
                print(f"      Cảnh báo lỗi đọc {full_p}: {e}")
    print(f"      -> Đã nạp thành công {len(sol_lines)} file .sol vào bộ nhớ.")

    # 4. Trích xuất câu bị lỗi cho từng mẫu
    print(f"\n[4/5] Đang thực hiện ánh xạ và trích xuất câu lỗi cho các mẫu DAppSCAN...")
    v12_data = []
    stats = Counter()

    for idx, row in enumerate(v11_data):
        src = row.get("source")
        lbl = row.get("label")
        code = row.get("code", "")

        # Bản sao giữ nguyên mọi trường ban đầu
        new_row = dict(row)

        # Mặc định cho các mẫu Safe hoặc nguồn không phải DAppSCAN
        new_row["swc_id"] = None
        new_row["swc_category"] = None
        new_row["raw_file_path"] = None
        new_row["vulnerable_lines"] = []
        new_row["vulnerable_line_numbers_in_file"] = []
        new_row["vulnerable_line_numbers_in_func"] = []

        if src != "dappscan":
            stats[f"non_dappscan_{src}"] += 1
            v12_data.append(new_row)
            continue

        if lbl == 0:
            stats["dappscan_safe"] += 1
            v12_data.append(new_row)
            continue

        # Xử lý DAppSCAN Vulnerable (lbl == 1)
        stats["dappscan_vuln_total"] += 1
        d_norm = clean_norm(code)
        fn = row.get("function_name")

        # Chuẩn hóa tên hàm nếu đang là "anonymous"
        if fn == "anonymous" or not fn:
            m_fn = re.search(r"\bfunction\s+([A-Za-z0-9_]+)", code)
            if m_fn:
                fn = m_fn.group(1)

        # Tìm hợp đồng tương ứng
        matched_contract = None
        # Ưu tiên 1: Hợp đồng có khai báo hàm này trong SWC
        for rel, swc_list in contract_swcs.items():
            if any(s.get("function") == fn for s in swc_list):
                if rel in sol_norms and d_norm in sol_norms[rel]:
                    matched_contract = rel
                    break

        # Ưu tiên 2: Bất kỳ hợp đồng nào chứa trọn vẹn code hàm
        if not matched_contract:
            for rel, c_norm in sol_norms.items():
                if d_norm in c_norm:
                    matched_contract = rel
                    break

        if not matched_contract:
            stats["dappscan_contract_not_found"] += 1
            v12_data.append(new_row)
            continue

        stats["dappscan_contract_matched"] += 1
        new_row["raw_file_path"] = matched_contract

        lines = sol_lines[matched_contract]
        swcs = contract_swcs[matched_contract]

        # Chọn SWC finding khớp nhất
        cand_swc = [s for s in swcs if s.get("function") == fn]
        if not cand_swc:
            cand_swc = swcs
        swc_entry = cand_swc[0]

        cat_str = swc_entry.get("category", "")
        swc_m = re.match(r"(SWC-\d+)", cat_str)
        swc_id = swc_m.group(1) if swc_m else None

        new_row["swc_id"] = swc_id
        new_row["swc_category"] = cat_str

        # Parse line numbers từ SWC
        raw_ln_str = swc_entry.get("lineNumber")
        file_line_nums = parse_line_numbers(raw_ln_str, len(lines))
        new_row["vulnerable_line_numbers_in_file"] = file_line_nums

        # Các dòng code trong hàm (để đối chiếu xác thực)
        func_code_lines = code.splitlines()
        func_stripped_lines = [l.strip() for l in func_code_lines]

        extracted_statements = []
        func_line_indices = []

        for ln in file_line_nums:
            if not (1 <= ln <= len(lines)):
                continue
            raw_statement = lines[ln - 1].strip()

            # Nếu dòng chỉ định là comment audit SWC, lấy dòng lệnh code tiếp theo
            if (raw_statement.startswith("//") or raw_statement.startswith("/*")) and "SWC-" in raw_statement:
                if ln < len(lines):
                    raw_statement = lines[ln].strip()

            # Bỏ qua dòng rỗng hoặc comment thuần túy
            if not raw_statement or raw_statement.startswith("//") or raw_statement.startswith("/*"):
                continue

            # Đối chiếu xem câu lệnh này có nằm trong hàm không
            for f_idx, f_line in enumerate(func_stripped_lines):
                # Khớp nếu câu lệnh trùng hoặc là thành phần chính
                if raw_statement == f_line or (len(raw_statement) > 15 and raw_statement in f_line) or (len(f_line) > 15 and f_line in raw_statement):
                    statement_text = f_line
                    if statement_text not in extracted_statements:
                        extracted_statements.append(statement_text)
                        func_line_indices.append(f_idx + 1)  # 1-indexed
                    break

        new_row["vulnerable_lines"] = extracted_statements
        new_row["vulnerable_line_numbers_in_func"] = func_line_indices

        if extracted_statements:
            stats["dappscan_lines_extracted_success"] += 1
        else:
            stats["dappscan_lines_unmatched_in_func"] += 1

        v12_data.append(new_row)

    # 5. Kiểm tra tính toàn vẹn (Integrity Check) & Ghi file
    print(f"\n[5/5] Kiểm định tính toàn vẹn dữ liệu (Integrity Validation)...")
    assert len(v12_data) == len(v11_data), f"Mismatch count: {len(v12_data)} vs {len(v11_data)}"

    # Kiểm tra split không đổi
    v11_splits = Counter(x["split"] for x in v11_data)
    v12_splits = Counter(x["split"] for x in v12_data)
    assert v11_splits == v12_splits, f"Split distribution changed: {v12_splits} vs {v11_splits}"

    # Kiểm tra nhãn không đổi
    v11_labels = Counter(x["label"] for x in v11_data)
    v12_labels = Counter(x["label"] for x in v12_data)
    assert v11_labels == v12_labels, f"Label distribution changed: {v12_labels} vs {v11_labels}"

    # Kiểm tra code không bị sửa đổi
    for r11, r12 in zip(v11_data, v12_data):
        assert r11["code"] == r12["code"], f"Code altered for id {r11['id']}"
        assert r11["label"] == r12["label"], f"Label altered for id {r11['id']}"

    # Ghi file mới final-dataset-v12-train.json
    print(f"      Đang ghi file dataset mới: {V12_PATH}")
    with open(V12_PATH, "w", encoding="utf-8") as f:
        json.dump(v12_data, f, ensure_ascii=False, indent=2)

    file_size_mb = os.path.getsize(V12_PATH) / (1024 * 1024)
    print(f"      ✅ Đã ghi thành công! Dung lượng file: {file_size_mb:.2f} MB")

    print("\n" + "=" * 72)
    print("  BÁO CÁO THỐNG KÊ KẾT QUẢ TRÍCH XUẤT DATASET V12")
    print("=" * 72)
    print(f"  • Tổng số mẫu toàn bộ dataset        : {len(v12_data):,}")
    print(f"    - Nguồn kaggle_ds2 (Synthetic)     : {stats['non_dappscan_kaggle_ds2']:,}")
    print(f"    - Nguồn iaudit_trustllm (Solodit)  : {stats['non_dappscan_iaudit_trustllm']:,}")
    print(f"    - Nguồn dappscan (Safe)            : {stats['dappscan_safe']:,}")
    print(f"    - Nguồn dappscan (Vulnerable)      : {stats['dappscan_vuln_total']:,}")
    print(f"  • DAppSCAN Vulnerable Mapping:")
    print(f"    - Khớp hợp đồng .sol gốc           : {stats['dappscan_contract_matched']:,} / {stats['dappscan_vuln_total']:,} ({stats['dappscan_contract_matched']/stats['dappscan_vuln_total']*100:.2f}%)")
    print(f"    - Trích xuất được CÂU LỖI cụ thể   : {stats['dappscan_lines_extracted_success']:,} / {stats['dappscan_vuln_total']:,} ({stats['dappscan_lines_extracted_success']/stats['dappscan_vuln_total']*100:.2f}%)")
    print(f"    - Mẫu lệch dòng báo cáo / span lệch: {stats['dappscan_lines_unmatched_in_func']:,}")
    print(f"  • Phân bổ Split:")
    for sp, cnt in sorted(v12_splits.items()):
        print(f"    - {sp:5s}: {cnt:,} mẫu")
    print("=" * 72)

    # In 3 mẫu ví dụ minh họa
    print("\n>>> VÍ DỤ 3 MẪU TRÍCH XUẤT THÀNH CÔNG:")
    examples = [x for x in v12_data if x.get("vulnerable_lines")][:3]
    for i, ex in enumerate(examples, 1):
        print(f"\n[{i}] ID: {ex['id']} | Original: {ex['original_id']} | Function: {ex['function_name']}")
        print(f"    SWC Category : {ex['swc_category']}")
        print(f"    Contract File: {ex['raw_file_path']}")
        print(f"    File Lines   : {ex['vulnerable_line_numbers_in_file']}")
        print(f"    Func Lines   : {ex['vulnerable_line_numbers_in_func']}")
        print(f"    Bug Statements:")
        for line in ex["vulnerable_lines"]:
            print(f"      -> {repr(line)}")


if __name__ == "__main__":
    build_dataset_v12()
