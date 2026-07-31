# DatasetBuild — dataset function-level SẠCH (v4)

Xây từ raw trong `../AllDataCrawl/` (DAppSCAN + iAudit/Solodit). Mục tiêu: nhãn cấp-**function** localize được, **không confound** (length/source/granularity), split **chống rò rỉ**.

## Kết quả cuối: `output/detect_v4_functionlevel.jsonl`
- **3,558 mẫu** function-level, cân bằng **1,779 Vulnerable / 1,779 Safe**.
- Cân bằng theo NGUỒN: `dappscan` 704/704 · `solodit` 1,075/1,075.
- Split chống rò rỉ: **train 2,848 / val 355 / test 355** (near-dup cluster giữ nguyên cụm; residual val/test→train = **0/0**).
- Schema mỗi dòng: `id, source, label, swc_id, swc_category, function, project, file, n_chars, split, code`.

### Kiểm định confound (đã PASS)
| Shortcut | AUC | |
|---|---|---|
| length-only | **0.491** | ~random (data cũ smartbug 0.66) |
| source-only | **0.473** | ~random |
| tfidf túi-từ | 0.826 | trần lexical — model v4 phải vượt |

Median n_chars Safe vs Vuln hội tụ: dappscan 642/643 · solodit 506/502. Rò rỉ marker/audit trong code = **0**.

## Pipeline (thư mục `scripts/`)
1. **`dappscan_extract.py`** — trích hàm vuln từ DAppSCAN, KIỂM CHỨNG lineNumber:
   - Bộ tìm hàm Solidity brace-matching (bỏ `{}` trong comment/string).
   - Chỉ lấy khi tên hàm khớp + số dòng nằm TRONG span hàm (`line_in_span`/`line_near`); N/A/rác/line-ảo → đánh dấu, không lấy bừa.
   - **Strip comment marker `// SWC-XXX: L..` DAppSCAN chèn sẵn** (chống rò rỉ nhãn) + dùng marker để xác nhận trích đúng hàm.
   - Ra `output/dappscan_vuln_functions.jsonl` (1,205 rows + cờ `match_status`). Sau lọc informational + tin cậy cao = **818 vuln thật**.
2. **`dappscan_safe.py`** — Safe = hàm KHÔNG bị gắn cờ trong CHÍNH file đã có vuln (negative khớp), dedup cấu trúc → `dappscan_safe_pool.jsonl` (4,402).
3. **`iaudit_parse.py`** — tách code (block ```Solidiy```) + label (completion) từ iAudit, dedup theo id → `iaudit_solodit.jsonl` (3,019: 1,575 safe/1,444 vuln, giữ split gốc + test 709).
4. **`merge_v4.py`** — dedup chéo nguồn → length-match + cân bằng safe/vuln theo nguồn → split chống rò rỉ (structural + tfidf cosine≥0.85) → báo cáo shortcut-AUC → `detect_v4_functionlevel.jsonl`.

Chạy lại: `python3 scripts/dappscan_extract.py --write && python3 scripts/dappscan_safe.py && python3 scripts/iaudit_parse.py && python3 scripts/merge_v4.py`

## Ghi chú chất lượng / hạn chế
- **DAppSCAN**: nhãn function+line từ audit thật; đã bỏ SWC informational (135/102/103/100/108/111/118/119/129/131). Chỉ giữ match tin cậy cao. `swc_id` có sẵn cho phần dappscan (704 mẫu) → phân tích per-SWC được.
- **Solodit/iAudit**: nhãn nhị phân (không có swc_id ở phần class); negative gồm cả bản GPT-4 augment của tác giả gốc.
- **Length-match** làm giảm số dappscan vuln 818→704 (bin dài thiếu safe khớp) — đánh đổi để giết confound độ dài.
- Safe DAppSCAN = "hàm audit không gắn cờ" → có thể còn lỗ hổng chưa bị bắt (nhiễu nhẹ phía safe, khó tránh).
- License DAppSCAN chưa rõ (xem `../AllDataCrawl/MANIFEST.json`).

## Bước tiếp có thể làm
- Mở rộng: nới cap/bổ sung safe dài để nâng dappscan vuln 704→818; thêm nguồn.
- Train detector v4 trên `detect_v4_functionlevel.jsonl` (function-level → không cần chunking/MIL nữa; ~85% ≤512 token) và so trần túi-từ 0.826.
