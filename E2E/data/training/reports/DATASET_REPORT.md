# BÁO CÁO XÂY DỰNG DATASET PHÁT HIỆN LỖ HỔNG SMART CONTRACT (v4, function-level)

**Ngày:** 2026-07-15 · **Đầu ra chính:** `output/detect_v4_functionlevel.jsonl` (và bản `.csv`)

---

## PHẦN 1 — LẤY DỮ LIỆU TỪ ĐÂU

### 1.1. Xuất phát điểm: bài báo LLM-BSCVM
Toàn bộ nguồn được xác định từ bài **LLM-BSCVM** (arXiv:2505.17416). Mục *IV.A Experimental Setup* nói rõ dataset của họ ghép từ **2 nguồn**, cộng các knowledge base:

| Ký hiệu trong bài | Nguồn | Vai trò |
|---|---|---|
| [13] + [30] | **TrustLLM/iAudit** ← Solodit | Dataset lỗ hổng (263 audit report) |
| [31] | **DAppScan** | Dataset lỗ hổng (audit report DApp) |
| [34] | SWC Registry | Knowledge base (taxonomy) |
| [32][33] | Best practices / EthTrust | Knowledge base |

### 1.2. Provenance từng nguồn (đã tải về `../AllDataCrawl/`)

**A. DAppSCAN** — `AllDataCrawl/DAppSCAN/`
- Tải từ: `github.com/InPlusLab/DAppSCAN`, commit `66a56619` (2025-03-25), chỉ lấy `DAppSCAN-source` (bỏ bytecode).
- Bài báo gốc: Zheng et al., *"DAppSCAN: Building Large-Scale Datasets for Smart Contract Weaknesses in DApp Projects"*, IEEE TSE 2024 (arXiv:2305.08456).
- **Nguồn gốc thật của dữ liệu:** rút từ **608 báo cáo audit thật** của **28 hãng bảo mật** (QuillAudits 124, Hacken 73, PeckShield 58, ConsenSys 47, OpenZeppelin 45, Trail of Bits 30, Iosiro, Inspex, CoinFabrik…). Nhóm tác giả **thủ công** ánh xạ từng weakness (SWC) vào đúng hàm/dòng mã nguồn (44 person-months). → nhãn do **auditor người** gán, chất lượng cao.
- Thành phần: 21,452 file `.sol` + 948 JSON nhãn (1,646 finding, **chỉ weakness — không có nhãn "Safe"**) + 807 PDF report gốc.
- ⚠️ License: repo **không ghi rõ** → cần xác nhận trước khi publish/thương mại.

**B. TrustLLM/iAudit (Solodit)** — `AllDataCrawl/iAudit-TrustLLM/`
- Tải từ: `sites.google.com/view/trustllm` → Google Drive (folder `1cAHxSu6...`); code từ `anonymous.4open.science/r/iAudit-324F`.
- Bài báo gốc: Ma et al., ICSE 2025 (arXiv:2403.16073).
- **Nguồn gốc thật:** 263 báo cáo audit từ **Solodit** (solodit.xyz) → 1,734 hàm lỗi; negative do tác giả tăng cường bằng GPT-4.
- Thành phần: train/val (class + reason) + test 709; định dạng instruction-tuning (code trong `prompt`, label trong `completion`).

**C. Phụ trợ:** SWC-registry (37 định nghĩa, MIT); LLM-BSCVM code (chỉ framework, **không data**); best-practices (yos.io + ConsenSys).

---

## PHẦN 2 — XỬ LÝ NHƯ THẾ NÀO & TẠI SAO

> Nguyên tắc bao trùm: data cũ (v2/v3) hỏng vì **nhãn cấp-contract** (lỗ hổng chìm trong contract khổng lồ), **confound** (độ dài/nguồn/pragma đoán được nhãn), **rò rỉ** near-dup train↔test, và **nguồn dễ áp đảo** làm điểm ảo. Mọi bước dưới đây nhắm diệt 4 lỗi đó.

### 2.1. DAppSCAN → hàm VULNERABLE
| Việc | Tại sao |
|---|---|
| Cắt đúng **HÀM** (bộ brace-matching, bỏ `{}` trong comment/string) thay vì lấy cả contract | Nhãn cũ cấp-contract 12k ký tự → model không định vị được lỗi (AUC 0.54). Cắt hàm → localize |
| **Kiểm chứng `lineNumber` nằm trong span của hàm khớp tên** | Nhiều lineNumber "ảo" (trỏ chỗ *gọi* / rác). Không kiểm sẽ cắt nhầm hàm |
| Bỏ 429 finding `N/A` (cấp-file), 61 line-lệch, 12 tên-là-event | Không localize được → ép vào tạo nhãn sai |
| **Strip comment `// SWC-107-Reentrancy: L95`** DAppSCAN chèn sẵn ở dòng lỗi | 🔴 Nếu giữ, model chỉ dò comment = rò rỉ nhãn 100%, vô dụng |
| Bỏ SWC *informational* (135/102/103/100/108/111/119/129/131) | Không phải lỗ hổng thật (code-no-effect, floating pragma…) |

→ **818 hàm vuln thật, tin cậy cao** (từ 1,646 finding thô).

### 2.2. DAppSCAN → hàm SAFE (phải TỰ tạo)
DAppSCAN **không có nhãn Safe** (positive-only). Nên:
- Lấy **các hàm KHÔNG bị gắn cờ trong CHÍNH file đã có vuln** → negative khớp.
- **Tại sao cùng file:** safe & vuln giống hệt project/thư viện/pragma/style → khác biệt duy nhất là lỗ hổng; model không gian lận được bằng đặc trưng nguồn.
- Dedup cấu trúc (bỏ 3,254 boilerplate OZ), cap 8/file → **pool 4,402**.

### 2.3. Solodit/iAudit → parse
- Tách code từ block ` ```Solidiy…``` `, label từ `completion`; dedup theo `id` (5 biến thể prompt/mẫu) + cấu trúc → **3,019 mẫu** (đã cắt sẵn dạng hàm, tôi không tự cắt).

### 2.4. GỘP — 4 thao tác diệt confound
1. **Dedup cấu trúc chéo nguồn** (giữ Vuln khi trùng).
2. **Length-match + cân bằng safe/vuln THEO NGUỒN** (chia bin độ dài, mỗi bin hạ về `min(safe,vuln)`) → giết confound độ dài (length-AUC 0.66→0.49) & làm nguồn không đoán được nhãn. *Chi phí:* dappscan vuln 818→704.
3. **Chia train/val/test theo cụm near-dup** (structural + TF-IDF cosine≥0.85, cả cụm về 1 split) → chống rò rỉ.
4. **Báo cáo shortcut-AUC** → chứng minh confound đã chết, không nói suông.

---

## PHẦN 3 — KẾT QUẢ

### 3.1. Dataset cuối: `detect_v4_functionlevel.jsonl` / `.csv`
- **3,558 mẫu** function-level, cân bằng **1,779 Vulnerable / 1,779 Safe**.
- Cân bằng theo nguồn: **dappscan 704/704 · solodit 1,075/1,075**.
- Split: **train 2,848 (1429/1419) · val 355 (178/177) · test 355 (172/183)**.
- Độ dài: p50 549 ký tự, p90 1,465; **~93% ≤ 512 token** → bỏ được chunking/MIL.

### 3.2. Kiểm định confound — ĐỀU ĐẠT
| Shortcut | AUC | Ý nghĩa |
|---|---|---|
| length-only | **0.491** | ~random (cũ smartbug 0.66) ✅ |
| source-only | **0.473** | ~random ✅ |
| tfidf túi-từ | 0.826 | trần lexical — model v4 phải VƯỢT |
| Median len Safe/Vuln | dappscan 642/643 · solodit 506/502 | hội tụ ✅ |
| Rò rỉ near-dup val/test→train | **0 / 0** | sạch ✅ |
| Rò rỉ marker/audit trong code | **0** | sạch ✅ |

### 3.3. Phân bố loại lỗ hổng (phần dappscan có `swc_id`, 704 mẫu)
SWC-101 (overflow) 148 · SWC-107 (reentrancy) 121 · SWC-104 88 · SWC-128 77 · SWC-114 69 · SWC-116 52 · SWC-105 36 · SWC-113 25 · SWC-120 18 · SWC-134 11 · SWC-123 9 · SWC-126 7. *(Phần solodit nhãn nhị phân, không có swc_id.)*

---

## PHẦN 4 — HẠN CHẾ & BƯỚC TIẾP
- Length-match làm dappscan vuln 818→704 (đánh đổi để giết confound độ dài).
- Safe DAppSCAN = "hàm audit không gắn cờ" → nhiễu nhẹ (có thể lỗ hổng bị bỏ sót), khó tránh khi không có nhãn safe tường minh.
- Solodit không có `swc_id` (chỉ nhị phân).
- License DAppSCAN chưa rõ.
- **Bước tiếp:** train detector v4 trên tập này (encoder gọn, không MIL) & so trần túi-từ 0.826; hoặc mở rộng safe dài để kéo dappscan 704→818.

---

## PHỤ LỤC — FILE ĐẦU RA (`output/`)
| File | Mô tả |
|---|---|
| `detect_v4_functionlevel.jsonl` / `.csv` | ⭐ Dataset cuối (3,558) — cột: id, source, label, swc_id, swc_category, function, project, file, n_chars, split, code |
| `detect_v4_preview.csv` | Bản xem nhanh (code rút gọn 140 ký tự/dòng) |
| `dappscan_vuln_functions.jsonl` | 1,205 hàm vuln (kèm cờ `match_status`, `swc_marker_confirmed`) |
| `dappscan_safe_pool.jsonl` | 4,402 negative khớp |
| `iaudit_solodit.jsonl` | 3,019 mẫu Solodit |

Pipeline tái lập: `scripts/` (`dappscan_extract.py` → `dappscan_safe.py` → `iaudit_parse.py` → `merge_v4.py`). Provenance raw: `../AllDataCrawl/MANIFEST.json`.
