# data/ — mọi dataset và kết quả chạy (chỉ ở máy local)

Thư mục này không được đẩy lên GitHub (trừ file này và `data_manifest.json`).
Mọi đường dẫn được khai báo một lần trong `E2E/config.py`.
Kiểm tra toàn vẹn: `.venv/bin/python -m E2E.data_tools.build_data_manifest` (SHA-256 của dataset + adapter).

```text
data/
├── knowledge_base/                 CƠ SỞ TRI THỨC LỖ HỔNG (Phase 1B RAG, Phase 2 Advisor)
│   ├── source/                     đầu vào của bước xây dựng
│   │   ├── final-vulnerability-knowledge.json   hàm lỗi + SWC + nguyên nhân + bản vá (DAppSCAN, iAudit, kaggle_ds2)
│   │   └── baseline_taxonomy.json               mô tả / cách khắc phục chuẩn của 37 SWC (SWC Registry)
│   └── built/                      đầu ra của E2E.phase1b_rag.build_knowledge_base, pipeline đọc khi chạy
│       ├── knowledge_items.json    11,869 ca lỗi có thể truy hồi
│       ├── swc_registry.json       37 SWC kèm bằng chứng từ dataset
│       ├── swc_vectorizer.joblib   TF-IDF (char 3-5 gram)
│       ├── swc_matrix.npz          ma trận TF-IDF của các ca
│       └── knowledge_manifest.json số lượng + nguồn
│
├── detector_dataset/               DATASET HUẤN LUYỆN DETECTOR (Phase 1C)
│   └── final-dataset-v12-train.json   10,356 hàm, split train/val/test (dappscan, iaudit_trustllm, kaggle_ds2)
│                                      `scripts/gpu_server.sh deploy` chép lên GPU server; evaluate_local_detectors dùng test split
│
├── smartbugs_wild/                 47,398 contract SmartBugs-Wild đã bỏ comment (+ LICENSE, README gốc)
│                                   đầu vào của VÒNG 1 (scripts/run_batch.sh) và evaluation/benchmark.py
├── solodit_eval/                   iaudit_solodit.jsonl — tập đánh giá benchmark 9 công cụ
│
└── results/                        KẾT QUẢ CHẠY
    ├── detector_v13_training/      log lần train v13 cuối (chỉ số test của detector)
    ├── round1/                     VÒNG 1: batch SmartBugs-Wild, single-pass (2,374 contract đã chạy)
    │   ├── state.sqlite            trạng thái batch (resume được) — nguồn của mọi file CSV bên dưới
    │   ├── runs/<run_id>/          artifact từng phase của mỗi contract (vòng 2 dùng lại)
    │   ├── round1_functions.csv    một dòng / hàm (export từ state.sqlite)
    │   ├── round1_contracts.csv    một dòng / contract
    │   ├── round1_applied_2628_functions[_compact].csv   2,628 hàm đã được vá — gửi giám khảo chấm
    │   └── progress.log, runner.out
    ├── round1_judge/               GIÁM KHẢO NGOÀI CHẤM VÒNG 1
    │   ├── danh_gia_ty_le_ham_co_loi_that.xlsx   đánh giá tỷ lệ hàm có lỗi thật
    │   └── ham_can_sua_lan_2.csv / .xlsx         632 hàm cần sửa lần 2 = đầu vào của vòng 2
    ├── round2/                     VÒNG 2, nhóm chính 614 hàm (không self-check)
    │   ├── state.sqlite, rescan/   trạng thái + quét lại static/RAG bản vá vòng 1
    │   └── ham_sua_lan_2_ket_qua.csv (+ _blind: bỏ cột nhận xét của giám khảo vòng 1)
    ├── round2_regress/             VÒNG 2, nhóm 18 hàm mà bản vá vòng 1 sinh lỗi mới (có self-check)
    ├── solodit_9tools/             kết quả benchmark 9 công cụ (details / summary CSV, report.md)
    ├── local_detector_eval.json    (sinh khi chạy evaluate_local_detectors) static & RAG trên test split
    └── audit_runs/<run_id>/        mỗi lần audit từ web app / CLI (web app đọc lịch sử từ đây)
```

Ghi chú:

- `ham_sua_lan_2_ket_qua.csv` của nhóm chính được export trước khi có các cột `selfcheck_*`;
  export lại (`bash scripts/run_round2.sh export`) cho cùng dữ liệu, thêm 5 cột self-check để trống.
- Dữ liệu đã xoá/thay thế trong các lần dọn dẹp nằm trong Thùng rác (`~/.Trash/RAG-cleanup-*`) cho tới khi dọn Thùng rác.
