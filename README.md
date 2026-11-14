# LLM-BSCVM — Kiểm toán lỗ hổng Smart Contract end-to-end

Pipeline kiểm toán Solidity theo khung **LLM-BSCVM** (arXiv:2505.17416): phát hiện lỗ hổng
(static + RAG + detector fine-tune), tư vấn sửa, đánh giá rủi ro CVSS, sinh bản vá,
kiểm chứng bản vá và xuất báo cáo audit (Markdown/HTML/PDF).

- **Detector:** Qwen2.5-Coder-14B-Instruct + LoRA v13 (phân loại từng hàm, ngưỡng 0.70) trên GPU server.
- **Agents (Advisor → Assessor → Fixer):** cùng model Qwen-14B trên GPU server (tắt adapter).
- **Giao diện:** web app FastAPI (`webapp/`) và CLI `python -m E2E.run`.

---

## 1. Cấu trúc thư mục

```text
RAG/
├── README.md, requirements.txt, .env.example
├── scripts/               các lệnh chạy (gọi từ thư mục gốc hoặc bất kỳ đâu)
│   ├── run_webapp.sh      chạy web app (http://127.0.0.1:8080)
│   └── gpu_server.sh      điều khiển GPU server: start | stop | status | logs | deploy | pull
│
├── E2E/                   toàn bộ code chạy trên máy local (package Python)
│   ├── config.py          NƠI DUY NHẤT khai báo đường dẫn, ngưỡng, trọng số, số lần gọi LLM
│   ├── run.py             CLI: python -m E2E.run <file.sol>
│   ├── pipeline/          LUỒNG CHẠY: orchestrator.py (Phase 0→6), gpu_pipeline.py (Phase 1C→5)
│   ├── phase0_parsing/    tách contract thành hàm (code_units.py), ngữ cảnh cho prompt (contract_context.py)
│   ├── phase1a_static/    phân tích tĩnh: static_analyzer.py (bộ luật), static_runner.py (+ Slither)
│   ├── phase1b_rag/       XÂY cơ sở tri thức (build_knowledge_base.py), 37 SWC agent, retriever TF-IDF, rag_runner
│   ├── phase1c_detector/  detector.py: gọi Qwen v13 chấm từng hàm
│   ├── phase1_fusion/     fusion.py + fusion_calibration.json
│   ├── phase2_advisor/    advisor.py (LLM + tri thức sửa lỗi), advisor_phase.py (lọc false positive)
│   ├── phase3_assessor/   assessor.py (CVSS v3.1), assessor_phase.py
│   ├── phase4_fixer/      fixer.py (prompt + vá), patching.py, behavior_guard.py
│   ├── phase5_verifier/   candidate_check.py (trong vòng sửa), final_verification.py, tool_matrix.py, behavior_verifier.py
│   ├── phase6_reporter/   report.py (MD/HTML/PDF), vulnerability_reference.py
│   └── common/            gpu_client.py, security_tools.py (Slither), solc.py, io_utils.py, regex_patterns.py (mọi regex), ...
│
├── gpu_server/            code chạy TRÊN GPU server (scripts/gpu_server.sh deploy đẩy lên ~/server_train)
│   ├── train_qwen25_v13.py, run_train.sh, setup.sh    huấn luyện detector v13
│   └── serve_v13_api.py, service.sh, run/stop/status_api_server.sh   API /health + /detect + /generate
│
├── webapp/                app.py (FastAPI) + static/ (giao diện)
├── demo_contracts/        VulnerableBank, SafeBank, AccessControl (web app "Mẫu", CLI, test)
├── tests/                 89 unit/integration test (gồm hồi quy static + fusion trên 3 contract SmartBugs-Wild)
├── data/                  (local) MỌI dataset + kết quả chạy — xem data/README.md
├── models/                (local) qwen25-v13-14b-detector-final/ — adapter LoRA + threshold_config.json
└── docs/                  (local) bài báo LLM-BSCVM (PDF)
```

*(local)* = nằm trong `.gitignore`, chỉ có trên máy này. Riêng `data/README.md` và `data/data_manifest.json` được đẩy lên.

---

## 2. Pipeline và file tương ứng

```text
Phase 0   tách hàm (code unit), SHA-256 nguồn        phase0_parsing/code_units.py
Phase 1A  static: luật regex/AST + Slither           phase1a_static/static_runner.py   ┐
Phase 1B  RAG: TF-IDF trên 11,869 ca + 37 SWC agent  phase1b_rag/rag_runner.py         ├ fusion theo từng finding
Phase 1C  detector Qwen v13 (P(vulnerable)/hàm)      phase1c_detector/detector.py      ┘ phase1_fusion/fusion.py
Phase 2   Advisor: đúng/sai, nguyên nhân, hướng sửa  phase2_advisor/
Phase 3   Assessor: CVSS v3.1, thứ tự ưu tiên        phase3_assessor/
Phase 4   Fixer: vá từng hàm                          phase4_fixer/fixer.py
Phase 5   kiểm chứng: compile + detector + static/RAG (vòng sửa ≤ 3 lần, candidate_check.py)
          rồi compile + Slither + so sánh AST + static/RAG (final_verification.py)
Phase 6   báo cáo Markdown / HTML / PDF              phase6_reporter/report.py
```

Điều phối: `pipeline/orchestrator.py` chạy Phase 0, 1A, 1B trên máy local → ghi `request.json` →
`pipeline/gpu_pipeline.py` chạy Phase 1C → 5 qua GPU server → orchestrator hợp nhất, kiểm chứng cuối, xuất báo cáo.

Mỗi lần audit ghi vào `data/results/audit_runs/<run_id>/`: `phase0_input.json`, `phase1a_static.json`,
`phase1b_rag.json`, `request.json`, `server_result.json`, `phase1_fused.json`, `phase2_advisor.json`,
`phase3_assessor.json`, `phase4_fixer.json`, `phase5_verification.json`, `phase6_audit_report.{md,html,pdf}`, `manifest.json`.

Ngữ nghĩa kết quả:

- Verdict `Safe` chỉ khi mọi kênh bắt buộc đều chạy xong và detector chấm được mọi hàm; thiếu/lỗi → `Inconclusive` (fail-closed).
- `manifest.status=complete` chỉ có nghĩa mọi phase đã chạy; bản vá được chấp nhận hay không xem
  `phase5_verification.json → overall_verdict`.

---

## 3. Cài đặt

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env            # điền SERVER_GPU_URL
```

Công cụ ngoài: `solc-select` (+ các bản solc cần dùng) và `slither-analyzer`.

> Không bọc `.venv/bin/python` bằng `timeout`/`gtimeout` (binary x86_64 làm python chạy sai kiến trúc, lỗi `pydantic_core`).

---

## 4. Các bước xây dựng (offline)

| Bước | Lệnh | Đầu vào → đầu ra |
| --- | --- | --- |
| Xây cơ sở tri thức (Phase 1B/2) | `.venv/bin/python -m E2E.phase1b_rag.build_knowledge_base` | `data/knowledge_base/source/` → `built/` (tất định: cùng đầu vào cho artifact giống hệt từng byte) |
| Huấn luyện detector (Phase 1C) | trên server: `cd ~/server_train && bash setup.sh && bash run_train.sh` | `data/detector_dataset/` → adapter v13; `./scripts/gpu_server.sh pull` kéo về `models/` |

---

## 5. Chạy

**GPU server**

```bash
./scripts/gpu_server.sh start | stop | status | logs
./scripts/gpu_server.sh deploy                 # đẩy gpu_server/ + dataset v12 lên server, rồi stop + start
SERVER_IP=100.122.153.35 ./scripts/gpu_server.sh status
```

**Web app** — `./scripts/run_webapp.sh` → http://127.0.0.1:8080 (`HOST=0.0.0.0` để mở LAN, `RELOAD=1` khi dev)

**CLI một contract**

```bash
.venv/bin/python -m E2E.run demo_contracts/VulnerableBank.sol
```

Tham số: `--run-id`, `--detector-threshold`, `--max-fix-attempts 1..3`, `--solidity-root`.

---

## 6. Đánh giá và kiểm thử

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -t .   # 89 test
```

Kết quả các lần chạy thực nghiệm (vòng 1, vòng 2, benchmark 9 công cụ) đã lưu trong `data/results/` (xem `data/README.md`).

Kết quả detector v13 trên test split (`data/results/detector_v13_training/train_v13_20261005_143623.log`):

| Tập | n | Accuracy | F1 |
| --- | ---: | ---: | ---: |
| Toàn bộ test (macro-F1) | 1,049 | 0.895 | 0.893 |
| dappscan | 223 | 0.655 | 0.624 |
| iaudit_trustllm | 326 | 0.899 | 0.893 |
| kaggle_ds2 (tổng hợp) | 500 | 1.000 | 1.000 |
| **Chỉ contract thực tế** | **549** | **0.800** | **0.786** |

---

## 7. Hạn chế cần biết

- Con số tổng 0.893 bị kéo lên bởi `kaggle_ds2`: dữ liệu tổng hợp từ ít template, test và train
  chung template. Con số trung thực để báo cáo là **real-world F1 0.786**; DAppSCAN (DeFi thật) chỉ 0.62.
- Detector là phân loại nhị phân, không tự gán SWC; nhãn SWC đến từ static/RAG.
- Trọng số fusion (`phase1_fusion/fusion_calibration.json`) đặt tay, chưa fit trên tập hiệu chuẩn.
- Thứ tự các finding Slither trả về không cố định giữa các lần chạy (nội dung giống nhau).
- SWC Registry không còn được cập nhật và không phủ lỗi business-logic/oracle/cross-chain.
- Bản vá do model sinh luôn cần review thủ công và test suite của dự án trước khi deploy.

Nguồn dữ liệu (tải lại nếu cần): DAppSCAN — github.com/InPlusLab/DAppSCAN;
iAudit/TrustLLM (Solodit) — sites.google.com/view/trustllm; SmartBugs-Wild — github.com/smartbugs/smartbugs-wild;
SWC Registry — swcregistry.io.
