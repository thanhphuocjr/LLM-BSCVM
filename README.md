# LLM-BSCVM — Kiểm toán lỗ hổng Smart Contract end-to-end

Pipeline kiểm toán Solidity theo khung **LLM-BSCVM** (arXiv:2505.17416): phát hiện lỗ hổng
(static + RAG + detector fine-tune), tư vấn sửa, đánh giá rủi ro CVSS, sinh bản vá,
kiểm chứng bản vá và xuất báo cáo audit (Markdown/HTML/PDF).

Stack đang chạy:

- **Detector:** Qwen2.5-Coder-14B-Instruct + LoRA (v13, sequence-classification, ngưỡng 0.70),
  phục vụ qua API trên GPU server (`server_train/serve_v13_api.py`).
- **Agents (Advisor → Assessor → Fixer → Verifier → Reporter):** cùng model trên GPU server
  (backend mặc định `server_gpu`); có thể thay bằng Gemini/Ollama/Kaggle.
- **Giao diện:** web app FastAPI (`webapp/`) và CLI `python -m E2E.run`.

---

## 1. Cấu trúc thư mục

```text
RAG/
├── README.md               tài liệu duy nhất của repo
├── requirements.txt        thư viện Python cho máy local
├── .env.example            mẫu cấu hình (copy thành .env)
├── run_webapp.sh           chạy web app (http://127.0.0.1:8080)
├── run_batch.sh            batch audit nền, resume được, cho cả corpus contract
├── gpu_server.sh           điều khiển GPU server: start | stop | status | logs | deploy | pull
│
├── E2E/                    package pipeline (toàn bộ code chạy local)
│   ├── config.py           NƠI DUY NHẤT khai báo đường dẫn data/model + ngưỡng + trọng số fusion
│   ├── run.py              entrypoint CLI  →  local/orchestrator.py
│   ├── kaggle_job.py       entrypoint submit thủ công cho backend kaggle
│   ├── local/              điều phối & runner từng backend
│   │   ├── orchestrator.py       luồng phase 0→6, ghi artifact từng phase
│   │   ├── server_gpu_runner.py  backend mặc định: detector + agents qua GPU server
│   │   ├── local_llm_runner.py   backend local/gemini/ollama (agents trong E2E/agents)
│   │   ├── fast_agent_runner.py  backend local_fast (agent tất định, không cần LLM)
│   │   ├── kaggle_job_client.py  backend kaggle (đẩy notebook E2E/kaggle lên Kaggle)
│   │   ├── static_runner.py      Phase 1A: static analyzer + Slither native
│   │   ├── rag_runner.py         Phase 1B: RAG TF-IDF
│   │   ├── tool_verifier.py      Phase 5: ma trận SmartBugs (slither/mythril/smartcheck…)
│   │   ├── behavior_verifier.py  Phase 5: compile + bảo toàn hành vi quan sát được
│   │   └── report.py             Phase 6: báo cáo Markdown/HTML/PDF
│   ├── agents/             agent LLM: advisor, assessor, reporter (bảng tham chiếu SWC), llm_client
│   ├── shared/             parser hàm Solidity, fusion, chọn solc theo pragma, định nghĩa phase, IO
│   ├── static_analysis/    static analyzer theo rule/AST cho Solidity
│   ├── rag/                retriever.py (TF-IDF), swc_agents.py (37 agent tín hiệu SWC), build_store.py
│   ├── kaggle/             notebook + runtime cho backend kaggle (CodeLlama, phương án dự phòng)
│   ├── tools/              batch_audit, benchmark, đánh giá detector local, manifest, strip_comments
│   ├── tests/              93 unit/integration test
│   ├── examples/           contract mẫu (VulnerableBank, SafeBank, …) cho test và web UI
│   ├── schemas/            JSON schema request/result của backend kaggle
│   ├── data_manifest.json  kích thước + SHA-256 của mọi data/model local (kiểm tra toàn vẹn)
│   ├── data/               (local, không push) — xem mục 4
│   ├── runs/               (local) mỗi lần audit một thư mục; web app đọc lịch sử từ đây
│   └── batch_runs/         (local) kết quả batch audit
│
├── server_train/           bundle chạy TRÊN GPU server (gpu_server.sh deploy rsync nguyên thư mục)
│   ├── serve_v13_api.py    API inference: /api/v1/health, /api/v1/detect, /api/v1/generate, /v1/chat/completions
│   ├── service.sh          start | stop | restart | status | logs (gọi run/stop/status_api_server.sh)
│   ├── train_qwen25_v13.py huấn luyện detector v13 (run_train.sh chạy nền)
│   ├── merge_lora_weights.py  gộp LoRA vào base model (vLLM/GGUF/AWQ)
│   ├── setup.sh            cài môi trường trên server
│   ├── final-dataset-v12-train.json   (local) dataset train/val/test của v13
│   └── train_v13_20261005_143623.log  (local) log lần train v13 cuối
│
├── webapp/                 app.py (FastAPI) + static/ (giao diện)
├── models/                 (local) qwen25-v13-14b-detector-final/ — adapter LoRA + threshold_config.json
└── docs/                   (local) bài báo LLM-BSCVM (PDF)
```

Những gì đánh dấu *(local)* nằm trong `.gitignore`: chỉ tồn tại trên máy này, không đẩy lên GitHub.

---

## 2. Pipeline

```text
Phase 0  Input        phân tích contract thành các unit (hàm/modifier), SHA-256 nguồn
Phase 1A Static       static analyzer + Slither native            ┐
Phase 1B RAG          TF-IDF trên 11,869 ca lỗi + 37 SWC agents   ├─ fusion theo từng finding
Phase 1C Detector     Qwen v13 trên GPU server (P(vulnerable))    ┘  (trọng số 0.65 / 0.20 / 0.15)
Phase 2  Advisor      nguyên nhân gốc + hướng sửa (RAG theo SWC)
Phase 3  Assessor     CVSS v3.1 → Critical/High/Medium/Low + thứ tự ưu tiên
Phase 4  Fixer        vá từng hàm theo ưu tiên
Phase 5  Verifier     compile (solc theo pragma) + static/RAG/detector chạy lại trên code đã vá
                      + ma trận SmartBugs + kiểm tra bảo toàn hành vi; tối đa 3 lần sửa
Phase 6  Reporter     báo cáo Markdown / HTML / PDF
```

Mỗi lần chạy ghi vào `E2E/runs/<run_id>/`: `phase0_input.json`, `phase1a_static.json`,
`phase1b_rag.json`, `phase1_fused.json`, `phase2_advisor.json`, `phase3_assessor.json`,
`phase4_fixer.json`, `phase5_verification.json`, `phase6_audit_report.{md,html,pdf}`, `manifest.json`.

Ngữ nghĩa kết quả:

- Verdict `Safe` chỉ khi mọi detector bắt buộc đều chạy xong; detector lỗi/thiếu → `Inconclusive` (fail-closed).
- `manifest.status=complete` chỉ có nghĩa mọi phase đã chạy; bản vá được chấp nhận hay không xem
  `phase5_verification.json → overall_verdict`.

Backend (`--backend`):

| Backend | Detector | Agents | Khi nào dùng |
| --- | --- | --- | --- |
| `server_gpu` (mặc định) | Qwen v13 trên GPU server | Qwen trên GPU server | chạy thật |
| `ollama` | không chạy | Advisor/Assessor qua Ollama local (`OLLAMA_MODEL`) | khi server tắt |
| `local` / `auto` / `local_fast` / `gemini` | không chạy | heuristic tất định, không gọi LLM | debug nhanh, offline |
| `kaggle` | CodeLlama adapter trên Kaggle | CodeLlama-Instruct trên Kaggle | dự phòng |

`GeminiClient` vẫn có trong `E2E/agents/llm_client.py` nhưng hiện không backend nào của
pipeline gọi tới (`--backend gemini` đi vào nhánh tất định như `local_fast`).

---

## 3. Cài đặt

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env            # điền SERVER_GPU_URL (và OLLAMA_MODEL nếu dùng --backend ollama)
```

Công cụ ngoài cho Phase 1/5:

- `solc-select` (+ các bản solc cần dùng) và `slither-analyzer` — pipeline tự chọn solc thấp nhất khớp pragma.
- SmartBugs tại `.tools/smartbugs` (hoặc đặt `SMARTBUGS_HOME`) + Docker cho ma trận mythril/smartcheck.

> Không bọc `.venv/bin/python` bằng `timeout`/`gtimeout` (binary x86_64 làm python chạy sai kiến trúc,
> lỗi `pydantic_core`). Gemini free tier chỉ ~20 request/ngày.

---

## 4. Dữ liệu và model (chỉ ở local)

Toàn bộ dataset nằm dưới `E2E/data/`; đường dẫn khai báo tại `E2E/config.py`.

| Đường dẫn | Nội dung | Dùng bởi |
| --- | --- | --- |
| `E2E/data/runtime/` | `knowledge_items.json` (11,869 ca), `swc_registry.json` (37 SWC), `swc_vectorizer.joblib`, `swc_matrix.npz`, `baseline_taxonomy.json`, `fusion_calibration.json` | RAG (Phase 1B), Advisor, fusion — **bắt buộc khi chạy** |
| `E2E/data/knowledge/final-vulnerability-knowledge.json` | kho tri thức lỗ hổng gốc (code lỗi + bản vá) | đầu vào để build lại `runtime/` |
| `E2E/data/contracts/smartbugs_wild/` | 47,398 contract SmartBugs-Wild đã bỏ comment (+ LICENSE gốc) | `run_batch.sh`, `E2E/tools/benchmark.py` |
| `E2E/data/eval/` | `iaudit_solodit.jsonl` + `results/` của benchmark 9 tool | `E2E/tools/benchmark_solodit_9tools.py` |
| `server_train/final-dataset-v12-train.json` | 10,356 hàm, split train/val/test (dappscan, iaudit_trustllm, kaggle_ds2) | train v13, `evaluate_local_detectors` |
| `models/qwen25-v13-14b-detector-final/` | adapter LoRA v13 + `threshold_config.json` | webapp, orchestrator (ngưỡng), GPU server |

Build lại RAG store (cho ra artifact giống hệt khi kho tri thức không đổi):

```bash
.venv/bin/python -m E2E.rag.build_store
```

Kiểm tra toàn vẹn / cập nhật manifest sau khi thay data:

```bash
.venv/bin/python -m E2E.tools.build_data_manifest
```

Nguồn gốc dữ liệu (tải lại nếu cần): DAppSCAN — github.com/InPlusLab/DAppSCAN;
iAudit/TrustLLM (Solodit) — sites.google.com/view/trustllm; SmartBugs-Wild — github.com/smartbugs/smartbugs-wild;
SWC Registry — swcregistry.io.

---

## 5. Chạy

**Web app**

```bash
./run_webapp.sh                       # http://127.0.0.1:8080 ; HOST=0.0.0.0 để mở LAN, RELOAD=1 khi dev
```

**CLI một contract**

```bash
.venv/bin/python -m E2E.run E2E/examples/VulnerableBank.sol                      # server_gpu
.venv/bin/python -m E2E.run Contract.sol --backend local_fast --run-id debug-1   # offline
.venv/bin/python -m E2E.run Contract.sol --tools slither mythril smartcheck      # ma trận Phase 5
```

Tham số hay dùng: `--mode {detector_only,agents_only,full}`, `--detector-threshold`,
`--max-fix-attempts 1..3`, `--full-matrix`, `--solidity-root` (contract có import).

**Batch audit cả corpus** (một contract một lúc, tự chờ server, dừng/tiếp tục bất kỳ lúc nào)

```bash
bash run_batch.sh start | stop | stop-now | status | log | export [N] | retry-failed
ACTIVE_HOURS=19:00-07:30 bash run_batch.sh start
```

Trạng thái ở `E2E/batch_runs/<batch>/state.sqlite`; kết quả `<batch>_functions.csv/.jsonl`.
Snapshot `<batch>_firstN_*.csv` được ghi một lần mỗi N contract; xoá đi để tiết kiệm dung lượng
thì không bị sinh lại — tạo lại khi cần bằng `bash run_batch.sh export N`.

**GPU server**

```bash
./gpu_server.sh start | stop | status | logs
./gpu_server.sh deploy                 # đẩy code server_train/ lên server, rồi stop + start
./gpu_server.sh pull                   # kéo adapter mới về models/ + log train v13
SERVER_IP=100.122.153.35 ./gpu_server.sh status
```

Huấn luyện lại detector: trên server, `cd ~/server_train && bash setup.sh` (một lần) rồi `bash run_train.sh`;
xong thì `./gpu_server.sh pull` ở local.

---

## 6. Đánh giá và kiểm thử

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s E2E/tests -t .   # 93 test
.venv/bin/python -m E2E.tools.benchmark                    # hồi quy static trên 3 contract SmartBugs
.venv/bin/python -m E2E.tools.evaluate_local_detectors     # static & RAG trên test split v12
.venv/bin/python -m E2E.tools.benchmark_solodit_9tools --split test --workers 4
```

Kết quả detector v13 trên test split (log `train_v13_20261005_143623.log`):

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
- Trọng số fusion (0.65/0.20/0.15) đặt tay, chưa fit trên tập hiệu chuẩn.
- SWC Registry không còn được cập nhật và không phủ lỗi business-logic/oracle/cross-chain.
- Bản vá do model sinh luôn cần review thủ công và test suite của dự án trước khi deploy.
