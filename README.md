# Smart Contract Vulnerability RAG

Project được chia thành các khu vực chính:

- `static_analysis/`: phân tích tĩnh bằng rule/regex cho Solidity.
- `rag/`: pipeline RAG, TF-IDF retriever, knowledge store và multi-agent SWC retriever.
- `agents/`: các agent của framework LLM-BSCVM (Advisor/Repair Suggestion, ...) + LLM client.
- `codelama_results/`: script và output phân loại SWC bằng LLM/CodeLlama-style workflow.
- `dataset/`: toàn bộ dữ liệu cần thiết cho project.

## Pipeline theo paper LLM-BSCVM

- **Phase 1 — Detection (Detector):** `phase1_integrated_detector.py` — fusion 3 thành phần
  (static analysis + RAG + CodeBERT LoRA) ra verdict + danh sách lỗ hổng.
- **Phase 2 — Repair Suggestion (Advisor):** `phase2_repair_suggestion.py` — nhận kết quả
  detection, retrieve remediation knowledge theo SWC, rồi sinh repair suggestion 5 phần
  (root cause, impact, repair steps, fixed code, prevention) bằng LLM.

## Dataset

- `dataset/raw/solodit/solodit.json`: Solodit nguyên bản, được giữ lại.
- `dataset/raw/dappscan/dappscan.json`: Dappscan nguyên bản, được giữ lại.
- `dataset/swc_registry.json`: registry SWC dùng cho agent/query.
- `dataset/processed/`: dữ liệu đã xử lý hoặc làm giàu.
- `dataset/auxiliary/`: dữ liệu phụ trợ khi merge/build dataset.
- `dataset/tools/`: script xử lý dataset.

## Lệnh chính

Chạy multi-agent RAG retriever:

```bash
python3 rag/retrieve_best_vuln.py --code-file path/to/Contract.sol
```

Mặc định lệnh này chạy đủ 37 SWC agents nhưng chỉ trả về top 3 finding liên quan nhất. Nếu cần đổi số lượng finding:

```bash
python3 rag/retrieve_best_vuln.py --code-file path/to/Contract.sol --max-findings 5
```

Nếu cần debug toàn bộ 37 agent:

```bash
python3 rag/retrieve_best_vuln.py --code-file path/to/Contract.sol --include-all-agents
```

Chạy retriever một query kiểu cũ:

```bash
python3 rag/retrieve_best_vuln.py --single-query --query "Check reentrancy" --code-file path/to/Contract.sol
```

Rebuild knowledge store:

```bash
python3 rag/build_knowledge_store.py
```

Rebuild TF-IDF store:

```bash
python3 rag/build_tfidf_corpus.py
```
Link Demo: https://drive.google.com/file/d/1l1RCZ1RqAwCsNHVRsutYDUeycCbhCs2V/view?usp=drive_link

## Cấu hình LLM backend (cho các agent phase 2/3)

Chọn backend qua `.env` bằng biến `LLM_BACKEND`:

```
# Chọn 1 trong 2: ollama (local Qwen) | gemini (API)
LLM_BACKEND=ollama

# --- Ollama (chạy local, không tốn quota) ---
OLLAMA_MODEL=qwen2.5-coder:7b
# OLLAMA_HOST=http://localhost:11434
# OLLAMA_TEMPERATURE=0.2

# --- Gemini (API) ---
GEMINI_API_KEY=...
GEMINI_GENERATION_MODEL=gemini-2.5-flash
GEMINI_TEMPERATURE=0.2
```

Cài đặt:

```bash
pip install google-genai ollama          # client libs
ollama pull qwen2.5-coder:7b             # tải model về local (chỉ cần 1 lần)
```

Đảm bảo Ollama server đang chạy (`ollama serve`). Có thể ghi đè backend cho từng lần chạy
bằng `--backend ollama` / `--backend gemini` (mặc định `auto` = đọc từ `.env`).

Chạy giống phase 1 — không cần tham số, dùng `CODE_TO_TEST` mặc định:

```bash
python3 phase2_repair_suggestion.py                    # full pipeline (như phase 1)
python3 phase2_repair_suggestion.py --fast-detection   # static-only detection (nhanh)
python3 phase2_repair_suggestion.py --code-file Contract.sol
```

Tái sử dụng kết quả detection đã lưu (không chạy lại phase 1):

```bash
python3 phase1_integrated_detector.py --code-file Contract.sol --output det.json
python3 phase2_repair_suggestion.py --code-file Contract.sol --detection-file det.json --output repair.json
```

Backend LLM mặc định là Gemini nhưng có thể thay (Ollama/CodeLlama) qua `agents/llm_client.py`
mà không phải sửa Advisor.

## Phase 3 — Risk Assessment (Assessor)

Đánh giá rủi ro từng lỗ hổng theo chuẩn **CVSS v3.1** và hệ 4 mức (Critical/High/Medium/Low),
sinh ra phân bố rủi ro + thứ tự ưu tiên sửa (repair priority). Đầu vào là output của phase 2.

Chuỗi đầy đủ: `Detection (phase 1) -> Advisor (phase 2) -> Assessor (phase 3)`.

```bash
python3 phase3_risk_assessment.py                    # full pipeline trên CODE_TO_TEST
python3 phase3_risk_assessment.py --fast-detection   # static-only detection (nhanh)
python3 phase3_risk_assessment.py --code-file Contract.sol
```

Tái sử dụng output đã lưu để chỉ chạy lại Assessor:

```bash
python3 phase2_repair_suggestion.py --output repair.json
python3 phase3_risk_assessment.py --repair-file repair.json --output risk.json
```

> Lưu ý quota: Gemini free tier giới hạn ~20 request/ngày cho `gemini-2.5-flash`.
> Mỗi lỗ hổng tốn 1 request ở mỗi agent (Advisor, Assessor), nên một hợp đồng nhiều
> lỗ hổng có thể chạm giới hạn. Dùng `--fast-detection`, `--repair-file` để tiết kiệm,
> hoặc nâng cấp billing / đổi model. Client đã tự động backoff theo `retryDelay` của API.
> (Dùng `LLM_BACKEND=ollama` để chạy local, không dính quota.)

## Phase 4 — Vulnerability Repair (Fixer)

Sắp xếp lỗ hổng theo **repair priority** (từ phase 3), rồi sinh ra **toàn bộ contract đã
được sửa** trong một lần (holistic), kèm **unified diff** (Original vs Fixed). Đầu vào là
output của phase 2 (Advisor) + phase 3 (Assessor).

Chuỗi đầy đủ: `Detection (1) -> Advisor (2) -> Assessor (3) -> Fixer (4)`.

```bash
python3 phase4_vulnerability_repair.py                      # full chain trên CODE_TO_TEST
python3 phase4_vulnerability_repair.py --fast-detection     # detection static-only
python3 phase4_vulnerability_repair.py --no-assessor        # bỏ phase 3, ưu tiên theo severity
python3 phase4_vulnerability_repair.py --code-file C.sol --fixed-output Fixed.sol
```

Tái sử dụng output đã lưu (Fixer chỉ tốn 1 request LLM):

```bash
python3 phase3_risk_assessment.py --include-repair --output risk.json
python3 phase4_vulnerability_repair.py \
    --repair-file risk.json --risk-file risk.json --fixed-output Fixed.sol
```

Output gồm: `fixed_code` (contract hoàn chỉnh), `diff`, và `changes[]` (mỗi lỗ hổng:
priority, risk, tóm tắt thay đổi, đã xử lý chưa). Khác với Advisor (chỉ gợi ý từng đoạn),
Fixer tạo MỘT contract sửa hoàn chỉnh nên các finding trùng chỗ gộp thành một bản vá đúng.
