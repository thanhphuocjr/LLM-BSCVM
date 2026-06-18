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

## Phase 2 — Repair Suggestion (Advisor)

Cần cấu hình Gemini trong `.env` (đã có sẵn):

```
GEMINI_API_KEY=...
GEMINI_GENERATION_MODEL=gemini-2.5-flash
GEMINI_TEMPERATURE=0.2
```

Cài SDK (một lần): `pip install google-genai`

Chạy detection + repair suggestion trong một lệnh (static-only cho nhanh):

```bash
python3 phase2_repair_suggestion.py --code-file path/to/Contract.sol --run-detection --fast-detection
```

Chạy đầy đủ (bật CodeBERT + RAG ở phase 1):

```bash
python3 phase2_repair_suggestion.py --code-file path/to/Contract.sol --run-detection
```

Tái sử dụng kết quả detection đã lưu (không chạy lại phase 1):

```bash
python3 phase1_integrated_detector.py --code-file Contract.sol --output det.json
python3 phase2_repair_suggestion.py --code-file Contract.sol --detection-file det.json --output repair.json
```

Backend LLM mặc định là Gemini nhưng có thể thay (Ollama/CodeLlama) qua `agents/llm_client.py`
mà không phải sửa Advisor.
